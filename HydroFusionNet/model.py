import os
import warnings
from Auxiliary_items import EarlyStopping, Metrics
from SHAPCount import DeepSHAP
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import h5py

warnings.filterwarnings('ignore')

class TimeEmbedding(nn.Module):
    def __init__(self, year_dim, month_dim, d_model):
        super(TimeEmbedding, self).__init__()
        self.year_embedding = nn.Embedding(year_dim, d_model // 2)
        self.month_embedding = nn.Embedding(month_dim, d_model // 2)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, year_indices, month_indices):
        year_indices = year_indices.view(-1)
        month_indices = month_indices.view(-1)
        year_emb = self.year_embedding(year_indices)
        month_emb = self.month_embedding(month_indices)
        time_emb = torch.cat([year_emb, month_emb], dim=-1)
        return self.proj(time_emb)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout_rate=0.3):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout_rate)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class EnhancedTimeSeriesModel(nn.Module):
    def __init__(self, input_features=9, hidden_size=128, num_layers=2, heads=8,
                 dropout_rate=0.3, year_dim=15, month_dim=12):
        super(EnhancedTimeSeriesModel, self).__init__()

        self.time_embedding = TimeEmbedding(
            year_dim=year_dim,
            month_dim=month_dim,
            d_model=hidden_size // 4
        )

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 4, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.gru = nn.GRU(
            hidden_size,
            hidden_size // 2,
            num_layers=1,
            batch_first=True,
            dropout=0
        )

        self.num_transformer_layers = max(1, num_layers - 1)
        transformer_heads = max(1, (hidden_size // 2) // 8)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=hidden_size // 2,
                heads=transformer_heads,
                dropout_rate=dropout_rate
            )
            for _ in range(self.num_transformer_layers)
        ])

        self.attention_weights = nn.Linear(hidden_size // 2, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 1.5),
            nn.Linear(hidden_size // 4, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, src, year_indices, month_indices):
        if src.dim() == 2:
            src = src.unsqueeze(1)

        batch_size, seq_len, feature_dim = src.shape

        if year_indices.dim() == 1:
            year_indices = year_indices.unsqueeze(1).expand(batch_size, seq_len)
        elif year_indices.dim() == 2 and year_indices.size(1) == 1:
            year_indices = year_indices.expand(batch_size, seq_len)

        if month_indices.dim() == 1:
            month_indices = month_indices.unsqueeze(1).expand(batch_size, seq_len)
        elif month_indices.dim() == 2 and month_indices.size(1) == 1:
            month_indices = month_indices.expand(batch_size, seq_len)

        src_flat = src.reshape(-1, feature_dim)
        features = self.feature_extractor(src_flat)

        year_flat = year_indices.reshape(-1)
        month_flat = month_indices.reshape(-1)
        time_emb = self.time_embedding(year_flat, month_flat)

        combined = torch.cat([features, time_emb], dim=-1)
        fused = self.fusion(combined)
        fused = fused.view(batch_size, seq_len, -1)

        gru_out, _ = self.gru(fused)

        transformer_out = gru_out.transpose(0, 1)
        for transformer_layer in self.transformer_layers:
            transformer_out = transformer_layer(transformer_out)
        transformer_out = transformer_out.transpose(0, 1)

        if seq_len > 1:
            attn_weights = torch.softmax(self.attention_weights(transformer_out), dim=1)
            pooled = torch.sum(transformer_out * attn_weights, dim=1)
        else:
            pooled = transformer_out.squeeze(1)

        output = self.classifier(pooled)
        return output.squeeze(-1)

class DifficultyAwareFocalLoss:
    """难度感知的Focal Loss"""

    def __init__(self, pos_weight=None, alpha=0.5, gamma=1.0, label_smoothing=0.02):
        self.pos_weight = pos_weight
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.easy_weight = 0.7
        self.hard_weight = 1.2

    def compute_loss(self, pred, labels, mask, model=None):
        pred = pred.view(-1)
        labels = labels.view(-1)
        mask = mask.view(-1).bool()
        pred = pred[mask]
        labels = labels[mask]

        if len(pred) == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)

        labels_smooth = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce_loss = F.binary_cross_entropy_with_logits(pred, labels_smooth, reduction='none')

        p_t = torch.sigmoid(pred)
        p_t = torch.where(labels > 0.5, p_t, 1 - p_t)

        difficulty = 1 - 2 * torch.abs(p_t - 0.5)
        sample_weight = self.easy_weight + (self.hard_weight - self.easy_weight) * difficulty
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = torch.where(labels > 0.5, self.alpha, 1 - self.alpha)

        loss = alpha_t * focal_weight * sample_weight * bce_loss
        return loss.mean()

class ModelInitializer:
    def __init__(self, best_params, device, script_dir):
        self.best_params = best_params
        self.device = device
        self.script_dir = script_dir

    def init_end_to_end_model(self, input_features=9, year_dim=15):
        model = EnhancedTimeSeriesModel(
            input_features=input_features,
            hidden_size=self.best_params.get('hidden_size', 128),
            num_layers=self.best_params.get('num_layers', 2),
            heads=self.best_params.get('heads', 4),
            dropout_rate=self.best_params.get('dropout_rate', 0.3),
            year_dim=year_dim,
            month_dim=12
        ).to(self.device)
        return model

    def init_optimizer_scheduler(self, model):
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.best_params.get('lr', 0.0005),
            weight_decay=self.best_params.get('weight_decay', 5e-4)
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=1e-6
        )
        return optimizer, scheduler

class ModelTrainer:
    def __init__(self, model, optimizer, scheduler, device, model_save_dir, best_params, loss_function):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_save_dir = model_save_dir
        self.loss_function = loss_function
        self.early_stopping = EarlyStopping(
            patience=best_params.get('patience', 40),
            max_epochs=2000
        )
        self.metrics = Metrics()

    def _process_batch(self, batch, mode='train'):
        features, labels, year_indices, month_indices, mask = batch
        features = features.to(self.device)
        labels = labels.to(self.device)
        year_indices = year_indices.to(self.device)
        month_indices = month_indices.to(self.device)
        mask = mask.to(self.device)

        if mode == 'train':
            self.optimizer.zero_grad()

        with torch.no_grad() if mode != 'train' else torch.enable_grad():
            pred = self.model(features, year_indices, month_indices)
            if pred.dim() > 1:
                pred = pred.squeeze(-1)

            loss = self.loss_function.compute_loss(pred, labels, mask, model=self.model if mode == 'train' else None)

            if mode == 'train' and loss.requires_grad:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

        pred_np = torch.sigmoid(pred).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()

        valid_pred = pred_np[mask_np.astype(bool)]
        valid_labels = labels_np[mask_np.astype(bool)]

        return loss.item(), valid_pred, valid_labels

    def _evaluate_on_loader(self, data_loader, mode='val'):
        self.model.eval()
        losses, valid_preds, valid_labels = [], [], []

        with torch.no_grad():
            for batch in data_loader:
                loss, valid_pred, valid_label = self._process_batch(batch, mode=mode)
                losses.append(loss)
                valid_preds.append(valid_pred)
                valid_labels.append(valid_label)

        valid_preds = np.concatenate(valid_preds)
        valid_labels = np.concatenate(valid_labels)

        accuracy = accuracy_score(valid_labels > 0.5, valid_preds > 0.5)
        precision = precision_score(valid_labels > 0.5, valid_preds > 0.5, zero_division=0)
        recall = recall_score(valid_labels > 0.5, valid_preds > 0.5, zero_division=0)
        f1 = f1_score(valid_labels > 0.5, valid_preds > 0.5, zero_division=0)

        return {
            'loss': np.mean(losses),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def train(self, train_loader, val_loader, watershed_name):
        print(f"开始训练流域: {watershed_name}")

        for epoch in range(1, self.early_stopping.max_epochs + 1):
            self.model.train()
            train_losses, train_valid_preds, train_valid_labels = [], [], []

            for batch in train_loader:
                loss, valid_pred, valid_label = self._process_batch(batch, mode='train')
                train_losses.append(loss)
                train_valid_preds.append(valid_pred)
                train_valid_labels.append(valid_label)

            train_valid_preds = np.concatenate(train_valid_preds)
            train_valid_labels = np.concatenate(train_valid_labels)
            train_f1 = f1_score(train_valid_labels > 0.5, train_valid_preds > 0.5, zero_division=0)
            train_recall = recall_score(train_valid_labels > 0.5, train_valid_preds > 0.5, zero_division=0)

            val_metrics = self._evaluate_on_loader(val_loader, mode='val')

            metrics_dict = {
                'train_loss': np.mean(train_losses),
                'train_f1': train_f1,
                'train_recall': train_recall,
                'val_loss': val_metrics['loss'],
                'val_f1': val_metrics['f1'],
                'val_accuracy': val_metrics['accuracy'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall']
            }

            self.metrics.record_metrics(epoch, metrics_dict)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train F1: {train_f1:.4f}, Train Recall: {train_recall:.4f}, "
                      f"Val F1: {val_metrics['f1']:.4f}, Val Recall: {val_metrics['recall']:.4f}")

            early_stop_metric = val_metrics['f1']
            self.early_stopping(early_stop_metric, self.model, epoch)

            if self.early_stopping.early_stop:
                print(f"早停触发于第 {epoch} 轮")
                break

            self.scheduler.step()

        self.model = self.early_stopping.restore_best_weights(self.model)

        save_path = os.path.join(self.model_save_dir, f'{watershed_name}_model.pth')
        torch.save({'model_state_dict': self.model.state_dict()}, save_path)

        metrics_file_path = os.path.join(self.model_save_dir, f'{watershed_name}_metrics.xlsx')
        self.metrics.save_to_excel(metrics_file_path)

    def evaluate(self, test_loader, watershed_name):
        print(f"评估流域 {watershed_name}...")
        test_metrics = self._evaluate_on_loader(test_loader, mode='test')

        print(f"Test F1: {test_metrics['f1']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")

        return test_metrics

class TimeSeriesAugmentation:
    """时序数据增强"""

    @staticmethod
    def temporal_jitter(X, y, years, months, strength=0.008):
        pos_idx = np.where(y > 0.5)[0]
        if len(pos_idx) == 0:
            return X, y, years, months

        n_augment = max(1, int(len(pos_idx) * 0.15))
        selected_idx = np.random.choice(pos_idx, n_augment, replace=False)

        augmented_X = []
        for idx in selected_idx:
            window = X[idx].copy()
            noise = np.random.normal(0, strength, window.shape)
            augmented_X.append(window + noise)

        if augmented_X:
            return (
                np.vstack([X, np.array(augmented_X)]),
                np.hstack([y, np.ones(len(augmented_X))]),
                np.hstack([years, years[selected_idx]]),
                np.hstack([months, months[selected_idx]])
            )
        return X, y, years, months

    @staticmethod
    def temporal_mixup(X, y, years, months, alpha=0.3):
        pos_idx = np.where(y > 0.5)[0]
        if len(pos_idx) < 2:
            return X, y, years, months

        n_augment = max(1, int(len(pos_idx) * 0.10))
        augmented_X = []
        augmented_years = []
        augmented_months = []

        for _ in range(n_augment):
            idx1, idx2 = np.random.choice(pos_idx, 2, replace=False)
            lam = np.random.beta(alpha, alpha)

            mixed = lam * X[idx1] + (1 - lam) * X[idx2]
            augmented_X.append(mixed)

            if lam > 0.5:
                augmented_years.append(years[idx1])
                augmented_months.append(months[idx1])
            else:
                augmented_years.append(years[idx2])
                augmented_months.append(months[idx2])

        if augmented_X:
            return (
                np.vstack([X, np.array(augmented_X)]),
                np.hstack([y, np.ones(len(augmented_X))]),
                np.hstack([years, np.array(augmented_years)]),
                np.hstack([months, np.array(augmented_months)])
            )
        return X, y, years, months

class DataProcessor:
    """时序数据处理器"""

    def __init__(self, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    def process_watershed_data(self, features, labels, years, months,
                               window_size=3, stride=2,
                               augment_positive=True, noise_level=0.008):

        X_seq, y_seq, year_seq, month_seq = self._create_time_windows_dynamic_stride(
            features, labels, years, months, window_size, stride
        )

        train_idx, val_idx, test_idx = self._temporal_split(year_seq, month_seq)

        X_train, y_train = X_seq[train_idx], y_seq[train_idx]
        year_train, month_train = year_seq[train_idx], month_seq[train_idx]

        X_val, y_val = X_seq[val_idx], y_seq[val_idx]
        year_val, month_val = year_seq[val_idx], month_seq[val_idx]

        X_test, y_test = X_seq[test_idx], y_seq[test_idx]
        year_test, month_test = year_seq[test_idx], month_seq[test_idx]

        if augment_positive:
            augmenter = TimeSeriesAugmentation()
            X_train, y_train, year_train, month_train = augmenter.temporal_jitter(
                X_train, y_train, year_train, month_train, strength=noise_level
            )
            X_train, y_train, year_train, month_train = augmenter.temporal_mixup(
                X_train, y_train, year_train, month_train, alpha=0.3
            )

        X_train, y_train, year_train, month_train = self._light_undersampling(
            X_train, y_train, year_train, month_train, max_ratio=4.5
        )

        train_loader = self._create_dataloader(X_train, y_train, year_train, month_train,
                                               batch_size=32, shuffle=True, weighted=True)
        val_loader = self._create_dataloader(X_val, y_val, year_val, month_val,
                                             batch_size=32, shuffle=False, weighted=False)
        test_loader = self._create_dataloader(X_test, y_test, year_test, month_test,
                                              batch_size=32, shuffle=False, weighted=False)

        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'stats': {
                'train_size': len(X_train),
                'train_pos': int(np.sum(y_train > 0.5)),
                'val_size': len(X_val),
                'val_pos': int(np.sum(y_val > 0.5)),
                'test_size': len(X_test),
                'test_pos': int(np.sum(y_test > 0.5))
            }
        }

    def _create_time_windows_dynamic_stride(self, features, labels, years, months, window_size, stride):
        time_idx = years * 12 + months
        sorted_idx = np.argsort(time_idx)

        features_sorted = features[sorted_idx]
        labels_sorted = labels[sorted_idx]
        years_sorted = years[sorted_idx]
        months_sorted = months[sorted_idx]
        time_sorted = time_idx[sorted_idx]

        if window_size <= 1:
            return features_sorted[:, np.newaxis, :], labels_sorted, years_sorted, months_sorted

        X_windows, y_windows, year_windows, month_windows = [], [], [], []

        i = 0
        while i < len(features_sorted) - window_size + 1:
            window_times = time_sorted[i:i + window_size]
            time_diffs = np.diff(window_times)

            if np.all(time_diffs <= 2) and np.all(time_diffs >= 0):
                window = features_sorted[i:i + window_size]
                label = labels_sorted[i + window_size - 1]

                X_windows.append(window)
                y_windows.append(label)
                year_windows.append(years_sorted[i + window_size - 1])
                month_windows.append(months_sorted[i + window_size - 1])
                i += stride
            else:
                i += 1

        return (np.array(X_windows), np.array(y_windows),
                np.array(year_windows), np.array(month_windows))

    def _temporal_split(self, years, months):
        n_samples = len(years)
        time_idx = years * 12 + months
        sorted_idx = np.argsort(time_idx)

        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))

        return sorted_idx[:train_end], sorted_idx[train_end:val_end], sorted_idx[val_end:]

    def _light_undersampling(self, X, y, years, months, max_ratio=4.5):
        y_binary = (y > 0.5).astype(int)
        pos_count = np.sum(y_binary == 1)
        neg_count = np.sum(y_binary == 0)

        current_ratio = neg_count / (pos_count + 1e-8)

        if current_ratio <= 2.0:
            print(f"    正负比例已平衡 ({current_ratio:.2f}:1)，跳过负采样")
            return X, y, years, months

        if current_ratio <= max_ratio:
            return X, y, years, months

        target_neg = int(pos_count * max_ratio)
        neg_idx = np.where(y_binary == 0)[0]
        pos_idx = np.where(y_binary == 1)[0]

        keep_neg = np.random.choice(neg_idx, target_neg, replace=False)
        keep_idx = np.concatenate([pos_idx, keep_neg])
        keep_idx = np.sort(keep_idx)

        print(f"    负采样: {len(X)} -> {len(keep_idx)} (比例 {current_ratio:.1f}:1 -> {max_ratio:.1f}:1)")

        return X[keep_idx], y[keep_idx], years[keep_idx], months[keep_idx]

    def _create_dataloader(self, X, y, years, months, batch_size, shuffle, weighted):
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(years, dtype=torch.long),
            torch.tensor(months, dtype=torch.long)
        )

        if weighted and shuffle:
            sampler = self._create_weighted_sampler(y)
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                              collate_fn=self._collate_fn)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                              collate_fn=self._collate_fn)

    def _create_weighted_sampler(self, labels):
        y_binary = (labels > 0.5).astype(int)
        pos_count = np.sum(y_binary == 1)
        neg_count = np.sum(y_binary == 0)

        if pos_count == 0 or neg_count == 0:
            return WeightedRandomSampler(torch.ones(len(labels)), len(labels), replacement=True)

        pos_ratio = pos_count / len(labels)

        if pos_ratio >= 0.3:
            pos_weight = 1.2
        else:
            pos_weight = (neg_count / pos_count) ** 0.4
            pos_weight = min(pos_weight, 2.0)

        weights = np.where(y_binary == 1, pos_weight, 1.0)
        return WeightedRandomSampler(torch.FloatTensor(weights), len(weights), replacement=True)

    @staticmethod
    def _collate_fn(batch):
        features, labels, years, months = zip(*batch)
        return (
            torch.stack(features, dim=0),
            torch.stack(labels, dim=0),
            torch.stack(years, dim=0),
            torch.stack(months, dim=0),
            torch.ones(len(labels), dtype=torch.bool)
        )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model_save_dir = './模型保存'
    os.makedirs(model_save_dir, exist_ok=True)

    h5_path = os.path.join(os.getcwd(), "model_data", "watershed_model_data.h5")

    with h5py.File(h5_path, 'r') as f:
        for watershed_name in f.keys():
            print(f"\n{'=' * 70}\n处理流域: {watershed_name}\n{'=' * 70}")

            group = f[watershed_name]
            features = group['features'][:]
            labels = group['targets'][:].astype(np.float32)
            years = group['years'][:]
            months = group['months'][:]

            unique_years = np.unique(years)
            year_mapping = {year: idx for idx, year in enumerate(sorted(unique_years))}
            year_indices = np.array([year_mapping[y] for y in years])
            month_indices = months - 1
            train_features = torch.FloatTensor(features[:, np.newaxis, :])
            train_labels = torch.FloatTensor(labels)
            train_years = torch.LongTensor(year_indices)
            train_months = torch.LongTensor(month_indices)

            train_dataset = TensorDataset(train_features, train_labels, train_years, train_months)
            train_loader = DataLoader(
                train_dataset,
                batch_size=32,
                shuffle=True,
                collate_fn=lambda batch: (
                    torch.stack([b[0] for b in batch]),
                    torch.stack([b[1] for b in batch]),
                    torch.stack([b[2] for b in batch]),
                    torch.stack([b[3] for b in batch]),
                    torch.ones(len(batch), dtype=torch.bool)
                )
            )
            n_samples = len(features)
            val_size = int(n_samples * 0.98)
            val_indices = np.random.choice(n_samples, val_size, replace=False)

            val_features = torch.FloatTensor(features[val_indices][:, np.newaxis, :])
            val_labels = torch.FloatTensor(labels[val_indices])
            val_years = torch.LongTensor(year_indices[val_indices])
            val_months = torch.LongTensor(month_indices[val_indices])

            val_dataset = TensorDataset(val_features, val_labels, val_years, val_months)
            val_loader = DataLoader(
                val_dataset,
                batch_size=32,
                shuffle=False,
                collate_fn=lambda batch: (
                    torch.stack([b[0] for b in batch]),
                    torch.stack([b[1] for b in batch]),
                    torch.stack([b[2] for b in batch]),
                    torch.stack([b[3] for b in batch]),
                    torch.ones(len(batch), dtype=torch.bool)
                )
            )
            np.random.seed(43)
            test_indices = np.random.choice(n_samples, val_size, replace=False)

            test_features = torch.FloatTensor(features[test_indices][:, np.newaxis, :])
            test_labels = torch.FloatTensor(labels[test_indices])
            test_years = torch.LongTensor(year_indices[test_indices])
            test_months = torch.LongTensor(month_indices[test_indices])

            test_dataset = TensorDataset(test_features, test_labels, test_years, test_months)
            test_loader = DataLoader(
                test_dataset,
                batch_size=32,
                shuffle=False,
                collate_fn=lambda batch: (
                    torch.stack([b[0] for b in batch]),
                    torch.stack([b[1] for b in batch]),
                    torch.stack([b[2] for b in batch]),
                    torch.stack([b[3] for b in batch]),
                    torch.ones(len(batch), dtype=torch.bool)
                )
            )

            # 统计信息
            train_pos = int(labels.sum())
            val_pos = int(labels[val_indices].sum())
            test_pos = int(labels[test_indices].sum())
            best_params = {
                'hidden_size': 128,
                'num_layers': 2,
                'heads': 4,
                'dropout_rate': 0.3,
                'lr': 0.0005,
                'weight_decay': 5e-4,
                'patience': 40
            }

            model_initializer = ModelInitializer(best_params, device, os.getcwd())
            model = model_initializer.init_end_to_end_model(
                input_features=9,
                year_dim=len(unique_years)
            )
            optimizer, scheduler = model_initializer.init_optimizer_scheduler(model)

            pos_ratio = train_pos / len(features)
            pos_weight = 1.0 if pos_ratio >= 0.25 else min((len(features) - train_pos) / (train_pos + 1e-8) ** 0.5, 2.0)

            loss_function = DifficultyAwareFocalLoss(
                pos_weight=pos_weight,
                alpha=0.50,
                gamma=1.0,
                label_smoothing=0.02
            )

            trainer = ModelTrainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                model_save_dir=model_save_dir,
                best_params=best_params,
                loss_function=loss_function
            )
            trainer.train(train_loader, val_loader, watershed_name)
            test_metrics = trainer.evaluate(test_loader, watershed_name)
            print(f"\n{'=' * 70}\n开始SHAP分析 - {watershed_name}\n{'=' * 70}")

            shap_save_dir = os.path.join(model_save_dir, 'shap_results')
            os.makedirs(shap_save_dir, exist_ok=True)

            try:
                shap_calculator = DeepSHAP(
                    model=model,
                    device=device,
                    train_dataset=train_loader,
                    feature_names=['NDVI', 'SM', 'TMP', 'ET', 'PRE', 'TMN', 'TMX']
                )

                print(f"收集测试数据用于SHAP分析...")
                shap_calculator.collect_data_from_dataloader(test_loader, num_samples=2000)

                print(f"计算SHAP值...")
                shap_results = shap_calculator.compute_shap_values(
                    sample_size=1000,  # 测试样本数
                    background_size=500,  # 背景样本数
                    nsamples=100
                )

                if shap_results:
                    shap_file = os.path.join(shap_save_dir, f'{watershed_name}_shap.pkl')
                    summary_file = os.path.join(shap_save_dir, f'{watershed_name}_summary.txt')

                    shap_calculator.save_results(shap_results, shap_file)
                    shap_calculator.save_shap_summary(shap_results, summary_file, watershed_name)

                    print(f"SHAP分析完成！")
                    print(f"  - SHAP值: {shap_file}")
                    print(f"  - 摘要: {summary_file}")
                else:
                    print(f"SHAP计算失败，跳过")

                shap_calculator.clear_data()

            except Exception as e:
                print(f"SHAP分析出错: {str(e)}")
                import traceback
                traceback.print_exc()
                print(f"继续执行后续流程...")

            print(f"{'=' * 70}\n")
            print(f"最终结果 - {watershed_name}")
            print(f"F1: {test_metrics['f1']:.4f}, Acc: {test_metrics['accuracy']:.4f}")
            print(f"Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}")
            print(f"{'=' * 70}\n")

if __name__ == "__main__":
    main()
