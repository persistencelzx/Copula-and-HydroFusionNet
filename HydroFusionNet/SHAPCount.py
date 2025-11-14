import time
from datetime import datetime
import warnings
import torch
import torch.nn as nn
import numpy as np
import shap
from torch.utils.data import DataLoader


class DeepSHAP:
    """计算和管理SHAP值的类 - 使用KernelExplainer稳定方案"""

    def __init__(self, model, device, train_dataset=None, feature_names=None):
        self.model = model
        self.device = device
        self.train_dataset = train_dataset
        self.feature_names = feature_names or ['NDVI', 'SM', 'TMP', 'ET', 'PRE', 'TMN', 'TMX']

        self.shap_data = {
            'features': [],
            'year_indices': [],
            'month_indices': [],
            'labels': [],
            'predictions': []
        }

    def collect_data_from_dataloader(self, dataloader, num_samples=2000):
        """从DataLoader中收集SHAP数据"""
        print(f"[{datetime.now()}] 开始从DataLoader收集SHAP数据...")
        start_time = time.time()

        collected_samples = 0
        self.model.eval()

        with torch.no_grad():
            for batch in dataloader:
                if collected_samples >= num_samples:
                    break

                features, labels, year_indices, month_indices, mask = batch
                batch_size = features.size(0)

                features_cpu = features.cpu().numpy()
                year_indices_cpu = year_indices.cpu().numpy()
                month_indices_cpu = month_indices.cpu().numpy()
                labels_cpu = labels.cpu().numpy()

                pred = self.model(features.to(self.device),
                                  year_indices.to(self.device),
                                  month_indices.to(self.device))
                pred_cpu = pred.cpu().numpy()

                self.shap_data['features'].append(features_cpu)
                self.shap_data['year_indices'].append(year_indices_cpu)
                self.shap_data['month_indices'].append(month_indices_cpu)
                self.shap_data['labels'].append(labels_cpu)
                self.shap_data['predictions'].append(pred_cpu)

                collected_samples += batch_size

        print(f"[{datetime.now()}] 收集了 {collected_samples} 个样本")
        print(f"[{datetime.now()}] 数据收集耗时: {time.time() - start_time:.2f} 秒")

    def compute_shap_values(self, sample_size=500, background_size=100, nsamples=100):
        """
        计算SHAP值（使用KernelExplainer - 稳定方案）

        Args:
            sample_size: 测试样本数量
            background_size: 背景样本数量
            nsamples: KernelExplainer采样次数（影响精度和速度）
        """
        print(f"[{datetime.now()}] 开始计算SHAP值...")
        start_time = time.time()

        if not self.shap_data['features']:
            print(f"[{datetime.now()}] 警告：没有收集到SHAP数据")
            return None

        try:
            # 合并数据
            print(f"[{datetime.now()}] 正在合并收集的数据...")
            all_features = np.concatenate(self.shap_data['features'], axis=0)
            all_year_indices = np.concatenate(self.shap_data['year_indices'], axis=0)
            all_month_indices = np.concatenate(self.shap_data['month_indices'], axis=0)
            all_labels = np.concatenate(self.shap_data['labels'], axis=0)
            all_predictions = np.concatenate(self.shap_data['predictions'], axis=0)

            print(f"[{datetime.now()}] 原始数据形状: features={all_features.shape}")

            # 展平为2D
            if all_features.ndim == 3:
                all_features = all_features.reshape(-1, all_features.shape[-1])
            all_year_indices = all_year_indices.flatten()
            all_month_indices = all_month_indices.flatten()
            all_labels = all_labels.flatten()
            all_predictions = all_predictions.flatten()

            n_samples = len(all_features)
            print(f"[{datetime.now()}] 总样本数: {n_samples}")

            if n_samples < 10:
                print(f"[{datetime.now()}] 数据样本太少，跳过SHAP计算")
                return None

            # 选择样本
            sample_size = min(sample_size, n_samples)
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)

            sample_features = all_features[sample_indices]
            sample_year_indices = all_year_indices[sample_indices]
            sample_month_indices = all_month_indices[sample_indices]

            # 获取背景数据
            if self.train_dataset is None:
                background_indices = np.random.choice(n_samples, min(background_size, n_samples), replace=False)
                background_features = all_features[background_indices]
                background_year_indices = all_year_indices[background_indices]
                background_month_indices = all_month_indices[background_indices]
            else:
                background_features, background_year_indices, background_month_indices = self._get_background_from_dataloader(
                    self.train_dataset, background_size
                )

            print(f"[{datetime.now()}] 背景数据: {background_features.shape}")
            print(f"[{datetime.now()}] 测试数据: {sample_features.shape}")

            # 使用KernelExplainer计算
            return self._compute_with_kernel_explainer(
                sample_features, sample_year_indices, sample_month_indices,
                background_features, background_year_indices, background_month_indices,
                all_labels, all_predictions, sample_indices, nsamples
            )

        except Exception as e:
            print(f"[{datetime.now()}] SHAP计算出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _get_background_from_dataloader(self, dataloader, background_size):
        """从DataLoader获取背景数据"""
        features_list = []
        year_list = []
        month_list = []
        collected = 0

        with torch.no_grad():
            for batch in dataloader:
                if collected >= background_size:
                    break

                features, _, year_indices, month_indices, _ = batch
                batch_size = features.size(0)

                features_list.append(features.cpu().numpy())
                year_list.append(year_indices.cpu().numpy())
                month_list.append(month_indices.cpu().numpy())

                collected += batch_size

        features = np.concatenate(features_list, axis=0)
        years = np.concatenate(year_list, axis=0)
        months = np.concatenate(month_list, axis=0)

        # 展平
        if features.ndim == 3:
            features = features.reshape(-1, features.shape[-1])
        years = years.flatten()
        months = months.flatten()

        indices = np.random.choice(len(features), min(background_size, len(features)), replace=False)

        return features[indices], years[indices], months[indices]

    def _compute_with_kernel_explainer(self, sample_features, sample_year_indices, sample_month_indices,
                                       background_features, background_year_indices, background_month_indices,
                                       all_labels, all_predictions, sample_indices, nsamples):
        """使用KernelExplainer计算SHAP值（按时间分组批量处理）"""
        print(f"[{datetime.now()}] 使用KernelExplainer计算SHAP值...")

        try:
            # 按时间分组
            unique_time_pairs = {}
            for i in range(len(sample_year_indices)):
                key = (int(sample_year_indices[i]), int(sample_month_indices[i]))
                if key not in unique_time_pairs:
                    unique_time_pairs[key] = []
                unique_time_pairs[key].append(i)

            print(f"[{datetime.now()}] 发现 {len(unique_time_pairs)} 个唯一时间组合")

            all_shap_values = []
            successful_count = 0

            for time_idx, (time_key, group_indices) in enumerate(unique_time_pairs.items()):
                year_val, month_val = time_key
                group_size = len(group_indices)

                print(f"[{datetime.now()}] 处理时间组 {time_idx + 1}/{len(unique_time_pairs)}: "
                      f"year={year_val}, month={month_val}, samples={group_size}")

                try:
                    # 创建当前时间组的预测函数
                    def predict_fn(features_2d):
                        """
                        预测函数，固定year和month
                        Args:
                            features_2d: (n_samples, n_features) numpy array
                        Returns:
                            predictions: (n_samples,) numpy array
                        """
                        n = len(features_2d)

                        # 转换为tensor
                        features_tensor = torch.FloatTensor(features_2d).to(self.device)
                        if features_tensor.ndim == 2:
                            features_tensor = features_tensor.unsqueeze(1)  # (n, 1, features)

                        year_tensor = torch.full((n,), year_val, dtype=torch.long, device=self.device)
                        month_tensor = torch.full((n,), month_val, dtype=torch.long, device=self.device)

                        # 模型预测
                        self.model.eval()
                        with torch.no_grad():
                            preds = self.model(features_tensor, year_tensor, month_tensor)
                            preds = torch.sigmoid(preds)  # 转换为概率

                        return preds.cpu().numpy()

                    # 准备当前组的数据
                    group_features = sample_features[group_indices]

                    # 创建KernelExplainer
                    explainer = shap.KernelExplainer(predict_fn, background_features)

                    # 计算SHAP值
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        group_shap_values = explainer.shap_values(
                            group_features,
                            nsamples=nsamples,
                            l1_reg='aic'  # 使用AIC正则化
                        )

                    # 存储结果（保持原始顺序）
                    for local_idx, global_idx in enumerate(group_indices):
                        all_shap_values.append((global_idx, group_shap_values[local_idx]))
                        successful_count += 1

                    print(f"[{datetime.now()}]   成功计算 {group_size} 个样本")

                except Exception as e:
                    print(f"[{datetime.now()}]   时间组计算失败: {str(e)}")
                    # 填充零向量
                    for global_idx in group_indices:
                        all_shap_values.append((global_idx, np.zeros(sample_features.shape[-1])))
                    continue

            if successful_count == 0:
                print(f"[{datetime.now()}] 所有样本计算失败")
                return None

            print(f"[{datetime.now()}] 成功计算 {successful_count}/{len(sample_features)} 个样本")

            # 按原始顺序重组
            all_shap_values.sort(key=lambda x: x[0])
            shap_values = np.array([item[1] for item in all_shap_values])

            print(f"[{datetime.now()}] SHAP值计算完成，形状: {shap_values.shape}")
            print(f"[{datetime.now()}] 总耗时: {time.time() - time.time():.2f} 秒")

            # 获取基准值
            try:
                year_val = int(sample_year_indices[0])
                month_val = int(sample_month_indices[0])

                def predict_fn_base(features_2d):
                    n = len(features_2d)
                    features_tensor = torch.FloatTensor(features_2d).to(self.device)
                    if features_tensor.ndim == 2:
                        features_tensor = features_tensor.unsqueeze(1)

                    year_tensor = torch.full((n,), year_val, dtype=torch.long, device=self.device)
                    month_tensor = torch.full((n,), month_val, dtype=torch.long, device=self.device)

                    self.model.eval()
                    with torch.no_grad():
                        preds = self.model(features_tensor, year_tensor, month_tensor)
                        preds = torch.sigmoid(preds)

                    return preds.cpu().numpy()

                base_value = float(np.mean(predict_fn_base(background_features)))

            except Exception as e:
                print(f"[{datetime.now()}] 获取基准值失败: {e}，使用0.5")
                base_value = 0.5

            return self._create_result_dict(
                shap_values, sample_features, sample_year_indices,
                sample_month_indices, all_labels, all_predictions,
                sample_indices, base_value
            )

        except Exception as e:
            print(f"[{datetime.now()}] KernelExplainer计算失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _create_result_dict(self, shap_values, sample_features, sample_year_indices,
                            sample_month_indices, all_labels, all_predictions,
                            sample_indices, base_value):
        """创建结果字典"""
        print(f"[{datetime.now()}] SHAP值形状: {shap_values.shape}")
        print(f"[{datetime.now()}] 基准值: {base_value}")

        return {
            'shap_values': shap_values,
            'features': sample_features,
            'year_indices': sample_year_indices,
            'month_indices': sample_month_indices,
            'labels': all_labels[sample_indices],
            'predictions': all_predictions[sample_indices],
            'feature_names': self.feature_names,
            'base_value': base_value
        }

    def save_shap_summary(self, shap_results, summary_file, watershed_name):
        """保存SHAP结果的简要统计信息"""
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"流域: {watershed_name}\n")
                f.write(f"SHAP值计算时间: {datetime.now()}\n")
                f.write("=" * 50 + "\n")

                if 'shap_values' in shap_results:
                    shap_values = shap_results['shap_values']
                    f.write(f"SHAP值形状: {shap_values.shape}\n")
                    f.write(f"样本数量: {shap_values.shape[0]}\n")
                    f.write(f"特征数量: {shap_values.shape[1]}\n")

                    feature_importance = np.abs(shap_values).mean(axis=0)
                    feature_names = shap_results.get('feature_names',
                                                     [f'Feature_{i}' for i in range(len(feature_importance))])

                    f.write("\n特征重要性排序:\n")
                    importance_pairs = list(zip(feature_names, feature_importance))
                    importance_pairs.sort(key=lambda x: x[1], reverse=True)

                    for i, (name, importance) in enumerate(importance_pairs):
                        f.write(f"{i + 1}. {name}: {importance:.4f}\n")

                    f.write(f"\n基准值: {shap_results.get('base_value', 'N/A')}\n")
                else:
                    f.write("SHAP值数据不可用\n")

            print(f"[{datetime.now()}] SHAP摘要已保存至: {summary_file}")

        except Exception as e:
            print(f"[{datetime.now()}] 保存SHAP摘要时出错: {str(e)}")

    def save_results(self, results, save_path):
        """保存SHAP结果到指定路径"""
        if results is not None:
            import pickle
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"[{datetime.now()}] SHAP结果已保存至 {save_path}")
        else:
            print(f"[{datetime.now()}] 没有SHAP结果可以保存")

    def clear_data(self):
        """清空收集的数据"""
        self.shap_data = {
            'features': [],
            'year_indices': [],
            'month_indices': [],
            'labels': [],
            'predictions': []
        }