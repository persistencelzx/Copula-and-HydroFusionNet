import os
import math
import warnings
from tqdm import tqdm
# ================== 科学计算库 ==================
import numpy as np
import pandas as pd
from scipy import stats
import pywt
from scipy.interpolate import LinearNDInterpolator, griddata, interp1d
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import norm, genextreme, multivariate_normal, multivariate_t, kendalltau, t

from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import joblib
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import warnings

# ================== PyTorch 生态系统 ==================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn.utils.prune as prune

# ================== 机器学习工具 ==================
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, root_mean_squared_error, explained_variance_score, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
import pickle

# ================== 可视化工具 ==================
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf, pacf
from mpl_toolkits.mplot3d import Axes3D
import shap

# ================== 实用工具 ==================
import joblib
from joblib import dump
import h5py
from scipy.optimize import minimize

class EarlyStopping:
    def __init__(self, patience, min_delta=1e-3, factor=1.2, max_epochs=2000, verbose=False):
        self.initial_patience = patience
        self.patience = patience
        self.min_delta = min_delta
        self.factor = factor
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.counter = 0
        self.best_val_f1 = -float('inf')
        self.early_stop = False
        self.best_model_wts = None

    def reset(self):
        self.patience = self.initial_patience
        self.counter = 0
        self.best_val_f1 = -float('inf')
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_f1, model, epoch):
        if val_f1 < self.best_val_f1 + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_val_f1 = val_f1
            self.best_model_wts = model.state_dict()
            self.counter = 0
            self.patience = int(self.initial_patience * self.factor)

        if epoch >= self.max_epochs:
            self.early_stop = True

    def restore_best_weights(self, model):
        if self.best_model_wts is not None:
            model.load_state_dict(self.best_model_wts)
        return model

class Metrics:
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.train_f1 = []
        self.val_losses = []
        self.val_f1 = []
        self.val_accuracy = []
        self.val_precision = []
        self.val_recall = []

    def record_metrics(self, epoch, metrics_dict):
        self.epochs.append(epoch)
        self.train_losses.append(metrics_dict['train_loss'])
        self.train_f1.append(metrics_dict['train_f1'])
        self.val_losses.append(metrics_dict['val_loss'])
        self.val_f1.append(metrics_dict['val_f1'])
        self.val_accuracy.append(metrics_dict['val_accuracy'])
        self.val_precision.append(metrics_dict['val_precision'])
        self.val_recall.append(metrics_dict['val_recall'])

    def save_to_excel(self, file_path):
        data = {
            'Epoch': self.epochs,
            'Train Loss': self.train_losses,
            'Train F1': self.train_f1,
            'Val Loss': self.val_losses,
            'Val F1': self.val_f1,
            'Val Accuracy': self.val_accuracy,
            'Val Precision': self.val_precision,
            'Val Recall': self.val_recall
        }
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)


# ssh -p 40514 root@connect.nmb2.seetacloud.com