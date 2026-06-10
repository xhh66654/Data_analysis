"""
模型验证工具模块。

为三个代理模型提供损失函数计算和拟合程度评估功能。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


class ModelValidationResult:
    """模型验证结果容器。"""

    def __init__(self):
        self.train_loss: float = 0.0
        self.val_loss: float = 0.0
        self.train_metrics: Dict[str, float] = {}
        self.val_metrics: Dict[str, float] = {}
        self.sample_count: int = 0
        self.train_sample_count: int = 0
        self.val_sample_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式输出。"""
        return {
            "sample_count": self.sample_count,
            "train_sample_count": self.train_sample_count,
            "val_sample_count": self.val_sample_count,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
        }


def split_data_for_validation(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    划分训练集和验证集。

    参数:
        X: 特征矩阵。
        y: 标签数组。
        val_size: 验证集比例。
        random_state: 随机种子。

    返回:
        X_train, X_val, y_train, y_val
    """
    return train_test_split(X, y, test_size=val_size, random_state=random_state)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    评估分类模型性能。

    参数:
        y_true: 真实标签。
        y_pred: 预测标签。
        sample_weight: 样本权重（可选）。

    返回:
        包含准确率和 F1 分数的字典。
    """
    accuracy = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    f1 = f1_score(y_true, y_pred, average="weighted", sample_weight=sample_weight)

    return {
        "accuracy": accuracy,
        "f1_weighted": f1,
    }


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    评估回归模型性能。

    参数:
        y_true: 真实值。
        y_pred: 预测值。
        sample_weight: 样本权重（可选）。

    返回:
        包含 MSE、MAE 和 R² 的字典。
    """
    mse = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
    mae = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight)

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
    }


def compute_classification_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """
    计算分类损失（交叉熵）。

    参数:
        y_true: 真实标签。
        y_pred: 预测标签。
        sample_weight: 样本权重（可选）。

    返回:
        交叉熵损失值。
    """
    # 计算每个样本的交叉熵
    n_classes = len(np.unique(y_true))
    if n_classes <= 2:
        # 二分类
        p = np.where(y_pred == y_true, 1.0, 0.0)
    else:
        # 多分类（简化版，使用预测正确的概率）
        p = np.where(y_pred == y_true, 1.0, 1e-15)

    loss = -np.log(p)
    if sample_weight is not None:
        loss = loss * sample_weight

    return float(np.mean(loss))


def compute_regression_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """
    计算回归损失（均方误差）。

    参数:
        y_true: 真实值。
        y_pred: 预测值。
        sample_weight: 样本权重（可选）。

    返回:
        MSE 损失值。
    """
    return float(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))