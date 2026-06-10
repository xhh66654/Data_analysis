"""
根据样本量与标签复杂度，自适应 VIPER / CART 训练超参。
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def resolve_train_params(
    n_samples: int,
    n_classes: int,
    *,
    action_item: Optional[str] = None,
    max_depth: int = 6,
    min_samples_leaf: int = 2,
    n_iters: int = 5,
    resample_augment: bool = True,
    penalty_factor: float = 2.0,
) -> Dict[str, Any]:
    """
    在调用方传入的基线参数上，按数据规模做保守放大或关闭增广。

    整体决策模式（``action_item`` 为 ``None``，一步 decision 为一个类）且类数多、
    样本量大时：略加深树、增大叶节点样本以减少过拟合；样本 >= 1500 时关闭
    重采样增广，避免 VIPER 在扩增集上过拟合。

    参数:
        n_samples: 训练样本数。
        n_classes: 动作类别数。
        action_item: 动作项名称；``None`` 表示整体决策模式。
        max_depth: 基线决策树最大深度。
        min_samples_leaf: 基线叶节点最少样本数。
        n_iters: 基线 VIPER 迭代轮数。
        resample_augment: 基线是否启用重采样增广。
        penalty_factor: VIPER 错误样本惩罚倍率。

    返回:
        调整后的超参字典，含 ``max_depth``、``min_samples_leaf``、
        ``n_iters``、``resample_augment``、``penalty_factor``、
        ``uniform_base_weights`` 等键。
    """
    is_full_label = action_item is None
    depth = max_depth
    leaf = min_samples_leaf
    iters = n_iters
    augment = resample_augment
    uniform_weights = False

    if is_full_label:
        if n_samples >= 300:
            depth = max(depth, min(14, 8 + n_classes // 2))
            leaf = max(leaf, max(4, n_samples // 1000))
        if n_classes >= 8:
            depth = max(depth, 12)
        if n_samples >= 2000:
            depth = max(depth, 12)
            leaf = min(leaf, max(4, n_samples // 1000))
        if n_samples >= 1500:
            augment = False
            iters = min(iters, 4)
        if n_samples >= 2000:
            uniform_weights = True
            iters = min(iters, 3)
    else:
        if n_samples >= 500:
            depth = max(depth, 8)
            leaf = max(leaf, max(5, n_samples // 600))
        if n_samples >= 1500:
            augment = False
            iters = min(iters, 4)

    return {
        "max_depth": int(depth),
        "min_samples_leaf": int(leaf),
        "n_iters": int(iters),
        "resample_augment": bool(augment),
        "penalty_factor": float(penalty_factor),
        "uniform_base_weights": bool(uniform_weights),
    }
