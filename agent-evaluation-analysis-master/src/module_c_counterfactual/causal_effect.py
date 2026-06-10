"""
因果效应量计算（对应方案文档算法 2，标量奖励版）。

机械论：状态特征 → 查询动作是否发生（分类器特征重要性）
目的论：发生 vs 未发生样本的标量 reward_scalar 均值差
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.module_c_counterfactual.cf_dataset import CFSample


@dataclass
class CausalFactor:
    """
    单个因果因素及其效应量（表 2 输出单元）。

    属性:
        name: 因素名称（特征名或 reward_scalar）。
        effect: 效应量数值（重要性或均值差）。
        rank: 重要性排名（从 1 开始）。
    """

    name: str
    effect: float
    rank: int = 0


def _build_xy(
    samples: List[CFSample],
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    从反事实样本构造分类器特征矩阵 X 与标签 y。

    参数:
        samples: K 采样反事实样本列表。
        feature_names: 状态特征名列表。

    返回:
        (X, y, ok) 三元组；ok 为 False 表示 y 无类别方差。
    """
    if not samples or not feature_names:
        return np.empty((0, 0)), np.empty(0), False
    n_feat = len(feature_names)
    X = np.vstack([s.state_features[:n_feat] for s in samples])
    y = np.array([1 if s.query_happened else 0 for s in samples], dtype=int)
    if len(np.unique(y)) < 2:
        return X, y, False
    return X, y, True


def mechanistic_effect(
    samples: List[CFSample],
    feature_names: List[str],
    t_window: Tuple[int, int] | None = None,
) -> List[CausalFactor]:
    """
    机械论因果效应：拟合 state → query_happened，按特征重要性排序。

    参数:
        samples: K 采样反事实样本列表。
        feature_names: 状态特征名列表。
        t_window: 保留兼容；观测空间 MVP 中未使用。

    返回:
        按效应量降序排列的 CausalFactor 列表。
    """
    del t_window  # 观测空间 MVP：样本已含扰动后 state
    X, y, ok = _build_xy(samples, feature_names)
    if not ok or X.shape[0] < 5:
        return _mechanistic_fallback(samples, feature_names)

    clf: RandomForestClassifier | LogisticRegression
    if X.shape[0] >= 20:
        clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=6,
            min_samples_leaf=2,
            random_state=0,
        )
        clf.fit(X, y)
        imp = getattr(clf, "feature_importances_", None)
    else:
        clf = LogisticRegression(max_iter=500, random_state=0)
        clf.fit(X, y)
        imp = np.abs(clf.coef_[0]) if clf.coef_.size else np.zeros(X.shape[1])

    if imp is None or len(imp) != len(feature_names):
        return _mechanistic_fallback(samples, feature_names)

    pairs = sorted(
        zip(feature_names, imp),
        key=lambda x: float(x[1]),
        reverse=True,
    )
    factors: List[CausalFactor] = []
    for rank, (name, val) in enumerate(pairs, start=1):
        if float(val) <= 1e-9:
            continue
        factors.append(CausalFactor(name=name, effect=float(val), rank=rank))
    return factors


def _mechanistic_fallback(
    samples: List[CFSample],
    feature_names: List[str],
) -> List[CausalFactor]:
    """样本 y 无方差时：用特征与 query_happened 的相关性近似。"""
    if not samples or not feature_names:
        return []
    X, y, _ = _build_xy(samples, feature_names)
    if X.size == 0:
        return []
    scores: List[tuple[str, float]] = []
    for j, name in enumerate(feature_names):
        col = X[:, j]
        if np.std(col) < 1e-9:
            continue
        corr = float(np.corrcoef(col, y)[0, 1]) if len(y) > 1 else 0.0
        if np.isnan(corr):
            corr = 0.0
        scores.append((name, abs(corr)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [
        CausalFactor(name=n, effect=e, rank=i + 1)
        for i, (n, e) in enumerate(scores[:10])
        if e > 1e-6
    ]


def teleological_effect(
    samples: List[CFSample],
    reward_dim_names: Optional[List[str]] = None,
) -> List[CausalFactor]:
    """
    目的论因果效应（标量奖励版）。

    计算 mean(reward | happened) - mean(reward | not happened)。

    参数:
        samples: K 采样反事实样本列表。
        reward_dim_names: 保留兼容；标量模式下忽略。

    返回:
        含 reward_scalar 均值差的 CausalFactor 列表（通常仅一条）。
    """
    del reward_dim_names
    if not samples:
        return []
    happened = [s.reward_scalar for s in samples if s.query_happened]
    not_happened = [s.reward_scalar for s in samples if not s.query_happened]
    if not happened or not not_happened:
        return []
    delta = float(np.mean(happened) - np.mean(not_happened))
    return [
        CausalFactor(
            name="reward_scalar",
            effect=delta,
            rank=1,
        )
    ]
