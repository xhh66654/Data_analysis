"""
特征预处理：量纲归一化 + 自动离散化。

================================================================================
说明（小白友好版）：
================================================================================

不同装备类型（飞机、雷达、侦察机）的观测字段完全不同：
    飞机：  自身状态.血量, 自身状态.速度_马赫, 敌机距离.水平距离_km ...
    雷达：  雷达状态.功率_kw, 雷达状态.方位角_deg, 目标信号.强度_dbm ...
    侦察机：位置.经度, 位置.纬度, 目标.数量, 目标.威胁等级 ...

因此预处理器不能有任何硬编码的特征名或分箱配置，
而是完全从数据中动态推断：

    1. 归一化（normalize）：
       从训练数据学习每列的均值和标准差，做 z-score 归一化。
       数值范围、单位各不相同也没关系，统统缩放到标准正态分布附近。

    2. 自动分箱（auto discretize）：
       对每个特征自动计算分位数边界（默认四分位：25%, 50%, 75%）。
       分箱标签自动命名为 "极低 / 低 / 中 / 高 / 极高" 等通用级别。
       不需要人工指定任何边界值，自动适配任何特征的数值范围。

    3. 自定义覆盖（可选）：
       如果对某个特征有专业的语义知识（比如"血量>0.5=良好"），
       可以通过 discretize_config 参数手动指定这个特征的分箱，
       其他特征仍然自动推断。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ==============================================================================
# 通用分箱标签（按分箱数量自动选择）
# ==============================================================================

_GENERIC_LABELS: Dict[int, List[str]] = {
    2: ["低", "高"],
    3: ["低", "中", "高"],
    4: ["极低", "低", "高", "极高"],
    5: ["极低", "低", "中", "高", "极高"],
    6: ["极低", "较低", "低", "高", "较高", "极高"],
}


def _generic_labels(n_bins: int) -> List[str]:
    """
    返回 n_bins 个区间的通用语义标签列表。

    参数:
        n_bins: 分箱区间数量。

    返回:
        对应数量的中文级别标签列表。
    """
    if n_bins in _GENERIC_LABELS:
        return _GENERIC_LABELS[n_bins]
    # 超出预设时，用数字编号
    return [f"级别{i+1}" for i in range(n_bins)]


@dataclass
class Preprocessor:
    """
    状态特征预处理器。

    支持任意观测空间结构，不依赖任何硬编码特征名或分箱配置。

    使用方式：
        pre = Preprocessor(feature_names=feature_names)
        X_norm = pre.fit_transform(X_raw)   # 训练时：学习参数 + 变换
        X_norm = pre.transform(X_new)       # 推理时：只变换（不重新学习）

    Attributes
    ----------
    feature_names     : 展平后的特征名列表（如 ["自身状态.血量", "敌机距离.水平距离_km"]），
                        顺序与数据列一致
    normalize         : 是否做 z-score 归一化（默认开启）
    n_quantiles       : 自动分箱时使用几个分位数边界（默认 3，即四分位分成4段）
    discretize_config : 手动覆盖某些特征的分箱配置（可选）：
                        key = 特征名（展平复合键），value = (bin_edges, labels)
                        e.g. {"自身状态.血量": ([0.25, 0.5, 0.75], ["危险","较低","良好","满血"])}
                        未指定的特征均使用自动分箱。
    """
    feature_names: List[str] = field(default_factory=list)
    normalize: bool = True
    n_quantiles: int = 3   # 默认三分位（把数据分成4段）
    discretize_config: Dict[str, Tuple] = field(default_factory=dict)

    # 以下在 fit() 后填充
    _mean:    Optional[np.ndarray] = field(default=None, repr=False)
    _std:     Optional[np.ndarray] = field(default=None, repr=False)
    _min:     Optional[np.ndarray] = field(default=None, repr=False)
    _max:     Optional[np.ndarray] = field(default=None, repr=False)
    # 自动推断的分箱配置（fit 后自动填充，手动配置优先）
    _auto_bins: Dict[str, Tuple] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # 训练：从数据中学习归一化参数 + 自动分箱
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "Preprocessor":
        """
        从训练数据中学习归一化参数和自动分箱边界。

        参数:
            X: (N, n_features) 原始特征矩阵（未归一化）。

        返回:
            自身实例，支持链式调用。
        """
        X = np.array(X, dtype=float)
        self._mean = X.mean(axis=0)
        self._std  = X.std(axis=0)
        self._min  = X.min(axis=0)
        self._max  = X.max(axis=0)
        # 标准差为零时用 1 替换（特征值不变的列，归一化后全是0，不影响训练）
        self._std = np.where(self._std == 0, 1.0, self._std)

        # 自动计算每列的分位数分箱边界
        self._auto_bins = {}
        quantile_probs = np.linspace(0, 100, self.n_quantiles + 2)[1:-1]  # 排除0%和100%
        n_bins = self.n_quantiles + 1
        labels = _generic_labels(n_bins)

        for i, feat_name in enumerate(self.feature_names):
            if feat_name in self.discretize_config:
                continue  # 已有手动配置，跳过自动推断
            col = X[:, i]
            # 计算分位数边界（去重，避免重复边界导致分箱无意义）
            edges = np.unique(np.percentile(col, quantile_probs)).tolist()
            if len(edges) == 0:
                # 特征值全部相同，无法分箱
                self._auto_bins[feat_name] = ([], [labels[0]])
            else:
                n_actual = len(edges) + 1
                actual_labels = _generic_labels(n_actual)
                self._auto_bins[feat_name] = (edges, actual_labels)

        return self

    # ------------------------------------------------------------------
    # 变换：应用归一化
    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        对输入数据做 z-score 归一化：x' = (x - mean) / std

        参数:
            X: (N, n_features) 或 (n_features,) 原始特征矩阵/向量。

        返回:
            归一化后的特征数组，形状与输入相同。
        """
        assert self._mean is not None, "请先调用 fit() 学习归一化参数"
        X = np.array(X, dtype=float)
        if not self.normalize:
            return X
        return (X - self._mean) / self._std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        先 fit 再 transform 的快捷方式。

        参数:
            X: (N, n_features) 原始特征矩阵。

        返回:
            归一化后的特征数组。
        """
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    # 反向工具：把归一化后的阈值还原为原始单位
    # ------------------------------------------------------------------
    def denormalize_threshold(self, feat_idx: int, thresh: float) -> float:
        """
        把决策树节点的归一化阈值还原到原始数值单位。

        还原公式：原始值 = thresh * std + mean

        Parameters
        ----------
        feat_idx : 特征的列索引
        thresh   : 决策树节点阈值（归一化后的值）

        Returns
        -------
        原始单位的阈值
        """
        assert self._mean is not None, "请先调用 fit()"
        return float(thresh * self._std[feat_idx] + self._mean[feat_idx])

    # ------------------------------------------------------------------
    # 语义标签：把特征值翻译成可读的区间名称
    # ------------------------------------------------------------------
    def discretize_label(self, feat_name: str, value: float) -> str:
        """
        给一个特征的原始值，返回对应的语义区间标签。

        优先使用手动配置（discretize_config），
        若没有手动配置则使用自动分箱（_auto_bins），
        若两者均无则返回数值字符串。

        Parameters
        ----------
        feat_name : 展平后的特征名，e.g. "自身状态.血量"
        value     : 原始特征值（未归一化）

        Returns
        -------
        语义标签字符串，e.g. "低" / "高" / "危险" / "良好"

        示例：
            # 自动分箱（三分位，四段）
            pre.discretize_label("敌机距离.水平距离_km", 30.0)  → "低"
            pre.discretize_label("敌机距离.水平距离_km", 80.0)  → "极高"

            # 手动配置（如果配置了的话）
            pre.discretize_label("自身状态.血量", 0.3)  → "危险"
        """
        # 1. 优先手动配置
        config = self.discretize_config.get(feat_name)
        # 2. 次选自动分箱
        if config is None:
            config = self._auto_bins.get(feat_name)
        # 3. 都没有，直接返回数值字符串
        if config is None:
            return f"{value:.3f}"

        edges, labels = config
        if not edges:
            return labels[0] if labels else f"{value:.3f}"

        idx = int(np.searchsorted(edges, value, side="right"))
        idx = max(0, min(idx, len(labels) - 1))
        return labels[idx]

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def get_feature_name(self, feat_idx: int) -> str:
        """
        根据列索引返回特征名。

        参数:
            feat_idx: 特征列索引。

        返回:
            特征名；越界时返回 ``feature_{feat_idx}``。
        """
        if feat_idx < len(self.feature_names):
            return self.feature_names[feat_idx]
        return f"feature_{feat_idx}"

    def get_bin_summary(self) -> Dict[str, Tuple]:
        """
        返回所有特征当前使用的分箱配置（手动 + 自动合并），便于调试和展示。

        返回:
            字典 ``{特征名: (bin_edges, labels)}``。
        """
        result = dict(self._auto_bins)
        result.update(self.discretize_config)   # 手动配置覆盖自动
        return result

    def export_fit_state(self) -> Dict[str, object]:
        """
        导出 fit 后的归一化与分箱参数（供 agent profile 持久化）。

        返回:
            含 ``mean``、``std``、``min``、``max``、``auto_bins`` 的字典。
        """
        assert self._mean is not None
        return {
            "mean": self._mean.tolist(),
            "std": self._std.tolist(),
            "min": self._min.tolist() if self._min is not None else None,
            "max": self._max.tolist() if self._max is not None else None,
            "auto_bins": {
                k: (list(edges), list(labels))
                for k, (edges, labels) in self._auto_bins.items()
            },
        }

    def import_fit_state(self, state: Dict[str, object]) -> "Preprocessor":
        """
        从 ``export_fit_state`` 的输出恢复预处理器状态（不执行 fit）。

        参数:
            state: 由 ``export_fit_state`` 产生的参数字典。

        返回:
            自身实例，支持链式调用。
        """
        self._mean = np.array(state["mean"], dtype=float)
        self._std = np.array(state["std"], dtype=float)
        if state.get("min") is not None:
            self._min = np.array(state["min"], dtype=float)
        if state.get("max") is not None:
            self._max = np.array(state["max"], dtype=float)
        self._auto_bins = {
            k: (list(v[0]), list(v[1]))
            for k, v in (state.get("auto_bins") or {}).items()
        }
        return self
