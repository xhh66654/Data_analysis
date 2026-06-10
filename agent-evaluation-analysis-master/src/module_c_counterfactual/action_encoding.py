"""联合动作标签编码（holistic JSON 整体决策类 → 数值特征）。"""
from __future__ import annotations

from typing import List

import numpy as np
from sklearn.preprocessing import LabelEncoder


class JointActionEncoder:
    """将联合动作字符串映射为单个数值特征，供转移/奖励回归使用。"""

    def __init__(self) -> None:
        """初始化编码器（未拟合状态）。"""
        self._encoder = LabelEncoder()
        self._is_fitted = False
        self._unknown_value: float = 0.0

    def fit(self, labels: List[str]) -> "JointActionEncoder":
        """
        从联合动作标签列表学习编码映射。

        参数:
            labels: 训练集中出现的联合动作字符串列表。

        返回:
            自身实例（支持链式调用）。
        """
        uniq = sorted(set(labels))
        if not uniq:
            uniq = ["__empty__"]
        self._encoder.fit(uniq)
        self._unknown_value = float(len(self._encoder.classes_))
        self._is_fitted = True
        return self

    def transform(self, label: str) -> float:
        """
        将单个联合动作标签编码为浮点特征值。

        参数:
            label: 联合动作字符串；未见过的标签映射为未知值。

        返回:
            编码后的浮点数。
        """
        if not self._is_fitted:
            raise RuntimeError("JointActionEncoder 尚未 fit。")
        if label in self._encoder.classes_:
            return float(self._encoder.transform([label])[0])
        return self._unknown_value

    def transform_row(self, label: str) -> np.ndarray:
        """
        将标签编码为长度为 1 的 numpy 向量（供特征拼接使用）。

        参数:
            label: 联合动作字符串。

        返回:
            shape=(1,) 的 float 数组。
        """
        return np.array([self.transform(label)], dtype=float)
