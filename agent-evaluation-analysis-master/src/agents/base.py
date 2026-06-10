"""
策略抽象基类。

所有可被溯因分析的策略都应实现此接口；
模块 A/B/C 都只依赖 BasePolicy，不绑定具体算法。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BasePolicy(ABC):
    """
    策略统一抽象接口。

    所有可被溯因分析的策略实现均应继承此类；模块 A/B/C 仅依赖此接口，
    不绑定具体强化学习算法。
    """

    @abstractmethod
    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """
        根据观测选择离散动作。

        参数
        ----
        obs : np.ndarray
            环境观测向量。
        deterministic : bool
            为 ``True`` 时取贪心动作；为 ``False`` 时可按策略随机采样。

        返回
        ----
        int
            离散动作编号。
        """
        raise NotImplementedError

    @abstractmethod
    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        """
        返回各动作的概率分布。

        模块 B 计算 KL 散度时需要此接口。确定性策略可返回 one-hot；
        随机策略应返回 softmax 后的概率向量。

        参数
        ----
        obs : np.ndarray
            环境观测向量。

        返回
        ----
        np.ndarray
            形状为 ``(n_actions,)`` 的概率分布，各元素非负且和为 1。
        """
        raise NotImplementedError

    # ---- 持久化 ----
    @abstractmethod
    def save(self, path: str) -> None:
        """
        将策略参数持久化到磁盘。

        参数
        ----
        path : str
            保存路径。
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """
        从磁盘加载策略参数。

        参数
        ----
        path : str
            模型文件路径。
        """
        raise NotImplementedError

    # ---- 可选：训练 ----
    def train(self, env: Any, total_steps: int, **kwargs: Any) -> None:
        """
        在环境中训练策略（可选实现）。

        默认未实现；子类按需覆盖。

        参数
        ----
        env : Any
            强化学习环境实例。
        total_steps : int
            训练总步数。
        **kwargs : Any
            算法相关的额外超参数。

        异常
        ----
        NotImplementedError
            子类未实现训练逻辑时抛出。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 未实现训练方法"
        )
