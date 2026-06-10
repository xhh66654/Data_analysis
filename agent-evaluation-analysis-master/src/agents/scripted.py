"""
脚本策略：用于敌方（红方）和非受训蓝方队友的基线行为。

提供若干简单但可工作的策略，便于环境先跑起来。
"""
from __future__ import annotations

import numpy as np

from src.agents.base import BasePolicy


class ScriptedPolicy(BasePolicy):
    """
    基于规则的脚本策略。

    支持追击、随机、规避等预设行为模式，无神经网络参数。
    """

    def __init__(self, mode: str = "pursue") -> None:
        """
        初始化脚本策略。

        参数
        ----
        mode : str
            行为模式，可选 ``"pursue"``（追击）、``"random"``（随机）、
            ``"evade"``（规避）。
        """
        self.mode = mode  # "pursue" / "random" / "evade"

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """
        根据观测与模式选择动作。

        参数
        ----
        obs : np.ndarray
            环境观测向量（含敌我相对位置等信息）。
        deterministic : bool
            脚本策略通常忽略此参数，行为由 ``mode`` 决定。

        返回
        ----
        int
            离散动作编号。
        """
        # TODO: 根据 obs 解析最近敌人相对位置，输出动作
        raise NotImplementedError

    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        """
        返回脚本策略的动作分布（通常为 one-hot）。

        参数
        ----
        obs : np.ndarray
            环境观测向量。

        返回
        ----
        np.ndarray
            形状 ``(n_actions,)`` 的概率向量，选中动作为 1，其余为 0。
        """
        # TODO: 对脚本策略返回 one-hot
        raise NotImplementedError

    def save(self, path: str) -> None:
        """
        将行为模式名称写入标记文件。

        脚本策略无可学习参数，仅持久化 ``mode`` 字符串。

        参数
        ----
        path : str
            输出文件路径。
        """
        # 脚本策略无参数，保存一个标记文件即可
        from pathlib import Path
        Path(path).write_text(self.mode, encoding="utf-8")

    def load(self, path: str) -> None:
        """
        从标记文件恢复行为模式。

        参数
        ----
        path : str
            此前 ``save`` 写入的文件路径。
        """
        from pathlib import Path
        self.mode = Path(path).read_text(encoding="utf-8").strip()
