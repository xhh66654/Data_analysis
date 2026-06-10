"""
PPO 包装器（基于 stable-baselines3）。

作为 DQN 的替代方案：训练更稳定，适合动作维度较大的场景。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.agents.base import BasePolicy


class PPOAgent(BasePolicy):
    """将 stable-baselines3 的 PPO 算法封装为 ``BasePolicy`` 接口。"""

    def __init__(self, env: Any, **ppo_kwargs: Any) -> None:
        """
        初始化 PPO 智能体。

        参数
        ----
        env : Any
            Gymnasium/Gym 兼容环境，用于构造 SB3 PPO 模型。
        **ppo_kwargs : Any
            传递给 ``stable_baselines3.PPO`` 的超参数。
        """
        # 延迟导入：避免未装 sb3 时报错
        # TODO: from stable_baselines3 import PPO; self.model = PPO(...)
        self.model: Any = None
        raise NotImplementedError

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """
        根据观测选择动作。

        参数
        ----
        obs : np.ndarray
            环境观测向量。
        deterministic : bool
            为 ``True`` 时确定性预测；为 ``False`` 时按策略分布采样。

        返回
        ----
        int
            离散动作编号。
        """
        # TODO: action, _ = self.model.predict(obs, deterministic=deterministic)
        raise NotImplementedError

    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        """
        返回策略在观测下的动作概率分布。

        参数
        ----
        obs : np.ndarray
            环境观测向量。

        返回
        ----
        np.ndarray
            形状 ``(n_actions,)`` 的概率向量。
        """
        # TODO: 用 self.model.policy.get_distribution(obs_tensor) 取概率
        raise NotImplementedError

    def save(self, path: str) -> None:
        """
        保存 PPO 模型到磁盘。

        参数
        ----
        path : str
            模型保存路径。
        """
        # TODO
        raise NotImplementedError

    def load(self, path: str) -> None:
        """
        从磁盘加载 PPO 模型。

        参数
        ----
        path : str
            模型文件路径。
        """
        # TODO
        raise NotImplementedError

    def train(self, env: Any, total_steps: int, **kwargs: Any) -> None:
        """
        调用 SB3 的 ``learn`` 方法训练策略。

        参数
        ----
        env : Any
            训练环境（通常与构造时相同）。
        total_steps : int
            训练总时间步数。
        **kwargs : Any
            传递给 ``model.learn`` 的额外参数。
        """
        # TODO: self.model.learn(total_timesteps=total_steps)
        raise NotImplementedError
