"""
DQN 智能体（PyTorch 实现）。

参考：Mnih et al. 2015, Human-level control through deep RL.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.agents.base import BasePolicy


# --------------------------------------------------------------------
# 经验回放
# --------------------------------------------------------------------
@dataclass
class Transition:
    """
    单步转移样本。

    属性
    ----
    obs : np.ndarray
        当前观测。
    action : int
        执行的动作编号。
    reward : float
        即时奖励。
    next_obs : np.ndarray
        下一时刻观测。
    done : bool
        是否终止（回合结束）。
    """
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    """环形经验回放池，用于 DQN 训练时存储与采样转移样本。"""

    def __init__(self, capacity: int) -> None:
        """
        初始化回放池。

        参数
        ----
        capacity : int
            池的最大容量；超出后以环形方式覆盖最旧样本。
        """
        self.capacity = capacity
        self._buf: List[Transition] = []
        self._ptr = 0

    def push(self, tr: Transition) -> None:
        """
        向回放池追加一条转移样本。

        参数
        ----
        tr : Transition
            待存入的转移元组。
        """
        # TODO
        raise NotImplementedError

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        随机采样一批转移数据。

        参数
        ----
        batch_size : int
            采样批量大小。

        返回
        ----
        Tuple[np.ndarray, ...]
            五元组 ``(obs, action, reward, next_obs, done)``，均为 NumPy 数组。
        """
        # TODO
        raise NotImplementedError

    def __len__(self) -> int:
        """
        返回当前池中样本数量。

        返回
        ----
        int
            已存储的转移条数（不超过 ``capacity``）。
        """
        return len(self._buf)


# --------------------------------------------------------------------
# Q 网络
# --------------------------------------------------------------------
class QNetwork(nn.Module):
    """多层感知机 Q 值网络，输出各动作的 Q 值。"""

    def __init__(self, obs_dim: int, n_actions: int, hidden: List[int]) -> None:
        """
        构建 Q 网络。

        参数
        ----
        obs_dim : int
            观测向量维度。
        n_actions : int
            离散动作空间大小。
        hidden : List[int]
            各隐藏层神经元数量列表。
        """
        super().__init__()
        # TODO: 用 nn.Sequential 搭 MLP
        raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        前向传播，计算各动作 Q 值。

        参数
        ----
        obs : torch.Tensor
            批量观测，形状 ``(batch, obs_dim)``。

        返回
        ----
        torch.Tensor
            各动作 Q 值，形状 ``(batch, n_actions)``。
        """
        # TODO
        raise NotImplementedError


# --------------------------------------------------------------------
# 智能体
# --------------------------------------------------------------------
class DQNAgent(BasePolicy):
    """
    深度 Q 网络（DQN）智能体。

    含在线 Q 网络、目标网络、经验回放池及 Adam 优化器；参考 Mnih et al. 2015。
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: List[int] | None = None,
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update_every: int = 1000,
        device: str = "cpu",
    ) -> None:
        """
        初始化 DQN 智能体。

        参数
        ----
        obs_dim : int
            观测维度。
        n_actions : int
            动作空间大小。
        hidden_sizes : List[int] | None
            Q 网络隐藏层配置，默认 ``[128, 128]``。
        lr : float
            Adam 学习率。
        gamma : float
            折扣因子。
        buffer_size : int
            经验回放池容量。
        batch_size : int
            每次梯度更新的批量大小。
        target_update_every : int
            目标网络同步间隔（训练步数）。
        device : str
            计算设备，如 ``"cpu"`` 或 ``"cuda"``。
        """
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.device = torch.device(device)

        hidden = hidden_sizes or [128, 128]
        self.q_net = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self._train_steps = 0

    # ---- BasePolicy ----
    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """
        根据当前 Q 值选择动作。

        参数
        ----
        obs : np.ndarray
            环境观测。
        deterministic : bool
            为 ``True`` 时取 argmax；为 ``False`` 时使用 epsilon-greedy 探索。

        返回
        ----
        int
            选中的动作编号。
        """
        # TODO: epsilon-greedy；deterministic=True 时返回 argmax
        raise NotImplementedError

    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        """
        将 Q 值转换为动作概率分布。

        参数
        ----
        obs : np.ndarray
            环境观测。

        返回
        ----
        np.ndarray
            形状 ``(n_actions,)`` 的概率向量（softmax 或贪心 one-hot）。
        """
        # TODO: 把 Q 值做 softmax / Boltzmann，或返回贪心 one-hot
        raise NotImplementedError

    def save(self, path: str) -> None:
        """
        保存在线 Q 网络权重。

        参数
        ----
        path : str
            模型文件路径。
        """
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str) -> None:
        """
        加载 Q 网络权重并同步至目标网络。

        参数
        ----
        path : str
            模型文件路径。
        """
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ---- 训练 ----
    def train(self, env: Any, total_steps: int, **kwargs: Any) -> None:
        """
        标准 DQN 训练主循环。

        参数
        ----
        env : Any
            强化学习环境。
        total_steps : int
            训练总步数。
        **kwargs : Any
            额外训练参数（如探索率调度等）。
        """
        # TODO:
        #   for step in range(total_steps):
        #       a = epsilon_greedy(obs)
        #       next_obs, r, done, _ = env.step(a)
        #       buffer.push(Transition(...))
        #       if len(buffer) >= batch_size:
        #           _update()
        #       if step % target_update_every == 0:
        #           target.load_state_dict(q.state_dict())
        raise NotImplementedError

    def _update(self) -> float:
        """
        从回放池采样并执行一次梯度更新。

        返回
        ----
        float
            本次更新的损失值。
        """
        # TODO
        raise NotImplementedError
