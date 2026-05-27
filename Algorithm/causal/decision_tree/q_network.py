"""
Q 值神经网络 QHatNetwork：FQE 阶段使用的 MLP 结构。

架构：Linear(state_dim → hidden) → ReLU → Linear(hidden → hidden) → ReLU
      → Linear(hidden → n_actions)
前向输出 shape (batch, n_actions)，第 a 维即 Q(s,a)。

方法：
  forward(s)           — 返回所有动作的 Q 值向量
  q_value(s, a)        — 取指定动作 a 的 Q(s,a)，用于 FQE TD 目标与 l_hat 计算

由 fqe.train_q_hat() 实例化；参数随 q_hat.pt 一并保存与加载。
默认 hidden=256，run_pipeline 中可通过 fqe_hidden 调整。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class QHatNetwork(nn.Module):
    """Q_hat(s) -> R^A；Q_hat(s,a) 为输出第 a 维。"""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)

    def q_value(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q = self.forward(s)
        return q.gather(1, a.long().view(-1, 1)).squeeze(1)
