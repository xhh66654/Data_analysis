"""集中式联合 Q：输入为联合观测向量 + 全体拼接连续动作向量，输出标量 Q。"""
from __future__ import annotations

import torch
import torch.nn as nn


class JointQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        d_in = int(state_dim) + int(action_dim)
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward_sa(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Q(s,a)，batch 上与 s,a 对齐，返回形状 (batch,)。"""
        x = torch.cat([s, a], dim=-1)
        return self.net(x).squeeze(-1)


def init_q_same_device(module: JointQNetwork, device: torch.device) -> JointQNetwork:
    return module.to(device)
