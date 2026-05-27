"""
基于反事实干预的邻居影响度：屏蔽某邻居的观测块（若在联合状态中）与其动作块，比较 Q 变化。
不改变仓库中其它模块，仅在本包内使用。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from .joint_q_network import JointQNetwork
from .trajectory_maddpg import JointTransitionBatch

logger = logging.getLogger(__name__)


def _agent_action_offset(agent_order_full: tuple[str, ...], agent: str, act_dim: int) -> slice:
    idx = agent_order_full.index(agent)
    start = idx * act_dim
    return slice(start, start + act_dim)


def _agent_state_offset(state_agents: tuple[str, ...], agent: str, obs_dim: int) -> slice | None:
    if agent not in state_agents:
        return None
    idx = state_agents.index(agent)
    start = idx * obs_dim
    return slice(start, start + obs_dim)


def apply_neighbor_ablation(
    s: torch.Tensor,
    a: torch.Tensor,
    *,
    victim: str,
    target_agent: str,
    agent_order_full: tuple[str, ...],
    state_agents: tuple[str, ...],
    obs_dim: int,
    act_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """拷贝并在 victim 上做「状态/动作置零」的反事实构造；target_agent 保持不变。"""
    if victim == target_agent:
        raise ValueError("反事实屏蔽对象不能是目标智能体")
    s2 = s.clone()
    a2 = a.clone()
    st_sl = _agent_state_offset(state_agents, victim, obs_dim)
    if st_sl is not None:
        s2[:, st_sl] = 0.0
    ac_sl = _agent_action_offset(agent_order_full, victim, act_dim)
    a2[:, ac_sl] = 0.0
    return s2, a2


def apply_neighbor_noise_intervention(
    s: torch.Tensor,
    a: torch.Tensor,
    *,
    victim: str,
    target_agent: str,
    agent_order_full: tuple[str, ...],
    state_agents: tuple[str, ...],
    obs_dim: int,
    act_dim: int,
    obs_std: dict[str, torch.Tensor],
    act_std: dict[str, torch.Tensor],
    noise_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 victim 的 obs/action 块加高斯噪声（替代置零），保持在训练分布内。"""
    if victim == target_agent:
        raise ValueError("反事实干预对象不能是目标智能体")
    s2 = s.clone()
    a2 = a.clone()

    st_sl = _agent_state_offset(state_agents, victim, obs_dim)
    if st_sl is not None and victim in obs_std:
        block = s2[:, st_sl]
        std = obs_std[victim]  # (obs_dim,)
        noise = torch.randn_like(block) * std.unsqueeze(0) * noise_scale
        s2[:, st_sl] = block + noise

    ac_sl = _agent_action_offset(agent_order_full, victim, act_dim)
    if victim in act_std:
        block_a = a2[:, ac_sl]
        std_a = act_std[victim]  # (act_dim,)
        noise_a = torch.randn_like(block_a) * std_a.unsqueeze(0) * noise_scale
        a2[:, ac_sl] = block_a + noise_a

    return s2, a2


@dataclass
class InfluenceRow:
    neighbor: str
    mean_abs_effect: float
    mean_signed_effect: float  # Q_full - Q_cf
    mean_abs_effect_normalized: float = 0.0  # mean_abs_effect / std(Q_full)


def _compute_agent_block_std(
    tensor: torch.Tensor, agent: str, agent_order: tuple[str, ...], dim_per_agent: int
) -> torch.Tensor:
    """计算某个 agent 在 tensor 中的逐维度标准差（全 batch）。"""
    idx = agent_order.index(agent)
    start = idx * dim_per_agent
    sl = slice(start, start + dim_per_agent)
    return tensor[:, sl].std(dim=0)  # (dim_per_agent,)


def estimate_neighbor_effects(
    q_net: JointQNetwork,
    batch: JointTransitionBatch,
    target_agent: str,
    *,
    neighbors: list[str] | None,
    device: str = "cpu",
    batch_size_infer: int = 2048,
    ablation_mode: str = "zero",
    noise_scale: float = 1.0,
) -> list[InfluenceRow]:
    """
    对 batch 中所有行估计每个邻居的反事实效应（绝对值均值与符号均值）。

    ablation_mode:
      - "zero": 传统置零消融（可能产生 OOD 输入）
      - "noise": 加高斯噪声 N(0, noise_scale * σ²)，保持在训练分布内
    noise_scale: 仅 noise 模式生效，控制噪声强度
    """
    dev = torch.device(device)
    q_net.eval()

    cand = neighbors
    if cand is None:
        cand = [a for a in batch.agent_order_full if a != target_agent]

    s = torch.from_numpy(batch.joint_obs_state).to(dev, dtype=torch.float32)
    a = torch.from_numpy(batch.joint_action_full).to(dev, dtype=torch.float32)

    # 噪声模式：预计算各 agent 的逐维度标准差
    obs_std: dict[str, torch.Tensor] = {}
    act_std: dict[str, torch.Tensor] = {}
    if ablation_mode == "noise":
        for nb in cand:
            obs_std[nb] = _compute_agent_block_std(s, nb, batch.state_agents, batch.obs_dim)
            act_std[nb] = _compute_agent_block_std(a, nb, batch.agent_order_full, batch.act_dim)

    results: list[InfluenceRow] = []
    n = s.shape[0]

    def eval_chunks(tensor_s: torch.Tensor, tensor_a: torch.Tensor) -> torch.Tensor:
        outs: list[torch.Tensor] = []
        for start in range(0, tensor_s.shape[0], batch_size_infer):
            sb = tensor_s[start : start + batch_size_infer]
            ab = tensor_a[start : start + batch_size_infer]
            with torch.no_grad():
                outs.append(q_net.forward_sa(sb, ab))
        return torch.cat(outs, dim=0)

    q_full = eval_chunks(s, a)
    q_std = float(q_full.std().item())  # 用于归一化影响度

    for nb in cand:
        s_cf_list: list[torch.Tensor] = []
        a_cf_list: list[torch.Tensor] = []
        for start in range(0, n, batch_size_infer):
            sb = s[start : start + batch_size_infer]
            ab = a[start : start + batch_size_infer]
            if ablation_mode == "noise":
                sb2, ab2 = apply_neighbor_noise_intervention(
                    sb, ab, victim=nb, target_agent=target_agent,
                    agent_order_full=batch.agent_order_full,
                    state_agents=batch.state_agents,
                    obs_dim=batch.obs_dim, act_dim=batch.act_dim,
                    obs_std=obs_std, act_std=act_std, noise_scale=noise_scale,
                )
            else:
                sb2, ab2 = apply_neighbor_ablation(
                    sb, ab, victim=nb, target_agent=target_agent,
                    agent_order_full=batch.agent_order_full,
                    state_agents=batch.state_agents,
                    obs_dim=batch.obs_dim, act_dim=batch.act_dim,
                )
            s_cf_list.append(sb2)
            a_cf_list.append(ab2)
        s_cf = torch.cat(s_cf_list, dim=0)
        a_cf = torch.cat(a_cf_list, dim=0)
        with torch.no_grad():
            q_cf = q_net.forward_sa(s_cf, a_cf)

        diff = q_full - q_cf
        abs_eff = float(diff.abs().mean().item())
        results.append(
            InfluenceRow(
                neighbor=nb,
                mean_abs_effect=abs_eff,
                mean_signed_effect=float(diff.mean().item()),
                mean_abs_effect_normalized=abs_eff / q_std if q_std > 0 else 0.0,
            )
        )

    logger.info("反事实影响评估完毕（模式=%s），邻居数=%s，Q_std=%.4f", ablation_mode, len(results), q_std)
    return results


def top_k_neighbors(rows: list[InfluenceRow], k: int) -> list[InfluenceRow]:
    """兼容旧接口：等价于 ``select_key_neighbors(..., min_abs_effect=None, max_count=k)``。"""
    return select_key_neighbors(rows, min_abs_effect=None, max_count=k)


def select_key_neighbors(
    rows: list[InfluenceRow],
    *,
    min_abs_effect: float | None = None,
    max_count: int | None = None,
    value_attr: str = "mean_abs_effect",
) -> list[InfluenceRow]:
    """
    从高到低排序后选取关键邻居。

    - **仅 max_count（min_abs_effect 为 None）**：取前 max_count 个；max_count<=0 则空。
    - **带阈值 min_abs_effect**：从高到低扫描，**一旦出现** value_attr < min_abs_effect，
      则该邻居及**之后所有更弱的**一律不要；再结合 max_count。

    value_attr: 用于排序和阈值比较的 InfluenceRow 字段名，默认 "mean_abs_effect"。
      设为 "mean_abs_effect_normalized" 时以归一化 σ 单位做阈值截断。
    """
    ranked = sorted(rows, key=lambda r: -getattr(r, value_attr))
    if min_abs_effect is None:
        cap = max_count if max_count is not None else 0
        if cap <= 0:
            return []
        return ranked[: min(cap, len(ranked))]

    chosen: list[InfluenceRow] = []
    mc = max_count if max_count is not None else 0
    for r in ranked:
        if getattr(r, value_attr) < min_abs_effect:
            break
        chosen.append(r)
        if mc > 0 and len(chosen) >= mc:
            break
    return chosen
