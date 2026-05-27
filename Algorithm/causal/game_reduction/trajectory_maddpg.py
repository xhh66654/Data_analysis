"""
读取 MADDPG 导出的 training_trajectory.csv（每环境步一行，各 agent 列内为 JSON）。
校验列名、解析 JSON，按回合边界构造 SARSA bootstrap 所需的下一步动作。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

META_COLS = ("episode", "global_step", "episode_step")


@dataclass(frozen=True)
class JointTransitionBatch:
    """与同 CSV 行对齐的一步联合样本（用于集中式连续联合 Q）。"""

    joint_obs_state: np.ndarray  # (N, D_state)：目标智能体观测 +（选定）邻居观测
    joint_action_full: np.ndarray  # (N, D_act_all)：全体智能体动作按 agent_order_full 拼接
    joint_next_obs_state: np.ndarray  # (N, D_state)
    joint_action_next: np.ndarray  # (N, D_act_full)：下一步全体动作（SARSA）；回合末填 0
    reward_signal: np.ndarray  # (N,)
    done: np.ndarray  # float32
    episode: np.ndarray  # int64
    agent_order_full: tuple[str, ...]
    state_agents: tuple[str, ...]  # 联合观测中包含的智能体，目标在前
    obs_dim: int
    act_dim: int


def _parse_agent_cell(x: Any) -> dict[str, Any]:
    if pd.isna(x):
        raise ValueError("agent 列为空")
    if isinstance(x, dict):
        return x
    s = x if isinstance(x, str) else str(x)
    return json.loads(s)


def load_maddpg_trajectory_csv(path: str | Path, *, encoding: str = "utf-8-sig") -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"轨迹文件不存在: {p.resolve()}")
    df = pd.read_csv(p, encoding=encoding)
    miss = [c for c in META_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"缺少元数据列: {miss}")
    agent_cols = sorted([c for c in df.columns if c.startswith("agent_")], key=lambda z: int(z.split("_")[1]))
    if len(agent_cols) < 1:
        raise ValueError("至少需要 1 个 agent_* 列")
    return df


def _resolve_state_agents(
    agent_cols: list[str],
    target: str,
    neighbor_subset: list[str] | None,
) -> tuple[str, ...]:
    if target not in agent_cols:
        raise ValueError(f"target {target} 不在列 {agent_cols}")
    others = [a for a in agent_cols if a != target]
    if neighbor_subset is None:
        neighbors = tuple(sorted(others, key=lambda z: int(z.split("_")[1])))
    else:
        for n in neighbor_subset:
            if n not in agent_cols:
                raise ValueError(f"邻居 {n} 不在 CSV 列中")
            if n == target:
                raise ValueError("neighbor_subset 不能与 target 相同")
        neighbors = tuple(sorted(neighbor_subset, key=lambda z: int(z.split("_")[1])))
    return (target,) + neighbors


def build_joint_batch(
    df: pd.DataFrame,
    target_agent: str,
    *,
    reward_mode: str = "target",
    neighbor_subset: list[str] | None = None,
) -> JointTransitionBatch:
    """
    reward_mode: target → 监督信号用目标体 reward；team_sum → 当步全员 reward 之和。
    neighbor_subset: None → 联合状态包含「目标 + 除目标外所有体」；
        传入列表 → 联合状态仅为「目标 + 列表中的邻居」的 obs 拼接；动作仍为全体拼接。
    """
    agent_cols = sorted([c for c in df.columns if c.startswith("agent_")], key=lambda z: int(z.split("_")[1]))
    state_agents = _resolve_state_agents(agent_cols, target_agent, neighbor_subset)

    parsed = {_a: [_parse_agent_cell(df.iloc[i][_a]) for i in range(len(df))] for _a in agent_cols}

    obs_dim = int(parsed[agent_cols[0]][0]["obs_dim"])
    act_dim = int(parsed[agent_cols[0]][0]["action_dim"])

    for a in agent_cols:
        row0 = parsed[a][0]
        if int(row0["obs_dim"]) != obs_dim or int(row0["action_dim"]) != act_dim:
            raise ValueError(f"{a} 的 obs_dim/action_dim 与其它智能体不一致")

    n = len(df)
    d_state = len(state_agents) * obs_dim
    d_act_all = len(agent_cols) * act_dim

    joint_obs_state = np.zeros((n, d_state), dtype=np.float32)
    joint_next_state = np.zeros((n, d_state), dtype=np.float32)
    joint_action_full = np.zeros((n, d_act_all), dtype=np.float32)

    rew_t = np.zeros(n, dtype=np.float32)

    episodes = pd.to_numeric(df["episode"], errors="raise").astype(np.int64).values
    done = np.zeros(n, dtype=np.float32)

    for i in range(n):
        off = 0
        for a in state_agents:
            rec = parsed[a][i]
            o = np.asarray(rec["obs"], dtype=np.float32).reshape(-1)
            nx = np.asarray(rec["next_obs"], dtype=np.float32).reshape(-1)
            if o.size != obs_dim or nx.size != obs_dim:
                raise ValueError(f"行{i} {a}: obs/next_obs 长度异常")
            joint_obs_state[i, off : off + obs_dim] = o
            joint_next_state[i, off : off + obs_dim] = nx
            off += obs_dim

        off = 0
        ra = []
        for a in agent_cols:
            rec = parsed[a][i]
            ac = np.asarray(rec["action"], dtype=np.float32).reshape(-1)
            if ac.size != act_dim:
                raise ValueError(f"行{i} {a}: action 长度异常")
            joint_action_full[i, off : off + act_dim] = ac
            off += act_dim
            ra.append(float(rec["reward"]))

        if reward_mode == "target":
            rew_t[i] = float(parsed[target_agent][i]["reward"])
        elif reward_mode == "team_sum":
            rew_t[i] = float(np.sum(ra))
        else:
            raise ValueError("reward_mode 须为 target 或 team_sum")

        dg = False
        for a in agent_cols:
            rec = parsed[a][i]
            dg = dg or bool(rec.get("terminated")) or bool(rec.get("truncated")) or bool(rec.get("done"))
        done[i] = 1.0 if dg else 0.0

    joint_action_next = np.zeros_like(joint_action_full)
    for i in range(n):
        if i == n - 1 or episodes[i + 1] != episodes[i]:
            joint_action_next[i, :] = 0.0
            done[i] = 1.0
        else:
            joint_action_next[i] = joint_action_full[i + 1]

    logger.info(
        "joint batch n=%s state_agents=%s d_state=%s d_act_full=%s done_rate=%.4f",
        n,
        state_agents,
        d_state,
        d_act_all,
        float(done.mean()),
    )

    return JointTransitionBatch(
        joint_obs_state=joint_obs_state,
        joint_action_full=joint_action_full,
        joint_next_obs_state=joint_next_state,
        joint_action_next=joint_action_next,
        reward_signal=rew_t,
        done=done,
        episode=episodes,
        agent_order_full=tuple(agent_cols),
        state_agents=tuple(state_agents),
        obs_dim=obs_dim,
        act_dim=act_dim,
    )


def subset_rows(batch: JointTransitionBatch, indices: np.ndarray) -> JointTransitionBatch:
    """按行下标取子批（评估反事实时用）。"""
    idx = np.asarray(indices, dtype=np.int64)
    return JointTransitionBatch(
        joint_obs_state=batch.joint_obs_state[idx],
        joint_action_full=batch.joint_action_full[idx],
        joint_next_obs_state=batch.joint_next_obs_state[idx],
        joint_action_next=batch.joint_action_next[idx],
        reward_signal=batch.reward_signal[idx],
        done=batch.done[idx],
        episode=batch.episode[idx],
        agent_order_full=batch.agent_order_full,
        state_agents=batch.state_agents,
        obs_dim=batch.obs_dim,
        act_dim=batch.act_dim,
    )
