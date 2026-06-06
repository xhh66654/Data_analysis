"""
轨迹 CSV 读入、列名校验与 MDP 转移样本构造（流水线数据层）。

定义轨迹 CSV 的标准列名常量：
  STATE_COLS / NEXT_STATE_COLS — s_0…s_7、s_next_0…s_next_7（8 维状态）
  ACTION_COL, REWARD_COL, EPISODE_COL, DW_COL, TRUNC_COL

主要类型与函数：
  TransitionBatch   — FQE 训练用的 numpy 批次：(s, a, r, s_next, done, a_next)
  load_trajectory_csv() — 读 CSV 并校验必需列是否存在
  build_transitions()   — 由 DataFrame 构造转移：
      · 同 episode 内：a_next[i] = action[i+1]（SARSA 链）
      · episode 末行 / dw / truncated → done=1
      · episode 断裂时 a_next 置 0（训练时由 done mask 忽略）

输入：causal/main.py 等导出的轨迹 CSV
输出：TransitionBatch（供 fqe.train_q_hat）或原始 DataFrame（供 VIPER 取 X,y）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STATE_COLS = [f"s_{i}" for i in range(8)]
NEXT_STATE_COLS = [f"s_next_{i}" for i in range(8)]
ACTION_COL = "action"
REWARD_COL = "reward"
EPISODE_COL = "episode"
DW_COL = "dw"
TRUNC_COL = "truncated"


@dataclass(frozen=True)
class TransitionBatch:
    s: np.ndarray
    a: np.ndarray
    r: np.ndarray
    s_next: np.ndarray
    done: np.ndarray
    a_next: np.ndarray

    @property
    def n(self) -> int:
        return int(self.s.shape[0])


@dataclass
class RewardNormConfig:
    clip_range: tuple[float, float] = (-10.0, 10.0)
    standardize: bool = True
    per_episode: bool = False


def normalize_rewards(
    df: pd.DataFrame,
    cfg: RewardNormConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    奖励归一化：提升 FQE 训练稳定性。
    
    处理策略：
    1. 裁剪极端奖励值到 [clip_min, clip_max]
    2. 可选：标准化为均值=0，标准差=1
    3. 可选：按 episode 分别归一化（保留相对奖励结构）
    """
    if cfg is None:
        cfg = RewardNormConfig()
    
    df = df.copy()
    rewards = df[REWARD_COL].values.astype(np.float64)
    
    clip_min, clip_max = cfg.clip_range
    original_min, original_max = rewards.min(), rewards.max()
    rewards = np.clip(rewards, clip_min, clip_max)
    
    clipped_count = np.sum((rewards == clip_min) | (rewards == clip_max))
    if clipped_count > 0:
        logger.info("奖励裁剪：%d 个值被裁剪 (范围 [%.2f, %.2f] → [%.2f, %.2f])",
                    clipped_count, original_min, original_max, clip_min, clip_max)
    
    if cfg.standardize:
        if cfg.per_episode:
            eps = df[EPISODE_COL].values
            normalized = np.zeros_like(rewards)
            for ep_id in np.unique(eps):
                mask = eps == ep_id
                ep_rewards = rewards[mask]
                mean = ep_rewards.mean()
                std = ep_rewards.std()
                if std > 1e-8:
                    normalized[mask] = (ep_rewards - mean) / std
                else:
                    normalized[mask] = ep_rewards - mean
            rewards = normalized
        else:
            mean = rewards.mean()
            std = rewards.std()
            if std > 1e-8:
                rewards = (rewards - mean) / std
            else:
                rewards = rewards - mean
    
    df[REWARD_COL] = rewards.astype(np.float32)
    
    final_mean = float(rewards.mean())
    final_std = float(rewards.std())
    logger.info("奖励归一化完成：均值=%.6f 标准差=%.6f", final_mean, final_std)

    stats = {
        "clipped_count": int(clipped_count),
        "clip_range": [float(clip_min), float(clip_max)],
        "standardize": cfg.standardize,
        "per_episode": cfg.per_episode,
    }
    return df, stats


def load_trajectory_csv(path: str) -> pd.DataFrame:
    """[DATA-CROP-00] 读入全量轨迹，不删行（行数 = CSV 数据行数）。"""
    need = STATE_COLS + NEXT_STATE_COLS + [ACTION_COL, REWARD_COL, EPISODE_COL, DW_COL, TRUNC_COL]
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"轨迹 CSV 缺少列: {missing}")
    return df


def build_transitions(df: pd.DataFrame) -> TransitionBatch:
    """
    逐步转移：
    - 若 episode[i+1]==episode[i]，则 a_next[i]=action[i+1]；
    - 否则视为终止（done=1），a_next 置 0（训练时 mask 掉）。
    - 同时 episode 末行、dw、truncated 任一成立则 done=1。
    """
    n = len(df)
    if n == 0:
        raise ValueError("轨迹 CSV 无数据行")

    s = df[STATE_COLS].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    s_next = df[NEXT_STATE_COLS].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    a = pd.to_numeric(df[ACTION_COL], errors="coerce").values.astype(np.int64)
    r = pd.to_numeric(df[REWARD_COL], errors="coerce").values.astype(np.float32)
    ep = pd.to_numeric(df[EPISODE_COL], errors="coerce").values.astype(np.int64)
    dw = pd.to_numeric(df[DW_COL], errors="coerce").fillna(0).values.astype(np.float32)
    tr = pd.to_numeric(df[TRUNC_COL], errors="coerce").fillna(0).values.astype(np.float32)

    if np.isnan(s).any() or np.isnan(s_next).any() or np.isnan(a).any() or np.isnan(r).any():
        raise ValueError("状态/动作/奖励存在 NaN，请清洗轨迹后再训练")

    done = np.zeros(n, dtype=np.float32)
    a_next = np.zeros(n, dtype=np.int64)

    for i in range(n):                    
        episode_break = i == n - 1 or ep[i + 1] != ep[i]
        if episode_break:
            done[i] = 1.0
            a_next[i] = 0
        else:
            a_next[i] = a[i + 1]
            if dw[i] != 0 or tr[i] != 0:
                done[i] = 1.0
            else:
                done[i] = 0.0

    logger.info(
        "转移构造完成 n=%d done_rate=%.4f",
        n,
        float(done.mean()),
    )
    return TransitionBatch(s=s, a=a, r=r, s_next=s_next, done=done, a_next=a_next)
