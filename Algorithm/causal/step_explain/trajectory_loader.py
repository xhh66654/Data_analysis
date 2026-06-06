"""
trajectory_loader — 轨迹 CSV 加载与单步定位。

复用 decision_tree.trajectory_io 的常量，额外支持：
  · 按 (episode, global_step) 精确定位一行
  · 按行号 (row_index) 定位一行
  · 宽松模式：CSV 缺少 dw/truncated 时自动填 0（推理时常见）

主要函数：
  load_csv(path)       → pd.DataFrame，校验必需列
  locate_row(df, ...)  → LocatedStep（含 Series + 行号）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 复用 decision_tree 常量
STATE_COLS = [f"s_{i}" for i in range(8)]
NEXT_STATE_COLS = [f"s_next_{i}" for i in range(8)]
ACTION_COL = "action"
REWARD_COL = "reward"
EPISODE_COL = "episode"
GLOBAL_STEP_COL = "global_step"
DW_COL = "dw"
TRUNC_COL = "truncated"

# 必需核心列（dw/truncated 可选，缺失自动补 0）
_REQUIRED_COLS = STATE_COLS + [ACTION_COL, REWARD_COL, EPISODE_COL]
_OPTIONAL_FILL = {DW_COL: 0, TRUNC_COL: 0}
# s_next 列为可选（没有也能解释，只是无法做下一步追溯）


@dataclass
class LocatedStep:
    """定位到的单步数据。"""
    row_index: int
    episode: int
    global_step: Optional[int]       # 若 CSV 无此列则为 None
    action: int
    reward: float
    state: np.ndarray                 # shape (8,)
    state_next: Optional[np.ndarray]  # shape (8,) 或 None
    raw: pd.Series                    # 原始行（调试用）


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    读取轨迹 CSV，校验必需列，自动补全可选列。

    支持编码：utf-8-sig（Windows Excel 导出带 BOM）和 utf-8。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"轨迹 CSV 不存在: {path}")

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="utf-8")

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"轨迹 CSV 缺少必需列: {missing}\n当前列: {df.columns.tolist()}")

    for col, fill_val in _OPTIONAL_FILL.items():
        if col not in df.columns:
            logger.warning("CSV 缺少列 %s，自动填充 %s", col, fill_val)
            df[col] = fill_val

    logger.info("已加载轨迹 CSV: %s  行数=%d  episode 数=%d",
                path.name, len(df), df[EPISODE_COL].nunique())
    return df


def locate_row(
    df: pd.DataFrame,
    *,
    episode: Optional[int] = None,
    global_step: Optional[int] = None,
    row_index: Optional[int] = None,
) -> LocatedStep:
    """
    定位一步数据，优先级：row_index > (episode, global_step) > (episode 首行)。

    参数
    ----
    df          : load_csv() 返回的 DataFrame
    episode     : 局号（int）
    global_step : 全局步号（int）；若 CSV 无此列，则取 episode 内第 global_step 行
    row_index   : 直接指定 DataFrame 绝对行号，优先于 episode/global_step
    """
    if row_index is not None:
        if row_index < 0 or row_index >= len(df):
            raise IndexError(f"row_index={row_index} 超出范围 [0, {len(df)-1}]")
        row = df.iloc[row_index]
        idx = row_index
    elif episode is not None:
        ep_mask = df[EPISODE_COL] == episode
        ep_df = df[ep_mask]
        if len(ep_df) == 0:
            raise ValueError(f"episode={episode} 在 CSV 中不存在")

        if global_step is not None:
            if GLOBAL_STEP_COL in df.columns:
                mask = ep_mask & (df[GLOBAL_STEP_COL] == global_step)
                matched = df[mask]
                if len(matched) == 0:
                    raise ValueError(
                        f"episode={episode}, global_step={global_step} 在 CSV 中不存在"
                    )
                idx = int(matched.index[0])
            else:
                # 无 global_step 列：按 episode 内偏移取第 global_step 行
                if global_step >= len(ep_df):
                    raise IndexError(
                        f"episode={episode} 仅有 {len(ep_df)} 步，"
                        f"step offset={global_step} 越界"
                    )
                idx = int(ep_df.index[global_step])
            row = df.loc[idx]
        else:
            # 未指定 step，取该局第一行
            idx = int(ep_df.index[0])
            row = df.loc[idx]
            logger.warning(
                "未指定 global_step，使用 episode=%d 的第一步 (row=%d)", episode, idx
            )
    else:
        raise ValueError("必须提供 row_index 或 episode（+可选 global_step）")

    state = np.array([float(row[c]) for c in STATE_COLS], dtype=np.float32)

    has_next = all(c in df.columns for c in NEXT_STATE_COLS)
    state_next = (
        np.array([float(row[c]) for c in NEXT_STATE_COLS], dtype=np.float32)
        if has_next else None
    )

    gs = int(row[GLOBAL_STEP_COL]) if GLOBAL_STEP_COL in df.columns else None

    return LocatedStep(
        row_index=int(idx),
        episode=int(row[EPISODE_COL]),
        global_step=gs,
        action=int(row[ACTION_COL]),
        reward=float(row[REWARD_COL]),
        state=state,
        state_next=state_next,
        raw=row,
    )
