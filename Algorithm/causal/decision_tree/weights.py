"""
将 l_hat / margin 转为 VIPER 抽样权重 weights（流水线阶段 3）。

支持三种 weight_mode：
  uniform   — weights = 1/n（VIPER 退化为均匀 bootstrap）
  advantage — w_raw = max(l_hat,0)+eps（旧实现：行为动作越差权重越高）
  margin    — w_raw = max(margin,0)+eps（真正的 VIPER：按决策重要性 top1-top2 加权，推荐）

weights 为归一化概率，供 viper_cart 中 np.random.choice(..., p=weights) 有放回抽样。
eps 防止全零权重，并给权重≈0 的样本极小但非零的抽样机会。

主要函数：
  compute_weights()           — advantage 模式：max(l_hat,0)+eps 后归一化
  compute_margin_weights()    — margin 模式：max(margin,0)+eps 后归一化（真正的 VIPER）
  compute_uniform_weights()   — 均匀抽样：weights=1/n
  weights_dataframe()         — 与 l_hat.csv 行对齐，附加 w_raw、weights 列
  save_weights_csv()          — 写入 CSV
  run_weights_from_l_hat_csv()— 从 l_hat.csv 按 weight_mode 一键生成 weights.csv
  sample_row_indices()        — 按 weights 有放回抽取行索引（VIPER 重采样用）

列名常量：LHAT_COL, MARGIN_COL, W_RAW_COL, WEIGHTS_COL
输入：l_hat.csv（需含 l_hat；margin 模式额外需 margin 列）
输出：{output_dir}/weights.csv
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .trajectory_io import ACTION_COL, EPISODE_COL

logger = logging.getLogger(__name__)

DEFAULT_EPS = 1e-6
LHAT_COL = "l_hat"
MARGIN_COL = "margin"
W_RAW_COL = "w_raw"
WEIGHTS_COL = "weights"

# 可选的抽样权重模式：
#   uniform   — 每条 1/n（VIPER 退化为均匀 bootstrap）
#   advantage — w_raw = max(l_hat,0)+eps（旧实现：行为动作越差权重越高，语义存疑）
#   margin    — w_raw = margin+eps（真正的 VIPER：按「决策重要性」top1-top2 加权）
WEIGHT_MODES = ("uniform", "advantage", "margin")


@dataclass(frozen=True)
class WeightsBatch:
    l_hat: np.ndarray
    w_raw: np.ndarray
    weights: np.ndarray

    @property
    def n(self) -> int:
        return int(self.weights.shape[0])


def compute_weights(l_hat: np.ndarray, eps: float = DEFAULT_EPS) -> WeightsBatch:
    """
    w_raw_i = max(l_hat_i, 0) + eps
    weights_i = w_raw_i / sum_j w_raw_j   （供 np.random.choice(..., p=weights)）
    """
    if eps <= 0:
        raise ValueError(f"eps 须为正，得到 {eps}")
    l = np.asarray(l_hat, dtype=np.float64).reshape(-1)
    if l.size == 0:
        raise ValueError("l_hat 为空")

    w_raw = np.maximum(l, 0.0) + float(eps)
    total = float(w_raw.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError(f"w_raw 之和无效: {total}")
    weights = w_raw / total

    logger.info(
        "weights 完成 n=%d w_raw_sum=%.6f weights_min=%.6e weights_max=%.6e",
        l.size,
        total,
        float(weights.min()),
        float(weights.max()),
    )
    return WeightsBatch(l_hat=l, w_raw=w_raw, weights=weights)


def compute_margin_weights(margin: np.ndarray, eps: float = DEFAULT_EPS) -> WeightsBatch:
    """
    真正的 VIPER 加权：按「决策重要性」margin = top1(Q) - top2(Q) 加权。

      w_raw_i   = max(margin_i, 0) + eps
      weights_i = w_raw_i / sum_j w_raw_j

    与 advantage（l_hat）模式的区别：margin 衡量「这一步选错代价多大」（决策是否关键），
    而非「记录的动作有多差」，因此高权重落在真正关键的决策点上，而不是噪声样本上。
    返回结构仍复用 WeightsBatch，其 l_hat 字段存放 margin 以便对齐。
    """
    if eps <= 0:
        raise ValueError(f"eps 须为正，得到 {eps}")
    m = np.asarray(margin, dtype=np.float64).reshape(-1)
    if m.size == 0:
        raise ValueError("margin 为空")

    w_raw = np.maximum(m, 0.0) + float(eps)
    total = float(w_raw.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError(f"w_raw 之和无效: {total}")
    weights = w_raw / total
    logger.info(
        "weights 完成（margin 模式） n=%d w_raw_sum=%.6f weights_min=%.6e weights_max=%.6e",
        m.size,
        total,
        float(weights.min()),
        float(weights.max()),
    )
    return WeightsBatch(l_hat=m, w_raw=w_raw, weights=weights)


def compute_uniform_weights(l_hat: np.ndarray) -> WeightsBatch:
    """
    均匀抽样权重：每条样本 w_raw=1，weights=1/n。
    VIPER 外循环退化为对全表的有放回 bootstrap，不强调高 l_hat 样本。
    """
    l = np.asarray(l_hat, dtype=np.float64).reshape(-1)
    if l.size == 0:
        raise ValueError("l_hat 为空")
    n = l.size
    w_raw = np.ones(n, dtype=np.float64)
    weights = np.full(n, 1.0 / n, dtype=np.float64)
    logger.info(
        "weights 完成（均匀抽样） n=%d w_raw_sum=%.6f weights_min=%.6e weights_max=%.6e",
        n,
        float(w_raw.sum()),
        float(weights.min()),
        float(weights.max()),
    )
    return WeightsBatch(l_hat=l, w_raw=w_raw, weights=weights)


def load_l_hat_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"未找到 l_hat CSV: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if LHAT_COL not in df.columns:
        raise ValueError(f"l_hat CSV 缺少列 {LHAT_COL!r}")
    return df


def weights_dataframe(l_hat_df: pd.DataFrame, batch: WeightsBatch) -> pd.DataFrame:
    if len(l_hat_df) != batch.n:
        raise ValueError(f"l_hat 行数 {len(l_hat_df)} 与 weights 行数 {batch.n} 不一致")

    cols = {}
    if EPISODE_COL in l_hat_df.columns:
        cols[EPISODE_COL] = l_hat_df[EPISODE_COL].values
    if ACTION_COL in l_hat_df.columns:
        cols[ACTION_COL] = l_hat_df[ACTION_COL].values
    if MARGIN_COL in l_hat_df.columns:
        cols[MARGIN_COL] = l_hat_df[MARGIN_COL].values
    cols[LHAT_COL] = batch.l_hat
    cols[W_RAW_COL] = batch.w_raw
    cols[WEIGHTS_COL] = batch.weights
    return pd.DataFrame(cols)


def save_weights_csv(path: str | Path, df_out: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("已保存 weights: %s (%d 行)", path.resolve(), len(df_out))
    return path


def run_weights_from_l_hat_csv(
    l_hat_path: str | Path,
    *,
    output_path: str | Path | None = None,
    eps: float = DEFAULT_EPS,
    weight_mode: str = "uniform",
    weighted_sampling: bool | None = None,
) -> tuple[WeightsBatch, pd.DataFrame, Path]:
    """根据 weight_mode 由 l_hat.csv 生成 weights.csv。

    weight_mode:
      uniform   — 均匀 1/n（默认，等价旧 weighted_sampling=False）
      advantage — max(l_hat,0)+eps（旧 VIPER 实现）
      margin    — max(margin,0)+eps（真正的 VIPER 决策重要性加权，推荐）
    """
    # 兼容旧参数 viper_weighted_sampling
    if weighted_sampling is not None:
        weight_mode = "margin" if weighted_sampling else "uniform"

    if weight_mode not in WEIGHT_MODES:
        raise ValueError(f"weight_mode 须为 {WEIGHT_MODES} 之一，得到 {weight_mode!r}")

    l_hat_path = Path(l_hat_path)
    df = load_l_hat_csv(l_hat_path)
    l_hat = pd.to_numeric(df[LHAT_COL], errors="coerce").values
    if np.isnan(l_hat).any():
        raise ValueError("l_hat 列含 NaN")

    if weight_mode == "uniform":
        batch = compute_uniform_weights(l_hat)
    elif weight_mode == "advantage":
        batch = compute_weights(l_hat, eps=eps)
    else:  # margin
        if MARGIN_COL not in df.columns:
            raise ValueError(
                f"weight_mode='margin' 需要 l_hat.csv 含 {MARGIN_COL!r} 列；"
                "请用新版 l_hat 重新生成 l_hat.csv"
            )
        margin = pd.to_numeric(df[MARGIN_COL], errors="coerce").values
        if np.isnan(margin).any():
            raise ValueError("margin 列含 NaN")
        batch = compute_margin_weights(margin, eps=eps)

    df_out = weights_dataframe(df, batch)
    if output_path is None:
        output_path = l_hat_path.parent / "weights.csv"
    out_path = save_weights_csv(output_path, df_out)
    return batch, df_out, out_path


def sample_row_indices(
    weights: np.ndarray,
    n_samples: int,
    *,
    rng: np.random.Generator | None = None,
    replace: bool = True,
) -> np.ndarray:
    """按 weights 有放回抽行索引，供后续 VIPER 轮次 CART 使用。"""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size == 0:
        raise ValueError("weights 为空")
    if n_samples < 1:
        raise ValueError(f"n_samples 须 >= 1，得到 {n_samples}")
    s = float(w.sum())
    if not np.isclose(s, 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError(f"weights 之和应为 1，得到 {s}")
    if (w < 0).any():
        raise ValueError("weights 含负值")
    gen = rng if rng is not None else np.random.default_rng()
    return gen.choice(w.size, size=n_samples, replace=replace, p=w)
