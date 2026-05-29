"""
将 l_hat 转为 VIPER 抽样权重 weights（流水线阶段 3）。

公式（与 VIPER 论文一致）：
  w_raw_i   = max(l_hat_i, 0) + eps
  weights_i = w_raw_i / sum_j(w_raw_j)

weights 为归一化概率，供 viper_cart 中 np.random.choice(..., p=weights) 有放回抽样。
eps 防止全零权重，并给 l_hat≈0 的样本极小但非零的抽样机会。

主要函数：
  compute_weights()           — VIPER 加权：max(l_hat,0)+eps 后归一化
  compute_uniform_weights()   — 均匀抽样：weights=1/n
  weights_dataframe()         — 与 l_hat.csv 行对齐，附加 w_raw、weights 列
  save_weights_csv()          — 写入 CSV
  run_weights_from_l_hat_csv()— 从 l_hat.csv 一键生成 weights.csv
  sample_row_indices()        — 按 weights 有放回抽取行索引（VIPER 重采样用）

列名常量：LHAT_COL, W_RAW_COL, WEIGHTS_COL
输入：l_hat.csv（或 l_hat 数组 + eps）
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
W_RAW_COL = "w_raw"
WEIGHTS_COL = "weights"


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


def compute_mixed_weights(l_hat: np.ndarray, alpha: float = 0.5, eps: float = DEFAULT_EPS) -> WeightsBatch:
    """
    混合策略权重：平衡 softmax 加权与均匀抽样。
    
    weights = alpha * softmax(l_hat) + (1 - alpha) * uniform
    
    参数：
        alpha: 混合系数，0=完全均匀，1=完全 softmax
        eps: 平滑项
    
    特点：
        - alpha=0.7 时：70% softmax + 30% 均匀，既利用高 l_hat 样本，又保持多样性
        - 使用 softmax 而非简单归一化，权重分布更平滑
    """
    if eps <= 0:
        raise ValueError(f"eps 须为正，得到 {eps}")
    if not (0 <= alpha <= 1):
        raise ValueError(f"alpha 须在 [0,1] 范围内，得到 {alpha}")
    
    l = np.asarray(l_hat, dtype=np.float64).reshape(-1)
    if l.size == 0:
        raise ValueError("l_hat 为空")
    
    n = l.size
    
    # Softmax 加权（使用非负部分）
    pos_lhat = np.maximum(l, 0.0)
    lhat_std = pos_lhat.std() + 1e-8
    
    if pos_lhat.max() > 0:
        # 用标准差归一化后再 softmax，避免极端值主导
        exp_lhat = np.exp(pos_lhat / lhat_std)
        softmax_weights = exp_lhat / exp_lhat.sum()
    else:
        softmax_weights = np.ones(n) / n
    
    # 均匀权重
    uniform_weights = np.ones(n) / n
    
    # 混合
    weights = alpha * softmax_weights + (1 - alpha) * uniform_weights
    
    # 添加平滑项并重新归一化
    weights = weights + eps
    weights = weights / weights.sum()
    
    w_raw = weights * n  # 伪 raw 权重（用于保持接口一致）
    
    logger.info(
        "混合权重完成 n=%d alpha=%.2f weights_min=%.6e weights_max=%.6e",
        n,
        alpha,
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
    weighted_sampling: bool = True,
    mixed_alpha: float = 0.7,  # 混合策略系数
) -> tuple[WeightsBatch, pd.DataFrame, Path]:
    l_hat_path = Path(l_hat_path)
    df = load_l_hat_csv(l_hat_path)
    l_hat = pd.to_numeric(df[LHAT_COL], errors="coerce").values
    if np.isnan(l_hat).any():
        raise ValueError("l_hat 列含 NaN")
    
    if weighted_sampling:
        # 使用混合策略
        batch = compute_mixed_weights(l_hat, alpha=mixed_alpha, eps=eps)
    else:
        batch = compute_uniform_weights(l_hat)
    
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
