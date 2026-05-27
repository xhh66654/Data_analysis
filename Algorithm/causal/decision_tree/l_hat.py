"""
逐行计算 VIPER 损失 l_hat，并写入 l_hat.csv（流水线阶段 2）。

在冻结的 Q_hat 上，对轨迹每一行 i 计算：
  q_all_i = Q_hat(s_i, ·)     — 所有动作的 Q 值
  V_hat_i = max(q_all_i)      — 状态价值上界
  Q_sa_i  = Q_hat(s_i, a_i)   — 实际执行动作的 Q 值
  l_hat_i = V_hat_i - Q_sa_i  — 价值差距，越大表示该步越「值得纠错」

l_hat 不修改 CSV 中的 action 标签，仅附加评分列，供 weights 模块转为抽样概率。

主要函数：
  compute_l_hat()      — 批量前向推理，返回 LHatBatch（含 q_all, V_hat, Q_sa, l_hat）
  l_hat_dataframe()    — 合并 episode/action 等列，组装 DataFrame
  save_l_hat_csv()     — 写入 CSV
  run_l_hat_from_csv() — 从轨迹 CSV + q_hat.pt 一键计算并保存

输入：q_hat.pt + 轨迹 states/actions
输出：{output_dir}/l_hat.csv
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .fqe import load_q_hat
from .q_network import QHatNetwork
from .trajectory_io import ACTION_COL, EPISODE_COL, STATE_COLS, load_trajectory_csv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LHatBatch:
    """逐行 VIPER 损失 l_hat = V_hat(s) - Q_hat(s, a)。"""

    q_all: np.ndarray  # (n, n_actions)
    V_hat: np.ndarray  # (n,)
    Q_sa: np.ndarray  # (n,)
    l_hat: np.ndarray  # (n,)

    @property
    def n(self) -> int:
        return int(self.l_hat.shape[0])


def compute_l_hat(
    q_net: QHatNetwork,
    states: np.ndarray,
    actions: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 4096,
) -> LHatBatch:
    """
    冻结 Q_hat，对每行 i：
      q_all_i = Q_hat(s_i, ·)
      V_hat_i = max(q_all_i)
      Q_sa_i = q_all_i[a_i]
      l_hat_i = V_hat_i - Q_sa_i
    """
    n = states.shape[0]
    if n == 0:
        raise ValueError("states 为空")
    if actions.shape[0] != n:
        raise ValueError("states 与 actions 行数不一致")

    dev = torch.device(device)
    q_net = q_net.to(dev)
    q_net.eval()
    for p in q_net.parameters():
        p.requires_grad = False

    n_actions = q_net.net[-1].out_features
    q_all = np.zeros((n, n_actions), dtype=np.float32)
    V_hat = np.zeros(n, dtype=np.float32)
    Q_sa = np.zeros(n, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            s_t = torch.from_numpy(states[start:end]).to(dev)
            a_t = torch.from_numpy(actions[start:end]).long().to(dev)
            q_t = q_net(s_t)
            q_np = q_t.cpu().numpy()
            q_all[start:end] = q_np
            V_hat[start:end] = q_t.max(dim=1).values.cpu().numpy()
            Q_sa[start:end] = q_net.q_value(s_t, a_t).cpu().numpy()

    l_hat = V_hat - Q_sa
    logger.info(
        "l_hat 完成 n=%d l_mean=%.6f l_std=%.6f l_max=%.6f",
        n,
        float(l_hat.mean()),
        float(l_hat.std()),
        float(l_hat.max()),
    )
    return LHatBatch(q_all=q_all, V_hat=V_hat, Q_sa=Q_sa, l_hat=l_hat)


def l_hat_dataframe(df: pd.DataFrame, batch: LHatBatch) -> pd.DataFrame:
    """在轨迹行上附加 Q 与 l_hat 列（与 CSV 行一一对齐）。"""
    if len(df) != batch.n:
        raise ValueError(f"DataFrame 行数 {len(df)} 与 l_hat 行数 {batch.n} 不一致")

    out = pd.DataFrame(
        {
            EPISODE_COL: df[EPISODE_COL].values,
            ACTION_COL: df[ACTION_COL].values,
            "V_hat": batch.V_hat,
            "Q_sa": batch.Q_sa,
            "l_hat": batch.l_hat,
        }
    )
    for j in range(batch.q_all.shape[1]):
        out[f"Q_hat_a{j}"] = batch.q_all[:, j]
    return out


def save_l_hat_csv(path: str | Path, df_out: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("已保存 l_hat: %s (%d 行)", path.resolve(), len(df_out))
    return path


def run_l_hat_from_csv(
    csv_path: str | Path,
    checkpoint: str | Path,
    *,
    output_path: str | Path | None = None,
    device: str = "cpu",
    batch_size: int = 4096,
) -> tuple[LHatBatch, pd.DataFrame, Path]:
    csv_path = Path(csv_path)
    checkpoint = Path(checkpoint)
    if not csv_path.is_file():
        raise FileNotFoundError(f"未找到轨迹 CSV: {csv_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"未找到 Q_hat 检查点: {checkpoint}")

    df = load_trajectory_csv(str(csv_path))
    states = df[STATE_COLS].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    actions = pd.to_numeric(df[ACTION_COL], errors="coerce").values.astype(np.int64)

    q_net, _meta = load_q_hat(checkpoint, device=device)
    batch = compute_l_hat(q_net, states, actions, device=device, batch_size=batch_size)
    df_out = l_hat_dataframe(df, batch)

    if output_path is None:
        output_path = checkpoint.parent / "l_hat.csv"
    out_path = save_l_hat_csv(output_path, df_out)
    return batch, df_out, out_path
