"""
fqe_train — 在 step_explain 内根据轨迹 CSV 现场训练 Q 网络（FQE）。

不依赖 decision_tree 事先产出的 q_hat.pt；训练结果写入本次输出目录。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

from causal.decision_tree.fqe import FQETrainConfig, save_q_hat, train_q_hat
from causal.decision_tree.trajectory_io import (
    build_transitions,
    load_trajectory_csv,
    normalize_rewards,
    RewardNormConfig,
)

logger = logging.getLogger(__name__)


def train_fqe_for_explain(
    trajectory_csv: str | Path,
    output_dir: str | Path,
    cfg: Dict[str, Any],
) -> Tuple[Path, Dict[str, Any]]:
    """
    读取轨迹 →（可选）奖励归一化 → FQE 训练 → 保存 output_dir/q_hat.pt。

    返回 (q_hat_path, meta)。
    """
    csv_path = Path(trajectory_csv).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "q_hat.pt"

    device = str(cfg.get("fqe_device") or cfg.get("device") or "cpu")
    seed = int(cfg.get("seed", 42))

    logger.info("阶段 1/2: 读取轨迹并训练 FQE（step_explain 内置，非加载历史模型）")
    logger.info("轨迹 CSV: %s", csv_path)
    df = load_trajectory_csv(str(csv_path))
    n = len(df)
    logger.info("样本行数: %d", n)

    if bool(cfg.get("enable_reward_norm", True)):
        reward_norm_cfg = RewardNormConfig(
            clip_range=(
                float(cfg.get("reward_clip_min", -10.0)),
                float(cfg.get("reward_clip_max", 10.0)),
            ),
            standardize=True,
            per_episode=bool(cfg.get("reward_norm_per_episode", False)),
        )
        df, _ = normalize_rewards(df, reward_norm_cfg)

    trans = build_transitions(df)
    fqe_cfg = FQETrainConfig(
        gamma=float(cfg.get("fqe_gamma", 0.99)),
        lr=float(cfg.get("fqe_lr", 1e-3)),
        batch_size=int(cfg.get("fqe_batch_size", 256)),
        epochs=int(cfg.get("fqe_epochs", 30)),
        hidden=int(cfg.get("fqe_hidden", 256)),
        device=device,
        target=str(cfg.get("fqe_target", "sarsa")),
        use_target_network=bool(cfg.get("fqe_use_target_network", False)),
        target_tau=float(cfg.get("fqe_target_tau", 0.005)),
        seed=seed,
    )

    logger.info(
        "FQE 训练开始: epochs=%d batch=%d hidden=%d target=%s device=%s",
        fqe_cfg.epochs,
        fqe_cfg.batch_size,
        fqe_cfg.hidden,
        fqe_cfg.target,
        device,
    )
    result = train_q_hat(trans, fqe_cfg)

    meta = {
        "csv": str(csv_path),
        "n_samples": n,
        "gamma": fqe_cfg.gamma,
        "target": fqe_cfg.target,
        "state_dim": trans.s.shape[1],
        "n_actions": int(trans.a.max()) + 1,
        "hidden": fqe_cfg.hidden,
        "final_loss": result.final_loss,
        "loss_history": result.history,
        "fqe_epochs": fqe_cfg.epochs,
        "fqe_device": device,
        "trained_by": "step_explain",
    }
    save_q_hat(ckpt_path, result.q_net, meta)
    logger.info("FQE 训练完成 final_loss=%.6f → %s", result.final_loss, ckpt_path)
    return ckpt_path, meta
