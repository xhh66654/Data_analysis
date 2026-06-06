"""离线 SARSA 式 Bellman：拟合标量联合 Q(s,a)，动作为连续向量拼接。"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .joint_q_network import JointQNetwork

logger = logging.getLogger(__name__)


@dataclass
class JointQTrainConfig:
    gamma: float = 0.95
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 40
    hidden: int = 256
    device: str = "cpu"
    use_target_network: bool = True
    target_tau: float = 0.005
    seed: int = 42


@dataclass
class JointQTrainResult:
    q_net: JointQNetwork
    final_loss: float
    history: list[float]


def train_joint_q(
    s: np.ndarray,
    a: np.ndarray,
    r: np.ndarray,
    s_next: np.ndarray,
    a_next: np.ndarray,
    done: np.ndarray,
    cfg: JointQTrainConfig,
) -> JointQTrainResult:
    if s.shape[0] < 2:
        raise ValueError("样本过少，无法训练联合 Q")

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    state_dim = int(s.shape[1])
    act_dim = int(a.shape[1])

    q = JointQNetwork(state_dim, act_dim, hidden=cfg.hidden).to(device)
    q_tgt = JointQNetwork(state_dim, act_dim, hidden=cfg.hidden).to(device)
    q_tgt.load_state_dict(q.state_dict())
    if cfg.use_target_network:
        for p in q_tgt.parameters():
            p.requires_grad = False
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(
        torch.from_numpy(s),
        torch.from_numpy(a),
        torch.from_numpy(r),
        torch.from_numpy(s_next),
        torch.from_numpy(a_next),
        torch.from_numpy(done),
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    history: list[float] = []
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for sb, ab, rb, snb, anb, dnb in loader:
            sb = sb.to(device, dtype=torch.float32)
            ab = ab.to(device, dtype=torch.float32)
            rb = rb.to(device, dtype=torch.float32)
            snb = snb.to(device, dtype=torch.float32)
            anb = anb.to(device, dtype=torch.float32)
            dnb = dnb.to(device, dtype=torch.float32)

            with torch.no_grad():
                qt = q_tgt.forward_sa(snb, anb)
                target = rb + cfg.gamma * (1.0 - dnb) * qt

            pred = q.forward_sa(sb, ab)
            loss = loss_fn(pred, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if cfg.use_target_network:
                with torch.no_grad():
                    for tp, sp in zip(q_tgt.parameters(), q.parameters(), strict=True):
                        tp.mul_(1.0 - cfg.target_tau).add_(sp, alpha=cfg.target_tau)

            epoch_loss += float(loss.item())
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        history.append(avg)
        if (epoch + 1) % max(1, cfg.epochs // 5) == 0 or epoch == 0:
            logger.info("joint Q epoch %s/%s loss=%.6f", epoch + 1, cfg.epochs, avg)

    if not history:
        final_loss = 0.0
    else:
        final_loss = float(history[-1])

    logger.info("joint Q 训练结束 final_loss=%.6f", final_loss)
    return JointQTrainResult(q_net=q, final_loss=final_loss, history=history)


def save_joint_q(path: str | Path, q: JointQNetwork, meta: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = {"state_dict": q.state_dict(), "meta": meta}
    torch.save(blob, p)
    logger.info("已保存 joint Q: %s", p.resolve())


def load_joint_q(path: str | Path, device: str = "cpu") -> tuple[JointQNetwork, dict]:
    p = Path(path)
    try:
        blob = torch.load(p, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(p, map_location=device)
    meta = blob.get("meta", {})
    state_dim = int(meta["state_dim"])
    act_dim = int(meta["action_dim"])
    hidden = int(meta.get("hidden", 256))
    q = JointQNetwork(state_dim, act_dim, hidden=hidden).to(torch.device(device))
    q.load_state_dict(blob["state_dict"])
    q.eval()
    return q, meta
