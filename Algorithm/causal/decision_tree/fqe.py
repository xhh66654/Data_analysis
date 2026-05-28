"""
FQE（Fitted Q Evaluation）：离线拟合 Q 值函数 Q_hat(s,a)。

用轨迹中的 (s, a, r, s', done, a') 转移样本，通过 Bellman 自举最小化 TD 误差，
训练 q_network.QHatNetwork。Q 网络不直接输出策略，而是为后续 l_hat/weights 提供
「这一步动作比价值最优差多少」的标尺。

主要类型与函数：
  FQETrainConfig  — 训练超参（gamma, lr, epochs, target=sarsa|max_q, 目标网络等）
  train_q_hat()   — 在 TransitionBatch 上训练，返回 Q 网络与 loss 曲线
  save_q_hat()    — 保存 q_hat.pt（含 state_dim、n_actions、meta）
  load_q_hat()    — 加载 checkpoint 并重建 QHatNetwork

target 模式：
  sarsa  — bootstrap 用轨迹真实下一步动作 a'（贴近行为策略，默认）
  max_q  — bootstrap 用 max_a' Q(s',a')（更乐观，l_hat 往往更大）

输入：trajectory_io.build_transitions() 产出的 TransitionBatch
输出：{output_dir}/q_hat.pt
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .q_network import QHatNetwork
from .trajectory_io import TransitionBatch

logger = logging.getLogger(__name__)


@dataclass
class FQETrainConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 30
    hidden: int = 256
    device: str = "cuda"
    target: str = "sarsa"  # "sarsa" | "max_q"
    use_target_network: bool = False
    target_tau: float = 0.005
    seed: int = 0


@dataclass
class FQETrainResult:
    q_net: QHatNetwork
    final_loss: float
    history: list[float]


def _bootstrap_target(
    q_online: QHatNetwork,
    q_target: QHatNetwork,
    s_next: torch.Tensor,
    a_next: torch.Tensor,
    r: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
    mode: str,
) -> torch.Tensor:
    with torch.no_grad():
        q_next = q_target(s_next)
        if mode == "sarsa":
            q_boot = q_next.gather(1, a_next.long().view(-1, 1)).squeeze(1)
        elif mode == "max_q":
            q_boot = q_next.max(dim=1).values
        else:
            raise ValueError(f"未知 FQE target 模式: {mode!r}，应为 sarsa 或 max_q")
        not_done = 1.0 - done
        return r + gamma * not_done * q_boot


def train_q_hat(data: TransitionBatch, cfg: FQETrainConfig) -> FQETrainResult:
    if data.n < 2:
        raise ValueError("转移样本过少，无法训练 Q_hat")
    if cfg.target not in ("sarsa", "max_q"):
        raise ValueError("cfg.target 须为 sarsa 或 max_q")

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    state_dim = data.s.shape[1]
    n_actions = int(data.a.max()) + 1
    amax = int(data.a.max())
    amin = int(data.a.min())
    if amin < 0:
        raise ValueError(f"动作 id 不能为负: min={amin}")
    if amax >= n_actions:
        raise ValueError(f"动作最大值 {amax} 超出推断的 n_actions={n_actions}")

    q_net = QHatNetwork(state_dim, n_actions, hidden=cfg.hidden).to(device)
    q_tgt = QHatNetwork(state_dim, n_actions, hidden=cfg.hidden).to(device)
    q_tgt.load_state_dict(q_net.state_dict())
    for p in q_tgt.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    
    # 添加学习率调度器（余弦退火）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    
    # 梯度裁剪阈值
    grad_clip_norm = 1.0

    ds = TensorDataset(
        torch.from_numpy(data.s),
        torch.from_numpy(data.a),
        torch.from_numpy(data.r),
        torch.from_numpy(data.s_next),
        torch.from_numpy(data.done),
        torch.from_numpy(data.a_next),
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    history: list[float] = []
    logger.info(
        "FQE 开始 n=%d n_actions=%d target=%s epochs=%d device=%s",
        data.n,
        n_actions,
        cfg.target,
        cfg.epochs,
        device,
    )

    for epoch in range(1, cfg.epochs + 1):
        epoch_losses: list[float] = []
        for s_b, a_b, r_b, sn_b, d_b, an_b in loader:
            s_b = s_b.to(device)
            a_b = a_b.to(device)
            r_b = r_b.to(device)
            sn_b = sn_b.to(device)
            d_b = d_b.to(device)
            an_b = an_b.to(device)

            tgt = _bootstrap_target(
                q_net,
                q_tgt if cfg.use_target_network else q_net,
                sn_b,
                an_b,
                r_b,
                d_b,
                cfg.gamma,
                cfg.target,
            )
            q_pred = q_net.q_value(s_b, a_b)
            loss = loss_fn(q_pred, tgt)
            opt.zero_grad()
            loss.backward()
            
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), grad_clip_norm)
            
            opt.step()
            scheduler.step()  # 每步更新学习率

            if cfg.use_target_network:
                with torch.no_grad():
                    for p, pt in zip(q_net.parameters(), q_tgt.parameters()):
                        pt.data.mul_(1.0 - cfg.target_tau)
                        pt.data.add_(cfg.target_tau * p.data)

            epoch_losses.append(float(loss.item()))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.append(mean_loss)
        logger.info("FQE epoch %d/%d loss=%.6f", epoch, cfg.epochs, mean_loss)

    return FQETrainResult(q_net=q_net, final_loss=history[-1] if history else float("nan"), history=history)


def save_q_hat(path: str | Path, q_net: QHatNetwork, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": q_net.state_dict(), "meta": meta}
    torch.save(payload, path)
    logger.info("已保存 Q_hat: %s", path.resolve())


def load_q_hat(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[QHatNetwork, dict]:
    """从 q_hat.pt 恢复网络并冻结参数。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"未找到 Q_hat 检查点: {path}")

    dev = torch.device(device)
    payload = torch.load(path, map_location=dev, weights_only=False)
    sd = payload["state_dict"]
    w0 = sd["net.0.weight"]
    w4 = sd["net.4.weight"]
    hidden = int(w0.shape[0])
    state_dim = int(w0.shape[1])
    n_actions = int(w4.shape[0])

    q_net = QHatNetwork(state_dim, n_actions, hidden=hidden).to(dev)
    q_net.load_state_dict(sd)
    q_net.eval()
    for p in q_net.parameters():
        p.requires_grad = False

    meta = dict(payload.get("meta") or {})
    logger.info(
        "已加载 Q_hat: %s state_dim=%d n_actions=%d hidden=%d",
        path.resolve(),
        state_dim,
        n_actions,
        hidden,
    )
    return q_net, meta
