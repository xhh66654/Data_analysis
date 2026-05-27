"""
结构因果建模（简化实现）：在历史轨迹上学结构方程

    a_target ≈ f( s_target, {群体平均观测/动作...}, env_pad )

约定来自 mean_field_schema.json 中的 blocks：从 mean_field_features.csv 的 mf_* 列取值；
**不把 a_target 块放入输入**（否则会与结局 Y=a_target 自回归泄露）。

边强度（可解释近似）：对已训练网络，逐个将某一父变量块置零后用 MSE 增量衡量
parent_block → a_target 的依赖强度（不等同于 Pearl 语义下的真实因果效应，需结合任务解读）。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def load_mean_field_bundle(
    mf_csv: str | Path,
    schema_json: str | Path,
    *,
    encoding: str = "utf-8-sig",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(mf_csv, encoding=encoding)
    schema = json.loads(Path(schema_json).read_text(encoding="utf-8"))
    return df, schema


def _mf_column_indices(schema: dict[str, Any]) -> list[int]:
    cols = schema.get("csv_columns") or []
    idx = []
    for j, c in enumerate(cols):
        if str(c).startswith("mf_"):
            idx.append(j)
    if not idx:
        raise ValueError("schema 中缺少 mf_* 列说明，请确认 mean_field 阶段写入了 csv_columns")
    return idx


def slices_from_schema(schema: dict[str, Any]) -> dict[str, slice]:
    out: dict[str, slice] = {}
    for b in schema.get("blocks", []):
        name = str(b["name"])
        start = int(b["start"])
        ln = int(b["length"])
        out[name] = slice(start, start + ln)
    return out


def build_scm_tensors(
    df: pd.DataFrame,
    schema: dict[str, Any],
    *,
    env_pad_dim: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """返回 X_parents (N, Dx), Y (N, Da), meta。"""
    col_idx = _mf_column_indices(schema)
    mf_mat = df.iloc[:, col_idx].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    if np.isnan(mf_mat).any():
        raise ValueError("mf_* 存在 NaN，请清洗数据")

    sl = slices_from_schema(schema)
    if "a_target" not in sl:
        raise ValueError("schema 缺少 a_target 块（结局变量 Y）")

    y = mf_mat[:, sl["a_target"]].copy()
    parent_parts: list[np.ndarray] = []
    parent_block_names: list[str] = []
    pa_sl: dict[str, slice] = {}
    cursor = 0

    ordered_keys = sorted(sl.keys(), key=lambda k: sl[k].start)
    for name in ordered_keys:
        if name == "a_target":
            continue  # Y 不作为父节点输入
        blk = mf_mat[:, sl[name]].copy()
        parent_parts.append(blk)
        L = blk.shape[1]
        pa_sl[name] = slice(cursor, cursor + L)
        cursor += L
        parent_block_names.append(name)

    if env_pad_dim > 0:
        parent_parts.append(np.zeros((mf_mat.shape[0], env_pad_dim), dtype=np.float32))
        pa_sl["env_pad"] = slice(cursor, cursor + env_pad_dim)
        cursor += env_pad_dim
        parent_block_names.append("env_pad")

    X = np.concatenate(parent_parts, axis=1)

    meta = {
        "parent_block_order": parent_block_names,
        "parent_slices_spec": {k: [sl.start, sl.stop] for k, sl in pa_sl.items()},
        "dx": int(X.shape[1]),
        "dy": int(y.shape[1]),
        "env_pad_dim": int(env_pad_dim),
        "formula": "Y=a_target <- f(parents excluding a_target)",
    }
    logger.info(
        "SCM 张量 shape X=%s Y=%s blocks=%s",
        X.shape,
        y.shape,
        parent_block_names,
    )
    return X, y, meta


def specs_to_slices(spec: dict[str, list[int]]) -> dict[str, slice]:
    """把 JSON 友好的 [start,end] 还原成 slice。"""
    return {k: slice(int(v[0]), int(v[1])) for k, v in spec.items()}


class StructuralEquationNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class SCMTrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 256
    hidden: int = 128
    device: str = "cpu"
    seed: int = 0
    val_fraction: float = 0.15


@dataclass
class SCMTrainResult:
    model: StructuralEquationNet
    final_train_mse: float
    val_mse: float
    meta: dict[str, Any]


def train_structural_equation(
    X: np.ndarray,
    y: np.ndarray,
    scm_meta_parent: dict[str, Any],
    cfg: SCMTrainConfig,
) -> SCMTrainResult:
    torch.manual_seed(cfg.seed)
    n = X.shape[0]
    if n < 16:
        raise ValueError("样本过少，暂不训练 SCM")
    dv = cfg.device

    idx = np.random.RandomState(cfg.seed).permutation(n)
    n_va = max(1, int(n * cfg.val_fraction))
    i_va = idx[:n_va]
    i_tr = idx[n_va:]

    x_tr = torch.from_numpy(X[i_tr]).to(dv, dtype=torch.float32)
    y_tr = torch.from_numpy(y[i_tr]).to(dv, dtype=torch.float32)
    x_va = torch.from_numpy(X[i_va]).to(dv, dtype=torch.float32)
    y_va = torch.from_numpy(y[i_va]).to(dv, dtype=torch.float32)

    model = StructuralEquationNet(X.shape[1], y.shape[1], hidden=cfg.hidden).to(dv)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(x_tr, y_tr)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    for ep in range(cfg.epochs):
        model.train()
        tot = 0.0
        nb = 0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        if (ep + 1) % max(1, cfg.epochs // 5) == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                vloss = loss_fn(model(x_va), y_va).item()
            logger.info(
                "SCM epoch %s/%s train_mse~%.6f val_mse=%.6f",
                ep + 1,
                cfg.epochs,
                tot / max(nb, 1),
                vloss,
            )

    model.eval()
    with torch.no_grad():
        tr_mse = float(loss_fn(model(x_tr), y_tr).item())
        val_mse = float(loss_fn(model(x_va), y_va).item())

    full_meta = dict(scm_meta_parent)
    full_meta.update(
        {
            "scm_hidden": cfg.hidden,
            "scm_epochs": cfg.epochs,
            "train_mse": tr_mse,
            "val_mse": val_mse,
            "interpretation_hint": (
                "神经结构方程近似；DAG/真实因果次序需额外假设与外生噪声建模"
            ),
        }
    )
    logger.info("SCM 训练结束 train_mse=%.6f val_mse=%.6f", tr_mse, val_mse)
    return SCMTrainResult(model=model, final_train_mse=tr_mse, val_mse=val_mse, meta=full_meta)


def edge_strength_via_ablation(
    model: StructuralEquationNet,
    X: np.ndarray,
    y: np.ndarray,
    parent_slices: dict[str, slice],
    *,
    batch_size: int = 4096,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """各父变量块整块置零，看验证 MSE 上升量。"""
    model.eval()
    dv = torch.device(device)
    xv = torch.from_numpy(X).to(dv, dtype=torch.float32)
    yv = torch.from_numpy(y).to(dv, dtype=torch.float32)
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        base_loss = float(loss_fn(model(xv), yv).item())

    edges: list[dict[str, Any]] = []
    for name, sl in parent_slices.items():
        x2 = xv.clone()
        x2[:, sl] = 0.0
        with torch.no_grad():
            lf = float(loss_fn(model(x2), yv).item())
        delta = max(0.0, lf - base_loss)
        edges.append({"parent_block": name, "mse_full_dataset": lf, "delta_mse_vs_baseline": delta})

    mx = max((e["delta_mse_vs_baseline"] for e in edges), default=1.0)
    for e in edges:
        e["relative_strength"] = float(e["delta_mse_vs_baseline"] / mx) if mx > 0 else 0.0

    edges.sort(key=lambda z: -z["delta_mse_vs_baseline"])
    logger.info("SCM 消融边强度排序: %s", [e["parent_block"] for e in edges[:5]])
    return edges, base_loss


def save_scm_checkpoint(path: str | Path, model: StructuralEquationNet, meta: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = {"state_dict": model.state_dict(), "meta": meta}
    torch.save(blob, p)
    logger.info("已保存 SCM: %s", p.resolve())


def load_scm_checkpoint(path: str | Path, device: str = "cpu") -> tuple[StructuralEquationNet, dict[str, Any]]:
    p = Path(path)
    try:
        blob = torch.load(p, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(p, map_location=device)
    meta = blob.get("meta") or {}
    dx = int(meta.get("dx", 0))
    dy = int(meta.get("dy", 0))
    if dx <= 0 or dy <= 0:
        raise ValueError(f"SCM checkpoint meta 缺少 dx/dy: {p}")
    hidden = int(meta.get("scm_hidden", 128))
    model = StructuralEquationNet(dx, dy, hidden=hidden)
    model.load_state_dict(blob["state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model, meta
