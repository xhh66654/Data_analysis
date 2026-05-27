"""
SCM 之上的反事实溯因：对父变量块做缩放干预（近似「削弱/移除」平均场因子），重推 a_hat，
并比较连续动作变化；可选「行为原型」将连续动作映射为伪概率分布（叙事用，非策略真 softmax）。

不依赖人工改表：干预规则由尺度集合 +（可选）边强度先验自动生成。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .scm_learning import (
    StructuralEquationNet,
    build_scm_tensors,
    specs_to_slices,
)

logger = logging.getLogger(__name__)


def load_behavior_prototypes_json(path: str | Path | None) -> dict[str, np.ndarray]:
    if path is None or (isinstance(path, str) and not path.strip()):
        return {}
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"行为原型文件不存在: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("行为原型 JSON 须为 {\"name\": [float,...], ...}")
    out: dict[str, np.ndarray] = {}
    for k, v in raw.items():
        out[str(k)] = np.asarray(v, dtype=np.float32).reshape(-1)
    return out


def _softmax_proto_logits(a: np.ndarray, protos: dict[str, np.ndarray], temperature: float) -> dict[str, float]:
    """负 L2 距离作 logit，越大越接近该原型。"""
    if not protos:
        return {}
    t = max(float(temperature), 1e-6)
    names: list[str] = []
    logits: list[float] = []
    for name, p in protos.items():
        if p.size != a.size:
            raise ValueError(f"原型 {name} 维数 {p.size} 与动作维 {a.size} 不符")
        d = float(np.linalg.norm(a - p))
        logits.append(-(d * d) / t)
        names.append(name)
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z)
    w = np.exp(z)
    w = w / (w.sum() + 1e-12)
    return {names[i]: float(w[i]) for i in range(len(names))}


def apply_parent_block_scales(
    x_row: np.ndarray,
    parent_slices: dict[str, slice],
    scales: dict[str, float],
) -> np.ndarray:
    """对每个父块乘以 scale（1=不变，0=移除，0.5=削弱）。未出现的块不改。"""
    z = x_row.copy().reshape(1, -1)
    for blk, sc in scales.items():
        if blk not in parent_slices:
            logger.warning("干预块未在 SCM 父变量中: %s", blk)
            continue
        sl = parent_slices[blk]
        z[:, sl] = z[:, sl] * float(sc)
    return z.reshape(-1)


def predict_action_vector(
    model: StructuralEquationNet,
    x_row: np.ndarray,
    *,
    device: str = "cpu",
) -> np.ndarray:
    model.eval()
    dv = torch.device(device)
    with torch.no_grad():
        xt = torch.from_numpy(x_row.reshape(1, -1)).to(dv, dtype=torch.float32)
        pred = model(xt).cpu().numpy().reshape(-1)
    return pred.astype(np.float32)


def _delta_metrics(a_base: np.ndarray, a_cf: np.ndarray, a_true: np.ndarray | None) -> dict[str, float]:
    out: dict[str, float] = {
        "l2_pred_change": float(np.linalg.norm(a_cf - a_base)),
        "cosine_similarity": _cosine(a_base, a_cf),
    }
    if a_true is not None:
        out["mse_counterfactual_vs_true_action"] = float(np.mean((a_cf - a_true) ** 2))
        out["mse_baseline_vs_true_action"] = float(np.mean((a_base - a_true) ** 2))
    return out


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u) * np.linalg.norm(v)
    if nu < 1e-12:
        return 1.0
    return float(np.dot(u, v) / nu)


def run_counterfactual_abduction(
    mf_df,
    mf_schema: dict[str, Any],
    model: StructuralEquationNet,
    scm_train_meta: dict[str, Any],
    *,
    row_indices: list[int],
    scales: list[float],
    device: str = "cpu",
    behavior_prototypes: dict[str, np.ndarray] | None = None,
    behavior_temperature: float = 1.0,
    edge_json_path: str | Path | None = None,
    max_blocks_from_edges: int = 6,
    max_joint_pairs: int = 10,
    env_pad_dim: int | None = None,
) -> dict[str, Any]:
    """
    对每个样本行：
      - 基线 x 上得到 a_hat_0；
      - 单边：对每个优先父块尝试 scale ∈ scales；
      - 联合：在非 env_pad 的前若干块中取两两组合，整块置零（scale=0），最多 max_joint_pairs 条。
    """
    env_pad = int(env_pad_dim if env_pad_dim is not None else scm_train_meta.get("env_pad_dim", 0))
    X, Y, scm_pre = build_scm_tensors(mf_df, mf_schema, env_pad_dim=env_pad)
    parent_slices = specs_to_slices(scm_pre["parent_slices_spec"])

    block_order: list[str] = []
    if edge_json_path and Path(edge_json_path).is_file():
        ej = json.loads(Path(edge_json_path).read_text(encoding="utf-8"))
        for e in ej.get("edges_sorted_by_delta_mse", []):
            b = str(e.get("parent_block", ""))
            if b and b in parent_slices and b not in block_order:
                block_order.append(b)
    for b in scm_pre["parent_block_order"]:
        if b not in block_order:
            block_order.append(b)
    block_order = block_order[: max(1, max_blocks_from_edges)]

    protos = behavior_prototypes or {}
    rows_out: list[dict[str, Any]] = []

    for ri in row_indices:
        if ri < 0 or ri >= len(mf_df):
            logger.warning("跳过越界行 %s", ri)
            continue
        x0 = X[ri].copy()
        y_true = Y[ri].copy()
        a0 = predict_action_vector(model, x0, device=device)

        row_pack: dict[str, Any] = {
            "row_index": ri,
            "episode": float(mf_df.iloc[ri].get("episode", np.nan)),
            "global_step": float(mf_df.iloc[ri].get("global_step", np.nan)),
            "true_action_y": y_true.tolist(),
            "baseline_predicted_action": a0.tolist(),
        }
        if protos:
            row_pack["baseline_behavior_pseudo_prob"] = _softmax_proto_logits(a0, protos, behavior_temperature)
            row_pack["true_behavior_pseudo_prob"] = _softmax_proto_logits(y_true, protos, behavior_temperature)

        singles: list[dict[str, Any]] = []
        for blk in block_order:
            for sc in scales:
                if blk == "env_pad" and sc == 0.0:
                    continue
                x1 = apply_parent_block_scales(x0, parent_slices, {blk: sc})
                a1 = predict_action_vector(model, x1, device=device)
                rec: dict[str, Any] = {
                    "intervention_type": "single_block_scale",
                    "block": blk,
                    "scale": float(sc),
                    "predicted_action": a1.tolist(),
                    **_delta_metrics(a0, a1, y_true),
                }
                if protos:
                    rec["behavior_pseudo_prob"] = _softmax_proto_logits(a1, protos, behavior_temperature)
                    if "baseline_behavior_pseudo_prob" in row_pack and rec["behavior_pseudo_prob"]:
                        k0 = row_pack["baseline_behavior_pseudo_prob"]
                        k1 = rec["behavior_pseudo_prob"]
                        common = set(k0) & set(k1)
                        if common:
                            top = max(common, key=lambda k: abs(k0.get(k, 0) - k1.get(k, 0)))
                            rec["largest_behavior_prob_drop"] = {
                                "behavior": top,
                                "baseline_p": k0.get(top, 0.0),
                                "counterfactual_p": k1.get(top, 0.0),
                                "delta_p": k1.get(top, 0.0) - k0.get(top, 0.0),
                            }
                singles.append(rec)

        joints: list[dict[str, Any]] = []
        topb = [b for b in block_order if b != "env_pad"]
        pair_list = [(topb[i], topb[j]) for i in range(len(topb)) for j in range(i + 1, len(topb))]
        for b1, b2 in pair_list[: max(0, int(max_joint_pairs))]:
            x1 = apply_parent_block_scales(x0, parent_slices, {b1: 0.0, b2: 0.0})
            a1 = predict_action_vector(model, x1, device=device)
            rec = {
                "intervention_type": "joint_zero_two_blocks",
                "blocks": [b1, b2],
                "predicted_action": a1.tolist(),
                **_delta_metrics(a0, a1, y_true),
            }
            if protos:
                rec["behavior_pseudo_prob"] = _softmax_proto_logits(a1, protos, behavior_temperature)
            joints.append(rec)

        row_pack["single_block_interventions"] = singles
        row_pack["joint_interventions_sample"] = joints
        rows_out.append(row_pack)

    return {
        "interpretation": (
            "对 SCM 父变量块做乘法尺度干预后重推动作；"
            "behavior_pseudo_prob 需行为原型 JSON，仅为与战术标签对齐的启发式"
        ),
        "scales_used": scales,
        "blocks_prioritized": block_order,
        "rows": rows_out,
    }
