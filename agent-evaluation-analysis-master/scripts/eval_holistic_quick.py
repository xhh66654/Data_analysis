"""
小样本快速验证 holistic 策略 π（状态 → 一步完整 decision_content）。

用于调参：先用少量 sim / 步数看验证指标，满意后再跑全量。

PowerShell 示例：
  # 默认：INF_A_001 / agent 1 / 前 3 局 / 每局 100 步
  py scripts/eval_holistic_quick.py

  # 更少数据（更快）
  $env:EVAL_MAX_SIMS="2"
  $env:EVAL_MAX_STEPS="60"
  py scripts/eval_holistic_quick.py

  # 调模式 / 树深度
  $env:ANALYSIS_CF_POLICY_MODE="joint"      # joint | composed | auto
  $env:ANALYSIS_CF_POLICY_AUTOTUNE="0"      # 小样本建议先关 autotune
  $env:POLICY_MAX_DEPTH="8"
  $env:POLICY_MIN_SAMPLES_LEAF="1"
  py scripts/eval_holistic_quick.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.module_c_counterfactual.data_loader import load_inference_records
from src.module_c_counterfactual.policy_model import PolicySurrogate
from src.module_c_counterfactual.surrogate_bundle import (
    _collect_policy_xyw,
    compute_policy_holdout_debug,
    split_records_train_val,
)
from src.module_c_counterfactual.training_data import subset_records_for_dev


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    """从环境变量读取正整数。

    参数:
        name: 环境变量名。
        default: 未设置或非法时的默认值。

    返回:
        正整数或 ``default``。
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except ValueError:
        return default


def _class_distribution(y: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
    """统计整体决策标签的频次分布（取前 top_k）。

    参数:
        y: 整体决策标签列表。
        top_k: 返回的最高频类别数。

    返回:
        含 ``share``、``count``、``label_preview`` 的字典列表。
    """
    c = Counter(y)
    total = len(y) or 1
    rows = []
    for label, cnt in c.most_common(top_k):
        rows.append(
            {
                "share": round(cnt / total, 4),
                "count": cnt,
                "label_preview": label[:72] + ("…" if len(label) > 72 else ""),
            }
        )
    return rows


def main() -> None:
    """在小样本子集上快速训练 holistic 策略 π 并输出验证指标 JSON。"""
    task_id = os.environ.get("EVAL_TASK_ID", "INF_A_001").strip()
    agent_id = int(os.environ.get("EVAL_AGENT_ID", "1"))
    max_sims = _env_int("EVAL_MAX_SIMS", 3)
    max_steps = _env_int("EVAL_MAX_STEPS", 100)
    max_depth = int(os.environ.get("POLICY_MAX_DEPTH", os.environ.get("EVAL_POLICY_MAX_DEPTH", "6")))
    min_leaf = int(
        os.environ.get("POLICY_MIN_SAMPLES_LEAF", os.environ.get("EVAL_POLICY_MIN_SAMPLES_LEAF", "2"))
    )
    mode = os.environ.get("ANALYSIS_CF_POLICY_MODE", "joint").strip().lower()
    if mode in ("holistic", "whole", "full"):
        mode = "joint"

    os.environ.setdefault("ANALYSIS_CF_POLICY_AUTOTUNE", "0")
    os.environ.setdefault("ANALYSIS_CF_VAL_RATIO", "0.2")

    all_records = load_inference_records(task_id)
    if not all_records:
        raise SystemExit(f"任务 {task_id} 无记录")

    records = subset_records_for_dev(
        all_records,
        max_sims=max_sims,
        max_steps_per_sim=max_steps,
    )
    train_recs, val_recs = split_records_train_val(records)

    X, y, _w = _collect_policy_xyw(records, agent_id)
    policy = PolicySurrogate(max_depth=max_depth, min_samples_leaf=min_leaf, mode=mode)  # type: ignore[arg-type]
    policy.fit_records(records, agent_id)

    dbg = compute_policy_holdout_debug(
        records,
        agent_id,
        policy,
        policy_max_depth=max_depth,
        policy_min_samples_leaf=min_leaf,
        mode=mode if mode != "auto" else getattr(policy, "mode", "joint"),
    )

    report: Dict[str, Any] = {
        "task_id": task_id,
        "agent_id": agent_id,
        "subset": {
            "max_sims": max_sims,
            "max_steps_per_sim": max_steps,
            "n_sims_used": len(records),
            "n_sims_total": len(all_records),
            "sim_ids": [r.sim_id for r in records],
            "n_policy_samples": len(y),
            "n_policy_classes": len(set(y)),
            "class_distribution_top5": _class_distribution(y),
        },
        "split": {
            "n_train_sims": len(train_recs),
            "n_val_sims": len(val_recs),
            "val_sim_ids": [r.sim_id for r in val_recs],
        },
        "config": {
            "policy_mode": mode,
            "max_depth": max_depth,
            "min_samples_leaf": min_leaf,
            "policy_estimator": os.environ.get("ANALYSIS_CF_POLICY_ESTIMATOR", "tree"),
            "policy_preprocess": os.environ.get("ANALYSIS_CF_POLICY_PREPROCESS", "1"),
            "policy_autotune": os.environ.get("ANALYSIS_CF_POLICY_AUTOTUNE", "0"),
        },
        "metrics": {
            "primary_metric": dbg.get("primary_metric"),
            "policy_learning_target": dbg.get("policy_learning_target"),
            "train_accuracy": dbg.get("policy_train_accuracy"),
            "train_weighted_accuracy": dbg.get("policy_train_weighted_accuracy"),
            "val_accuracy": dbg.get("policy_val_accuracy"),
            "val_weighted_accuracy": dbg.get("policy_val_weighted_accuracy"),
            "val_per_item_accuracy": dbg.get("policy_val_per_item_accuracy"),
            "val_per_item_breakdown": dbg.get("policy_val_per_item_breakdown"),
            "majority_baseline_val": dbg.get("majority_baseline_val_accuracy"),
        },
        "hints": [
            "val_accuracy = 整体决策 JSON 完全匹配率（holistic joint-exact）",
            "val_per_item_accuracy = 各动作项平均命中率（通常更高）",
            "若 val ≈ majority_baseline，说明少数几类整体决策占主导，可加大 EVAL_MAX_SIMS 再看",
            "瓶颈常在「雷达方向控制」等难维度，可看 val_per_item_breakdown",
        ],
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
