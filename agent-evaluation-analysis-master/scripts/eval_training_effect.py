"""
TODO(remove before release): 训练效果评估脚本（模块 A + 模块 C）。

用途：把“模型训练得怎么样”用可量化指标打印出来，便于调参/优化/重构。

运行示例（PowerShell）：
  # 评估全部任务
  py scripts/eval_training_effect.py

  # 只评估某个任务
  $env:EVAL_TASK_ID="INF_A_006"
  py scripts/eval_training_effect.py

  # 只评估某个智能体
  $env:EVAL_AGENT_ID="1"
  py scripts/eval_training_effect.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score

from src.module_c_counterfactual.data_loader import list_inference_task_ids, load_inference_records
from src.module_c_counterfactual.surrogate_bundle import SurrogateBundle
from src.module_c_counterfactual.training_data import iter_transitions
from src.module_a_rules.collect_data import collect_from_records, compute_return_to_go
from src.module_a_rules.extract_rules import extract_rules_from_tree
from src.module_a_rules.merge_rules import merge_rules, rules_coverage
from src.module_a_rules.viper import VIPERData


def _env_int(name: str) -> Optional[int]:
    """从环境变量读取整数；未设置或为空时返回 ``None``。

    参数:
        name: 环境变量名。

    返回:
        解析后的整数，或 ``None``。
    """
    v = os.environ.get(name, "").strip()
    return int(v) if v else None


def _safe_float(x: Any) -> Optional[float]:
    """安全地将值转为浮点数。

    参数:
        x: 任意可转换对象。

    返回:
        浮点值；无法转换时为 ``None``。
    """
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def eval_module_c_bundle(records, agent_id: int) -> Dict[str, Any]:
    """评估模块 C 代理模型（π/T/R）的训练与拟合指标。

    参数:
        records: 推理记录列表。
        agent_id: 目标智能体 ID。

    返回:
        含 policy、transition、reward 各子指标的字典。
    """
    bundle = SurrogateBundle.fit(records, agent_id)

    # --- policy 指标（来自 SurrogateBundle.training_debug 临时字段）---
    policy_debug = bundle.training_debug or {}

    # --- transition / reward 拟合误差（训练集）---
    n = 0
    mse_next = 0.0
    mse_reward = 0.0
    for obs_t, action_label, obs_t1, r_t in iter_transitions(records, agent_id):
        pred_next = bundle.transition.predict(obs_t, action_label)
        pred_r = bundle.reward.predict(obs_t, action_label, obs_t1)
        a = np.array(pred_next, dtype=float)
        b = np.array(obs_t1, dtype=float)
        mse_next += float(np.mean((a - b) ** 2))
        mse_reward += float((float(pred_r) - float(r_t)) ** 2)
        n += 1

    return {
        "policy": {
            "mode": policy_debug.get("policy_mode"),
            "mode_requested": policy_debug.get("policy_mode_requested"),
            "mode_auto_report": policy_debug.get("policy_mode_auto_report"),
            "train_accuracy": _safe_float(policy_debug.get("policy_train_accuracy")),
            "train_weighted_accuracy": _safe_float(policy_debug.get("policy_train_weighted_accuracy")),
            "val_accuracy": _safe_float(policy_debug.get("policy_val_accuracy")),
            "val_per_item_accuracy": _safe_float(policy_debug.get("policy_val_per_item_accuracy")),
            "val_weighted_accuracy": _safe_float(policy_debug.get("policy_val_weighted_accuracy")),
            "majority_baseline_val_accuracy": _safe_float(
                policy_debug.get("majority_baseline_val_accuracy")
            ),
            "cv_per_item_accuracy_mean": _safe_float(
                policy_debug.get("policy_cv_per_item_accuracy_mean")
            ),
            "grid_best": policy_debug.get("policy_grid_best"),
            "n_policy_samples": policy_debug.get("n_policy_samples"),
            "n_policy_classes": policy_debug.get("n_policy_classes"),
            "tree_depth": policy_debug.get("policy_tree_depth"),
            "tree_leaves": policy_debug.get("policy_tree_leaves"),
        },
        "transition": {
            "mse_next_obs": (mse_next / n) if n else None,
            "n_transitions": n,
            "standardized": policy_debug.get("transition_eval"),
        },
        "reward": {
            "mse_reward": (mse_reward / n) if n else None,
            "n_transitions": n,
        },
    }


def _viper_train(records, agent_id: int, *, max_depth: int, min_samples_leaf: int, n_iters: int, penalty_factor: float):
    """使用 VIPER 在记录上训练决策树并返回结果对象。

    参数:
        records: 推理记录列表。
        agent_id: 智能体 ID。
        max_depth: 决策树最大深度。
        min_samples_leaf: 叶节点最小样本数。
        n_iters: VIPER 迭代轮数。
        penalty_factor: 错分惩罚系数。

    返回:
        ``(vres, X_raw, y, rewards)`` 四元组。
    """
    X_raw, y, rewards, feature_names = collect_from_records(records, agent_id, action_item=None)
    viper = VIPERData(
        X_raw=X_raw,
        y=y,
        rewards=rewards,
        feature_names=feature_names,
        action_item="联合动作",
        action_space=[],
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    vres = viper.run(
        n_iters=n_iters,
        penalty_factor=penalty_factor,
        resample_augment=True,
    )
    return vres, X_raw, y, rewards


def _viper_eval(vres, X_raw_val, y_val, rewards_val) -> Dict[str, Optional[float]]:
    """在验证集上评估 VIPER 决策树的准确率。

    参数:
        vres: VIPER 训练结果（含预处理器与最佳树）。
        X_raw_val: 验证集原始特征矩阵。
        y_val: 验证集标签列表。
        rewards_val: 验证集奖励序列（用于 RTG 加权准确率）。

    返回:
        含 ``acc`` 与 ``acc_weighted`` 的字典。
    """
    if len(y_val) == 0:
        return {"acc": None, "acc_weighted": None}
    X_val_pre = vres.preprocessor.transform(X_raw_val)
    y_pred = vres.best_tree.predict(X_val_pre)
    acc = float(accuracy_score(y_val, y_pred))
    w = compute_return_to_go(np.array(rewards_val, dtype=float))
    acc_w = float(accuracy_score(y_val, y_pred, sample_weight=w))
    return {"acc": acc, "acc_weighted": acc_w}


def eval_module_a(records, agent_id: int) -> Dict[str, Any]:
    """评估模块 A 规则抽取（VIPER + 合并 + holdout/CV/网格搜索）。

    参数:
        records: 推理记录列表。
        agent_id: 智能体 ID。

    返回:
        含准确率、覆盖率、holdout、CV、网格最优等键的指标字典。
    """
    from src.module_a_rules.verify_tree_rules import verify_tree_and_rules
    # baseline（与当前服务默认一致）
    base_params = {
        "max_depth": 6,
        "min_samples_leaf": 2,
        "n_iters": 5,
        "penalty_factor": 2.0,
    }
    vres, X_raw, y, rewards = _viper_train(records, agent_id, **base_params)
    raw_rules = extract_rules_from_tree(vres.best_tree, preprocessor=vres.preprocessor)
    merged = merge_rules(raw_rules)
    X_pre = vres.preprocessor.transform(X_raw)
    cov = rules_coverage(merged, X_pre, y) if len(y) else None
    tree_check = verify_tree_and_rules(vres.best_tree, X_pre, y, merged_rules=merged)

    # hold-out（按 sim 切分）
    n_total = len(records)
    n_val = max(1, int(round(n_total * 0.2))) if n_total >= 2 else 0
    holdout = {"acc": None, "acc_weighted": None, "n_val_records": n_val, "n_val_samples": 0}
    if n_total - n_val >= 1 and n_val >= 1:
        train_records = records[: n_total - n_val]
        val_records = records[n_total - n_val :]
        v_tr, _, _, _ = _viper_train(train_records, agent_id, **base_params)
        Xv, yv, rv, _ = collect_from_records(val_records, agent_id, action_item=None)
        ev = _viper_eval(v_tr, Xv, yv, rv)
        holdout = {
            "acc": ev["acc"],
            "acc_weighted": ev["acc_weighted"],
            "n_val_records": n_val,
            "n_val_samples": int(len(yv)),
        }

    # K-fold by sim + 小网格搜索
    cv_folds = max(2, min(5, len(records))) if len(records) >= 2 else 0
    cv_default_scores: List[float] = []
    grid_best: Optional[Dict[str, Any]] = None
    if cv_folds >= 2:
        grid = []
        for d in sorted({max(3, base_params["max_depth"] - 1), base_params["max_depth"], base_params["max_depth"] + 1}):
            for leaf in sorted({max(1, base_params["min_samples_leaf"] - 1), base_params["min_samples_leaf"], base_params["min_samples_leaf"] + 1}):
                for n_iters in (4, 5):
                    for penalty in (1.5, 2.0):
                        grid.append(
                            {
                                "max_depth": d,
                                "min_samples_leaf": leaf,
                                "n_iters": n_iters,
                                "penalty_factor": penalty,
                            }
                        )
        best = -1.0
        for params in grid:
            fold_scores: List[float] = []
            for fold_idx in range(cv_folds):
                val_idx = set(range(fold_idx, len(records), cv_folds))
                tr = [r for i, r in enumerate(records) if i not in val_idx]
                va = [r for i, r in enumerate(records) if i in val_idx]
                if not tr or not va:
                    continue
                v_tr, _, _, _ = _viper_train(tr, agent_id, **params)
                Xv, yv, rv, _ = collect_from_records(va, agent_id, action_item=None)
                ev = _viper_eval(v_tr, Xv, yv, rv)
                if ev["acc_weighted"] is not None:
                    fold_scores.append(float(ev["acc_weighted"]))
            if not fold_scores:
                continue
            mean_score = float(np.mean(np.array(fold_scores, dtype=float)))
            if params == base_params:
                cv_default_scores = fold_scores
            if mean_score > best:
                best = mean_score
                grid_best = {
                    **params,
                    "cv_weighted_accuracy_mean": mean_score,
                    "cv_weighted_accuracy_std": float(np.std(np.array(fold_scores, dtype=float))),
                    "cv_folds": len(fold_scores),
                }

    return {
        "accuracy": round(float(vres.best_accuracy), 4),
        "coverage": round(float(cov), 4) if cov is not None else None,
        "n_records": len(records),
        "n_samples": int(len(y)),
        "n_rules": int(len(merged)),
        "tree_rules_raw_matches_tree": bool(tree_check.raw_matches_tree_predict),
        "tree_rules_merged_matches_tree": bool(tree_check.merged_matches_tree_predict),
        "n_leaves": int(tree_check.n_leaves),
        "viper_augmentation_iters": len(vres.augmentation_history),
        "holdout_accuracy": holdout["acc"],
        "holdout_weighted_accuracy": holdout["acc_weighted"],
        "n_val_records": holdout["n_val_records"],
        "n_val_samples": holdout["n_val_samples"],
        "cv_weighted_accuracy_mean": float(np.mean(np.array(cv_default_scores, dtype=float))) if cv_default_scores else None,
        "cv_weighted_accuracy_std": float(np.std(np.array(cv_default_scores, dtype=float))) if cv_default_scores else None,
        "grid_best": grid_best,
    }


def _write_module_a_report(results: List[Dict[str, Any]]) -> None:
    """将模块 A 评估结果写入 ``experiments/MODULE_A_EVAL_report.md``。

    参数:
        results: ``eval_module_a`` / ``eval_module_c_bundle`` 汇总后的结果列表。
    """
    root = Path(__file__).resolve().parent.parent
    path = root / "experiments" / "MODULE_A_EVAL_report.md"
    lines = [
        "# Module A 评估快照（自动生成）\n",
        "| task | agent | accuracy | coverage | holdout_w | raw≡tree | merged≡tree | n_rules |",
        "|------|-------|----------|----------|-----------|----------|-------------|---------|",
    ]
    for row in results:
        a = row.get("module_a") or {}
        lines.append(
            f"| {row.get('task_id')} | {row.get('agent_id')} | "
            f"{a.get('accuracy')} | {a.get('coverage')} | {a.get('holdout_weighted_accuracy')} | "
            f"{a.get('tree_rules_raw_matches_tree')} | {a.get('tree_rules_merged_matches_tree')} | "
            f"{a.get('n_rules')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    """遍历任务与智能体，输出模块 A/C 训练效果 JSON，可选写报告与门禁检查。"""
    task_filter = os.environ.get("EVAL_TASK_ID", "").strip()
    agent_filter = _env_int("EVAL_AGENT_ID")

    task_ids = [task_filter] if task_filter else list_inference_task_ids()
    results: List[Dict[str, Any]] = []

    for task_id in task_ids:
        records = load_inference_records(task_id)
        if not records:
            continue
        agent_ids = sorted({aid for r in records for aid in r.agent_ids})
        if agent_filter is not None:
            agent_ids = [aid for aid in agent_ids if aid == agent_filter]

        for agent_id in agent_ids:
            a = eval_module_a(records, agent_id=agent_id)
            c = eval_module_c_bundle(records, agent_id=agent_id)
            results.append(
                {
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "module_a": a,
                    "module_c": c,
                }
            )

    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))

    if os.environ.get("EVAL_WRITE_REPORT", "").strip().lower() in ("1", "true", "yes"):
        _write_module_a_report(results)

    # 可选门禁（环境变量未设置则跳过）
    min_policy_acc = os.environ.get("EVAL_POLICY_ACC_MIN", "").strip()
    max_t_mse = os.environ.get("EVAL_T_MSE_MAX", "").strip()
    max_r_mse = os.environ.get("EVAL_R_MSE_MAX", "").strip()
    if not (min_policy_acc or max_t_mse or max_r_mse):
        return

    failures: List[str] = []
    for row in results:
        c = row.get("module_c") or {}
        pol = c.get("policy") or {}
        acc = pol.get("train_accuracy")
        if min_policy_acc and acc is not None and float(acc) < float(min_policy_acc):
            failures.append(
                f"{row['task_id']}/agent{row['agent_id']}: policy acc {acc} < {min_policy_acc}"
            )
        t_mse = _safe_float((c.get("transition") or {}).get("mse_next_obs"))
        if max_t_mse and t_mse is not None and t_mse > float(max_t_mse):
            failures.append(
                f"{row['task_id']}/agent{row['agent_id']}: T mse {t_mse} > {max_t_mse}"
            )
        r_mse = _safe_float((c.get("reward") or {}).get("mse_reward"))
        if max_r_mse and r_mse is not None and r_mse > float(max_r_mse):
            failures.append(
                f"{row['task_id']}/agent{row['agent_id']}: R mse {r_mse} > {max_r_mse}"
            )
    if failures:
        raise SystemExit("EVAL GATE FAILED:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()

