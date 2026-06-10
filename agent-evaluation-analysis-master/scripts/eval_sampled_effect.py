"""
TODO(remove before release): 抽样检查模型训练效果（模块A + 模块C）。

你现在的核心诉求是：
1) 每次解释都会临时训练代理模型（π/T/R 和 VIPER），所以你想“看得见”训练效果；
2) 不希望一次优化影响下一个完全不同的智能体，因此需要“严格保守”——只在指标显著变好时才采用。

这个脚本做的事：
- 从推理数据里随机抽样 N 条“可解释的决策”（保证不是最后一步）
- 对每条样本，跑两次反事实服务（模块C）：
    (A) 默认配置（不优化）
    (B) optimize + strict（policy_mode=auto + T_autotune，但严格门控）
  并把 SurrogateBundle 的训练指标（train/val/CV、T 的 N-MSE/MAE、是否回退）结构化输出。
- 对每个 (task, agent) 也跑一次模块A的规则抽取健康检查：
    - train_accuracy / coverage / merge_check（是否回退 raw rules）
    - 简易 holdout（按 sim 切 80/20）
    - 简易 CV（按 sim 做 K-fold，指标用 return-to-go 加权准确率）

运行示例（PowerShell）：
  # 抽样 5 条（默认）
  py scripts/eval_sampled_effect.py

  # 指定抽样数量
  $env:SAMPLE_N="10"
  py scripts/eval_sampled_effect.py

  # 只检查某个任务/智能体
  $env:SAMPLE_TASK_ID="INF_A_006"
  $env:SAMPLE_AGENT_ID="1"
  py scripts/eval_sampled_effect.py
"""

from __future__ import annotations

import sys
import json
import os
import random
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------
# 兼容性处理（Windows/脚本直接运行时的 import 路径）
# ------------------------------------------------------------
# 目标：允许直接运行 `py scripts/eval_sampled_effect.py` 时正常 `import src.*`
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.module_a_rules.collect_data import (
    collect_from_records,
    collect_from_records_with_segments,
    compute_return_to_go,
)
from src.module_a_rules.viper import VIPERData
from src.module_c_counterfactual.data_loader import list_inference_task_ids, load_inference_records
from src.service import counterfactual_service, rule_extraction_service


def _env_int(name: str, default: int) -> int:
    """从环境变量读取整数，失败时返回默认值。

    参数:
        name: 环境变量名。
        default: 解析失败或未设置时的默认值。

    返回:
        整数值。
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_str(name: str) -> str:
    """读取并去除首尾空白的环境变量字符串。

    参数:
        name: 环境变量名。

    返回:
        环境变量值；未设置时为空字符串。
    """
    return os.environ.get(name, "").strip()


def _set_env_flag(name: str, enabled: bool) -> None:
    """设置或清除布尔型环境变量（``"1"`` 表示启用）。

    参数:
        name: 环境变量名。
        enabled: 为 ``True`` 时设为 ``"1"``，否则删除该变量。
    """
    if enabled:
        os.environ[name] = "1"
    else:
        os.environ.pop(name, None)


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


def _pick_decision_samples(
    records,
    *,
    agent_id: int,
    n: int,
    seed: int,
) -> List[Tuple[str, str, int, Dict[str, Any]]]:
    """从多局记录中抽样可解释决策步。

    参数:
        records: 推理记录列表。
        agent_id: 智能体 ID。
        n: 目标抽样数量。
        seed: 随机种子。

    返回:
        ``(task_id, sim_id, query_step, decision_content)`` 元组列表。
    """
    rng = random.Random(seed)
    candidates: List[Tuple[str, str, int, Dict[str, Any]]] = []
    for rec in records:
        # 不选最后一步：one_step/multi_step 需要 t+1
        for t in range(max(0, rec.total_steps - 1)):
            dec = rec.get_decision_at(t, agent_id)
            if dec is None or not dec.content:
                continue
            candidates.append((rec.task_id, rec.sim_id, int(t), dict(dec.content)))

    if not candidates:
        return []
    rng.shuffle(candidates)
    return candidates[: max(1, min(n, len(candidates)))]


def _summarize_train_debug(train_debug: Dict[str, Any]) -> Dict[str, Any]:
    """裁剪 train_debug，仅保留评估相关字段。

    参数:
        train_debug: 反事实服务返回的训练调试字典。

    返回:
        精简后的子集字典。
    """
    out: Dict[str, Any] = {}
    for k in (
        "policy_mode",
        "policy_mode_requested",
        "policy_mode_auto_report",
        "policy_train_accuracy",
        "policy_val_accuracy",
        "policy_cv_weighted_accuracy_mean",
        "policy_cv_weighted_accuracy_std",
        "transition_autotune",
        "transition_eval",
        "optimization_summary",
    ):
        out[k] = train_debug.get(k)
    return out


def _eval_module_a_generalization(records, agent_id: int) -> Dict[str, Any]:
    """模块 A 泛化健康检查（按 sim 留 holdout 与 K-fold CV）。

    参数:
        records: 推理记录列表。
        agent_id: 智能体 ID。

    返回:
        含 holdout 与 CV 加权准确率的字典。
    """
    base = {"max_depth": 6, "min_samples_leaf": 2, "n_iters": 5, "penalty_factor": 2.0}

    def _train(recs):
        """在子集记录上训练 VIPER 并返回运行结果。

        参数:
            recs: 用于训练的推理记录子列表。

        返回:
            ``VIPERData.run`` 的结果对象。
        """
        X, y, r, feat, seg = collect_from_records_with_segments(recs, agent_id, action_item=None)
        v = VIPERData(
            X_raw=X,
            y=y,
            rewards=r,
            feature_names=feat,
            action_item="联合动作",
            action_space=[],
            max_depth=base["max_depth"],
            min_samples_leaf=base["min_samples_leaf"],
            episode_lengths=seg,
        )
        return v.run(n_iters=base["n_iters"], penalty_factor=base["penalty_factor"])

    n = len(records)
    holdout = {"acc": None, "w_acc": None, "val_samples": 0, "val_records": 0}
    if n >= 2:
        n_val = max(1, int(round(n * 0.2)))
        if n - n_val >= 1:
            tr = records[: n - n_val]
            va = records[n - n_val :]
            res = _train(tr)
            Xv, yv, rv, _ = collect_from_records(va, agent_id, action_item=None)
            if len(yv):
                Xv_pre = res.preprocessor.transform(Xv)
                ypred = res.best_tree.predict(Xv_pre)
                holdout = {
                    "acc": float(accuracy_score(yv, ypred)),
                    "w_acc": float(accuracy_score(yv, ypred, sample_weight=compute_return_to_go(np.array(rv, dtype=float)))),
                    "val_samples": int(len(yv)),
                    "val_records": int(len(va)),
                }

    k = max(2, min(5, n)) if n >= 2 else 0
    cv_scores: List[float] = []
    if k >= 2:
        for fold in range(k):
            val_idx = set(range(fold, n, k))
            tr = [r for i, r in enumerate(records) if i not in val_idx]
            va = [r for i, r in enumerate(records) if i in val_idx]
            if not tr or not va:
                continue
            res = _train(tr)
            Xv, yv, rv, _ = collect_from_records(va, agent_id, action_item=None)
            if len(yv) == 0:
                continue
            Xv_pre = res.preprocessor.transform(Xv)
            ypred = res.best_tree.predict(Xv_pre)
            w = compute_return_to_go(np.array(rv, dtype=float))
            cv_scores.append(float(accuracy_score(yv, ypred, sample_weight=w)))

    return {
        "holdout": holdout,
        "cv_weighted_accuracy_mean": float(np.mean(np.array(cv_scores, dtype=float))) if cv_scores else None,
        "cv_weighted_accuracy_std": float(np.std(np.array(cv_scores, dtype=float))) if cv_scores else None,
        "cv_folds": int(len(cv_scores)),
    }


def _mean(xs: List[Optional[float]]) -> Optional[float]:
    """计算忽略 ``None`` 后的算术均值。

    参数:
        xs: 可含 ``None`` 的浮点列表。

    返回:
        均值；无有效值时为 ``None``。
    """
    vals = [float(x) for x in xs if x is not None]
    if not vals:
        return None
    return float(np.mean(np.array(vals, dtype=float)))


def _build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总抽样评估结果，生成可读性强的统计摘要。

    参数:
        results: 各 task/agent 的完整评估条目列表。

    返回:
        含模块 A/C 均值指标与短板提示的字典。
    """
    total_samples = 0
    policy_switched = 0
    t_autotune_enabled = 0
    t_autotune_applied = 0

    base_policy_val: List[Optional[float]] = []
    opt_policy_val: List[Optional[float]] = []
    base_policy_cv: List[Optional[float]] = []
    opt_policy_cv: List[Optional[float]] = []
    base_t_nmse: List[Optional[float]] = []
    opt_t_nmse: List[Optional[float]] = []

    a_holdout_w: List[Optional[float]] = []
    a_cv_w: List[Optional[float]] = []

    for entry in results:
        a_gen = (entry.get("module_a") or {}).get("generalization") or {}
        hold = (a_gen.get("holdout") or {}) if isinstance(a_gen, dict) else {}
        a_holdout_w.append(_safe_float(hold.get("w_acc")))
        a_cv_w.append(_safe_float(a_gen.get("cv_weighted_accuracy_mean")))

        for s in entry.get("module_c_samples") or []:
            total_samples += 1
            b = (s.get("baseline") or {}).get("train_debug") or {}
            o = (s.get("optimized_strict") or {}).get("train_debug") or {}

            base_policy_val.append(_safe_float(b.get("policy_val_accuracy")))
            opt_policy_val.append(_safe_float(o.get("policy_val_accuracy")))
            base_policy_cv.append(_safe_float(b.get("policy_cv_weighted_accuracy_mean")))
            opt_policy_cv.append(_safe_float(o.get("policy_cv_weighted_accuracy_mean")))

            bt = (b.get("transition_eval") or {}) if isinstance(b, dict) else {}
            ot = (o.get("transition_eval") or {}) if isinstance(o, dict) else {}
            base_t_nmse.append(_safe_float(bt.get("mean_nmse")))
            opt_t_nmse.append(_safe_float(ot.get("mean_nmse")))

            base_mode = (b.get("policy_mode") or (s.get("baseline") or {}).get("policy_mode"))
            opt_mode = (o.get("policy_mode") or (s.get("optimized_strict") or {}).get("policy_mode"))
            if base_mode and opt_mode and str(base_mode) != str(opt_mode):
                policy_switched += 1

            opt_sum = (o.get("optimization_summary") or {}) if isinstance(o, dict) else {}
            tinfo = (opt_sum.get("transition_autotune") or {}) if isinstance(opt_sum, dict) else {}
            if tinfo.get("enabled"):
                t_autotune_enabled += 1
                if tinfo.get("applied"):
                    t_autotune_applied += 1

    # 简单“短板提示”：先看 policy，再看 A 的泛化
    hints: List[str] = []
    if (m := _mean(opt_policy_val)) is not None and m < 0.4:
        hints.append("模块C.policy 的验证准确率偏低：优先提高 π 的泛化（建议 per_item/减少联合类别/增样本）。")
    if (m := _mean(a_holdout_w)) is not None and m < 0.4:
        hints.append("模块A 的 holdout 加权准确率偏低：优先用 CV/holdout 指标驱动调参，而不是训练准确率。")
    if total_samples > 0 and t_autotune_enabled > 0 and t_autotune_applied == 0:
        hints.append("严格保守下 T autotune 全部回退：说明候选参数没显著提升，先别扩大候选，优先改特征/数据。")

    return {
        "n_task_agent": int(len(results)),
        "n_samples_total": int(total_samples),
        "policy_mode_switched_rate": (policy_switched / total_samples) if total_samples else None,
        "transition_autotune_enabled_rate": (t_autotune_enabled / total_samples) if total_samples else None,
        "transition_autotune_applied_rate": (t_autotune_applied / total_samples) if total_samples else None,
        "module_c": {
            "baseline": {
                "policy_val_accuracy_mean": _mean(base_policy_val),
                "policy_cv_weighted_accuracy_mean": _mean(base_policy_cv),
                "transition_mean_nmse_mean": _mean(base_t_nmse),
            },
            "optimized_strict": {
                "policy_val_accuracy_mean": _mean(opt_policy_val),
                "policy_cv_weighted_accuracy_mean": _mean(opt_policy_cv),
                "transition_mean_nmse_mean": _mean(opt_t_nmse),
            },
        },
        "module_a": {
            "holdout_weighted_accuracy_mean": _mean(a_holdout_w),
            "cv_weighted_accuracy_mean": _mean(a_cv_w),
        },
        "hints": hints,
    }


def main() -> None:
    """抽样对比默认与 optimize+strict 配置下的模块 A/C 训练效果。"""
    # -----------------------------
    # 0) 读取用户配置（环境变量）
    # -----------------------------
    n_samples = _env_int("SAMPLE_N", 5)
    seed = _env_int("SAMPLE_SEED", 0)
    task_filter = _env_str("SAMPLE_TASK_ID")
    agent_filter_raw = _env_str("SAMPLE_AGENT_ID")
    agent_filter = int(agent_filter_raw) if agent_filter_raw.isdigit() else None

    # -----------------------------
    # 1) 列出任务并加载 records
    # -----------------------------
    task_ids = [task_filter] if task_filter else list_inference_task_ids()
    if not task_ids:
        print(json.dumps({"error": "no tasks found"}, ensure_ascii=False, indent=2))
        return

    results: List[Dict[str, Any]] = []

    for task_id in task_ids:
        records = load_inference_records(task_id)
        if not records:
            continue
        # 对这个任务的所有智能体分别抽样（或只抽指定 agent）
        agent_ids = sorted({aid for r in records for aid in r.agent_ids})
        if agent_filter is not None:
            agent_ids = [a for a in agent_ids if a == agent_filter]
        if not agent_ids:
            continue

        for agent_id in agent_ids:
            # -----------------------------
            # 2) 模块A：规则抽取效果（按 task+agent 只算一次）
            # -----------------------------
            a = rule_extraction_service(agent_id=agent_id, inference_task_id=task_id)
            a_general = _eval_module_a_generalization(records, agent_id=agent_id)

            # -----------------------------
            # 3) 模块C：抽样对比（默认 vs optimize+strict）
            # -----------------------------
            samples = _pick_decision_samples(records, agent_id=agent_id, n=n_samples, seed=seed)
            c_samples: List[Dict[str, Any]] = []

            for _, sim_id, query_step, decision_content in samples:
                # 先确保 train_debug 打开，否则拿不到训练指标
                _set_env_flag("ANALYSIS_CF_TRAIN_DEBUG", True)

                # (A) baseline：清掉优化开关
                for k in (
                    "ANALYSIS_CF_POLICY_MODE",
                    "ANALYSIS_CF_T_AUTOTUNE",
                    "ANALYSIS_STRICT_CONSERVATIVE",
                    "ANALYSIS_CF_T_GROUPED",
                ):
                    os.environ.pop(k, None)

                base = counterfactual_service(
                    agent_id=agent_id,
                    inference_task_id=task_id,
                    sim_id=sim_id,
                    decision_content=decision_content,
                    query_step=query_step,
                    cf_level="one_step",
                    horizon=5,
                )
                base_td = base.get("train_debug", {}) or {}

                # (B) optimized strict：只在显著变好时才采用
                os.environ["ANALYSIS_CF_POLICY_MODE"] = "auto"
                os.environ["ANALYSIS_CF_T_AUTOTUNE"] = "1"
                os.environ["ANALYSIS_STRICT_CONSERVATIVE"] = "1"

                opt = counterfactual_service(
                    agent_id=agent_id,
                    inference_task_id=task_id,
                    sim_id=sim_id,
                    decision_content=decision_content,
                    query_step=query_step,
                    cf_level="one_step",
                    horizon=5,
                )
                opt_td = opt.get("train_debug", {}) or {}

                c_samples.append(
                    {
                        "task_id": task_id,
                        "sim_id": sim_id,
                        "agent_id": agent_id,
                        "query_step": int(query_step),
                        "decision_content": decision_content,
                        "baseline": {
                            "t_query": base.get("t_query"),
                            "policy_mode": base_td.get("policy_mode"),
                            "train_debug": _summarize_train_debug(base_td) if isinstance(base_td, dict) else {},
                        },
                        "optimized_strict": {
                            "t_query": opt.get("t_query"),
                            "policy_mode": opt_td.get("policy_mode"),
                            "train_debug": _summarize_train_debug(opt_td) if isinstance(opt_td, dict) else {},
                        },
                    }
                )

            results.append(
                {
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "module_a": {
                        "accuracy": a.get("accuracy"),
                        "coverage": a.get("coverage"),
                        "n_rules": a.get("n_rules"),
                        "merge_check": a.get("merge_check"),
                        "generalization": a_general,
                    },
                    "module_c_samples": c_samples,
                }
            )

    summary = _build_summary(results)
    print(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

