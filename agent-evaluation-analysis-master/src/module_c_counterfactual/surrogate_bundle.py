"""
三模型统一训练入口（一步/多步反事实共用）。

小白版比喻：
    - policy（π）  ：「在这种态势下，智能体一般会怎么打」—— 决策树
    - transition（T）：「打完这一步，态势会变成什么样」—— 回归/树
    - reward（R）   ：「这一步大概能得多少分」—— 回归/树

SurrogateBundle.fit() 会用任务下所有仿真记录，把这三个模型一起训好，
后面 one_step / multi_step 反事实都复用这一套，不用每次重训。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.module_c_counterfactual.inference_record import InferenceRecord
from src.module_c_counterfactual.policy_model import PolicySurrogate
from src.module_c_counterfactual.reward_model import RewardModel
from src.module_c_counterfactual.training_data import compute_obs_feature_means, iter_transitions
from src.module_c_counterfactual.transition_model import TransitionModel


def _is_strict_conservative_enabled() -> bool:
    """
    TODO(remove before release): 严格保守模式开关。
    """
    import os

    v = os.environ.get("ANALYSIS_STRICT_CONSERVATIVE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _strict_improvement_margin() -> float:
    """
    TODO(remove before release): 严格保守模式最小改进阈值（默认 1e-3）。
    通过 ANALYSIS_STRICT_MARGIN 覆盖，例如 0.002 / 0.005。
    """
    import os

    raw = os.environ.get("ANALYSIS_STRICT_MARGIN", "").strip()
    if not raw:
        return 1e-3
    try:
        v = float(raw)
        return v if v >= 0.0 else 1e-3
    except Exception:
        return 1e-3


def _strict_robust_z() -> float:
    """
    TODO(remove before release): 严格保守模式的“鲁棒性系数”（默认 1.0）。

    说明：少数据时 CV 的方差很大，只看均值容易误判。
    这里采用一个简单的保守判定：

      baseline_mean - best_mean >= margin + z * (baseline_std + best_std)

    z 越大越保守。
    """
    import os

    raw = os.environ.get("ANALYSIS_STRICT_Z", "").strip()
    if not raw:
        return 1.0
    try:
        v = float(raw)
        return v if v >= 0.0 else 1.0
    except Exception:
        return 1.0


def _policy_mode_from_env() -> str:
    """
    策略代理模式（默认 composed：分项学习、holistic JSON 输出）。

    - composed / holistic_compose : 各动作项一棵树，预测组装为整体决策标签（推荐）
    - joint / holistic            : 单棵树直接预测整体类（类不均衡时 joint-exact 偏低）
    - per_item                    : 调试，旧 tuple 标签
    - auto                        : joint 与 composed 间 CV 选择
    """
    import os

    v = os.environ.get("ANALYSIS_CF_POLICY_MODE", "joint").strip().lower()
    if v in ("composed", "holistic_compose", "factorized", "compose"):
        return "composed"
    if v in ("holistic", "whole", "full"):
        return "joint"
    if v in ("per_item", "peritem", "item", "items"):
        return "per_item"
    if v in ("auto", "adaptive"):
        return "auto"
    return "joint"


def _is_transition_autotune_enabled() -> bool:
    """
    TODO(remove before release): T 模型自动调参开关。
    """
    import os

    v = os.environ.get("ANALYSIS_CF_T_AUTOTUNE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _is_transition_grouped_enabled() -> bool:
    """
    TODO(remove before release): T 模型分维建模候选开关。
    """
    import os

    v = os.environ.get("ANALYSIS_CF_T_GROUPED", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _collect_policy_xyw(
    records: List[InferenceRecord],
    agent_id: int,
) -> Tuple[List[List[float]], List[str], List[float]]:
    """
    从多条记录收集策略训练样本 (X, y, return-to-go 权重)。

    参数:
        records: 推理记录列表。
        agent_id: 目标智能体编号。

    返回:
        (观测向量列表, holistic 标签列表, RTG 权重列表)。
    """
    from src.module_c_counterfactual.training_data import joint_action_label

    X: List[List[float]] = []
    y: List[str] = []
    w: List[float] = []
    for rec in records:
        rewards = getattr(rec, "rewards", [])
        running = 0.0
        rtg = [0.0] * rec.total_steps
        for i in range(rec.total_steps - 1, -1, -1):
            running += float(rewards[i]) if i < len(rewards) else 0.0
            rtg[i] = running
        for i in range(rec.total_steps):
            obs = rec.get_obs_vector(i, agent_id)
            dec = rec.get_decision_at(i, agent_id)
            if not obs or dec is None:
                continue
            X.append(obs)
            y.append(joint_action_label(dec.content, rec, agent_id))
            w.append(rtg[i] if i < len(rtg) else 0.0)
    return X, y, w


def _safe_weighted_accuracy(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> Optional[float]:
    """
    计算加权准确率；权重长度不匹配或全零时安全降级。

    参数:
        y_true: 真实标签数组。
        y_pred: 预测标签数组。
        w: 样本权重数组。

    返回:
        加权准确率；无法计算时 None。
    """
    if len(w) != len(y_true):
        return None
    w_abs = np.abs(w.astype(float))
    denom = float(np.sum(w_abs))
    if denom <= 1e-12:
        if len(w_abs) == 0:
            return None
        w_abs = np.ones_like(w_abs)
        denom = float(len(w_abs))
    return float(np.sum((y_pred == y_true).astype(float) * w_abs) / denom)


def _val_ratio_from_env() -> float:
    """从环境变量读取 holdout 验证集比例（默认 0.2，限制在 0.05～0.5）。"""
    raw = os.environ.get("ANALYSIS_CF_VAL_RATIO", "0.2").strip()
    try:
        v = float(raw)
        return max(0.05, min(0.5, v))
    except Exception:
        return 0.2


def split_records_train_val(
    records: List[InferenceRecord],
    val_ratio: Optional[float] = None,
) -> Tuple[List[InferenceRecord], List[InferenceRecord]]:
    """
    按仿真局（sim）划分训练/验证集，避免同局步样本泄露。

    参数:
        records: 推理记录列表。
        val_ratio: 验证集比例；None 时读环境变量。

    返回:
        (train_records, val_records) 元组。
    """
    ratio = _val_ratio_from_env() if val_ratio is None else val_ratio
    n_total = len(records)
    if n_total < 2:
        return list(records), []
    n_val = max(1, int(round(n_total * ratio)))
    n_train = n_total - n_val
    if n_train < 1:
        return list(records), []
    return records[:n_train], records[n_train:]


def _split_transition_rows_by_sim(
    rows: List[Dict[str, Any]],
    val_ratio: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """转移行按 sim_id 划分；无 sim_id 时回退为随机行划分。"""
    from collections import defaultdict

    ratio = _val_ratio_from_env() if val_ratio is None else val_ratio
    by_sim: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    no_sim: List[Dict[str, Any]] = []
    for row in rows:
        sid = row.get("sim_id")
        if sid:
            by_sim[str(sid)].append(row)
        else:
            no_sim.append(row)

    sims = sorted(by_sim.keys())
    if len(sims) >= 2:
        n_val_sims = max(1, int(round(len(sims) * ratio)))
        val_sims = sims[-n_val_sims:]
        train_sims = sims[:-n_val_sims] or sims[:1]
        train_rows = [r for s in train_sims for r in by_sim[s]]
        val_rows = [r for s in val_sims for r in by_sim[s]]
        return train_rows + no_sim, val_rows

    all_rows = rows
    if len(all_rows) < 4:
        return all_rows, []
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(all_rows))
    n_val = max(1, int(round(len(all_rows) * ratio)))
    val_idx = set(int(i) for i in perm[:n_val])
    train_rows = [all_rows[i] for i in range(len(all_rows)) if i not in val_idx]
    val_rows = [all_rows[i] for i in val_idx]
    return train_rows, val_rows


def _policy_accuracy_on_samples(
    policy: PolicySurrogate,
    X: List[List[float]],
    y: List[str],
    w: List[float],
) -> Tuple[Optional[float], Optional[float]]:
    """
    在样本集上评估策略模型的准确率与加权准确率。

    参数:
        policy: 已拟合的策略模型。
        X: 观测向量列表。
        y: holistic 标签列表。
        w: 样本权重列表。

    返回:
        (accuracy, weighted_accuracy) 元组。
    """
    if not X:
        return None, None
    y_pred = np.array([policy.predict(x) for x in X])
    y_true = np.array(y)
    acc = float(np.mean(y_pred == y_true))
    wacc = _safe_weighted_accuracy(y_true, y_pred, np.array(w, dtype=float))
    return acc, wacc


def _parse_joint_label(label: str) -> Dict[str, Any]:
    """将 holistic JSON 或旧 tuple 字符串标签解析为动作项 dict。"""
    import ast
    import json

    try:
        obj = json.loads(label)
        if isinstance(obj, dict):
            if len(obj) == 1 and "__default__" in obj:
                inner = obj["__default__"]
                return dict(inner) if isinstance(inner, dict) else {}
    except json.JSONDecodeError:
        pass
    try:
        pairs = ast.literal_eval(label)
        return dict(pairs) if isinstance(pairs, list) else {}
    except (ValueError, SyntaxError, TypeError):
        return {}


def _per_item_mean_accuracy(
    policy: PolicySurrogate,
    X: List[List[float]],
    y_joint: List[str],
) -> Tuple[Optional[float], Dict[str, float]]:
    """各动作项预测正确的平均比例（比联合标签完全匹配更反映代理质量）。"""
    if not X or not y_joint:
        return None, {}
    hits: Dict[str, int] = {}
    totals: Dict[str, int] = {}
    for x, y_str in zip(X, y_joint):
        true_d = _parse_joint_label(y_str)
        if not true_d:
            continue
        pred = policy.predict(x)
        pred_d = _parse_joint_label(pred) if isinstance(pred, str) else dict(pred)
        for k, v in true_d.items():
            totals[k] = totals.get(k, 0) + 1
            if pred_d.get(k) == v:
                hits[k] = hits.get(k, 0) + 1
    if not totals:
        return None, {}
    breakdown = {k: hits.get(k, 0) / totals[k] for k in totals}
    return float(np.mean(list(breakdown.values()))), breakdown


def _majority_baseline_joint_accuracy(y_joint: List[str]) -> Optional[float]:
    """计算联合标签多数类基线准确率。"""
    from collections import Counter

    if not y_joint:
        return None
    c = Counter(y_joint)
    return float(c.most_common(1)[0][1] / len(y_joint))


def _is_policy_tree_autotune_enabled() -> bool:
    """是否启用策略树深度/叶节点参数的 CV 自动调参。"""
    v = os.environ.get("ANALYSIS_CF_POLICY_AUTOTUNE", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _cv_policy_score(
    records: List[InferenceRecord],
    agent_id: int,
    *,
    mode: str,
    max_depth: int,
    min_samples_leaf: int,
) -> Optional[float]:
    """按 sim 折 CV；joint 用 RTG 加权整体决策准确率，per_item 用动作项均值。"""
    folds = max(2, min(5, len(records))) if len(records) >= 2 else 0
    if folds < 2:
        return None
    scores: List[float] = []
    for fold_idx in range(folds):
        val_idx = set(range(fold_idx, len(records), folds))
        tr_records = [r for i, r in enumerate(records) if i not in val_idx]
        va_records = [r for i, r in enumerate(records) if i in val_idx]
        if not tr_records or not va_records:
            continue
        pol = PolicySurrogate(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            mode=mode,  # type: ignore[arg-type]
        )
        pol.fit_records(tr_records, agent_id)
        X_va, y_va, w_va = _collect_policy_xyw(va_records, agent_id)
        if not y_va:
            continue
        if mode in ("joint", "composed"):
            _, wacc = _policy_accuracy_on_samples(pol, X_va, y_va, w_va)
            if wacc is not None:
                scores.append(wacc)
        else:
            acc, _ = _per_item_mean_accuracy(pol, X_va, y_va)
            if acc is not None:
                scores.append(acc)
    if not scores:
        return None
    return float(np.mean(np.array(scores, dtype=float)))


def _search_policy_tree_params(
    records: List[InferenceRecord],
    agent_id: int,
    mode: str,
    base_depth: int,
    base_leaf: int,
) -> Tuple[int, int, Optional[Dict[str, Any]]]:
    """按 sim 折 CV 选树参数；joint 优化整体决策加权准确率。"""
    if not _is_policy_tree_autotune_enabled() or len(records) < 3:
        return base_depth, base_leaf, None

    _, y_probe, _ = _collect_policy_xyw(records, agent_id)
    n_classes = len(set(y_probe)) if y_probe else 0
    depth_extra = {7, 8, 9} if mode == "joint" and n_classes >= 6 else set()
    depth_candidates = sorted(
        {max(3, base_depth - 1), base_depth, base_depth + 1, min(base_depth + 2, 12)}
        | depth_extra
    )
    leaf_candidates = sorted(
        {1, max(2, base_leaf - 2), max(1, base_leaf - 1), base_leaf, base_leaf + 2, base_leaf + 5}
    )
    best_score = -1.0
    best_report: Optional[Dict[str, Any]] = None
    metric_key = (
        "cv_holistic_weighted_accuracy_mean"
        if mode in ("joint", "composed")
        else "cv_per_item_accuracy_mean"
    )
    for d in depth_candidates:
        for leaf in leaf_candidates:
            score = _cv_policy_score(
                records,
                agent_id,
                mode=mode,
                max_depth=d,
                min_samples_leaf=leaf,
            )
            if score is not None and score > best_score:
                best_score = score
                best_report = {
                    "max_depth": d,
                    "min_samples_leaf": leaf,
                    metric_key: score,
                    "policy_mode": mode,
                }
    if best_report is None:
        return base_depth, base_leaf, None
    return int(best_report["max_depth"]), int(best_report["min_samples_leaf"]), best_report


def _enrich_policy_metrics(
    debug: Dict[str, Any],
    policy: PolicySurrogate,
    X: List[List[float]],
    y_joint: List[str],
    *,
    prefix: str,
) -> None:
    """向 debug 字典写入 per-item 准确率指标（原地修改）。"""
    acc, breakdown = _per_item_mean_accuracy(policy, X, y_joint)
    debug[f"{prefix}_per_item_accuracy"] = acc
    debug[f"{prefix}_per_item_breakdown"] = breakdown


def compute_policy_holdout_debug(
    records: List[InferenceRecord],
    agent_id: int,
    policy_fitted_on_all: PolicySurrogate,
    *,
    policy_max_depth: int,
    policy_min_samples_leaf: int,
    mode: str,
    val_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """
    计算策略 π 的训练/验证 holdout 指标。

    参数:
        records: 推理记录列表。
        agent_id: 智能体编号。
        policy_fitted_on_all: 在全量数据上已拟合的 policy。
        policy_max_depth: 策略树最大深度。
        policy_min_samples_leaf: 叶节点最小样本数。
        mode: 策略模式（joint/composed/per_item）。
        val_ratio: 验证集比例；None 时读环境变量。

    返回:
        含 train/val 准确率、per-item 分解、多数类基线等的 debug 字典。
    """
    ratio = _val_ratio_from_env() if val_ratio is None else val_ratio
    train_records, val_records = split_records_train_val(records, ratio)

    eff_depth = int(getattr(policy_fitted_on_all, "max_depth", policy_max_depth))
    eff_leaf = int(getattr(policy_fitted_on_all, "min_samples_leaf", policy_min_samples_leaf))

    X_all, y_all, w_all = _collect_policy_xyw(records, agent_id)
    train_acc, train_wacc = _policy_accuracy_on_samples(policy_fitted_on_all, X_all, y_all, w_all)
    train_pi, train_pi_bd = _per_item_mean_accuracy(policy_fitted_on_all, X_all, y_all)

    val_acc: Optional[float] = None
    val_wacc: Optional[float] = None
    val_pi: Optional[float] = None
    val_pi_bd: Dict[str, float] = {}
    val_majority: Optional[float] = None
    n_val_samples = 0
    if train_records and val_records:
        holdout_policy = PolicySurrogate(
            max_depth=eff_depth,
            min_samples_leaf=eff_leaf,
            mode=mode,  # type: ignore[arg-type]
        )
        holdout_policy.fit_records(train_records, agent_id)
        X_va, y_va, w_va = _collect_policy_xyw(val_records, agent_id)
        n_val_samples = len(y_va)
        val_majority = _majority_baseline_joint_accuracy(y_va)
        if y_va:
            val_acc, val_wacc = _policy_accuracy_on_samples(holdout_policy, X_va, y_va, w_va)
            val_pi, val_pi_bd = _per_item_mean_accuracy(holdout_policy, X_va, y_va)

    return {
        "primary_metric": (
            "policy_val_weighted_accuracy"
            if mode in ("joint", "composed")
            else "policy_val_per_item_accuracy"
        ),
        "policy_learning_target": (
            "holistic_decision_content_composed"
            if mode == "composed"
            else "holistic_decision_content"
        ),
        "agent_id": agent_id,
        "val_split_ratio": ratio,
        "n_train_records": len(train_records),
        "n_val_records": len(val_records),
        "val_sim_ids": [r.sim_id for r in val_records],
        "n_policy_samples": len(y_all),
        "n_policy_classes": len(set(y_all)) if y_all else 0,
        "n_val_samples": n_val_samples,
        "policy_train_accuracy": train_acc,
        "policy_train_weighted_accuracy": train_wacc,
        "policy_train_per_item_accuracy": train_pi,
        "policy_train_per_item_breakdown": train_pi_bd,
        "policy_val_accuracy": val_acc,
        "policy_val_weighted_accuracy": val_wacc,
        "policy_val_per_item_accuracy": val_pi,
        "policy_val_per_item_breakdown": val_pi_bd,
        "majority_baseline_val_accuracy": val_majority,
        "accuracy_note": (
            "π 学习「状态 → 一步完整 decision_content」（整体动作类）；"
            "按 agent_id 单独训练。验证指标为整体决策完全匹配（加权）；"
            "若接近 majority_baseline，说明 mock 中少数几类整体决策占主导。"
        ),
    }


def compute_policy_holdout_debug_from_rows(
    rows: List[Dict[str, Any]],
    policy_fitted_on_all: PolicySurrogate,
    *,
    feature_names: List[str],
    action_space: List[str],
    policy_max_depth: int,
    policy_min_samples_leaf: int,
    mode: str,
    val_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """转移 reservoir 路径的 holdout 指标（按 sim_id 划分）。"""
    ratio = _val_ratio_from_env() if val_ratio is None else val_ratio
    train_rows, val_rows = _split_transition_rows_by_sim(rows, ratio)

    def _xyw_from_rows(subset: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[str], List[float]]:
        """
        从转移样本行中提取策略训练用的 X、y、权重。

        参数:
            subset: 含 ``obs`` 与 ``action`` 字段的样本字典列表。

        返回:
            三元组 (观测矩阵, 动作标签列表, 全 1 权重列表)。
        """
        X = [r["obs"] for r in subset]
        y = [r["action"] for r in subset]
        w = [1.0] * len(subset)
        return X, y, w

    eff_depth = int(getattr(policy_fitted_on_all, "max_depth", policy_max_depth))
    eff_leaf = int(getattr(policy_fitted_on_all, "min_samples_leaf", policy_min_samples_leaf))

    X_all, y_all, w_all = _xyw_from_rows(rows)
    train_acc, train_wacc = _policy_accuracy_on_samples(policy_fitted_on_all, X_all, y_all, w_all)
    train_pi, train_pi_bd = _per_item_mean_accuracy(policy_fitted_on_all, X_all, y_all)

    val_acc: Optional[float] = None
    val_wacc: Optional[float] = None
    val_pi: Optional[float] = None
    val_pi_bd: Dict[str, float] = {}
    val_majority: Optional[float] = None
    n_val_samples = 0
    val_sim_ids: List[str] = sorted({str(r["sim_id"]) for r in val_rows if r.get("sim_id")})

    if train_rows and val_rows:
        holdout = PolicySurrogate(
            max_depth=eff_depth,
            min_samples_leaf=eff_leaf,
            mode=mode,  # type: ignore[arg-type]
        )
        holdout.fit_transition_rows(
            train_rows,
            feature_names=feature_names,
            action_space=action_space,
        )
        X_va, y_va, w_va = _xyw_from_rows(val_rows)
        n_val_samples = len(y_va)
        val_majority = _majority_baseline_joint_accuracy(y_va)
        if y_va:
            val_acc, val_wacc = _policy_accuracy_on_samples(holdout, X_va, y_va, w_va)
            val_pi, val_pi_bd = _per_item_mean_accuracy(holdout, X_va, y_va)

    train_sims = sorted({str(r["sim_id"]) for r in train_rows if r.get("sim_id")})
    return {
        "primary_metric": "policy_val_per_item_accuracy",
        "accuracy_note": (
            "联合标签完全匹配在 mock 中常被 3 类主导动作拖到 ~33%（≈多数类基线）；"
            "主看 policy_val_per_item_accuracy。"
        ),
        "val_split_ratio": ratio,
        "n_train_sims": len(train_sims),
        "n_val_sims": len(val_sim_ids),
        "val_sim_ids": val_sim_ids,
        "n_policy_samples": len(y_all),
        "n_policy_classes": len(set(y_all)) if y_all else 0,
        "n_val_samples": n_val_samples,
        "policy_train_accuracy": train_acc,
        "policy_train_weighted_accuracy": train_wacc,
        "policy_train_per_item_accuracy": train_pi,
        "policy_train_per_item_breakdown": train_pi_bd,
        "policy_val_accuracy": val_acc,
        "policy_val_weighted_accuracy": val_wacc,
        "policy_val_per_item_accuracy": val_pi,
        "policy_val_per_item_breakdown": val_pi_bd,
        "majority_baseline_val_accuracy": val_majority,
        "holdout_split": "by_sim",
    }


def _mode_cv_score(
    records: List[InferenceRecord],
    agent_id: int,
    *,
    mode: str,
    max_depth: int,
    min_samples_leaf: int,
    folds: int,
) -> Optional[float]:
    """
    按仿真局折 CV 评估指定策略模式的加权准确率均值。

    参数:
        records: 推理记录列表。
        agent_id: 智能体编号。
        mode: 策略模式（joint/composed/per_item）。
        max_depth: 树最大深度。
        min_samples_leaf: 叶节点最小样本数。
        folds: 折数。

    返回:
        CV 加权准确率均值；无法计算时 None。
    """
    scores: List[float] = []
    for fold_idx in range(folds):
        val_idx = set(range(fold_idx, len(records), folds))
        tr_records = [r for i, r in enumerate(records) if i not in val_idx]
        va_records = [r for i, r in enumerate(records) if i in val_idx]
        if not tr_records or not va_records:
            continue
        Xva, yva, wva = _collect_policy_xyw(va_records, agent_id)
        if not yva:
            continue
        pol = PolicySurrogate(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            mode=mode,  # type: ignore[arg-type]
        )
        pol.fit_records(tr_records, agent_id)
        pred = np.array([pol.predict(x) for x in Xva])
        s = _safe_weighted_accuracy(np.array(yva), pred, np.array(wva, dtype=float))
        if s is not None:
            scores.append(float(s))
    if not scores:
        return None
    return float(np.mean(np.array(scores, dtype=float)))


def _select_policy_mode_auto(
    records: List[InferenceRecord],
    agent_id: int,
    *,
    max_depth: int,
    min_samples_leaf: int,
) -> Tuple[str, Dict[str, Any]]:
    """
    用 K-fold(by sim) 比较 joint/per_item，返回更优模式与报告。

    说明：
    - 之前为了“快”，这里用 2~3 折快速 CV。
    - 但你现在的需求是“稳定可靠”，所以这里改成与训练调试一致的 2~5 折 CV，
      避免出现 auto 选出来的模式与后续 CV 报表相互矛盾的情况。
    """
    folds = max(2, min(5, len(records)))

    joint_score = _cv_policy_score(
        records, agent_id, mode="joint", max_depth=max_depth, min_samples_leaf=min_samples_leaf
    )
    composed_score = _cv_policy_score(
        records, agent_id, mode="composed", max_depth=max_depth, min_samples_leaf=min_samples_leaf
    )

    strict = _is_strict_conservative_enabled()
    margin = _strict_improvement_margin() if strict else 0.0
    if composed_score is None and joint_score is None:
        chosen = "composed"
    elif composed_score is None:
        chosen = "joint"
    elif joint_score is None:
        chosen = "composed"
    else:
        chosen = "composed" if composed_score >= (joint_score + margin) else "joint"
    return chosen, {
        "folds": folds,
        "strict_conservative": strict,
        "improvement_margin": margin,
        "joint_cv_weighted_accuracy_mean": joint_score,
        "composed_cv_weighted_accuracy_mean": composed_score,
        "selected_mode": chosen,
    }


def _build_transition_diagnosis(
    transition_eval: Optional[Dict[str, Any]],
    *,
    nmse_warn: float = 0.35,
    nmse_bad: float = 0.6,
) -> Optional[Dict[str, Any]]:
    """
    TODO(remove before release): 基于标准化误差给出可执行诊断结论。
    """
    if not transition_eval:
        return None
    mean_nmse = transition_eval.get("mean_nmse")
    feats = transition_eval.get("feature_metrics") or []
    if mean_nmse is None:
        return {
            "recommend_replace": False,
            "severity": "unknown",
            "reason": "mean_nmse 不可用，暂不建议替换模型。",
            "top_risky_features": [],
            "next_actions": [
                "检查 transition_eval 输入样本是否为空或方差为 0 的特征过多",
                "补充更多有效 transition 样本后再评估",
            ],
        }

    feats_valid = [f for f in feats if f.get("nmse") is not None]
    feats_sorted = sorted(feats_valid, key=lambda x: float(x.get("nmse", 0.0)), reverse=True)
    top3 = [
        {
            "feature": f.get("feature"),
            "nmse": float(f.get("nmse")),
            "mae": float(f.get("mae", 0.0)),
        }
        for f in feats_sorted[:3]
    ]

    if float(mean_nmse) >= nmse_bad:
        return {
            "recommend_replace": True,
            "severity": "high",
            "reason": f"mean_nmse={float(mean_nmse):.4f} >= {nmse_bad:.2f}，建议替换或重构 T 模型。",
            "top_risky_features": top3,
            "next_actions": [
                "优先替换 T 模型（例如 LightGBM/XGBoost 或分维模型）",
                "对 top_risky_features 进行单独建模或特征工程",
                "改用时间窗特征（t-1, t-2）增强状态转移可预测性",
                "替换后按同一任务重跑 N-MSE/MAE 与 rollout 对比",
            ],
        }
    if float(mean_nmse) >= nmse_warn:
        return {
            "recommend_replace": True,
            "severity": "medium",
            "reason": f"mean_nmse={float(mean_nmse):.4f} >= {nmse_warn:.2f}，建议先做特征分组/参数增强，再评估是否替换。",
            "top_risky_features": top3,
            "next_actions": [
                "先做参数增强：提高 n_estimators、调整 max_depth/min_samples_leaf",
                "按 top_risky_features 增加交互特征或归一化策略",
                "对观测维度做分组建模（运动学/敌机状态分开）",
                "若两轮调参后 mean_nmse 仍 >= 阈值，再替换模型",
            ],
        }
    return {
        "recommend_replace": False,
        "severity": "low",
        "reason": f"mean_nmse={float(mean_nmse):.4f}，当前 T 模型可继续使用。",
        "top_risky_features": top3,
        "next_actions": [
            "保持当前 T 模型，优先优化策略代理 π 的泛化能力",
            "持续监控 top_risky_features 的 nmse 是否上升",
            "在新任务或数据分布变化后重跑标准化评估",
        ],
    }


def _compute_transition_standardized_eval(
    transition: TransitionModel,
    records: List[InferenceRecord],
    agent_id: int,
    feature_names: List[str],
) -> Optional[Dict[str, Any]]:
    """
    计算 T 模型在验证集上的标准化误差（N-MSE/MAE）及诊断信息。

    参数:
        transition: 已拟合的转移模型。
        records: 推理记录列表。
        agent_id: 智能体编号。
        feature_names: 展平特征名列表。

    返回:
        含 mean_nmse、feature_metrics、diagnosis 的字典；无样本时 None。
    """
    trans_rows: List[np.ndarray] = []
    trans_pred_rows: List[np.ndarray] = []
    for obs_t, a_label, obs_t1, _ in iter_transitions(records, agent_id):
        pred_t1 = transition.predict(obs_t, a_label)
        trans_rows.append(np.array(obs_t1, dtype=float))
        trans_pred_rows.append(np.array(pred_t1, dtype=float))
    if not trans_rows or not trans_pred_rows:
        return None

    y_true_next = np.vstack(trans_rows)
    y_pred_next = np.vstack(trans_pred_rows)
    err = y_pred_next - y_true_next
    mse_dim = np.mean(err ** 2, axis=0)
    mae_dim = np.mean(np.abs(err), axis=0)
    var_dim = np.var(y_true_next, axis=0)
    nmse_dim = np.array(
        [float(m / v) if float(v) > 1e-12 else None for m, v in zip(mse_dim, var_dim)],
        dtype=object,
    )
    feature_metrics: List[Dict[str, Any]] = []
    for idx in range(len(mse_dim)):
        fname = feature_names[idx] if idx < len(feature_names) else f"f_{idx}"
        feature_metrics.append(
            {
                "feature": fname,
                "mse": float(mse_dim[idx]),
                "mae": float(mae_dim[idx]),
                "nmse": None if nmse_dim[idx] is None else float(nmse_dim[idx]),
            }
        )
    valid_nmse = [float(x) for x in nmse_dim.tolist() if x is not None]
    out = {
        "n_samples": int(y_true_next.shape[0]),
        "n_features": int(y_true_next.shape[1]),
        "mean_mae": float(np.mean(mae_dim)),
        "mean_nmse": float(np.mean(np.array(valid_nmse, dtype=float))) if valid_nmse else None,
        "feature_metrics": feature_metrics,
    }
    out["diagnosis"] = _build_transition_diagnosis(out)
    return out


def _fit_transition_model(
    records: List[InferenceRecord],
    agent_id: int,
    feature_names: List[str],
) -> Tuple[TransitionModel, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    拟合 T 模型；开启 autotune 时会做小网格搜索并选最优参数。
    Returns: (best_model, autotune_report, transition_eval)
    """
    if not _is_transition_autotune_enabled():
        t = TransitionModel()
        t.fit(records, agent_id)
        return t, None, _compute_transition_standardized_eval(t, records, agent_id, feature_names)

    candidates = []
    variants = ["multioutput"]
    if _is_transition_grouped_enabled():
        variants.append("per_feature")
    for model_variant in variants:
        for n_estimators in (120, 180):
            for max_depth in (8, 10, 12):
                for min_leaf in (2, 3, 5):
                    candidates.append(
                        {
                            "model_variant": model_variant,
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "min_samples_leaf": min_leaf,
                        }
                    )

    folds = max(2, min(3, len(records)))

    def _cv_mean_nmse_for_params(params: Dict[str, Any]) -> Optional[float]:
        """
        对给定超参做 K 折交叉验证，返回各折平均 NMSE 的均值。

        参数:
            params: 转移模型超参字典（model_variant、n_estimators 等）。

        返回:
            折间平均 NMSE；无有效折时返回 None。
        """
        fold_scores: List[float] = []
        for fold_idx in range(folds):
            val_idx = set(range(fold_idx, len(records), folds))
            tr = [r for i, r in enumerate(records) if i not in val_idx]
            va = [r for i, r in enumerate(records) if i in val_idx]
            if not tr or not va:
                continue
            m = TransitionModel(
                model_variant=str(params.get("model_variant", "multioutput")),
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
            )
            m.fit(tr, agent_id)
            ev = _compute_transition_standardized_eval(m, va, agent_id, feature_names)
            mean_nmse = None if ev is None else ev.get("mean_nmse")
            if mean_nmse is not None:
                fold_scores.append(float(mean_nmse))
        if not fold_scores:
            return None
        return float(np.mean(np.array(fold_scores, dtype=float)))

    # 默认参数作为保守基线：只有显著优于基线时才采用新参数。
    default_params = {
        "model_variant": "multioutput",
        "n_estimators": 120,
        "max_depth": 10,
        "min_samples_leaf": 2,
    }
    # 为 strict 判定计算 baseline 的 std（鲁棒比较）
    def _cv_nmse_scores_for_params(params: Dict[str, Any]) -> List[float]:
        """
        对给定超参做 K 折交叉验证，返回每折的平均 NMSE 列表。

        参数:
            params: 转移模型超参字典。

        返回:
            各折 mean_nmse 浮点列表；跳过无效折。
        """
        fold_scores: List[float] = []
        for fold_idx in range(folds):
            val_idx = set(range(fold_idx, len(records), folds))
            tr = [r for i, r in enumerate(records) if i not in val_idx]
            va = [r for i, r in enumerate(records) if i in val_idx]
            if not tr or not va:
                continue
            m = TransitionModel(
                model_variant=str(params.get("model_variant", "multioutput")),
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
            )
            m.fit(tr, agent_id)
            ev = _compute_transition_standardized_eval(m, va, agent_id, feature_names)
            mean_nmse = None if ev is None else ev.get("mean_nmse")
            if mean_nmse is not None:
                fold_scores.append(float(mean_nmse))
        return fold_scores

    baseline_scores = _cv_nmse_scores_for_params(default_params)
    baseline_cv_mean_nmse = float(np.mean(np.array(baseline_scores, dtype=float))) if baseline_scores else None
    baseline_cv_std_nmse = float(np.std(np.array(baseline_scores, dtype=float))) if baseline_scores else None
    strict = _is_strict_conservative_enabled()
    improvement_margin = _strict_improvement_margin() if strict else 0.0
    z = _strict_robust_z() if strict else 0.0
    best_score = float("inf")
    best_params: Optional[Dict[str, Any]] = None
    best_model: Optional[TransitionModel] = None
    best_eval: Optional[Dict[str, Any]] = None

    for params in candidates:
        fold_scores: List[float] = []
        for fold_idx in range(folds):
            val_idx = set(range(fold_idx, len(records), folds))
            tr = [r for i, r in enumerate(records) if i not in val_idx]
            va = [r for i, r in enumerate(records) if i in val_idx]
            if not tr or not va:
                continue
            m = TransitionModel(
                model_variant=str(params.get("model_variant", "multioutput")),
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
            )
            m.fit(tr, agent_id)
            ev = _compute_transition_standardized_eval(m, va, agent_id, feature_names)
            mean_nmse = None if ev is None else ev.get("mean_nmse")
            if mean_nmse is not None:
                fold_scores.append(float(mean_nmse))
        if not fold_scores:
            continue
        score = float(np.mean(np.array(fold_scores, dtype=float)))
        if score < best_score:
            best_score = score
            best_params = {
                **params,
                "cv_mean_nmse": score,
                "cv_std_nmse": float(np.std(np.array(fold_scores, dtype=float))),
                "cv_folds": len(fold_scores),
            }

    if best_params is None:
        t = TransitionModel()
        t.fit(records, agent_id)
        return t, {"enabled": True, "best": None, "fallback_default": True}, _compute_transition_standardized_eval(
            t, records, agent_id, feature_names
        )

    # 仅当候选参数在 CV 上优于默认参数（留最小裕量）时，才真正应用候选参数。
    # strict 判定：加入 std/鲁棒项，避免少数据时“看似更好”导致误采用
    use_tuned = True
    if baseline_cv_mean_nmse is not None:
        if strict:
            best_std = float(best_params.get("cv_std_nmse", 0.0) or 0.0)
            base_std = float(baseline_cv_std_nmse or 0.0)
            use_tuned = best_score <= (baseline_cv_mean_nmse - improvement_margin - z * (base_std + best_std))
        else:
            use_tuned = best_score <= (baseline_cv_mean_nmse - improvement_margin)

    if not use_tuned:
        t = TransitionModel(
            model_variant=str(default_params["model_variant"]),
            n_estimators=default_params["n_estimators"],
            max_depth=default_params["max_depth"],
            min_samples_leaf=default_params["min_samples_leaf"],
        )
        t.fit(records, agent_id)
        report = {
            "enabled": True,
            "best": best_params,
            "fallback_default": True,
            "strict_conservative": strict,
            "grouped_candidate_enabled": _is_transition_grouped_enabled(),
            "baseline_cv_mean_nmse": baseline_cv_mean_nmse,
            "baseline_cv_std_nmse": baseline_cv_std_nmse,
            "improvement_margin": improvement_margin,
            "robust_z": z,
            "reason": "tuned params not significantly better than default baseline",
        }
        return t, report, _compute_transition_standardized_eval(t, records, agent_id, feature_names)

    best_model = TransitionModel(
        model_variant=str(best_params.get("model_variant", "multioutput")),
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
    )
    best_model.fit(records, agent_id)
    best_eval = _compute_transition_standardized_eval(best_model, records, agent_id, feature_names)
    report = {
        "enabled": True,
        "best": best_params,
        "fallback_default": False,
        "strict_conservative": strict,
        "grouped_candidate_enabled": _is_transition_grouped_enabled(),
        "baseline_cv_mean_nmse": baseline_cv_mean_nmse,
        "baseline_cv_std_nmse": baseline_cv_std_nmse,
        "improvement_margin": improvement_margin,
        "robust_z": z,
    }
    return best_model, report, best_eval


@dataclass
class SurrogateBundle:
    """
  一步/多步反事实的「工具箱」：三个学到的近似模型 + 辅助数据。

    policy            : π，输入态势向量，输出动作
    transition        : T，输入 (态势, 动作)，输出下一帧态势
    reward            : R，输入 (态势, 动作, 下一态势)，输出奖励
    obs_feature_means : 各特征在训练集上的平均值（做 train_mean 扰动时用）
    feature_names     : 态势向量每一维的名字，和观测展平顺序一致
    n_training_transitions : 训练用了多少条「步与步之间」的样本（仅统计用）
    """

    policy: PolicySurrogate
    transition: TransitionModel
    reward: RewardModel
    obs_feature_means: List[float]
    feature_names: List[str]
    n_training_transitions: int = 0
    # TODO(remove before release): 临时训练观测指标，用于人工确认训练效果。
    training_debug: Optional[Dict[str, Any]] = None

    @classmethod
    def fit(
        cls,
        records: List[InferenceRecord],
        agent_id: int,
        *,
        policy_max_depth: int = 6,
        policy_min_samples_leaf: int = 2,
    ) -> "SurrogateBundle":
        """
        用多条仿真记录，为指定智能体训练 π、T、R 三个代理模型。

        参数：
            records   : 同一 inference_task 下的多局 InferenceRecord（合并训练更稳）
            agent_id  : 只学习这个智能体的数据
            policy_max_depth / policy_min_samples_leaf : 决策树复杂度控制

        返回：
            可直接传给 one_step_counterfactual / multi_step_counterfactual 的 SurrogateBundle

        小白提示：训练一次即可；解释单条决策时不要再 fit，否则很慢。
        """
        if not records:
            raise ValueError("records 不能为空。")
        from src.module_c_counterfactual.agent_schema import (
            assert_same_agent_schema,
            discover_holistic_action_space,
        )
        from src.module_c_counterfactual.training_data import joint_action_label

        agent_schema = assert_same_agent_schema(records, agent_id)
        holistic_labels: List[str] = []
        for rec in records:
            for t in range(rec.total_steps):
                dec = rec.get_decision_at(t, agent_id)
                if dec is None:
                    continue
                holistic_labels.append(joint_action_label(dec.content, rec, agent_id))
        holistic_action_space = discover_holistic_action_space(holistic_labels)

        t0 = perf_counter()

        requested_mode = _policy_mode_from_env()
        mode_select_report: Optional[Dict[str, Any]] = None
        if requested_mode == "auto":
            selected_mode, mode_select_report = _select_policy_mode_auto(
                records,
                agent_id,
                max_depth=policy_max_depth,
                min_samples_leaf=policy_min_samples_leaf,
            )
        else:
            selected_mode = requested_mode

        eff_depth = policy_max_depth
        eff_leaf = policy_min_samples_leaf
        policy_tree_autotune_report: Optional[Dict[str, Any]] = None
        if _is_policy_tree_autotune_enabled() and len(records) >= 3:
            eff_depth, eff_leaf, policy_tree_autotune_report = _search_policy_tree_params(
                records,
                agent_id,
                selected_mode,
                policy_max_depth,
                policy_min_samples_leaf,
            )

        policy = PolicySurrogate(
            max_depth=eff_depth,
            min_samples_leaf=eff_leaf,
            mode=selected_mode,  # type: ignore[arg-type]
        )
        policy.fit_records(records, agent_id)

        feat_names, obs_means = compute_obs_feature_means(records, agent_id)
        if not feat_names:
            feat_names = records[0].get_flat_feature_names(agent_id)

        transition, transition_autotune_report, transition_eval = _fit_transition_model(
            records,
            agent_id,
            feat_names,
        )

        reward = RewardModel()
        reward.fit(records, agent_id)

        n_trans = sum(1 for r in records for _ in range(max(r.total_steps - 1, 0)))
        t1 = perf_counter()

        mode = selected_mode
        holdout_dbg = compute_policy_holdout_debug(
            records,
            agent_id,
            policy,
            policy_max_depth=eff_depth,
            policy_min_samples_leaf=eff_leaf,
            mode=mode,
        )
        policy_acc = holdout_dbg.get("policy_train_accuracy")
        policy_acc_weighted = holdout_dbg.get("policy_train_weighted_accuracy")
        policy_val_acc = holdout_dbg.get("policy_val_accuracy")
        policy_val_acc_weighted = holdout_dbg.get("policy_val_weighted_accuracy")
        n_policy_samples = int(holdout_dbg.get("n_policy_samples") or 0)
        n_policy_classes = int(holdout_dbg.get("n_policy_classes") or 0)
        n_val_records = int(holdout_dbg.get("n_val_records") or 0)
        n_val_samples = int(holdout_dbg.get("n_val_samples") or 0)

        grid_best = policy_tree_autotune_report
        cv_final = _cv_policy_score(
            records, agent_id, mode=mode, max_depth=eff_depth, min_samples_leaf=eff_leaf
        )

        # 树结构信息：joint 取单树；per_item 取各子树的均值/最大值
        tree = getattr(getattr(policy, "_clf", None), "tree_", None)
        per_item_depths: List[int] = []
        per_item_leaves: List[int] = []
        if tree is None and getattr(policy, "_item_clfs", None):
            for clf_i in getattr(policy, "_item_clfs", {}).values():
                t_i = getattr(clf_i, "tree_", None)
                if t_i is None:
                    continue
                per_item_depths.append(int(t_i.max_depth))
                per_item_leaves.append(int(t_i.n_leaves))
        training_debug = {
            "fit_seconds": round(t1 - t0, 4),
            "n_records": len(records),
            "agent_id": agent_id,
            "agent_schema_fingerprint": agent_schema.fingerprint_payload(),
            "holistic_action_space_size": len(holistic_action_space),
            "holistic_action_space": holistic_action_space[:20],
            "policy_estimator": os.environ.get("ANALYSIS_CF_POLICY_ESTIMATOR", "tree"),
            "policy_preprocess": os.environ.get("ANALYSIS_CF_POLICY_PREPROCESS", "1"),
            "policy_viper_iters": os.environ.get("ANALYSIS_CF_POLICY_VIPER_ITERS", "0"),
            "policy_mode": mode,
            "policy_mode_requested": requested_mode,
            "policy_mode_auto_report": mode_select_report,
            "strict_conservative": _is_strict_conservative_enabled(),
            "strict_improvement_margin": _strict_improvement_margin(),
            "primary_metric": holdout_dbg.get("primary_metric"),
            "policy_learning_target": holdout_dbg.get("policy_learning_target"),
            "accuracy_note": holdout_dbg.get("accuracy_note"),
            "val_split_ratio": holdout_dbg.get("val_split_ratio"),
            "n_train_records": holdout_dbg.get("n_train_records"),
            "n_val_records": n_val_records,
            "val_sim_ids": holdout_dbg.get("val_sim_ids"),
            "n_policy_samples": n_policy_samples,
            "n_policy_classes": n_policy_classes,
            "n_transitions": n_trans,
            "policy_effective_max_depth": eff_depth,
            "policy_effective_min_samples_leaf": eff_leaf,
            "policy_tree_autotune": policy_tree_autotune_report,
            "policy_train_accuracy": policy_acc,
            "policy_train_weighted_accuracy": policy_acc_weighted,
            "policy_train_per_item_accuracy": holdout_dbg.get("policy_train_per_item_accuracy"),
            "policy_train_per_item_breakdown": holdout_dbg.get("policy_train_per_item_breakdown"),
            "policy_val_accuracy": policy_val_acc,
            "policy_val_weighted_accuracy": policy_val_acc_weighted,
            "policy_val_per_item_accuracy": holdout_dbg.get("policy_val_per_item_accuracy"),
            "policy_val_per_item_breakdown": holdout_dbg.get("policy_val_per_item_breakdown"),
            "majority_baseline_val_accuracy": holdout_dbg.get("majority_baseline_val_accuracy"),
            "n_val_samples": n_val_samples,
            "policy_cv_score_mean": cv_final,
            "policy_cv_per_item_accuracy_mean": cv_final,
            "policy_grid_best": grid_best,
            "policy_tree_depth": int(tree.max_depth) if tree is not None else (int(np.mean(per_item_depths)) if per_item_depths else None),
            "policy_tree_leaves": int(tree.n_leaves) if tree is not None else (int(np.mean(per_item_leaves)) if per_item_leaves else None),
            "transition_autotune": transition_autotune_report,
            "transition_eval": transition_eval,
        }
        policy_auto_report = mode_select_report or {}
        transition_report = transition_autotune_report or {}
        training_debug["optimization_summary"] = {
            "strict_conservative": _is_strict_conservative_enabled(),
            "strict_improvement_margin": _strict_improvement_margin(),
            "policy_mode": {
                "requested": requested_mode,
                "selected": mode,
                "changed": bool(mode != requested_mode),
                "reason": (
                    "policy_mode_auto_selected"
                    if requested_mode == "auto"
                    else "policy_mode_fixed_by_request"
                ),
                "report": policy_auto_report,
            },
            "transition_autotune": {
                "enabled": bool(transition_report.get("enabled")),
                "applied": bool(
                    transition_report.get("enabled")
                    and not transition_report.get("fallback_default", False)
                ),
                "fallback_default": bool(transition_report.get("fallback_default", False)),
                "reason": transition_report.get("reason"),
                "report": transition_report,
            },
        }

        return cls(
            policy=policy,
            transition=transition,
            reward=reward,
            obs_feature_means=obs_means,
            feature_names=feat_names,
            n_training_transitions=n_trans,
            training_debug=training_debug,
        )

    @classmethod
    def fit_from_transition_rows(
        cls,
        rows: List[Dict[str, Any]],
        agent_id: int,
        record: InferenceRecord,
        *,
        policy_max_depth: int = 6,
        policy_min_samples_leaf: int = 2,
    ) -> "SurrogateBundle":
        """
        从转移 reservoir 重训 π/T/R（增量 profile 用，跳过完整 CV 调试链）。
        """
        if not rows:
            raise ValueError("transition rows 不能为空。")

        requested_mode = _policy_mode_from_env()
        if requested_mode == "auto":
            selected_mode = "joint"
        else:
            selected_mode = requested_mode

        feat_names = record.get_flat_feature_names(agent_id)
        if not feat_names and rows:
            feat_names = [f"f{i}" for i in range(len(rows[0]["obs"]))]

        policy = PolicySurrogate(
            max_depth=policy_max_depth,
            min_samples_leaf=policy_min_samples_leaf,
            mode=selected_mode,  # type: ignore[arg-type]
        )
        policy.fit_transition_rows(
            rows,
            feature_names=feat_names,
            action_space=list(record.action_space),
        )

        t_pending = [(r["obs"], r["action"], r["next_obs"]) for r in rows]
        r_pending = [
            (r["obs"], r["action"], r["next_obs"], float(r["reward"])) for r in rows
        ]

        transition = TransitionModel()
        transition.fit_transition_tuples(t_pending)
        reward = RewardModel()
        reward.fit_reward_tuples(r_pending)

        obs_sums = np.zeros(len(feat_names), dtype=float)
        obs_count = 0
        for row in rows:
            for vec in (row["obs"], row["next_obs"]):
                if len(vec) == len(feat_names):
                    obs_sums += np.array(vec, dtype=float)
                    obs_count += 1
        obs_means = (obs_sums / obs_count).tolist() if obs_count else []

        holdout_dbg = compute_policy_holdout_debug_from_rows(
            rows,
            policy,
            feature_names=feat_names,
            action_space=list(record.action_space),
            policy_max_depth=policy_max_depth,
            policy_min_samples_leaf=policy_min_samples_leaf,
            mode=selected_mode,
        )
        training_debug = {
            "n_transitions": len(rows),
            "policy_mode": selected_mode,
            **holdout_dbg,
        }

        return cls(
            policy=policy,
            transition=transition,
            reward=reward,
            obs_feature_means=obs_means,
            feature_names=feat_names,
            n_training_transitions=len(rows),
            training_debug=training_debug,
        )
