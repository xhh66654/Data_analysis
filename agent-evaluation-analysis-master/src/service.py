"""
溯因分析系统服务入口。

================================================================================
两个服务说明（小白友好版）：
================================================================================

服务A：规则抽取服务 rule_extraction_service()
    输入：
        agent_id           - 你要分析哪个智能体的决策，e.g. 1
        inference_task_id  - 推理任务 id，e.g. "INF_A_001"
                             系统会加载该任务下全部推理数据，再提取该智能体的样本合并训练
        pdf_path (可选)     - 决策树 PDF 保存路径，不传则自动生成

    处理流程：
        1. 从数据库加载 inference_task_id 对应的全部推理数据记录
        2. 提取 agent_id 在各局中的 (观测特征, 动作) 样本并合并
        3. 用 VIPER 算法（CART 决策树 + return-to-go 权重 + 迭代加权）拟合策略
        4. 从训练好的决策树 DFS 提取 IF-THEN 规则集
        5. 合并冗余规则，计算每条规则的覆盖率
        6. 导出决策树 PDF 可视化
        7. 返回规则集文本 + 结构化数据

    输出：
        {
          "inference_task_id": "INF_A_001",
          "agent_id": 1,
          "label_name": "联合动作",
          "accuracy": 0.94,
          "n_records": 3,
          "sim_ids": ["SIM_A_0001", "SIM_A_0002", "SIM_A_0003"],
          "pdf_path": "output/rule_tree_INF_A_001_agent1_联合动作.png",
          "rules_text": "规则1: IF ... THEN [('机动控制','规避'), ...]\n...",
          "rules": [Rule(...), ...],
          "n_rules": 8,
          "n_samples": 132,
          "feature_names": ["自身状态.血量", ...],
        }

服务B：反事实推理服务 counterfactual_service()
    输入：
        agent_id, inference_task_id, sim_id, decision_content[, query_step]
        cf_level: local | one_step | multi_step
        use_k_sampling: one_step/multi_step 默认 True（K 次代理采样 + 表2）
        k_samples, k_noise_scale, k_seed

    处理流程：
        1. 加载推理记录（全任务训练 + 单局定位）
        2. ObservationRollback 定位 t_query
        3. local 仅训练 π；one_step/multi_step 训练 SurrogateBundle（π/T/R）
        4. 反事实：单特征扰动 或 K 采样路径
        5. 渲染机械/目的解释 + 可选 LLM 润色

    输出：
        {
          "task_id": "INF_A_001",
          "sim_id": "SIM_A_0001",
          "agent_id": 1,
          "t_query": 3,
          "original_action": "机动控制=发射导弹  武器控制=发射导弹  ...",
          "mechanistic": "【机械性解释】该决策的状态原因分析：...",
          "teleological": "【目的性解释】该决策的意图解读：...",
          "key_features": [{"feature": "敌机距离.水平距离_km", "value": 40.0, "label": "极低", "changed": True}, ...],
          "n_key_features_changed": 3,
        }

================================================================================
从外部调用（Python API）：
================================================================================

    from src.service import rule_extraction_service, counterfactual_service

    # --- 基于规则的策略提取 ---
    result = rule_extraction_service(
        agent_id=1,
        inference_task_id="INF_A_001",
    )
    print(result["rules_text"])
    print(f"决策树 PDF：{result['pdf_path']}")

    # --- 反事实推理 ---
    result = counterfactual_service(
        agent_id=1,
        inference_task_id="INF_A_001",
        sim_id="SIM_A_0001",
        decision_content={"机动控制": "规避"},
    )
    print(result["mechanistic"])
    print(result["teleological"])

================================================================================
从命令行调用：
================================================================================

    # 基于规则的策略提取
    py main.py --mode explain_a --inference_task_id INF_A_001 --agent_id 1

    # 反事实推理
    py main.py --mode explain_c --inference_task_id INF_A_001 --sim_id SIM_A_0001 --agent_id 1 ^
               --decision 机动控制=规避
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ==============================================================================
# 输出目录（可通过环境变量 ANALYSIS_OUTPUT_DIR 覆盖）
# ==============================================================================
_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def _output_dir() -> Path:
    """
    获取分析结果输出目录，不存在时自动创建。

    可通过环境变量 ``ANALYSIS_OUTPUT_DIR`` 覆盖默认路径 ``output/``。

    返回
    ----
    Path
        输出目录路径对象。
    """
    d = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", str(_DEFAULT_OUTPUT_DIR)))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _is_train_debug_enabled() -> bool:
    """
    检测是否开启代理模型训练效果调试输出。

    读取环境变量 ``ANALYSIS_CF_TRAIN_DEBUG``；值为 ``1``/``true``/``yes``/``on`` 时启用。

    返回
    ----
    bool
        是否启用训练调试信息。

    备注
    ----
    TODO(remove before release): 临时开关，发布前移除。
    """
    flag = os.environ.get("ANALYSIS_CF_TRAIN_DEBUG", "").strip().lower()
    return flag in ("1", "true", "yes", "on")


def _is_a_autotune_enabled() -> bool:
    """
    检测是否开启模块 A（规则抽取）VIPER 自动调参。

    读取环境变量 ``ANALYSIS_A_AUTOTUNE``。

    返回
    ----
    bool
        是否启用自动调参。

    备注
    ----
    TODO(remove before release): 临时开关。
    """
    flag = os.environ.get("ANALYSIS_A_AUTOTUNE", "").strip().lower()
    return flag in ("1", "true", "yes", "on")


def _is_strict_conservative_enabled() -> bool:
    """
    检测是否开启严格保守调参模式。

    读取环境变量 ``ANALYSIS_STRICT_CONSERVATIVE``；启用后仅在交叉验证
    显著优于默认基线时才采用调参结果。

    返回
    ----
    bool
        是否启用严格保守模式。

    备注
    ----
    TODO(remove before release): 临时开关。
    """
    flag = os.environ.get("ANALYSIS_STRICT_CONSERVATIVE", "").strip().lower()
    return flag in ("1", "true", "yes", "on")


def _merge_max_coverage_drop() -> float:
    """
    获取规则合并后覆盖率的最大允许下降阈值。

    读取环境变量 ``ANALYSIS_A_MERGE_MAX_DROP``，默认 0.01；
    超过阈值时回退到未合并规则，防止可读性提升但语义退化。

    返回
    ----
    float
        覆盖率下降阈值（非负浮点数）。

    备注
    ----
    TODO(remove before release): 临时配置项。
    """
    raw = os.environ.get("ANALYSIS_A_MERGE_MAX_DROP", "").strip()
    if not raw:
        return 0.01
    try:
        v = float(raw)
        return v if v >= 0.0 else 0.01
    except Exception:
        return 0.01


def _strict_improvement_margin() -> float:
    """
    获取严格保守模式的最小改进阈值。

    读取环境变量 ``ANALYSIS_STRICT_MARGIN``，默认 ``1e-3``。

    返回
    ----
    float
        调参结果相对基线 CV 均值需超过的最小差值。

    备注
    ----
    TODO(remove before release): 临时配置项。
    """
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
    获取严格保守模式的鲁棒性系数 z。

    读取环境变量 ``ANALYSIS_STRICT_Z``，默认 1.0。判定规则：

    ``best_mean - base_mean >= margin + z * (base_std + best_std)``

    z 越大越保守；z=0 时退化为仅比较均值差。

    返回
    ----
    float
        鲁棒性系数（非负浮点数）。

    备注
    ----
    TODO(remove before release): 临时配置项。
    """
    raw = os.environ.get("ANALYSIS_STRICT_Z", "").strip()
    if not raw:
        return 1.0
    try:
        v = float(raw)
        return v if v >= 0.0 else 1.0
    except Exception:
        return 1.0


def _weighted_accuracy(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """
    计算加权准确率；权重长度不匹配时退化为未加权准确率。

    参数
    ----
    y_true : np.ndarray
        真实标签数组。
    y_pred : np.ndarray
        预测标签数组。
    weights : np.ndarray
        样本权重（如 return-to-go）；全零时等权处理。

    返回
    ----
    float
        加权准确率，范围 [0, 1]。
    """
    from sklearn.metrics import accuracy_score

    if len(weights) != len(y_true):
        return float(accuracy_score(y_true, y_pred))
    w = np.abs(np.array(weights, dtype=float))
    if float(np.sum(w)) <= 1e-12:
        w = np.ones_like(w)
    return float(accuracy_score(y_true, y_pred, sample_weight=w))


def _search_viper_params(
    records,
    agent_id: int,
    *,
    base_n_iters: int,
    base_max_depth: int,
    base_min_samples_leaf: int,
    action_item: Optional[str] = None,
) -> Dict[str, Any]:
    """
    按仿真局做 K-fold 交叉验证，小网格搜索 VIPER 超参数。

    参数
    ----
    records
        推理记录列表，每条对应一局仿真。
    agent_id : int
        目标智能体 ID。
    base_n_iters : int
        基线 VIPER 迭代次数。
    base_max_depth : int
        基线决策树最大深度。
    base_min_samples_leaf : int
        基线叶节点最小样本数。
    action_item : str | None
        可选单动作项标签；为 ``None`` 时使用联合动作。

    返回
    ----
    Dict[str, Any]
        含 ``enabled``、基线 CV 统计、``best`` 最优参数及可选 ``strict_conservative`` 判定。

    备注
    ----
    TODO(remove before release): 临时调参逻辑。
    """
    from src.module_a_rules.collect_data import (
        collect_from_records,
        collect_from_records_with_segments,
        compute_return_to_go,
    )
    from src.module_a_rules.viper import VIPERData

    k = max(2, min(5, len(records)))
    grid = []
    # 泛化优先：扩大“更保守树复杂度”候选（浅树 + 更大叶子），
    # 同时保留默认附近候选，避免只朝一个方向偏置。
    depth_candidates = sorted(
        {
            max(3, base_max_depth - 2),
            max(3, base_max_depth - 1),
            base_max_depth,
            base_max_depth + 1,
        }
    )
    leaf_candidates = sorted(
        {
            max(1, base_min_samples_leaf - 1),
            base_min_samples_leaf,
            base_min_samples_leaf + 1,
            base_min_samples_leaf + 2,
        }
    )
    iter_candidates = sorted({max(3, base_n_iters - 1), base_n_iters})
    for d in depth_candidates:
        for leaf in leaf_candidates:
            for n_iters in iter_candidates:
                for penalty in (1.5, 2.0):
                    grid.append(
                        {
                            "max_depth": d,
                            "min_samples_leaf": leaf,
                            "n_iters": n_iters,
                            "penalty_factor": penalty,
                        }
                    )

    base_key = (base_max_depth, base_min_samples_leaf, base_n_iters, 2.0)
    base_scores: List[float] = []
    best: Optional[Dict[str, Any]] = None
    best_score = -1.0
    best_std = float("inf")

    for params in grid:
        fold_scores: List[float] = []
        for fold_idx in range(k):
            val_idx = set(range(fold_idx, len(records), k))
            train_records = [r for i, r in enumerate(records) if i not in val_idx]
            val_records = [r for i, r in enumerate(records) if i in val_idx]
            if not train_records or not val_records:
                continue

            Xtr, ytr, rtr, fn, seg_tr = collect_from_records_with_segments(
                train_records, agent_id, action_item=action_item
            )
            Xva, yva, rva, _ = collect_from_records(val_records, agent_id, action_item=action_item)
            if len(ytr) == 0 or len(yva) == 0:
                continue

            v = VIPERData(
                X_raw=Xtr,
                y=ytr,
                rewards=rtr,
                feature_names=fn,
                action_item=action_item or "联合动作",
                action_space=[],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                episode_lengths=seg_tr,
            )
            vr = v.run(n_iters=params["n_iters"], penalty_factor=params["penalty_factor"])
            Xva_pre = vr.preprocessor.transform(Xva)
            ypred = vr.best_tree.predict(Xva_pre)
            wva = compute_return_to_go(np.array(rva, dtype=float))
            fold_scores.append(_weighted_accuracy(np.array(yva), np.array(ypred), np.array(wva)))

        if not fold_scores:
            continue
        mean_s = float(np.mean(np.array(fold_scores, dtype=float)))
        std_s = float(np.std(np.array(fold_scores, dtype=float)))

        key = (params["max_depth"], params["min_samples_leaf"], params["n_iters"], params["penalty_factor"])
        if key == base_key:
            base_scores = fold_scores
        # 先看均值，再看方差：均值持平时优先更稳（std 更小）。
        if (mean_s > best_score) or (abs(mean_s - best_score) <= 1e-12 and std_s < best_std):
            best_score = mean_s
            best_std = std_s
            best = {
                **params,
                "cv_weighted_accuracy_mean": mean_s,
                "cv_weighted_accuracy_std": std_s,
                "cv_folds": len(fold_scores),
            }

    report = {
        "enabled": True,
        "base_cv_weighted_accuracy_mean": float(np.mean(np.array(base_scores, dtype=float))) if base_scores else None,
        "base_cv_weighted_accuracy_std": float(np.std(np.array(base_scores, dtype=float))) if base_scores else None,
        "best": best,
    }
    if _is_strict_conservative_enabled():
        base_mean = report.get("base_cv_weighted_accuracy_mean")
        base_std = report.get("base_cv_weighted_accuracy_std")
        best_mean = (best or {}).get("cv_weighted_accuracy_mean")
        best_std = (best or {}).get("cv_weighted_accuracy_std")
        margin = _strict_improvement_margin()
        z = _strict_robust_z()
        should_apply = (
            best is not None
            and best_mean is not None
            and base_mean is not None
            and float(best_mean) >= float(base_mean) + margin + z * (float(base_std or 0.0) + float(best_std or 0.0))
        )
        report["strict_conservative"] = {
            "enabled": True,
            "improvement_margin": margin,
            "robust_z": z,
            "apply_best": bool(should_apply),
            "reason": (
                "best parameters significantly better than baseline"
                if should_apply
                else "best parameters not significantly better than baseline"
            ),
        }
    return report


# ==============================================================================
# 服务A：规则抽取服务
# ==============================================================================

def rule_extraction_service(
    agent_id: int,
    inference_task_id: Optional[str] = None,
    pdf_path: Optional[str] = None,
    n_iters: int = 5,
    max_depth: int = 6,
    min_samples_leaf: int = 2,
    *,
    task_id: Optional[str] = None,
    action_item: Optional[str] = None,
    action_items: Optional[List[str]] = None,
    unit_id: Optional[str] = None,
    update_agent_profile: bool = True,
    save_rules: bool = True,
) -> Dict[str, Any]:
    """
    规则抽取服务主入口。

    从推理任务中提取 CART 决策树，DFS 生成 IF-THEN 规则集，并导出 PDF。

    标签模式（三选一）：
    - 默认：整体决策单树（一步完整 decision 视为一个动作类，按 agent_id 单独训练）
    - action_item：单动作项一棵树，e.g. "机动控制"；多装备时可配合 unit_id
    - action_items：多动作项各一棵树，返回 mode=multi

    参数
    ----
    agent_id : int
        待分析的智能体 ID。
    inference_task_id : str | None
        推理任务 ID，如 ``"INF_A_001"``。
    pdf_path : str | None
        决策树 PDF 保存路径；不传则自动生成。
    n_iters : int
        VIPER 迭代次数，默认 5。
    max_depth : int
        决策树最大深度，默认 6。
    min_samples_leaf : int
        叶节点最小样本数，默认 2。
    task_id : str | None
        ``inference_task_id`` 的已废弃别名。
    action_item : str | None
        单动作项标签，如 ``"机动控制"``。
    action_items : List[str] | None
        多动作项列表，各训练一棵树，返回 ``mode=multi``。
    unit_id : str | None
        多装备场景下限定某一装备个体。
    update_agent_profile : bool
        是否更新 ``output/agent_profiles/``。
    save_rules : bool
        是否写入 ``output/rules/*.json``。

    返回
    ----
    Dict[str, Any]
        单树模式含规则文本、准确率、PDF 路径等；多树模式含 ``trees`` 字典。
    """
    if inference_task_id is None:
        inference_task_id = task_id
    if inference_task_id is None:
        raise ValueError("必须提供 inference_task_id（或已废弃的 task_id）。")
    if action_item is not None and action_items is not None:
        raise ValueError("action_item 与 action_items 不能同时指定。")

    from src.module_c_counterfactual.data_loader import load_inference_records
    from src.module_a_rules.pipeline import run_rule_extraction_for_label

    records = load_inference_records(inference_task_id)
    if not records:
        raise ValueError(f"推理任务 {inference_task_id} 不存在，请检查 inference_task_id。")

    all_agent_ids = sorted({aid for r in records for aid in r.agent_ids})
    if agent_id not in all_agent_ids:
        raise ValueError(
            f"agent_id={agent_id} 不在推理任务 {inference_task_id} 中。"
            f"可用的智能体 id：{all_agent_ids}"
        )

    autotune_report = None
    penalty_factor = 2.0
    if _is_a_autotune_enabled() and len(records) >= 3:
        autotune_report = _search_viper_params(
            records,
            agent_id,
            base_n_iters=n_iters,
            base_max_depth=max_depth,
            base_min_samples_leaf=min_samples_leaf,
            action_item=action_item if action_items is None else None,
        )
        best = (autotune_report or {}).get("best") or {}
        strict_apply = (autotune_report or {}).get("strict_conservative", {}).get("apply_best")
        if best and (strict_apply is None or bool(strict_apply)):
            max_depth = int(best.get("max_depth", max_depth))
            min_samples_leaf = int(best.get("min_samples_leaf", min_samples_leaf))
            n_iters = int(best.get("n_iters", n_iters))
            penalty_factor = float(best.get("penalty_factor", penalty_factor))

    project_root = Path(__file__).resolve().parent.parent
    out_dir = _output_dir()
    max_drop = _merge_max_coverage_drop()

    def _run_one(item: Optional[str], item_pdf: Optional[str] = None) -> Dict[str, Any]:
        """
        对单个动作标签执行一次完整规则抽取流水线。

        参数
        ----
        item : str | None
            动作项名称；``None`` 表示联合动作（整体决策）。
        item_pdf : str | None
            该标签对应的 PDF 输出路径；仅联合动作模式使用。

        返回
        ----
        Dict[str, Any]
            单次抽取结果，含规则、准确率及优化摘要等字段。
        """
        result = run_rule_extraction_for_label(
            records,
            agent_id,
            inference_task_id,
            action_item=item,
            unit_id=unit_id,
            pdf_path=item_pdf if item is None else None,
            n_iters=n_iters,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            penalty_factor=penalty_factor,
            output_dir=out_dir,
            project_root=project_root,
            merge_max_coverage_drop=max_drop,
            autotune_report=autotune_report,
            update_agent_profile=update_agent_profile,
            save_rules=save_rules,
        )
        result["optimization_summary"] = {
            "strict_conservative": _is_strict_conservative_enabled(),
            "strict_improvement_margin": _strict_improvement_margin(),
            "module_a_autotune_enabled": _is_a_autotune_enabled(),
            "module_a_autotune_applied": bool(
                (autotune_report or {}).get("best")
                and (
                    (autotune_report or {}).get("strict_conservative", {}).get("apply_best")
                    is not False
                )
            ),
            "module_a_autotune_reason": (autotune_report or {}).get("strict_conservative", {}).get(
                "reason"
            ),
        }
        return result

    if action_items:
        trees: Dict[str, Any] = {}
        for item in action_items:
            trees[item] = _run_one(item)
        return {
            "mode": "multi",
            "inference_task_id": inference_task_id,
            "agent_id": agent_id,
            "trees": trees,
            "n_action_items": len(trees),
        }

    single_item = action_item
    result = _run_one(single_item, pdf_path)
    result["mode"] = "single"
    return result


# ==============================================================================
# 服务B：反事实推理服务
# ==============================================================================

def counterfactual_service(
    agent_id: int,
    inference_task_id: str,
    sim_id: str,
    decision_content: Dict[str, Any],
    top_k: int = 5,
    max_depth: int = 6,
    min_samples_leaf: int = 2,
    change_score_mode: str = "action_change",
    cf_level: str = "local",
    horizon: int = 5,
    perturb_strategy: Optional[str] = None,
    explain_with_llm: Optional[bool] = None,
    query_step: Optional[int] = None,
    use_k_sampling: Optional[bool] = None,
    k_samples: int = 100,
    k_noise_scale: float = 0.1,
    k_seed: Optional[int] = None,
    bidirectional_perturb: bool = False,
    alternative_decision_content: Optional[Dict[str, Any]] = None,
    update_agent_profile: bool = True,
    update_surrogate_profile: bool = True,
) -> Dict[str, Any]:
    """
    反事实推理服务主入口（前端/命令行都调这个函数）。

    你做的一件事：指定「哪一局、哪个智能体、哪一步、什么决策内容」，
    系统返回可读解释 + 结构化 key_features。

    三种难度（cf_level）：
        local      — 只问「改一个因素，决策变不变」（最快，只需决策树）
        one_step   — 再问「改完这一步奖励会不会变」（要 π+T+R，看 1 步）
        multi_step — 再问「改完随后 3～5 步累计奖励会不会变」（要 π+T+R，滚几步）

    共同约定：
        - decision_content：用户选中的完整决策组合（非单个动作维度）
        - 每次仍只扰动「一个」观测特征，逐个排查原因
        - nl_explanation：问答式主文案（为什么 / 回答 / 或者回答）

    参数
    ----
    agent_id : int
        智能体 ID，如 1。
    inference_task_id : str
        推理任务 ID，如 ``"INF_A_001"``。
    sim_id : str
        仿真局 ID，如 ``"SIM_A_0001"``，用于定位具体推理记录。
    decision_content : Dict[str, Any]
        完整决策组合（与 ``decision_json`` 一致）；全键匹配定位时间步。
    top_k : int
        解释中展示的关键特征上限，默认 5。
    max_depth : int
        策略近似决策树最大深度，默认 6。
    min_samples_leaf : int
        策略近似决策树叶节点最少样本数，默认 2。
    change_score_mode : str
        局部反事实评分模式：``"action_change"`` 或 ``"prob_delta_l1"``。
    cf_level : str
        反事实层级：``"local"`` | ``"one_step"`` | ``"multi_step"``。
    horizon : int
        多步反事实滚动步数（仅 ``multi_step``），默认 5。
    perturb_strategy : str | None
        特征扰动策略；默认按层级自动选择。
    explain_with_llm : bool | None
        是否 LLM 润色解释；``None`` 时读环境变量 ``ANALYSIS_LLM_EXPLAIN``。
    query_step : int | None
        0-based 时间步；同一决策组合重复出现时必填。
    use_k_sampling : bool | None
        是否使用 K 次代理采样；``one_step``/``multi_step`` 默认 ``True``。
    k_samples : int
        K 采样次数，默认 100。
    k_noise_scale : float
        K 采样噪声尺度，默认 0.1。
    k_seed : int | None
        K 采样随机种子；为 ``None`` 时使用 ``t_query``。
    bidirectional_perturb : bool
        是否双向扰动特征（增大与减小均尝试）。
    alternative_decision_content : Dict[str, Any] | None
        可选对照决策，用于动作标签对比说明。
    update_agent_profile : bool
        是否更新观测预处理器 agent profile。
    update_surrogate_profile : bool
        是否更新代理模型 profile 缓存。

    返回
    ----
    Dict[str, Any]
        含机械性/目的性解释、关键特征、定位步号等，详见模块头部文档。

    异常
    ----
    ValueError
        找不到匹配决策步、参数非法或无法构造反事实样本时抛出。
    """
    if not decision_content:
        raise ValueError("decision_content 不能为空，请传入完整决策组合（与 decision_json 一致）。")
    cf_level = (cf_level or "local").strip().lower()
    if cf_level not in ("local", "one_step", "multi_step"):
        raise ValueError(f"不支持的 cf_level: {cf_level}，请使用 local、one_step 或 multi_step。")
    if perturb_strategy is None:
        perturb_strategy = "train_mean"
    if use_k_sampling is None:
        use_k_sampling = cf_level in ("one_step", "multi_step")

    agent_profile_version: Optional[int] = None
    surrogate_profile_version: Optional[int] = None
    surrogate_profile_hit = False

    # ---- 步骤1：加载推理记录（定位用单局 + 训练用全任务） ----
    from src.module_c_counterfactual.data_loader import load_inference_records
    records_all = load_inference_records(inference_task_id)
    if not records_all:
        raise ValueError(f"任务 {inference_task_id} 下无推理记录，无法训练代理模型。")
    record = next((r for r in records_all if r.sim_id == sim_id), None)
    if record is None:
        raise ValueError(
            f"找不到推理数据：inference_task_id={inference_task_id}, sim_id={sim_id}"
        )

    # ---- 步骤1b：Preprocessor 标尺（全 task 观测，与 Module A 共用 agent profile） ----
    from src.module_a_rules.agent_profile import fit_preprocessor_with_profile
    from src.module_a_rules.collect_data import collect_from_records

    X_raw, _, _, feature_names = collect_from_records(
        records_all, agent_id, action_item=None
    )
    pre, agent_prof, _agent_prof_path = fit_preprocessor_with_profile(
        X_raw,
        feature_names,
        agent_id,
        records_all[0],
        update_profile=update_agent_profile,
    )
    if agent_prof is not None:
        agent_profile_version = agent_prof.version

    # ---- 步骤2：用 ObservationRollback 定位决策时间步 ----
    from src.module_c_counterfactual.rollback import ObservationRollback
    rb = ObservationRollback(record)
    ctx = rb.from_frontend_input(
        agent_id=agent_id,
        decision_content=decision_content,
        query_step=query_step,
    )

    if ctx is None:
        snapshots = record.list_decision_snapshots(agent_id, limit=12)
        hint = "\n".join(f"  step {t}: {label}" for t, label in snapshots) or "  （无决策记录）"
        raise ValueError(
            f"在任务 {inference_task_id} / 仿真 {sim_id} 中，agent_id={agent_id} "
            f"找不到匹配 decision_content={decision_content}"
            + (f" 且 query_step={query_step}" if query_step is not None else "")
            + f" 的决策步。\n"
            f"本局前若干步决策示例：\n{hint}\n"
            f"若同一组合出现多次，请传 query_step（0-based）。"
        )

    from src.module_c_counterfactual.surrogate_cache import (
        get_or_fit_policy_surrogate,
        get_or_fit_surrogate_bundle,
    )

    policy_cache_hit = False
    bundle = None
    surrogate_cache_hit = False

    if cf_level == "local":
        policy, policy_cache_hit, surrogate_profile_hit, surrogate_profile_version = (
            get_or_fit_policy_surrogate(
                records_all,
                agent_id,
                inference_task_id,
                policy_max_depth=max_depth,
                policy_min_samples_leaf=min_samples_leaf,
                update_profile=update_surrogate_profile,
            )
        )
    else:
        bundle, surrogate_cache_hit, surrogate_profile_hit, surrogate_profile_version = (
            get_or_fit_surrogate_bundle(
                records_all,
                agent_id,
                inference_task_id,
                policy_max_depth=max_depth,
                policy_min_samples_leaf=min_samples_leaf,
                update_profile=update_surrogate_profile,
            )
        )

    # ---- 步骤3-4：反事实推理 ----
    original_reward = 0.0
    k_sampling_meta: Optional[Dict[str, Any]] = None

    if use_k_sampling and cf_level in ("one_step", "multi_step") and bundle is not None:
        from src.module_c_counterfactual.cf_dataset import generate_cf_dataset
        from src.module_c_counterfactual.causal_effect import mechanistic_effect, teleological_effect
        from src.module_c_counterfactual.explain_nl import render_k_sampling_explanation

        reward_mode = "step" if cf_level == "one_step" else "cumulative"
        eff_h = 1 if cf_level == "one_step" else horizon
        if cf_level == "one_step" and ctx.t_query >= record.total_steps - 1:
            raise ValueError(
                f"t_query={ctx.t_query} 为最后一步，无法构造 s_{{t+1}}，请换一条非末尾决策。"
            )
        samples = generate_cf_dataset(
            ctx,
            bundle,
            K=k_samples,
            horizon=eff_h,
            reward_mode=reward_mode,
            noise_scale=k_noise_scale,
            seed=k_seed if k_seed is not None else ctx.t_query,
        )
        flat_names = record.get_flat_feature_names(agent_id)
        mech_factors = mechanistic_effect(samples, flat_names)
        tele_factors = teleological_effect(samples)
        k_sampling_meta = {
            "K": k_samples,
            "horizon": eff_h,
            "reward_mode": reward_mode,
            "noise_scale": k_noise_scale,
            "seed": k_seed if k_seed is not None else ctx.t_query,
            "n_samples": len(samples),
        }
        explanation = render_k_sampling_explanation(
            mechanistic_factors=mech_factors,
            teleological_factors=tele_factors,
            action_t=ctx.action_t,
            k_meta=k_sampling_meta,
            top_k=top_k,
        )
        cf_results = samples
    elif cf_level == "one_step":
        if ctx.t_query >= record.total_steps - 1:
            raise ValueError(
                f"t_query={ctx.t_query} 为最后一步，无法构造 s_{{t+1}}，请换一条非末尾决策。"
            )
        from src.module_c_counterfactual.counterfactual import one_step_counterfactual
        from src.module_c_counterfactual.explain_nl import render_one_step_explanation

        rewards = getattr(record, "rewards", [])
        original_reward = float(rewards[ctx.t_query]) if ctx.t_query < len(rewards) else 0.0
        cf_results = one_step_counterfactual(
            ctx,
            bundle,
            perturb_strategy=perturb_strategy,
        )
    elif cf_level == "multi_step":
        from src.module_c_counterfactual.counterfactual import multi_step_counterfactual
        from src.module_c_counterfactual.explain_nl import render_multi_step_explanation

        cf_results = multi_step_counterfactual(
            ctx,
            bundle,
            horizon=horizon,
            perturb_strategy=perturb_strategy,
        )
    else:
        from src.module_c_counterfactual.counterfactual import local_counterfactual
        from src.module_c_counterfactual.explain_nl import render_cf_explanation

        ref_value = None
        if perturb_strategy == "train_mean":
            from src.module_c_counterfactual.training_data import compute_obs_feature_means

            if agent_prof is not None and len(agent_prof.mean) == len(ctx.obs_t):
                ref_value = list(agent_prof.mean)
            else:
                _fn, ref_value = compute_obs_feature_means(records_all, agent_id)
                ref_value = ref_value or None
        cf_results = local_counterfactual(
            ctx,
            policy,
            perturb_strategy=perturb_strategy,
            ref_value=ref_value,
            change_score_mode=change_score_mode,
            bidirectional=bidirectional_perturb,
        )

    flat_names = record.get_flat_feature_names(agent_id)
    if use_k_sampling and cf_level in ("one_step", "multi_step") and k_sampling_meta is not None:
        pass  # explanation 已在 K 采样分支生成
    elif cf_level == "one_step":
        explanation = render_one_step_explanation(
            results=cf_results,
            obs_t=ctx.obs_t,
            feature_names=flat_names,
            action_t=ctx.action_t,
            original_reward=original_reward,
            preprocessor=pre,
            top_k=top_k,
            perturb_strategy=perturb_strategy,
        )
    elif cf_level == "multi_step":
        explanation = render_multi_step_explanation(
            results=cf_results,
            obs_t=ctx.obs_t,
            feature_names=flat_names,
            action_t=ctx.action_t,
            preprocessor=pre,
            top_k=top_k,
            perturb_strategy=perturb_strategy,
        )
    else:
        explanation = render_cf_explanation(
            results=cf_results,
            obs_t=ctx.obs_t,
            feature_names=flat_names,
            action_t=ctx.action_t,
            preprocessor=pre,
            top_k=top_k,
        )

    from src.module_c_counterfactual.explain_nl import attach_natural_language_qa

    explanation = attach_natural_language_qa(
        explanation,
        agent_id=agent_id,
        decision_content=decision_content,
        cf_level=cf_level,
        t_query=ctx.t_query,
    )

    if alternative_decision_content:
        alt_label = str(sorted(alternative_decision_content.items()))
        explanation["alternative_comparison"] = {
            "queried_action": explanation.get("original_action"),
            "alternative_action": alt_label,
            "note": "完整 A vs B 对比需结合单特征扰动或 K 采样结果；当前提供动作标签对照。",
        }

    # ---- 整理返回值 ----
    if use_k_sampling and k_sampling_meta is not None:
        n_changed = sum(1 for s in cf_results if getattr(s, "query_happened", False))
    else:
        n_changed = sum(1 for r in cf_results if getattr(r, "action_changed", False))
    key_factors = explanation["key_features"]
    payload: Dict[str, Any] = {
        "inference_task_id":      inference_task_id,
        "task_id":                inference_task_id,  # 兼容旧字段
        "sim_id":                 sim_id,
        "agent_id":               agent_id,
        "decision_content":       decision_content,
        "query_step":             query_step,
        "cf_level":               cf_level,
        "horizon": (
            explanation.get("horizon")
            if cf_level == "multi_step" and not use_k_sampling
            else (k_sampling_meta.get("horizon") if k_sampling_meta else None)
        ),
        "perturb_strategy":       perturb_strategy,
        "n_training_records":     len(records_all),
        "n_training_transitions": bundle.n_training_transitions if bundle is not None else 0,
        "surrogate_cache_hit":    surrogate_cache_hit,
        "surrogate_profile_hit":  surrogate_profile_hit,
        "surrogate_profile_version": surrogate_profile_version,
        "agent_profile_version":  agent_profile_version,
        "policy_cache_hit":       policy_cache_hit if cf_level == "local" else None,
        "use_k_sampling":         use_k_sampling,
        "k_sampling_meta":        k_sampling_meta,
        "teleological_effect_scalar": explanation.get("teleological_effect_scalar"),
        "disclaimer":             explanation.get("disclaimer"),
        "t_query":                ctx.t_query,
        "original_action":        explanation["original_action"],
        "headline":               explanation.get("headline"),
        "explained_decision":     explanation.get("explained_decision"),
        "nl_question":            explanation.get("nl_question"),
        "nl_answer_teleological": explanation.get("nl_answer_teleological"),
        "nl_answer_mechanistic":  explanation.get("nl_answer_mechanistic"),
        "nl_explanation":         explanation.get("nl_explanation"),
        "teleological_factors":   explanation.get("teleological_factors"),
        "mechanistic_factors":    explanation.get("mechanistic_factors"),
        "mechanistic":            explanation["mechanistic"],
        "teleological":           explanation["teleological"],
        "key_features":           key_factors,
        "key_factors":            key_factors,
        "n_key_features_changed": n_changed,
        "n_key_factors_changed":  n_changed,
        "n_features_total":       len(cf_results),
    }
    # TODO(remove before release): 训练效果可视化辅助字段，默认关闭。
    if _is_train_debug_enabled():
        if bundle is not None:
            payload["train_debug"] = bundle.training_debug or {}
        elif cf_level == "local" and policy is not None:
            from src.module_c_counterfactual.surrogate_bundle import compute_policy_holdout_debug

            payload["train_debug"] = compute_policy_holdout_debug(
                records_all,
                agent_id,
                policy,
                policy_max_depth=max_depth,
                policy_min_samples_leaf=min_samples_leaf,
                mode=getattr(policy, "mode", "joint"),
            )
    if cf_level == "local":
        payload["change_score_mode"] = change_score_mode
    if cf_level == "one_step":
        payload["original_reward"] = explanation.get("original_reward", original_reward)
    if cf_level == "multi_step":
        payload["original_cumulative_reward"] = explanation.get("original_cumulative_reward")
        payload["original_action_seq"] = explanation.get("original_action_seq")
        payload["cf_action_seq"] = explanation.get("cf_action_seq")
        payload["top_feature"] = explanation.get("top_feature")
        payload["original_final_obs"] = explanation.get("original_final_obs")
        payload["cf_final_obs"] = explanation.get("cf_final_obs")
        payload["disclaimer"] = explanation.get("disclaimer")

    from src.module_c_counterfactual.llm_explain import enhance_cf_explanation

    payload = enhance_cf_explanation(
        payload,
        cf_level=cf_level,
        inference_task_id=inference_task_id,
        sim_id=sim_id,
        agent_id=agent_id,
        t_query=ctx.t_query,
        decision_content=decision_content,
        perturb_strategy=perturb_strategy,
        n_training_records=len(records_all),
        enabled=explain_with_llm,
    )
    return payload


# ==============================================================================
# 工具函数
# ==============================================================================

