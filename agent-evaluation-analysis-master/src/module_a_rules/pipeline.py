"""
规则抽取单标签流水线（VIPER → 规则 → PDF），供 service 调用。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.module_a_rules.agent_profile import fit_preprocessor_with_profile
from src.module_a_rules.collect_data import collect_from_records_with_segments
from src.module_a_rules.extract_rules import extract_rules_from_tree, rules_to_text
from src.module_a_rules.merge_rules import merge_rules, rules_coverage
from src.module_a_rules.run_report import (
    DataFlowTracker,
    attach_file_logger,
    build_viper_metrics,
    evaluate_holdout,
    make_run_dir,
    save_run_artifacts,
)
from src.utils.logger import get_logger
from src.module_a_rules.rule_match import save_rules_json
from src.module_a_rules.rule_tree import (
    build_rule_tree,
    build_rule_tree_from_sklearn_tree,
    explain_rule_merge_steps,
    export_rule_tree_pdf,
    merge_steps_to_dict,
    rule_tree_to_dict,
)
from src.module_a_rules.train_params import resolve_train_params
from src.module_a_rules.verify_tree_rules import verify_tree_and_rules
from src.module_a_rules.viper import VIPERData
from src.module_c_counterfactual.inference_record import InferenceRecord

log = get_logger(__name__)


def _label_name_for(action_item: Optional[str]) -> str:
    """
    将动作项名称转换为流水线使用的标签名。

    参数:
        action_item: 动作项名称；为 ``None`` 时表示整体决策模式。

    返回:
        动作项名称或默认字符串「整体决策」。
    """
    return action_item if action_item else "整体决策"


def _assert_records_for_agent(records: List[InferenceRecord], agent_id: int) -> None:
    """
    校验所有推理记录均包含指定智能体。

    参数:
        records: 待用于规则抽取的推理记录列表。
        agent_id: 目标智能体 ID。

    抛出:
        ValueError: 任一记录中不存在该 ``agent_id`` 时。
    """
    missing = [r.sim_id for r in records if agent_id not in r.agent_ids]
    if missing:
        raise ValueError(
            f"agent_id={agent_id} 在以下仿真局中不存在: {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}"
        )


def _safe_filename_part(name: str) -> str:
    """
    将标签名转为安全的文件名片段。

    参数:
        name: 原始标签名，可能含路径分隔符。

    返回:
        替换 ``/`` 与 ``\\`` 后的安全字符串。
    """
    return name.replace("/", "_").replace("\\", "_")


def run_rule_extraction_for_label(
    records: List[InferenceRecord],
    agent_id: int,
    inference_task_id: str,
    *,
    action_item: Optional[str] = None,
    unit_id: Optional[str] = None,
    pdf_path: Optional[str] = None,
    n_iters: int = 5,
    max_depth: int = 6,
    min_samples_leaf: int = 2,
    penalty_factor: float = 2.0,
    resample_augment: bool = True,
    output_dir: Path,
    project_root: Path,
    merge_max_coverage_drop: float,
    autotune_report: Optional[Dict[str, Any]] = None,
    update_agent_profile: bool = True,
    save_rules: bool = True,
) -> Dict[str, Any]:
    """
    对单个智能体完成端到端规则抽取流水线（VIPER → 规则 → PDF）。

    默认将一步完整决策视为一个动作类（整体决策模式）。

    参数:
        records: 用于训练的推理记录列表。
        agent_id: 目标智能体 ID。
        inference_task_id: 推理任务标识，用于输出文件命名。
        action_item: 动作项名称；``None`` 表示整体决策标签。
        unit_id: 多装备个体时与 ``action_item`` 联用的个体 ID。
        pdf_path: 决策树 PDF 输出路径前缀；``None`` 时自动生成。
        n_iters: VIPER 迭代轮数。
        max_depth: 决策树最大深度。
        min_samples_leaf: 叶节点最少样本数。
        penalty_factor: VIPER 错误样本惩罚倍率。
        resample_augment: 是否在 VIPER 中启用重采样增广。
        output_dir: 输出目录（PDF、规则 JSON 等）。
        project_root: 项目根目录，用于生成相对路径。
        merge_max_coverage_drop: 允许合并规则相对原始规则的最大覆盖率下降。
        autotune_report: 可选的自适应调参报告，写入返回结果。
        update_agent_profile: 是否更新本地预处理器 profile。
        save_rules: 是否将规则集保存为 JSON。

    返回:
        包含准确率、覆盖率、规则文本、PDF 路径、校验结果等的字典。
    """
    label_name = _label_name_for(action_item)
    record0 = records[0]
    _assert_records_for_agent(records, agent_id)

    run_dir = make_run_dir(output_dir, inference_task_id, agent_id, label_name)
    file_handler = attach_file_logger(run_dir)
    log.info("规则抽取开始: task=%s agent=%s label=%s", inference_task_id, agent_id, label_name)

    X_raw, y, rewards, feature_names, segment_lengths = collect_from_records_with_segments(
        records, agent_id, action_item=action_item, unit_id=unit_id
    )
    flow = DataFlowTracker(baseline=len(y))
    log.info("======== 样本流转汇总（相对训练池全量） ========")
    flow.add(
        "00",
        "加载推理记录",
        len(records),
        len(y),
        note=f"{len(records)} 局仿真 → {len(y)} 训练步；agent_id={agent_id}",
    )
    if len(y) == 0:
        raise ValueError(
            f"agent_id={agent_id} label={label_name} 在任务 {inference_task_id} 中无样本。"
        )

    pre, profile, profile_abs_path = fit_preprocessor_with_profile(
        X_raw,
        feature_names,
        agent_id,
        record0,
        update_profile=update_agent_profile,
    )
    flow.add(
        "01",
        "特征归一化+分箱标尺",
        len(y),
        len(y),
        note=f"{len(feature_names)} 维特征；行数不变",
    )

    from src.module_c_counterfactual.agent_schema import discover_holistic_action_space

    action_space: List[str] = []
    if action_item:
        for item in record0.action_items:
            if item.name == action_item:
                action_space = [str(v) for v in item.possible_values]
                break
    else:
        action_space = discover_holistic_action_space(y)

    n_classes = len({str(v) for v in y})
    tuned = resolve_train_params(
        len(y),
        n_classes,
        action_item=action_item,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_iters=n_iters,
        resample_augment=resample_augment,
        penalty_factor=penalty_factor,
    )
    max_depth = tuned["max_depth"]
    min_samples_leaf = tuned["min_samples_leaf"]
    n_iters = tuned["n_iters"]
    resample_augment = tuned["resample_augment"]
    penalty_factor = tuned["penalty_factor"]
    uniform_base_weights = tuned.get("uniform_base_weights", False)

    viper = VIPERData(
        X_raw=X_raw,
        y=y,
        rewards=rewards,
        feature_names=feature_names,
        action_item=label_name,
        action_space=action_space,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        episode_lengths=segment_lengths,
        preprocessor=pre,
        uniform_base_weights=uniform_base_weights,
    )
    flow.add(
        "02",
        "VIPER 训练池准备",
        len(y),
        len(y),
        note=f"n_iters={n_iters} max_depth={max_depth} n_classes={n_classes}",
    )
    viper_result = viper.run(
        n_iters=n_iters,
        penalty_factor=penalty_factor,
        resample_augment=resample_augment,
    )
    viper_metrics = build_viper_metrics(viper_result)
    log.info(
        "VIPER 训练完成: best_acc=%.4f best_loss=%.6f best_iter=%s",
        viper_metrics["best_accuracy"],
        viper_metrics["best_loss"],
        viper_metrics["best_iter"],
    )
    flow.add(
        "03",
        f"VIPER 重采样→CART.fit",
        len(y),
        len(y),
        note=f"共 {n_iters} 轮；末轮 weighted_loss={viper_metrics.get('final_weighted_loss')}",
    )

    raw_rules = extract_rules_from_tree(
        viper_result.best_tree,
        preprocessor=viper_result.preprocessor,
    )
    flow.add(
        "04",
        "决策树 DFS 提取规则",
        len(raw_rules),
        len(raw_rules),
        note="叶节点数=原始规则数",
        kind="crop",
    )
    merged = merge_rules(raw_rules)
    X_pre = viper_result.preprocessor.transform(X_raw)
    coverage_raw = rules_coverage(raw_rules, X_pre, y)
    coverage_merged = rules_coverage(merged, X_pre, y)
    use_merged_rules = (coverage_raw - coverage_merged) <= merge_max_coverage_drop
    all_rules = merged if use_merged_rules else raw_rules
    confs = [r.confidence for r in all_rules] if all_rules else []
    log.info(
        "规则合并完成: 原始=%d 最终=%d use_merged=%s 置信度范围=%s",
        len(raw_rules),
        len(all_rules),
        use_merged_rules,
        f"[{min(confs):.4f}, {max(confs):.4f}]" if confs else "N/A",
    )
    flow.add(
        "05",
        "规则合并",
        len(raw_rules),
        len(all_rules),
        note=f"覆盖率 {coverage_raw:.4f}→{coverage_merged:.4f}；采用={'合并' if use_merged_rules else '原始'}",
        kind="crop",
    )
    holdout = evaluate_holdout(
        viper_result.best_tree,
        all_rules,
        X_pre,
        np.asarray(y),
        segment_lengths,
    )
    if holdout.get("enabled"):
        log.info(
            "Holdout 测试: train_acc=%.4f val_acc=%.4f train_loss=%.4f val_loss=%.4f",
            holdout["train_tree_accuracy"],
            holdout["val_tree_accuracy"],
            holdout["train_loss"],
            holdout["val_loss"],
        )
    else:
        log.info("Holdout 测试: %s", holdout.get("reason", "跳过"))

    tree_rules_check = verify_tree_and_rules(
        viper_result.best_tree,
        X_pre,
        np.asarray(y),
        merged_rules=merged,
    )
    display_max_depth = int(os.environ.get("TREE_VIZ_MAX_DEPTH", "6"))
    merge_steps = explain_rule_merge_steps(raw_rules)
    merged_rule_tree = build_rule_tree(
        all_rules,
        feature_names,
        preprocessor=viper_result.preprocessor,
    )
    model_rule_tree = build_rule_tree_from_sklearn_tree(
        viper_result.best_tree,
        feature_names,
        preprocessor=viper_result.preprocessor,
        max_depth=display_max_depth,
    )
    rules_text = rules_to_text(all_rules, preprocessor=viper_result.preprocessor)

    if pdf_path is None:
        pdf_path = str(
            output_dir
            / f"rule_tree_{inference_task_id}_agent{agent_id}_{_safe_filename_part(label_name)}"
        )

    from src.viz.tree_plot import export_tree_pdf

    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    title = (
        f"inference_task_id={inference_task_id}  agent_id={agent_id}  "
        f"label={label_name}  n_records={len(records)}"
    )
    actual_pdf = export_tree_pdf(
        tree=viper_result.best_tree,
        out_path=pdf_path,
        feature_names=feature_names,
        class_names=[str(c) for c in viper_result.best_tree.classes_],
        preprocessor=viper_result.preprocessor,
        title=title,
        display_max_depth=display_max_depth,
    )
    try:
        actual_pdf_rel = str(Path(actual_pdf).resolve().relative_to(project_root.resolve()))
    except Exception:
        actual_pdf_rel = actual_pdf

    legend_rel = None
    legend_candidate = Path(actual_pdf).with_name(f"{Path(actual_pdf).stem}_legend.txt")
    if legend_candidate.exists():
        try:
            legend_rel = str(legend_candidate.resolve().relative_to(project_root.resolve()))
        except Exception:
            legend_rel = str(legend_candidate)

    rule_tree_pdf_rel = None
    try:
        rule_tree_out = str(
            output_dir
            / f"rule_tree_merged_{inference_task_id}_agent{agent_id}_{_safe_filename_part(label_name)}"
        )
        rule_tree_pdf = export_rule_tree_pdf(
            merged_rule_tree,
            rule_tree_out,
            title=f"合并规则树 ({merged_rule_tree.tree_kind})",
        )
        rule_tree_pdf_rel = str(Path(rule_tree_pdf).resolve().relative_to(project_root.resolve()))
    except Exception as e:
        print(f"[rule_tree] 合并规则树 PDF 导出失败: {e}")

    rules_json_rel = None
    if save_rules:
        rules_dir = output_dir / "rules"
        rules_path = rules_dir / (
            f"{inference_task_id}_agent{agent_id}_{_safe_filename_part(label_name)}.json"
        )
        save_rules_json(
            rules_path,
            all_rules,
            feature_names=feature_names,
            metadata={
                "inference_task_id": inference_task_id,
                "agent_id": agent_id,
                "label_name": label_name,
                "action_item": action_item,
                "accuracy": round(viper_result.best_accuracy, 4),
            },
        )
        try:
            rules_json_rel = str(rules_path.resolve().relative_to(project_root.resolve()))
        except Exception:
            rules_json_rel = str(rules_path)

    profile_rel = None
    if profile_abs_path is not None:
        try:
            profile_rel = str(profile_abs_path.resolve().relative_to(project_root.resolve()))
        except Exception:
            profile_rel = str(profile_abs_path)

    sim_ids = [r.sim_id for r in records]
    schema = record0.get_agent_schema(agent_id)

    result_body: Dict[str, Any] = {
        "inference_task_id": inference_task_id,
        "task_id": inference_task_id,
        "agent_id": agent_id,
        "label_name": label_name,
        "label_mode": "holistic_decision" if action_item is None else "action_item",
        "action_item": action_item,
        "unit_id": unit_id,
        "agent_schema": {
            "equipment_units": list(schema.equipment_units),
            "observation_space": list(schema.observation_space),
            "action_items": list(schema.action_item_names),
            "holistic_action_space": action_space if action_item is None else None,
            "n_holistic_classes": len(action_space) if action_item is None else None,
            "is_multi_unit": schema.is_multi_unit,
            "n_units": len(schema.equipment_units),
            "feature_dim": len(feature_names),
        },
        "accuracy": round(viper_result.best_accuracy, 4),
        "coverage": round(coverage_merged if use_merged_rules else coverage_raw, 4),
        "pdf_path": actual_pdf_rel,
        "viz_legend_path": legend_rel,
        "rules_text": rules_text,
        "rules": all_rules,
        "_preprocessor": viper_result.preprocessor,
        "n_rules": len(all_rules),
        "n_samples": len(y),
        "n_records": len(records),
        "sim_ids": sim_ids,
        "feature_names": feature_names,
        "viper_history": viper_result.history,
        "viper_loss_history": viper_result.loss_history,
        "viper_acc_orig_history": viper_result.acc_orig_history,
        "viper_best_loss": viper_result.best_loss,
        "viper_metrics": viper_metrics,
        "holdout_eval": holdout,
        "viper_augmentation_history": viper_result.augmentation_history,
        "autotune": autotune_report,
        "preprocessor_profile": {
            "profile_id": profile.profile_id if profile else None,
            "profile_path": profile_rel,
            "schema_fingerprint": profile.schema_fingerprint if profile else None,
            "n_samples": profile.n_samples if profile else None,
            "version": profile.version if profile else None,
        },
        "rules_json_path": rules_json_rel,
        "rule_merge": {
            "steps": merge_steps_to_dict(merge_steps),
            "use_merged_rules": bool(use_merged_rules),
            "summary": (
                "合并后规则集可组装为树；若 tree_kind=decision_tree_equivalent 则与决策树遍历一致，"
                "否则为 prefix_tree_with_conflicts（需按最长路径或规则优先级匹配）。"
            ),
        },
        "rule_tree": {
            "from_merged_rules": rule_tree_to_dict(merged_rule_tree.tree),
            "tree_kind": merged_rule_tree.tree_kind,
            "is_decision_tree_compatible": merged_rule_tree.is_decision_tree_compatible,
            "overlap_conflicts": merged_rule_tree.overlap_conflicts,
            "pdf_path": rule_tree_pdf_rel,
        },
        "model_tree": {
            "from_sklearn_tree": rule_tree_to_dict(model_rule_tree.tree),
            "tree_kind": model_rule_tree.tree_kind,
            "note": "模型原生决策树结构，保证二叉树；与 merge 后规则树可对照查看。",
        },
        "tree_rules_verification": {
            "n_leaves": tree_rules_check.n_leaves,
            "n_raw_rules": tree_rules_check.n_raw_rules,
            "n_merged_rules": tree_rules_check.n_merged_rules,
            "raw_paths_one_to_one": tree_rules_check.raw_paths_one_to_one,
            "raw_matches_tree_predict": tree_rules_check.raw_matches_tree_predict,
            "merged_matches_tree_predict": tree_rules_check.merged_matches_tree_predict,
            "tree_accuracy_on_train": round(float(tree_rules_check.tree_accuracy_on_train), 6),
            "raw_rules_coverage": round(float(tree_rules_check.raw_rules_coverage), 6),
            "merged_rules_coverage": round(float(tree_rules_check.merged_rules_coverage), 6),
            "details": tree_rules_check.details,
        },
        "merge_check": {
            "coverage_raw": round(float(coverage_raw), 6),
            "coverage_merged": round(float(coverage_merged), 6),
            "coverage_drop": round(float(coverage_raw - coverage_merged), 6),
            "max_allowed_drop": float(merge_max_coverage_drop),
            "use_merged_rules": bool(use_merged_rules),
        },
        "train_params": {
            "n_iters": n_iters,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "penalty_factor": penalty_factor,
            "resample_augment": resample_augment,
            "n_classes": n_classes,
            "uniform_base_weights": uniform_base_weights,
            "adaptive": True,
        },
    }
    log.info("====================================================")
    artifact_paths = save_run_artifacts(
        run_dir,
        rules=all_rules,
        preprocessor=viper_result.preprocessor,
        result=result_body,
        flow=flow,
        viper_metrics=viper_metrics,
        holdout=holdout,
        n_iters=n_iters,
    )
    result_body["run_artifacts"] = artifact_paths
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()
    return result_body
