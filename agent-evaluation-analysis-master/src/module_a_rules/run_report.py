"""
规则抽取流水线运行报告：样本流转日志、结果落盘、终端摘要。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score

from src.module_a_rules.extract_rules import Rule, rules_to_text
from src.module_a_rules.merge_rules import rules_coverage
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class DataFlowStep:
    """单步样本/规则流转记录。"""

    step_id: str
    name: str
    count_in: int
    count_out: int
    pct_of_baseline: float
    note: str = ""
    kind: str = "pool"  # pool | crop


@dataclass
class DataFlowTracker:
    """记录规则抽取流水线各阶段数量变化。"""

    baseline: int
    steps: List[DataFlowStep] = field(default_factory=list)

    def add(
        self,
        step_id: str,
        name: str,
        count_in: int,
        count_out: int,
        *,
        note: str = "",
        kind: str = "pool",
    ) -> None:
        pct = (count_out / self.baseline * 100.0) if self.baseline > 0 else 0.0
        self.steps.append(
            DataFlowStep(step_id, name, count_in, count_out, pct, note, kind)
        )
        tag = "[裁剪]" if kind == "crop" else "[全量/池内]"
        log.info(
            "  %s %s %6d → %8d  (占全量 %5.1f%%)  %s",
            tag,
            f"{step_id} {name}".ljust(28),
            count_in,
            count_out,
            pct,
            note,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_samples": self.baseline,
            "steps": [
                {
                    "step_id": s.step_id,
                    "name": s.name,
                    "kind": s.kind,
                    "count_in": s.count_in,
                    "count_out": s.count_out,
                    "pct_of_baseline": round(s.pct_of_baseline, 4),
                    "note": s.note,
                }
                for s in self.steps
            ],
        }


def rules_to_full_text(
    rules: Sequence[Rule],
    preprocessor: Any,
) -> str:
    """将全部规则写入文件的纯文本（无截断）。"""
    lines = [f"共 {len(rules)} 条规则\n"]
    for i, rule in enumerate(rules, start=1):
        lines.append(f"【规则 {i}】")
        lines.append(rule.to_text(pre=preprocessor))
        lines.append("-" * 50)
    return "\n".join(lines)


def rule_to_oneline(rule: Rule, preprocessor: Any, *, n_iters: int) -> str:
    """终端预览用的单行规则格式。"""
    if rule.conditions:
        if preprocessor is not None:
            cond_parts = [c.to_text(preprocessor) for c in rule.conditions]
        else:
            cond_parts = [
                f"feat[{c.feature_idx}]{c.op}{c.threshold:.4f}" for c in rule.conditions
            ]
        cond_str = " AND ".join(cond_parts)
        prefix = f"IF {cond_str}"
    else:
        prefix = "IF 无条件"
    action_str = str(rule.action)
    return (
        f"{prefix} THEN {action_str} "
        f"[置信度={rule.confidence:.2f}, 支持度={rule.support}, n_iters={n_iters}]"
    )


def split_holdout_by_episode(
    segment_lengths: Sequence[int],
    val_ratio: float = 0.2,
) -> Tuple[int, int]:
    """
    按仿真局（episode 段）划分训练/验证样本边界。

    返回 (train_end, val_start) 索引；val_start == len(y) 表示无验证集。
    """
    segs = list(segment_lengths)
    if len(segs) < 2:
        return len(segs) and sum(segs) or 0, sum(segs)
    n_val = max(1, int(round(len(segs) * val_ratio)))
    n_train = len(segs) - n_val
    if n_train < 1:
        return sum(segs), sum(segs)
    train_end = sum(segs[:n_train])
    return train_end, train_end


def evaluate_holdout(
    tree: Any,
    rules: Sequence[Rule],
    X_pre: np.ndarray,
    y: np.ndarray,
    segment_lengths: Sequence[int],
    *,
    val_ratio: float = 0.2,
) -> Dict[str, Any]:
    """在按仿真局划分的验证集上评估决策树与规则集。"""
    n = len(y)
    train_end, val_start = split_holdout_by_episode(segment_lengths, val_ratio)
    if val_start >= n or train_end <= 0:
        return {
            "enabled": False,
            "reason": "仿真局不足 2，跳过 holdout",
            "n_train": n,
            "n_val": 0,
        }

    X_tr, y_tr = X_pre[:train_end], y[:train_end]
    X_va, y_va = X_pre[val_start:], y[val_start:]

    y_pred_tr = tree.predict(X_tr)
    y_pred_va = tree.predict(X_va)
    train_acc = float(accuracy_score(y_tr, y_pred_tr))
    val_acc = float(accuracy_score(y_va, y_pred_va))
    train_cov = float(rules_coverage(list(rules), X_tr, y_tr))
    val_cov = float(rules_coverage(list(rules), X_va, y_va))

    return {
        "enabled": True,
        "val_ratio": val_ratio,
        "n_train": int(train_end),
        "n_val": int(n - val_start),
        "n_val_sims": max(1, len(segment_lengths) - max(1, int(round(len(segment_lengths) * val_ratio)))),
        "train_tree_accuracy": round(train_acc, 6),
        "val_tree_accuracy": round(val_acc, 6),
        "train_rules_coverage": round(train_cov, 6),
        "val_rules_coverage": round(val_cov, 6),
        "train_loss": round(1.0 - train_acc, 6),
        "val_loss": round(1.0 - val_acc, 6),
    }


def build_viper_metrics(viper_result: Any) -> Dict[str, Any]:
    """从 VIPERResult 汇总损失与迭代指标。"""
    history = list(getattr(viper_result, "history", []) or [])
    loss_history = list(getattr(viper_result, "loss_history", []) or [])
    acc_orig = list(getattr(viper_result, "acc_orig_history", []) or [])
    best_acc = float(viper_result.best_accuracy)
    best_loss = float(getattr(viper_result, "best_loss", 1.0 - best_acc))

    weighted_accs = [float(a) for _, a in history]
    orig_accs = [float(a) for _, a in acc_orig]
    losses = [float(l) for _, l in loss_history]

    return {
        "best_accuracy": round(best_acc, 6),
        "best_loss": round(best_loss, 6),
        "final_weighted_accuracy": round(weighted_accs[-1], 6) if weighted_accs else None,
        "final_weighted_loss": round(losses[-1], 6) if losses else None,
        "final_orig_accuracy": round(orig_accs[-1], 6) if orig_accs else None,
        "best_iter": int(np.argmax(orig_accs)) if orig_accs else 0,
        "n_iters": len(history),
        "history": [{"iter": i, "weighted_accuracy": round(a, 6), "weighted_loss": round(1.0 - a, 6)} for i, a in history],
        "acc_orig_history": [{"iter": i, "accuracy": round(a, 6)} for i, a in acc_orig],
        "augmentation_history": list(getattr(viper_result, "augmentation_history", []) or []),
    }


def make_run_dir(
    output_dir: Path,
    inference_task_id: str,
    agent_id: int,
    label_name: str,
) -> Path:
    """为单次规则抽取创建独立输出子目录。"""
    safe = label_name.replace("/", "_").replace("\\", "_")
    run_dir = output_dir / "viper_out" / f"{inference_task_id}_agent{agent_id}_{safe}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def attach_file_logger(run_dir: Path) -> logging.Handler:
    """将详细日志写入 pipeline.log，返回 handler 供调用方移除。"""
    log_path = run_dir / "pipeline.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)
    return handler


def save_run_artifacts(
    run_dir: Path,
    *,
    rules: Sequence[Rule],
    preprocessor: Any,
    result: Dict[str, Any],
    flow: DataFlowTracker,
    viper_metrics: Dict[str, Any],
    holdout: Dict[str, Any],
    n_iters: int,
) -> Dict[str, str]:
    """落盘规则文本、流转报告与汇总 JSON。"""
    rules_path = run_dir / "rules.txt"
    rules_path.write_text(rules_to_full_text(rules, preprocessor), encoding="utf-8")
    log.info("已保存规则文件: %s (%d 条)", rules_path, len(rules))

    flow_path = run_dir / "data_flow_report.json"
    flow_path.write_text(
        json.dumps(flow.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("已写入样本流转报告: %s", flow_path)

    confs = [r.confidence for r in rules] if rules else []
    final_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inference_task_id": result.get("inference_task_id"),
        "agent_id": result.get("agent_id"),
        "label_name": result.get("label_name"),
        "n_samples": result.get("n_samples"),
        "n_records": result.get("n_records"),
        "sim_ids": result.get("sim_ids"),
        "n_rules_raw": result.get("tree_rules_verification", {}).get("n_raw_rules"),
        "n_rules_final": result.get("n_rules"),
        "confidence_range": [round(min(confs), 4), round(max(confs), 4)] if confs else None,
        "accuracy": result.get("accuracy"),
        "coverage": result.get("coverage"),
        "viper_metrics": viper_metrics,
        "holdout_eval": holdout,
        "merge_check": result.get("merge_check"),
        "train_params": result.get("train_params"),
        "artifacts": {
            "rules_txt": str(rules_path),
            "rules_json": result.get("rules_json_path"),
            "pdf_path": result.get("pdf_path"),
            "rule_tree_pdf": result.get("rule_tree", {}).get("pdf_path"),
            "pipeline_log": str(run_dir / "pipeline.log"),
        },
    }
    final_path = run_dir / "final_result.json"
    final_path.write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("已写入汇总结果: %s", final_path)

    return {
        "run_dir": str(run_dir),
        "rules_txt": str(rules_path),
        "data_flow_report": str(flow_path),
        "final_result_json": str(final_path),
        "pipeline_log": str(run_dir / "pipeline.log"),
    }


def print_terminal_summary(
    result: Dict[str, Any],
    *,
    artifact_paths: Dict[str, str],
    viper_metrics: Dict[str, Any],
    holdout: Dict[str, Any],
    preprocessor: Any = None,
    preview_k: int = 10,
) -> None:
    """终端仅输出运行状态、训练成果与规则预览。"""
    rules: List[Rule] = list(result.get("rules") or [])
    n_iters = int(viper_metrics.get("n_iters") or result.get("train_params", {}).get("n_iters") or 0)

    sep = "=" * 60
    print(sep)
    print("规则抽取流水线 · 运行完成")
    print(sep)
    print(f"推理任务     : {result.get('inference_task_id')}")
    print(f"智能体 ID    : {result.get('agent_id')}")
    print(f"标签模式     : {result.get('label_name')}")
    print(f"样本数 N     : {result.get('n_samples')}  （{result.get('n_records')} 局仿真）")
    if holdout.get("enabled"):
        print(
            f"Holdout 测试 : 训练 {holdout['n_train']} 步 / 验证 {holdout['n_val']} 步  "
            f"验证准确率={holdout['val_tree_accuracy']:.4f}  验证损失={holdout['val_loss']:.4f}"
        )
    else:
        print(f"Holdout 测试 : {holdout.get('reason', '未执行')}")
    print(f"输出目录     : {artifact_paths.get('run_dir', '')}")
    print(f"VIPER 损失   : best_loss={viper_metrics.get('best_loss'):.6f}  "
          f"(best_acc={viper_metrics.get('best_accuracy'):.4f}, iter={viper_metrics.get('best_iter')})")
    if viper_metrics.get("final_weighted_loss") is not None:
        print(f"VIPER 末轮   : weighted_loss={viper_metrics['final_weighted_loss']:.6f}  "
              f"weighted_acc={viper_metrics.get('final_weighted_accuracy'):.4f}")
    print(f"规则覆盖率   : {result.get('coverage'):.4f}  决策树准确率: {result.get('accuracy'):.4f}")
    print(f"规则条数     : {result.get('n_rules')}  "
          f"(原始 {result.get('tree_rules_verification', {}).get('n_raw_rules', '?')} → "
          f"合并后 {result.get('n_rules')})")
    print(f"规则文件     : {artifact_paths.get('rules_txt', '')}")
    print(f"决策树 PDF   : {result.get('pdf_path', '')}")
    print(f"汇总 JSON    : {artifact_paths.get('final_result_json', '')}")
    print(f"流转报告     : {artifact_paths.get('data_flow_report', '')}")
    print(sep)
    if rules:
        print(f"规则预览（前 {min(preview_k, len(rules))} 条）:")
        for i, rule in enumerate(rules[:preview_k], start=1):
            line = rule_to_oneline(rule, preprocessor, n_iters=n_iters)
            print(f"  {i}. {line}")
        if len(rules) > preview_k:
            print(f"  … 共 {len(rules)} 条，完整内容见 rules.txt")
    print(sep)
