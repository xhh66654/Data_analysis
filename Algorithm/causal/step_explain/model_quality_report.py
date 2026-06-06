"""
model_quality_report — 展示 decision_tree 训练产物的可读摘要。

读取 q_hat.pt 内 meta，并尽量加载同目录 final_result.json / viper_out/metrics.json，
在 step_explain 运行时打印与 decision_tree 类似的训练效果面板。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _norm_path(p: str | Path) -> str:
    try:
        return str(Path(p).resolve()).lower()
    except OSError:
        return str(p).lower()


def load_pipeline_summary(q_hat_path: str | Path) -> Optional[Dict[str, Any]]:
    """从 q_hat.pt 所在 fqe_out 目录加载 decision_tree 流水线汇总。"""
    root = Path(q_hat_path).resolve().parent
    final_json = root / "final_result.json"
    if not final_json.is_file():
        return None
    try:
        return json.loads(final_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _loss_trend(history: List[float]) -> str:
    if len(history) < 2:
        return "样本不足"
    first, last = history[0], history[-1]
    drop = (first - last) / (abs(first) + 1e-8)
    if drop > 0.15:
        return "明显下降（拟合正常）"
    if drop > 0.05:
        return "略有下降"
    if abs(drop) <= 0.05:
        return "基本持平（可增加 epoch）"
    return "未下降（建议检查数据或超参）"


def _fqe_quality_label(final_loss: float, history: List[float]) -> str:
    if final_loss < 1.0:
        base = "较好"
    elif final_loss < 5.0:
        base = "一般"
    else:
        base = "偏差"
    trend = _loss_trend(history)
    if "明显下降" in trend and final_loss < 2.0:
        return "优"
    if base == "较好":
        return "良"
    if base == "一般":
        return "中"
    return "差"


def build_model_training_report(
    q_hat_path: str | Path,
    q_meta: Dict[str, Any],
    *,
    explain_trajectory_csv: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    组装价值模型 / VIPER 训练摘要（可写入 explain.json）。
    """
    q_path = Path(q_hat_path).resolve()
    pipeline = load_pipeline_summary(q_path) or {}

    history = list(q_meta.get("loss_history") or [])
    final_loss = float(q_meta.get("final_loss") or (history[-1] if history else 0.0))

    train_csv = str(q_meta.get("csv") or pipeline.get("trajectory_csv") or "")
    explain_csv = str(explain_trajectory_csv or "")
    same_csv = (
        bool(train_csv and explain_csv)
        and _norm_path(train_csv) == _norm_path(explain_csv)
    )

    fqe_epochs = []
    for i, loss in enumerate(history, start=1):
        fqe_epochs.append({"epoch": i, "loss": round(float(loss), 6)})

    trained_by = str(q_meta.get("trained_by") or "external")
    report: Dict[str, Any] = {
        "q_hat_path": str(q_path),
        "trained_by": trained_by,
        "fqe": {
            "train_trajectory_csv": train_csv,
            "final_loss": round(final_loss, 6),
            "loss_trend_zh": _loss_trend(history),
            "quality_grade": _fqe_quality_label(final_loss, history),
            "gamma": q_meta.get("gamma"),
            "target": q_meta.get("target"),
            "state_dim": q_meta.get("state_dim"),
            "n_actions": q_meta.get("n_actions"),
            "hidden": q_meta.get("hidden"),
            "epochs": fqe_epochs,
            "n_epochs": len(history),
        },
        "trajectory_match": {
            "explain_csv": explain_csv,
            "same_as_training": same_csv,
            "warning_zh": (
                None
                if same_csv or not train_csv or not explain_csv
                else (
                    f"当前解释的轨迹与训练 Q 网络用的轨迹不一致：\n"
                    f"  训练: {train_csv}\n"
                    f"  解释: {explain_csv}\n"
                    f"  建议二者使用同一 CSV，否则评分可能不准。"
                )
            ),
        },
    }

    if q_meta.get("n_samples") is not None and not pipeline:
        report["fqe"]["n_samples"] = q_meta.get("n_samples")

    if pipeline:
        report["decision_tree_pipeline"] = {
            "n_samples": pipeline.get("n_samples"),
            "finished_at": pipeline.get("finished_at"),
            "fqe_final_loss": pipeline.get("fqe_final_loss"),
            "viper_selected_round": pipeline.get("viper_selected_round"),
            "viper_val_metric_name": pipeline.get("eval_selection_metric"),
            "viper_val_metric": pipeline.get("eval_val_metric"),
            "viper_acc_full": pipeline.get("viper_last_acc_full"),
            "viper_acc_resampled": pipeline.get("viper_last_acc_resampled"),
            "test_accuracy": pipeline.get("test_accuracy"),
            "test_macro_f1": pipeline.get("test_macro_f1"),
            "n_rules": pipeline.get("n_rules"),
            "output_dir": pipeline.get("output_dir"),
        }
        cfg = pipeline.get("config") or {}
        report["decision_tree_pipeline"]["fqe_epochs_config"] = cfg.get("fqe_epochs")
        report["decision_tree_pipeline"]["fqe_device"] = cfg.get("fqe_device")
        report["decision_tree_pipeline"]["oracle_relabel"] = cfg.get("oracle_relabel")

    metrics_path = q_path.parent / "viper_out" / "metrics.json"
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            val_m = metrics.get("val") or {}
            test_m = metrics.get("test") or {}
            report["viper_metrics"] = {
                "selection_metric": metrics.get("selection_metric"),
                "val_accuracy": val_m.get("accuracy"),
                "val_macro_f1": val_m.get("macro_f1"),
                "test_accuracy": test_m.get("accuracy"),
                "test_macro_f1": test_m.get("macro_f1"),
                "split": metrics.get("split"),
            }
        except (json.JSONDecodeError, OSError):
            pass

    return report


def format_model_training_console(report: Dict[str, Any]) -> str:
    """与 decision_tree 日志风格接近的控制台摘要。"""
    lines: List[str] = []
    fqe = report.get("fqe") or {}
    trained_by = report.get("trained_by", "external")
    if trained_by == "step_explain":
        lines.append("【价值模型训练情况】（本次运行 step_explain 根据轨迹 CSV 现场训练 FQE）")
    else:
        lines.append("【价值模型训练情况】（加载已有 q_hat.pt；可能来自 decision_tree）")
    lines.append(f"Q 网络文件   : {report.get('q_hat_path', '')}")
    if fqe.get("train_trajectory_csv"):
        lines.append(f"训练轨迹     : {fqe['train_trajectory_csv']}")
    dt = report.get("decision_tree_pipeline") or {}
    n_samples = dt.get("n_samples") or fqe.get("n_samples")
    if n_samples is not None:
        lines.append(f"训练样本数   : {n_samples}")

    lines.append(
        f"FQE 结构     : {fqe.get('state_dim')} 维状态 → hidden={fqe.get('hidden')} "
        f"→ {fqe.get('n_actions')} 动作 | target={fqe.get('target')} gamma={fqe.get('gamma')}"
    )

    epochs = fqe.get("epochs") or []
    if epochs:
        lines.append("FQE 训练 loss（逐 epoch，与 decision_tree 日志一致）:")
        for row in epochs:
            lines.append(f"  epoch {row['epoch']:>2}/{len(epochs)}  loss={row['loss']:.6f}")
        lines.append(
            f"  → 最终 loss={fqe.get('final_loss'):.6f}  "
            f"趋势={fqe.get('loss_trend_zh')}  评级={fqe.get('quality_grade')}"
        )
    else:
        lines.append(f"FQE 最终 loss : {fqe.get('final_loss', '—')}  评级={fqe.get('quality_grade', '—')}")

    if dt:
        lines.append("VIPER / 决策树（decision_tree 流水线附带产物，step_explain 不训练此项）:")
        rnd = dt.get("viper_selected_round")
        metric_name = dt.get("viper_val_metric_name") or "metric"
        val_m = dt.get("viper_val_metric")
        if rnd is not None:
            lines.append(
                f"  选用轮次     : 第 {rnd} 轮（共配置 {dt.get('fqe_epochs_config', '?')} epoch FQE 后训练）"
            )
        if val_m is not None:
            lines.append(
                f"  验证集       : {metric_name}={float(val_m):.4f}  "
                f"acc_full={float(dt.get('viper_acc_full', 0)):.4f}  "
                f"acc_resampled={float(dt.get('viper_acc_resampled', 0)):.4f}"
            )
        if dt.get("test_macro_f1") is not None:
            lines.append(
                f"  测试集       : acc={float(dt.get('test_accuracy', 0)):.4f}  "
                f"macro_f1={float(dt.get('test_macro_f1', 0)):.4f}"
            )
        if dt.get("n_rules") is not None:
            lines.append(f"  规则条数     : {dt['n_rules']}")

    vm = report.get("viper_metrics") or {}
    if vm and not dt:
        lines.append(
            f"VIPER 验证集 : macro_f1={float(vm.get('val_macro_f1', 0)):.4f}  "
            f"acc={float(vm.get('val_accuracy', 0)):.4f}"
        )

    tm = report.get("trajectory_match") or {}
    if tm.get("warning_zh"):
        lines.append("[注意] " + str(tm["warning_zh"]).replace("\n", "\n       "))

    return "\n".join(lines)
