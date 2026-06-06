"""
effect_validation — 单步解释效果自检。

不依赖人工标注，用可计算的指标判断「这一步的解释是否可信、是否有区分度」：
  · 动作一致性：模型是否认为当前动作合理
  · 动作区分度：各备选评分是否拉开差距
  · 归因强度：关键因素的反事实影响是否足够大
  · 归因可分性：前几名因素是否容易区分
  · 结果吻合：即时奖励与模型判断是否矛盾

输出 validation_report（写入 validation_report.json），并在控制台打印中文摘要。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from causal.step_explain.action_explain import ActionComparison
from causal.step_explain.dim_reduce import BlockImportance


@dataclass
class CheckResult:
    name: str
    status: str          # pass | warn | fail
    score: float         # 0~1
    message_zh: str
    detail: Dict[str, Any]


def _status_from_score(score: float, pass_th: float = 0.7, warn_th: float = 0.4) -> str:
    if score >= pass_th:
        return "pass"
    if score >= warn_th:
        return "warn"
    return "fail"


def _grade(overall: float) -> str:
    if overall >= 0.85:
        return "优"
    if overall >= 0.65:
        return "良"
    if overall >= 0.45:
        return "中"
    return "差"


def validate_explanation(
    action_cmp: ActionComparison,
    block_importances: List[BlockImportance],
    reward: float,
    *,
    q_meta: Optional[Dict[str, Any]] = None,
    min_block_delta: float = 0.05,
    min_action_spread: float = 0.1,
) -> Dict[str, Any]:
    """
    对单步解释做效果验证，返回可序列化的 validation_report。
    """
    checks: List[CheckResult] = []
    q_vals = np.array([s.q_value for s in action_cmp.all_actions], dtype=np.float64)
    q_spread = float(q_vals.max() - q_vals.min()) if len(q_vals) else 0.0
    q_std = float(q_vals.std()) if len(q_vals) > 1 else 0.0

    # 1. 动作一致性
    if action_cmp.is_optimal:
        action_score = 1.0
        action_msg = "模型认为当前动作是几种选择里最合适的一步。"
    else:
        rel_margin = action_cmp.margin / (q_spread + 1e-8)
        action_score = max(0.0, 1.0 - min(rel_margin, 1.0))
        action_msg = (
            f"模型更推荐「{action_cmp.best.label}」，与当前动作评分相差 "
            f"{action_cmp.margin:.3f}（相对差距 {rel_margin:.0%}）。"
        )
    checks.append(CheckResult(
        name="action_consistency",
        status=_status_from_score(action_score),
        score=action_score,
        message_zh=action_msg,
        detail={
            "is_optimal": action_cmp.is_optimal,
            "margin": round(action_cmp.margin, 6),
            "chosen_rank": action_cmp.chosen.rank,
            "n_actions": len(action_cmp.all_actions),
        },
    ))

    # 2. 动作区分度（spread）
    spread_score = min(1.0, q_spread / max(min_action_spread, 1e-8))
    if q_spread < min_action_spread:
        spread_msg = "各备选动作的评分非常接近，解释里「为何选这个而非那个」说服力较弱。"
    else:
        spread_msg = f"各动作评分有一定差距（极差 {q_spread:.3f}），备选对比较有意义。"
    checks.append(CheckResult(
        name="action_discrimination",
        status=_status_from_score(spread_score, pass_th=0.6, warn_th=0.3),
        score=spread_score,
        message_zh=spread_msg,
        detail={"q_spread": round(q_spread, 6), "q_std": round(q_std, 6)},
    ))

    # 3. 归因强度（top block）
    if block_importances:
        top = block_importances[0]
        strength_score = min(1.0, top.abs_delta / max(min_block_delta, 1e-8))
        if top.abs_delta < min_block_delta:
            strength_msg = (
                f"首要因素「{top.block_name}」影响偏弱（|Δ|={top.abs_delta:.3f}），"
                "归因结论可能不够稳定。"
            )
        else:
            strength_msg = (
                f"首要因素「{top.block_name}」影响明显（|Δ|={top.abs_delta:.3f}），"
                "归因较有依据。"
            )
        checks.append(CheckResult(
            name="attribution_strength",
            status=_status_from_score(strength_score, pass_th=0.65, warn_th=0.35),
            score=strength_score,
            message_zh=strength_msg,
            detail={"top_block": top.block_name, "top_abs_delta": round(top.abs_delta, 6)},
        ))

        # 4. 归因可分性（top1 vs top2）
        if len(block_importances) >= 2:
            r1, r2 = block_importances[0].abs_delta, block_importances[1].abs_delta
            ratio = r1 / (r2 + 1e-8)
            sep_score = min(1.0, (ratio - 1.0) / 2.0) if ratio > 1 else 0.3
            if ratio < 1.2:
                sep_msg = "前两个因素的影响几乎一样大，难以明确区分「最关键」的因素。"
            else:
                sep_msg = f"首要因素明显强于次要因素（强度比约 {ratio:.1f} 倍）。"
            checks.append(CheckResult(
                name="attribution_separability",
                status=_status_from_score(sep_score, pass_th=0.55, warn_th=0.25),
                score=max(0.0, sep_score),
                message_zh=sep_msg,
                detail={"top1_abs": round(r1, 6), "top2_abs": round(r2, 6), "ratio": round(ratio, 4)},
            ))
    else:
        checks.append(CheckResult(
            name="attribution_strength",
            status="fail",
            score=0.0,
            message_zh="未计算状态块归因，无法评估关键因素可信度。",
            detail={},
        ))

    # 5. 奖励与模型判断吻合
    reward_score = 1.0
    reward_msg = "即时奖励与模型判断方向一致。"
    if reward > 0 and not action_cmp.is_optimal and action_cmp.margin > 0.15 * max(q_spread, 1e-8):
        reward_score = 0.45
        reward_msg = "这一步实际拿到了正奖励，但模型更看好其它动作，解释与结果略有张力。"
    elif reward < -0.5 and action_cmp.is_optimal:
        reward_score = 0.5
        reward_msg = "这一步即时奖励较差，但模型仍认为当前动作最合适，需结合长期收益理解。"
    elif reward < -1.0 and not action_cmp.is_optimal:
        reward_score = 0.7
        reward_msg = "即时奖励较差，且模型也不优先推荐当前动作，与数据表现一致。"
    checks.append(CheckResult(
        name="reward_alignment",
        status=_status_from_score(reward_score),
        score=reward_score,
        message_zh=reward_msg,
        detail={"reward": round(float(reward), 6)},
    ))

    # 6. Q 网络训练质量（若有 meta）
    if q_meta and "final_loss" in q_meta:
        loss = float(q_meta["final_loss"])
        # 启发式：loss 过大则降权（不硬编码绝对阈值，用相对）
        loss_score = 1.0 if loss < 1.0 else (0.7 if loss < 10.0 else 0.4)
        loss_msg = (
            f"价值模型离线拟合损失为 {loss:.4f}，"
            + ("估计较可靠。" if loss_score >= 0.7 else "估计误差偏大，解释仅供参考。")
        )
        checks.append(CheckResult(
            name="q_model_fit",
            status=_status_from_score(loss_score, pass_th=0.7, warn_th=0.4),
            score=loss_score,
            message_zh=loss_msg,
            detail={"fqe_final_loss": loss},
        ))

    scores = [c.score for c in checks]
    overall = float(np.mean(scores)) if scores else 0.0
    grade = _grade(overall)

    n_fail = sum(1 for c in checks if c.status == "fail")
    n_warn = sum(1 for c in checks if c.status == "warn")

    if overall >= 0.85 and n_fail == 0:
        summary = "整体可信度高，解释与模型、数据较为一致，可直接用于汇报。"
    elif overall >= 0.65:
        summary = "整体可用，但有个别指标偏弱，建议结合业务语境阅读。"
    elif overall >= 0.45:
        summary = "解释区分度一般，建议调大训练数据或重新训练价值模型后再解释。"
    else:
        summary = "解释可信度偏低，不宜单独作为决策依据，请先检查 q_hat 与轨迹是否匹配。"

    return {
        "overall_score": round(overall, 4),
        "overall_percent": round(overall * 100, 1),
        "grade": grade,
        "summary_zh": summary,
        "n_pass": sum(1 for c in checks if c.status == "pass"),
        "n_warn": n_warn,
        "n_fail": n_fail,
        "checks": [
            {
                "name": c.name,
                "status": c.status,
                "score": round(c.score, 4),
                "message_zh": c.message_zh,
                "detail": c.detail,
            }
            for c in checks
        ],
    }


def format_validation_console(report: Dict[str, Any]) -> str:
    """控制台打印用的验证摘要。"""
    lines = [
        f"效果评分     : {report['overall_percent']}%（{report['grade']}）",
        f"结论         : {report['summary_zh']}",
        f"检查项       : 通过 {report['n_pass']} / 警告 {report['n_warn']} / 未通过 {report['n_fail']}",
    ]
    for c in report.get("checks") or []:
        tag = {"pass": "通过", "warn": "注意", "fail": "未过"}.get(c["status"], "?")
        lines.append(f"  [{tag}] {c['message_zh']}")
    return "\n".join(lines)
