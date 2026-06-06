"""
narrative — 中文自然语言段落生成。

提供两种风格：
  · plain（默认）：面向业务用户，不出现 Q 值、ΔQ 等术语
  · technical：保留模型评分与数值，供研发排查

主要函数：
  build_narrative()       → 用户可读段落
  build_narrative_technical() → 技术版（可选存档）
  build_explain_json()    → 结构化 explain.json
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from causal.step_explain.action_explain import ActionComparison
from causal.step_explain.dim_reduce import BlockImportance

# 通俗动作名（叙述用，不含「动作N」编号）
PLAIN_ACTION_LABELS: Dict[int, str] = {
    0: "保持当前航向",
    1: "向前推进",
    2: "向左转向",
    3: "向右转向",
}


def _plain_action_label(label: str, action_id: int) -> str:
    if action_id in PLAIN_ACTION_LABELS:
        return PLAIN_ACTION_LABELS[action_id]
    m = re.search(r"（([^）]+)）", label)
    return m.group(1) if m else label


def _format_importance_plain(imp: float) -> str:
    if imp < 0.01:
        return "影响很小"
    if imp < 0.05:
        return "有一定影响"
    if imp < 0.15:
        return "影响较明显"
    if imp < 0.5:
        return "影响较大"
    return "起了关键作用"


def _format_importance_technical(imp: float) -> str:
    if imp < 0.01:
        return "极低"
    if imp < 0.05:
        return "较低"
    if imp < 0.15:
        return "中等"
    if imp < 0.5:
        return "较高"
    return "显著"


def _block_effect_plain(bi: BlockImportance) -> str:
    """用自然语言描述反事实方向，避免 ΔQ。"""
    if bi.delta_q < -0.02:
        return "若缺少该信息，对当前选择的支撑会明显减弱"
    if bi.delta_q > 0.02:
        return "该信息存在时，反而略微拉低了对当前选择的支撑（可能存在交互）"
    return "对当前选择有一定参考价值"


def build_narrative(
    episode: int,
    step: Optional[int],
    reward: float,
    action_cmp: ActionComparison,
    block_importances: List[BlockImportance],
    top_k_blocks: int = 2,
    style: str = "plain",
) -> str:
    if style == "technical":
        return build_narrative_technical(
            episode, step, reward, action_cmp, block_importances, top_k_blocks
        )
    return _build_narrative_plain(
        episode, step, reward, action_cmp, block_importances, top_k_blocks
    )


def _build_narrative_plain(
    episode: int,
    step: Optional[int],
    reward: float,
    action_cmp: ActionComparison,
    block_importances: List[BlockImportance],
    top_k_blocks: int,
) -> str:
    chosen = action_cmp.chosen
    best = action_cmp.best
    chosen_name = _plain_action_label(chosen.label, chosen.action_id)

    step_str = f"第 {step} 步" if step is not None else "该步"
    reward_hint = "这一步获得了较好的即时回报" if reward > 0.3 else (
        "这一步即时回报一般" if reward > -0.3 else "这一步即时回报偏差"
    )

    header = (
        f"在第 {episode} 局 {step_str}，系统选择了「{chosen_name}」。{reward_hint}。"
    )

    if action_cmp.is_optimal:
        action_eval = (
            "根据离线策略评估，在所考虑的几种操作里，这是相对最合适的一步，"
            "其它选择的综合评分都更低。"
        )
    else:
        best_name = _plain_action_label(best.label, best.action_id)
        q_spread = max(
            (s.q_value for s in action_cmp.all_actions),
            default=1.0,
        ) - min((s.q_value for s in action_cmp.all_actions), default=0.0)
        rel = action_cmp.margin / (q_spread + 1e-8)
        if rel < 0.2:
            action_eval = (
                f"评估显示「{best_name}」略优于当前选择，但差距不大，"
                "实际策略在探索或受其它因素影响时仍可能执行当前操作。"
            )
        else:
            action_eval = (
                f"评估更推荐「{best_name}」，与当前选择有一定差距；"
                "若需完全对齐最优策略，可考虑调整决策逻辑。"
            )

    top_blocks = block_importances[:top_k_blocks]
    if top_blocks:
        parts = []
        for i, bi in enumerate(top_blocks):
            level = _format_importance_plain(bi.abs_delta)
            effect = _block_effect_plain(bi)
            if i == 0:
                parts.append(f"主要是因为「{bi.block_name}」{level}（{bi.block_desc}；{effect}）")
            else:
                parts.append(f"「{bi.block_name}」也{level}（{effect}）")
        factor_str = "从态势信息看，" + "；".join(parts) + "。"
    else:
        factor_str = "未能从态势分块中识别出明确的关键因素。"

    alt_parts = []
    for alt in action_cmp.alternatives:
        alt_name = _plain_action_label(alt.label, alt.action_id)
        gap = alt.q_value - chosen.q_value
        if gap >= -0.05:
            alt_parts.append(f"若改为「{alt_name}」，综合评分与当前接近")
        else:
            alt_parts.append(f"若改为「{alt_name}」，综合评分会明显低于当前选择")
    alt_str = ("相比之下：" + "；".join(alt_parts) + "。") if alt_parts else ""

    note = (
        "（说明：上述「综合评分」由离线轨迹训练的价值模型估计，"
        "反映长期收益倾向，不是单步即时奖励。）"
    )

    parts = [header, action_eval, factor_str]
    if alt_str:
        parts.append(alt_str)
    parts.append(note)
    return "".join(parts)


def build_narrative_technical(
    episode: int,
    step: Optional[int],
    reward: float,
    action_cmp: ActionComparison,
    block_importances: List[BlockImportance],
    top_k_blocks: int = 2,
) -> str:
    chosen = action_cmp.chosen
    best = action_cmp.best
    step_str = f"第 {step} 步" if step is not None else "某步"
    header = (
        f"第 {episode} 局 {step_str}，智能体选择了「{chosen.label}」（动作编号 {chosen.action_id}），"
        f"当前奖励为 {reward:.4f}。"
    )

    if action_cmp.is_optimal:
        action_eval = (
            f"该动作 Q 估值为 {chosen.q_value:.3f}，"
            f"是当前状态下 Q 值最高的选择（最优动作）。"
        )
    else:
        action_eval = (
            f"该动作 Q 估值为 {chosen.q_value:.3f}，"
            f"最优备选为「{best.label}」（Q = {best.q_value:.3f}），"
            f"差距为 {action_cmp.margin:.3f}。"
        )

    top_blocks = block_importances[:top_k_blocks]
    if top_blocks:
        factor_parts = []
        for bi in top_blocks:
            level = _format_importance_technical(bi.abs_delta)
            factor_parts.append(
                f"「{bi.block_name}」（{bi.block_desc}，影响程度 {level}，ΔQ={bi.delta_q:+.3f}）"
            )
        factor_str = "决策主要受以下状态因素影响：" + "、".join(factor_parts) + "。"
    else:
        factor_str = "未能计算状态因素影响。"

    alt_parts = []
    for alt in action_cmp.alternatives:
        gap = alt.q_value - chosen.q_value
        direction = "高" if gap > 0 else "低"
        alt_parts.append(
            f"「{alt.label}」Q 值 {alt.q_value:.3f}（比当前选择{direction} {abs(gap):.3f}）"
        )
    alt_str = ("备选动作参考：" + "；".join(alt_parts) + "。") if alt_parts else ""

    parts = [header, action_eval, factor_str]
    if alt_str:
        parts.append(alt_str)
    return " ".join(parts)


def build_explain_json(
    episode: int,
    step: Optional[int],
    row_index: int,
    reward: float,
    action_cmp: ActionComparison,
    block_importances: List[BlockImportance],
    top_k_blocks: int = 2,
    narrative_zh: str = "",
    narrative_technical: str = "",
    validation_report: Optional[Dict[str, Any]] = None,
    model_training_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """生成结构化 explain.json 内容。"""
    payload: Dict[str, Any] = {
        "query": {
            "episode": episode,
            "global_step": step,
            "row_index": row_index,
        },
        "chosen_action": {
            "id": action_cmp.chosen.action_id,
            "label": action_cmp.chosen.label,
            "plain_label": _plain_action_label(
                action_cmp.chosen.label, action_cmp.chosen.action_id
            ),
            "q_value": round(action_cmp.chosen.q_value, 6),
            "rank": action_cmp.chosen.rank,
        },
        "best_action": {
            "id": action_cmp.best.action_id,
            "label": action_cmp.best.label,
            "plain_label": _plain_action_label(
                action_cmp.best.label, action_cmp.best.action_id
            ),
            "q_value": round(action_cmp.best.q_value, 6),
        },
        "is_optimal": action_cmp.is_optimal,
        "margin": round(action_cmp.margin, 6),
        "reward": round(reward, 6),
        "all_actions": [
            {
                "id": s.action_id,
                "label": s.label,
                "plain_label": _plain_action_label(s.label, s.action_id),
                "q_value": round(s.q_value, 6),
                "rank": s.rank,
            }
            for s in action_cmp.all_actions
        ],
        "block_importances": [
            {
                "block_name": bi.block_name,
                "block_desc": bi.block_desc,
                "dims": bi.dims,
                "delta_q": round(bi.delta_q, 6),
                "abs_delta": round(bi.abs_delta, 6),
                "baseline": bi.baseline,
            }
            for bi in block_importances[:top_k_blocks]
        ],
        "narrative_zh": narrative_zh,
        "narrative_technical": narrative_technical,
    }
    if validation_report is not None:
        payload["validation"] = validation_report
    if model_training_report is not None:
        payload["model_training"] = model_training_report
    return payload
