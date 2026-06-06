"""
action_explain — 动作维约简：Q 值全动作比较。

给定当前状态 s 和智能体选择的 action，用 Q 网络计算所有动作的 Q 值，
得出：
  · chosen 动作的 Q 值与排名
  · 最优动作（Q 最高）及其 Q 值
  · 与最优动作的差距 margin
  · Top-K 个备选动作（供写入叙述）

主要类型与函数：
  ActionScore        — 单个动作的 Q 值信息
  ActionComparison   — 完整动作比较结果
  action_labels      — 默认动作标签（可由外部 YAML/JSON 覆盖）
  compare_actions()  — 对一个状态做全动作 Q 评估
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from causal.decision_tree.q_network import QHatNetwork

logger = logging.getLogger(__name__)

# 默认动作标签；可通过 compare_actions(action_labels=...) 覆盖
DEFAULT_ACTION_LABELS: Dict[int, str] = {
    0: "动作0（保持）",
    1: "动作1（前进）",
    2: "动作2（转向左）",
    3: "动作3（转向右）",
}


@dataclass
class ActionScore:
    action_id: int
    label: str
    q_value: float
    rank: int        # 1 = 最优


@dataclass
class ActionComparison:
    chosen: ActionScore
    best: ActionScore           # Q 最高的动作
    margin: float               # best.q_value - chosen.q_value（≥0）
    is_optimal: bool            # chosen == best
    all_actions: List[ActionScore]          # 按 Q 值降序
    alternatives: List[ActionScore]         # 除 chosen 外 Top-K 备选（按 Q 降序）


def compare_actions(
    q_net: QHatNetwork,
    state: np.ndarray,
    chosen_action: int,
    action_labels: Optional[Dict[int, str]] = None,
    top_k_alternatives: int = 2,
    device: str | torch.device = "cpu",
) -> ActionComparison:
    """
    计算所有动作在 state 下的 Q 值，返回 ActionComparison。

    参数
    ----
    q_net              : 已加载的 QHatNetwork
    state              : 当前步状态，shape (state_dim,)
    chosen_action      : 智能体实际执行的动作 id
    action_labels      : 动作 id → 中文标签；None 时用默认标签
    top_k_alternatives : 除 chosen 外返回多少个备选动作
    device             : 计算设备
    """
    labels = action_labels or DEFAULT_ACTION_LABELS

    dev = torch.device(device)
    q_net = q_net.to(dev)
    q_net.eval()

    s_t = torch.from_numpy(state).unsqueeze(0).float().to(dev)
    with torch.no_grad():
        q_all = q_net(s_t)[0].cpu().numpy()  # shape (n_actions,)

    n_actions = len(q_all)

    ranked_ids = np.argsort(-q_all)  # 降序
    scores: List[ActionScore] = []
    for rank_idx, aid in enumerate(ranked_ids):
        scores.append(ActionScore(
            action_id=int(aid),
            label=labels.get(int(aid), f"动作{aid}"),
            q_value=float(q_all[aid]),
            rank=rank_idx + 1,
        ))

    id_to_score: Dict[int, ActionScore] = {s.action_id: s for s in scores}

    if chosen_action not in id_to_score:
        logger.warning(
            "chosen_action=%d 不在 Q 网络的动作范围 [0, %d)，将使用最近有效动作",
            chosen_action, n_actions,
        )
        chosen_action = int(np.clip(chosen_action, 0, n_actions - 1))

    chosen_score = id_to_score[chosen_action]
    best_score = scores[0]  # rank=1

    alternatives = [s for s in scores if s.action_id != chosen_action][:top_k_alternatives]

    return ActionComparison(
        chosen=chosen_score,
        best=best_score,
        margin=float(best_score.q_value - chosen_score.q_value),
        is_optimal=chosen_action == best_score.action_id,
        all_actions=scores,
        alternatives=alternatives,
    )
