"""
溯因查询对象。

例如："为什么 ego 在 t=42 时选择了攻击 Red2 而不是 Red1？"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class AbductiveQuery:
    """
    溯因查询数据类。

    描述「为何某智能体在某时刻采取了特定行为」的结构化查询对象。

    参数:
        query_id: 查询唯一标识（字符串，便于日志追踪）。
        t_query: 被解释行为发生的时间步索引。
        agent_id: 被解释的智能体编号。
        observed_action: 该时刻实际发生的离散动作编码。
        predicate: 给定完整轨迹，判断查询事件是否成立的谓词函数。
        description: 查询的自然语言描述（可选）。
    """
    query_id: str
    t_query: int
    agent_id: int
    observed_action: int
    predicate: Callable[[Any], bool]
    description: str = ""
