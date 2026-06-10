"""从多条 InferenceRecord 构造转移/奖励训练样本与扰动统计量。"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterator, List, Optional, Tuple

import numpy as np

from src.module_c_counterfactual.inference_record import InferenceRecord


def joint_action_label(decision_content: dict, record: InferenceRecord, agent_id: int) -> str:
    """
    将一步完整 decision_content 规范化为 holistic 动作标签字符串。

    参数:
        decision_content: 决策内容 dict。
        record: 推理记录（用于获取 schema）。
        agent_id: 智能体编号。

    返回:
        稳定排序的 JSON 格式整体动作标签。
    """
    from src.module_c_counterfactual.agent_schema import canonical_decision_label
    schema = record.get_agent_schema(agent_id)
    return canonical_decision_label(decision_content, schema)


def iter_transitions(
    records: List[InferenceRecord],
    agent_id: int,
) -> Iterator[Tuple[List[float], str, List[float], float]]:
    """
    遍历多局记录中的 (s_t, a_t, s_{t+1}, r_t) 转移样本。

    参数:
        records: 推理记录列表。
        agent_id: 目标智能体编号。

    生成:
        四元组 (obs_t, action_label, obs_t1, reward) 迭代器。
    """
    for record in records:
        rewards = getattr(record, "rewards", [])
        for t in range(record.total_steps - 1):
            obs_t = record.get_obs_vector(t, agent_id)
            obs_t1 = record.get_obs_vector(t + 1, agent_id)
            dec = record.get_decision_at(t, agent_id)
            if not obs_t or not obs_t1 or dec is None:
                continue
            r_t = float(rewards[t]) if t < len(rewards) else 0.0
            yield obs_t, joint_action_label(dec.content, record, agent_id), obs_t1, r_t


def compute_obs_feature_means(
    records: List[InferenceRecord],
    agent_id: int,
) -> Tuple[List[str], List[float]]:
    """
    计算该智能体在训练集上的逐特征均值（用于 train_mean 扰动）。

    参数:
        records: 推理记录列表。
        agent_id: 目标智能体编号。

    返回:
        (feature_names, means) 元组；无样本时 means 为空列表。
    """
    feature_names: List[str] = []
    sums: np.ndarray | None = None
    count = 0

    for record in records:
        names = record.get_flat_feature_names(agent_id)
        for t in range(record.total_steps):
            obs = record.get_obs_vector(t, agent_id)
            if not obs:
                continue
            vec = np.array(obs, dtype=float)
            if sums is None:
                feature_names = names
                sums = np.zeros(len(vec), dtype=float)
            if len(vec) != len(sums):
                continue
            sums += vec
            count += 1

    if sums is None or count == 0:
        return feature_names, []
    return feature_names, (sums / count).tolist()


def truncate_inference_record(record: InferenceRecord, max_steps: int) -> InferenceRecord:
    """
    截取推理记录的前 max_steps 步，用于小样本快速试训。

    参数:
        record: 原始推理记录。
        max_steps: 保留的最大步数。

    返回:
        截断后的新 InferenceRecord（原记录不变）。
    """
    n = max(1, min(int(max_steps), int(record.total_steps)))
    if n >= record.total_steps:
        return record
    decisions = [d for d in record.decisions if int(d.step) < n]
    observations = [o for o in record.observations if int(o.step) < n]
    return replace(
        record,
        decisions=decisions,
        observations=observations,
        rewards=list(record.rewards[:n]),
        total_steps=n,
    )


def subset_records_for_dev(
    records: List[InferenceRecord],
    *,
    max_sims: Optional[int] = None,
    max_steps_per_sim: Optional[int] = None,
) -> List[InferenceRecord]:
    """
    开发/调参用小样本：限制仿真局数与每局步数。

    示例：前 3 局、每局最多 80 步 → 数秒内可看完 holistic π 指标。
    """
    out = list(records)
    if max_sims is not None and max_sims > 0:
        out = out[: int(max_sims)]
    if max_steps_per_sim is not None and max_steps_per_sim > 0:
        out = [truncate_inference_record(r, max_steps_per_sim) for r in out]
    return out
