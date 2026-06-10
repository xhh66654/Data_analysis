"""
数据采集：从 InferenceRecord 中提取 (观测向量, 动作, 奖励) 训练样本。

================================================================================
说明（小白友好版）：
================================================================================

这个文件做的事：
    从 InferenceRecord 的每步决策记录里，
    按指定智能体提取展平的观测特征向量和动作标签，
    整理成 numpy 矩阵格式供后续决策树训练使用。

观测展平规则：
    新版观测是嵌套字典，例如：
        "自身状态": {"血量": 0.8, "速度_马赫": 1.2, "高度_km": 8.0}
        "敌机距离": {"水平距离_km": 40.0, "高度差_km": 0.5}
        "敌机状态": {"存活": 1, "锁定中": 0, "威胁等级": 0.4}

    展平后（按 observation_space 顺序，子字段字母序）：
        → [0.8, 8.0, 1.2,  40.0, 0.5,  0,  1, 0.4]
    对应特征名（复合键格式：观测项.子字段）：
        → ["自身状态.血量", "自身状态.高度_km", "自身状态.速度_马赫",
           "敌机距离.水平距离_km", "敌机距离.高度差_km",
           "敌机状态.存活", "敌机状态.威胁等级", "敌机状态.锁定中"]

动作标签（默认：整体决策）：
    action_item=None 时，一步的完整 decision_content 视为**一个整体动作类** y
    （holistic_decision_label），学习「状态 → 整体决策」映射。
    action_item 指定时，仅取该维度（单动作项子树，可选）。

训练范围：
    始终按 agent_id 分开采集；不同智能体 schema/策略分别训练，不混样本。
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from src.module_c_counterfactual.agent_schema import (
    action_label_from_content,
    validate_step_decision,
)
from src.module_c_counterfactual.inference_record import InferenceRecord


def collect_from_record(
    record: InferenceRecord,
    agent_id: int,
    action_item: Optional[str] = None,
    unit_id: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    从单条 InferenceRecord 中提取单个智能体的训练样本。

    Parameters
    ----------
    record      : 推理数据记录（从数据库加载）
    agent_id    : 要提取哪个智能体的数据
    action_item : None=整体决策（一步完整 decision 作为一个类）；否则为单动作项子树。
    unit_id     : 多装备个体时与 action_item 联用，解释单个体单维度。

    Returns
    -------
    X            : (T, n_features) 展平后的观测特征矩阵
                   T = 有效时间步数，n_features = 展平后特征总数
    y            : (T,) 动作标签数组。
                   action_item != None 时：["追击", "规避", ...]
                   action_item is None 时：整体决策 JSON 字符串，每类一种完整决策
    rewards      : (T,) 每步奖励值
    feature_names: 与 X 列对应的特征名列表，e.g. ["自身状态.血量", ...]

    使用示例：
        X, y, rewards, feat_names = collect_from_record(record, agent_id=1, action_item="机动控制")
        print(X.shape)      # (12, 8)  → 12步，8个展平特征
        print(feat_names)   # ["自身状态.血量", "自身状态.高度_km", ...]
        print(y[:3])        # ["追击", "追击", "规避"]
    """
    # 用列表暂存每步的特征向量和标签，最后统一转成 numpy 数组
    X_list: List[List[float]] = []   # 观测特征行列表，每行对应一个时间步
    y_list: List = []                # 动作标签列表，与 X_list 等长
    feature_names: Optional[List[str]] = None  # 特征名，只记录一次（每步结构一致）
    if agent_id not in record.agent_ids:
        raise ValueError(
            f"agent_id={agent_id} 不在记录 {record.sim_id} 中，"
            f"可用: {sorted(record.agent_ids)}"
        )
    schema = record.get_agent_schema(agent_id)

    # 遍历所有时间步，提取有效样本
    for t in range(record.total_steps):
        obs_vec = record.get_obs_vector(t, agent_id)
        if not obs_vec:
            continue

        dec = record.get_decision_at(t, agent_id)
        if dec is None:
            continue
        if action_item is None:
            try:
                validate_step_decision(dec.content, schema)
            except ValueError:
                continue
        action_val: Optional[Any] = action_label_from_content(
            dec.content,
            schema,
            action_item=action_item,
            unit_id=unit_id,
        )
        if action_val is None:
            continue

        X_list.append(obs_vec)
        y_list.append(action_val)

        if feature_names is None:
            feature_names = record.get_flat_feature_names(agent_id)

    # 转成 numpy 数组格式（决策树训练需要 numpy 矩阵）
    X = np.array(X_list, dtype=float)           # shape: (T, n_features)
    y = np.array(y_list)                         # shape: (T,)
    # 奖励只取前 T 步（与有效样本数对齐）
    rewards = np.array(record.rewards[:len(y_list)], dtype=float)
    feature_names = feature_names or []          # 如果一步都没有，返回空列表

    return X, y, rewards, feature_names


def collect_from_records(
    records: List[InferenceRecord],
    agent_id: int,
    action_item: Optional[str] = None,
    unit_id: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    从多条 InferenceRecord 中合并提取单个智能体的训练样本。

    使用场景：
        有多个任务（多次仿真）的数据，想合并在一起训练更健壮的决策树，
        让规则覆盖更多场景。

    Parameters
    ----------
    records     : 多条推理数据记录（必须来自同一类装备，否则特征结构不匹配）
    agent_id    : 要提取哪个智能体的数据
    action_item : 为 None 时使用整体决策标签（一步一整体动作类）

    Returns
    -------
    X, y, rewards, feature_names（所有记录的数据上下拼接后的结果）
    """
    X, y, rewards, feature_names, _ = collect_from_records_with_segments(
        records, agent_id=agent_id, action_item=action_item, unit_id=unit_id
    )
    return X, y, rewards, feature_names


def collect_from_records_with_segments(
    records: List[InferenceRecord],
    agent_id: int,
    action_item: Optional[str] = None,
    unit_id: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[int]]:
    """
    从多条记录提取样本，并返回每条记录贡献的样本数（用于分段权重计算）。

    参数:
        records: 推理数据记录列表。
        agent_id: 目标智能体 ID。
        action_item: 动作项名称；``None`` 表示整体决策标签。
        unit_id: 多装备个体时与 ``action_item`` 联用的个体 ID。

    返回:
        五元组 ``(X, y, rewards, feature_names, segment_lengths)``，
        其中 ``segment_lengths`` 为每条记录的有效样本数列表。
    """
    from src.module_c_counterfactual.agent_schema import assert_same_agent_schema

    if records:
        assert_same_agent_schema(records, agent_id)

    # 分别收集每条记录的数据，最后拼接
    X_all, y_all, r_all = [], [], []
    feature_names: List[str] = []
    segment_lengths: List[int] = []

    for rec in records:
        X, y, rewards, fn = collect_from_record(
            rec, agent_id, action_item, unit_id=unit_id
        )
        if len(X) == 0:
            # 该记录没有有效样本，跳过
            continue
        X_all.append(X)
        y_all.append(y)
        r_all.append(rewards)
        segment_lengths.append(int(len(y)))
        if not feature_names:
            # 只记录一次特征名（所有同类记录特征名应相同）
            feature_names = fn

    # 如果所有记录都没有有效样本，返回空数组
    if not X_all:
        return np.array([]), np.array([]), np.array([]), feature_names, segment_lengths

    # np.concatenate：沿第0轴（行方向）把多个数组拼接成一个
    return (
        np.concatenate(X_all, axis=0),   # 行拼接：(T1+T2+..., n_features)
        np.concatenate(y_all, axis=0),   # 行拼接：(T1+T2+...,)
        np.concatenate(r_all, axis=0),   # 行拼接：(T1+T2+...,)
        feature_names,
        segment_lengths,
    )


def collect_multi_agent(
    record: InferenceRecord,
    action_item: Optional[str] = None,
    agent_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    从单条 InferenceRecord 中提取多个智能体的训练样本并合并。

    适用于多个智能体行为相似，希望用同一棵决策树描述所有智能体策略的场景。
    例如：一个编队中有 3 架飞机，它们的策略相近，可以合并训练一棵共用决策树。

    Parameters
    ----------
    record      : 推理数据记录
    action_item : 目标动作项名称
    agent_ids   : 要提取的智能体 id 列表；为 None 时提取所有智能体

    返回:
        四元组 ``(X, y, rewards, feature_names)``，多智能体样本沿行方向拼接。
    """
    if agent_ids is None:
        # 未指定时，取记录中所有智能体的 id
        agent_ids = record.agent_ids

    X_all, y_all, r_all = [], [], []
    feature_names: List[str] = []

    for aid in agent_ids:
        # 逐个智能体提取数据
        X, y, rewards, fn = collect_from_record(record, aid, action_item)
        if len(X) == 0:
            continue
        X_all.append(X)
        y_all.append(y)
        r_all.append(rewards)
        if not feature_names:
            feature_names = fn

    if not X_all:
        return np.array([]), np.array([]), np.array([]), feature_names

    return (
        np.concatenate(X_all, axis=0),
        np.concatenate(y_all, axis=0),
        np.concatenate(r_all, axis=0),
        feature_names,
    )


def compute_return_to_go(rewards: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    计算每步的 return-to-go（从当前步到末尾的折扣累计奖励）。

    ── 为什么要用 return-to-go 作为训练权重？ ──
        普通训练把每步样本同等对待。
        但在强化学习中，带来高收益的时间步（后续奖励高）才是"关键决策"。
        用 return-to-go 作为权重，让决策树更关注那些"关键时刻"的决策，
        而不是那些"无关紧要"的步骤。

    ── 计算公式 ──
        return-to-go[t] = r[t] + γ·r[t+1] + γ²·r[t+2] + ...
        （从第 t 步开始到最后一步的折扣累计奖励）

    ── 示例 ──
        rewards = [1, 2, 3, 4],  gamma=1.0
        return_to_go = [10, 9, 7, 4]   （10=1+2+3+4，9=2+3+4，7=3+4，4=4）
        归一化后  = [1.0, 0.833, 0.5, 0.0] + 1e-6

    Parameters
    ----------
    rewards : 每步奖励值数组，shape (T,)
    gamma   : 折扣因子，默认 1.0（不折扣，简单累加）
              设为 0.95 则越远的步奖励折扣越多，更关注近期收益

    Returns
    -------
    weights : 归一化到 (0, 1] 的 return-to-go 权重，shape (T,)
              加了 1e-6 的小偏移，确保所有权重 > 0（sklearn 要求权重非零）
    """
    T = len(rewards)
    rtg = np.zeros(T, dtype=float)   # 初始化 return-to-go 数组

    # 从后往前累加：rtg[t] = rewards[t] + gamma * rtg[t+1]
    for t in reversed(range(T)):
        next_val = rtg[t + 1] if t + 1 < T else 0.0   # 最后一步之后没有奖励
        rtg[t] = rewards[t] + gamma * next_val

    # ---- min-max 归一化到 [0, 1] ----
    # 让最大 return-to-go 的步骤权重=1，最小的=0，其余按比例缩放
    r_min, r_max = rtg.min(), rtg.max()
    if r_max > r_min:
        # 正常情况：有差异，按比例缩放
        rtg = (rtg - r_min) / (r_max - r_min)
    # else: 所有奖励一样，归一化后全0，后面加 1e-6 变成全 1e-6（相当于均等权重）

    # 加小偏移：防止零权重（sklearn 会忽略权重=0 的样本，导致关键样本被丢弃）
    rtg = rtg + 1e-6
    return rtg
