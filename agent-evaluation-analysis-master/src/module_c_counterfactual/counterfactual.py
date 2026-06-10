"""
反事实推理核心模块（当前阶段：局部反事实）。

================================================================================
什么是反事实推理？（小白友好版）
================================================================================

反事实推理就是问"如果当时不一样，结果会怎样"这类问题。

举个生活中的例子：
    真实情况：今天带了雨伞，没有淋雨。
    反事实问：如果我当时没带雨伞，我会不会淋雨？
    → 通过这个对比，我们可以解释"带雨伞"是"没淋雨"的原因。

在智能体的决策解释中：
    真实情况：智能体1在 t=2 时刻，看到敌机距离=40km，选择了"发射导弹"。
    反事实问：如果当时敌机距离是 85km，智能体还会选择发射导弹吗？
    → 把敌机距离改为 85km，送给决策树，看预测出什么动作。
    → 如果预测动作变成了"开启雷达"，说明"敌机距离"就是影响这次决策的关键原因。

================================================================================
局部反事实推理的完整流程（5步）：
================================================================================

步骤1：[选择要解释的决策]
        前端用户选择：(推理任务id, 智能体id, 动作, 具体决策内容)
        → data_loader 根据 task_id 从 Doris 数据库加载 InferenceRecord
        → ObservationRollback 根据 (agent_id, action, decision_content) 定位 t_query

步骤2：[回溯到决策之前的状态]
        从 InferenceRecord 中取出 t_query 时刻该智能体的真实观测向量
        → 得到 obs_t = [自身血量=1.0, 敌机距离=40.0, 敌机状态=1.0, 自身速度=1.3, 导弹剩余=4.0]
        → 得到 action_t = "发射导弹"
        这两个值组成 CFContext（反事实上下文）

步骤3：[选择候选原因]
        候选原因就是观测空间中的特征，比如 ["自身血量", "敌机距离", ...]
        每次只修改一个特征，逐一检验

步骤4：[修改状态]
        对每个候选特征：把该特征改成"反事实值"（当前实现：置零或替换为均值）
        → 比如把"敌机距离"从 40.0 改成 0.0（置零策略）
        → 得到反事实观测 cf_obs

步骤5：[用决策树预测反事实动作，比较结果]
        把 cf_obs 送入决策树 → 预测反事实动作 cf_action
        比较 cf_action 和真实 action_t：
            如果不一样 → action_changed=True → 该特征是影响这次决策的关键原因（强解释因子）
            如果一样   → action_changed=False → 该特征对这次决策影响不大

最终输出每个候选特征对应的 LocalCFResult，列出哪些特征是强解释因子。

================================================================================
已实现层次：
================================================================================
    - 一步反事实（OneStepCFResult）：单特征扰动 + π/T/R 推 1 步
    - 多步反事实（MultiStepCFResult）：单特征扰动 + π/T/R 推 3～5 步（可配置 horizon）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os

import numpy as np

from src.module_c_counterfactual.inference_record import InferenceRecord
from src.module_c_counterfactual.policy_model import PolicySurrogate
from src.module_c_counterfactual.surrogate_bundle import SurrogateBundle


# ==============================================================================
# 数据结构
# ==============================================================================

@dataclass
class CFContext:
    """
    反事实推理的上下文。

    描述"我们要解释哪一步决策"，是所有反事实推理函数的输入。

    Attributes
    ----------
    record    : 完整的推理数据记录（包含所有步骤的观测、动作、奖励）
    agent_id  : 被解释的智能体编号，e.g. 1
    t_query   : 被解释的时间步索引（0-based），e.g. 2 表示第3步
    obs_t     : t_query 时刻该智能体的真实观测特征向量
                e.g. [1.0, 40.0, 1.0, 1.3, 4.0]（对应 observation_space 的顺序）
    action_t  : t_query 时刻该智能体的真实动作
                e.g. "发射导弹"
    """
    record: InferenceRecord
    agent_id: int
    t_query: int
    obs_t: List[Any]
    action_t: Any


@dataclass
class LocalCFResult:
    """
    局部反事实推理的单条结果。

    对一个候选特征的检验结果：
        如果修改该特征后，决策树预测的动作发生了变化（action_changed=True），
        说明该特征是影响本次决策的关键原因（强解释因子）。

    Attributes
    ----------
    candidate_feature : 被检验的候选特征名，e.g. "敌机距离"
    original_action   : 真实世界中的动作，e.g. "发射导弹"
    cf_action         : 修改该特征后决策树预测的动作，e.g. "开启雷达"
    action_changed    : 动作是否发生变化（True=强解释因子，False=影响不大）
    cf_obs            : 修改后的反事实观测向量（便于调试和进一步分析）
    """
    candidate_feature: str
    original_action: Any
    cf_action: Any
    action_changed: bool
    # 变化幅度分数：当前离散动作采用 0/1（变化=1.0，不变=0.0）
    change_score: float
    change_score_mode: str
    cf_obs: List[Any]


# ==============================================================================
# 步骤4：特征扰动——生成反事实观测
# ==============================================================================

def perturb_obs_features(
    obs_vector: List[Any],
    feature_names: List[str],
    candidate_features: List[str],
    strategy: str = "zero",
    ref_value: Optional[List[Any]] = None,
    feature_to_idx: Optional[Dict[str, int]] = None,
) -> List[Any]:
    """
    对观测向量中的候选特征进行修改，生成反事实观测向量。

    这就是"反事实干预"的操作：假设某个特征的值不一样，其他特征保持不变。

    Parameters
    ----------
    obs_vector        : 原始观测特征向量，e.g. [1.0, 40.0, 1.0, 1.3, 4.0]
    feature_names     : 特征名列表（与 obs_vector 一一对应），e.g. ["自身血量", "敌机距离", ...]
    candidate_features: 要修改的特征名列表，e.g. ["敌机距离"]
    strategy          : 修改策略（决定把特征改成什么值）：
                        "zero"        → 把候选特征值改成 0
                        "replace"     → 用 ref_value 中对应位置的值替换（需传 ref_value）
                        "mean"        → 用 ref_value 中对应位置值的均值替换（需传 ref_value）
                        "train_mean"  → 用训练集特征均值向量替换（ref_value 为逐维均值）
    ref_value         : strategy="replace" 或 "mean" 时需要提供的参考值向量

    Returns
    -------
    cf_obs : 修改后的反事实观测向量（与 obs_vector 等长，只有候选特征位置被修改）

    示例：
        obs = [1.0, 40.0, 1.0, 1.3, 4.0]
        feature_names = ["自身血量", "敌机距离", "敌机状态", "自身速度", "导弹剩余数量"]
        perturb_obs_features(obs, feature_names, ["敌机距离"], strategy="zero")
        → [1.0, 0.0, 1.0, 1.3, 4.0]   ← 只有"敌机距离"变成了0
    """
    cf_obs = list(obs_vector)   # 先复制一份，不改动原始数据

    for fname in candidate_features:
        # 找到该特征在向量中的位置索引
        if feature_to_idx is not None:
            idx = feature_to_idx.get(fname)
            if idx is None:
                continue
        else:
            if fname not in feature_names:
                continue  # 找不到就跳过（容错处理）
            idx = feature_names.index(fname)

        if strategy == "zero":
            # 置零策略：把这个特征的值改成 0
            cf_obs[idx] = 0.0

        elif strategy == "replace" and ref_value is not None:
            # 替换策略：用参考值替换
            cf_obs[idx] = ref_value[idx]

        elif strategy == "mean" and ref_value is not None:
            # 均值策略：如果参考值是列表（多个样本），取均值
            v = ref_value[idx]
            cf_obs[idx] = float(np.mean(v)) if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)) else float(v)

        elif strategy == "train_mean" and ref_value is not None:
            cf_obs[idx] = float(ref_value[idx])

    return cf_obs


# ==============================================================================
# 步骤5：局部反事实推理主函数
# ==============================================================================

def local_counterfactual(
    ctx: CFContext,
    policy_model: PolicySurrogate,
    candidate_features: Optional[List[str]] = None,
    perturb_strategy: str = "zero",
    ref_value: Optional[List[Any]] = None,
    change_score_mode: str = "action_change",
    prefilter_top_k: Optional[int] = None,
    bidirectional: bool = False,
) -> List[LocalCFResult]:
    """
    局部反事实推理：解释"为什么这一步会这么选择"。

    对每个候选特征逐一检验：
        1. 把该特征改为反事实值，其他特征保持不变 → 得到 cf_obs
        2. 把 cf_obs 送给决策树 → 预测反事实动作 cf_action
        3. 比较 cf_action 和真实 action_t：
           - 不一样 → 该特征是强解释因子（action_changed=True）
           - 一样   → 该特征影响不大（action_changed=False）

    Parameters
    ----------
    ctx               : 反事实推理上下文（由 ObservationRollback 构建）
    policy_model      : 已经训练好的策略近似决策树（由 PolicySurrogate.fit() 得到）
    candidate_features: 要检验的候选特征列表；为 None 时自动遍历所有观测特征
    perturb_strategy  : 特征扰动策略，默认置零
    change_score_mode : 变化评分模式：
                        - "action_change": 只看动作是否变化（0/1）
                        - "prob_delta_l1": 比较扰动前后类别概率分布的 L1 差值

    Returns
    -------
    results : 每个候选特征对应一条 LocalCFResult，按 action_changed=True 的排在前面

    典型输出示例：
        候选特征="敌机距离"，原始动作="发射导弹"，反事实动作="开启雷达"，action_changed=True
        候选特征="自身血量"，原始动作="发射导弹"，反事实动作="发射导弹"，action_changed=False
        候选特征="自身速度"，原始动作="发射导弹"，反事实动作="发射导弹"，action_changed=False
        → 解释结论：智能体选择"发射导弹"是因为"敌机距离"足够近。
    """
    # 如果没有指定候选特征，就遍历所有展平后的特征
    # 注意：必须用展平后的复合键（如 "自身状态.血量"），而非顶层键（如 "自身状态"），
    # 因为 obs_t 和 perturb_obs_features 都操作的是展平向量
    if candidate_features is None:
        candidate_features = ctx.record.get_flat_feature_names(ctx.agent_id)
    feature_names = ctx.record.get_flat_feature_names(ctx.agent_id)
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}

    if prefilter_top_k is None:
        prefilter_top_k = _multi_prefilter_topk_from_env()
    if prefilter_top_k and prefilter_top_k > 0:
        candidate_features = prefilter_features_for_cf(
            ctx=ctx,
            policy=policy_model,
            candidate_features=list(candidate_features),
            feature_names=feature_names,
            perturb_strategy=perturb_strategy,
            ref_value=ref_value,
            top_k=prefilter_top_k,
        )

    results: List[LocalCFResult] = []
    base_proba: Optional[Dict[Any, float]] = None
    if change_score_mode == "prob_delta_l1":
        base_proba = policy_model.predict_proba(ctx.obs_t)

    def _eval_perturb(cf_obs: List[Any]) -> tuple[Any, bool, float]:
        """评估单次扰动后的预测动作、是否变化及变化分数。"""
        cf_action = policy_model.predict(cf_obs)
        action_changed = cf_action != ctx.action_t
        if change_score_mode == "action_change":
            change_score = 1.0 if action_changed else 0.0
        elif change_score_mode == "prob_delta_l1":
            cf_proba = policy_model.predict_proba(cf_obs)
            change_score = _l1_prob_delta(base_proba or {}, cf_proba)
        else:
            raise ValueError(f"不支持的 change_score_mode: {change_score_mode}")
        return cf_action, action_changed, change_score

    for feat in candidate_features:
        best_obs = list(ctx.obs_t)
        best_action = ctx.action_t
        best_changed = False
        best_score = 0.0

        strategies = [perturb_strategy]
        if bidirectional:
            strategies = [perturb_strategy, "delta_up", "delta_down"]

        for strat in strategies:
            if strat in ("delta_up", "delta_down"):
                cf_obs = list(ctx.obs_t)
                idx = feature_to_idx.get(feat)
                if idx is None:
                    continue
                base_v = float(cf_obs[idx])
                delta = max(abs(base_v) * 0.5, 1.0)
                cf_obs[idx] = base_v + delta if strat == "delta_up" else base_v - delta
            else:
                cf_obs = perturb_obs_features(
                    obs_vector=ctx.obs_t,
                    feature_names=feature_names,
                    candidate_features=[feat],
                    strategy=strat,
                    ref_value=ref_value,
                    feature_to_idx=feature_to_idx,
                )
            cf_action, action_changed, change_score = _eval_perturb(cf_obs)
            if change_score > best_score or (change_score == best_score and action_changed and not best_changed):
                best_score = change_score
                best_changed = action_changed
                best_action = cf_action
                best_obs = cf_obs

        results.append(
            LocalCFResult(
                candidate_feature=feat,
                original_action=ctx.action_t,
                cf_action=best_action,
                action_changed=best_changed,
                change_score=best_score,
                change_score_mode=change_score_mode,
                cf_obs=best_obs,
            )
        )

    # 把“变化更大”的因子排在前面，再按是否改变动作稳定排序
    results.sort(key=lambda r: (-r.change_score, not r.action_changed))
    return results


def _l1_prob_delta(base_proba: Dict[Any, float], cf_proba: Dict[Any, float]) -> float:
    """
    计算两个类别概率分布的 L1 差值，范围 [0, 2]。
    值越大表示扰动对策略输出分布影响越大。
    """
    keys = set(base_proba.keys()) | set(cf_proba.keys())
    return float(sum(abs(base_proba.get(k, 0.0) - cf_proba.get(k, 0.0)) for k in keys))


# ==============================================================================
# 一步反事实（单特征扰动 + 向前推 1 步）
# ==============================================================================

@dataclass
class OneStepCFResult:
    """
    一步反事实里「检验某一个特征」的完整结果（小白版）。

    想象只改一个开关（例如敌机距离），看下一步会怎样：
    - original_* ：仿真里真实发生的情况
    - cf_*       ：改了这个特征后，用近似模型推出来的情况
    - reward_delta：反事实一步奖励 − 真实一步奖励（>0 表示改完后模型觉得更赚）
    """
    candidate_feature: str
    original_action: Any
    original_next_obs: List[float]
    original_reward: float
    cf_action: Any
    cf_next_obs: List[float]
    cf_reward: float
    action_changed: bool
    reward_delta: float
    cf_obs: List[Any]


def one_step_counterfactual(
    ctx: CFContext,
    bundle: SurrogateBundle,
    candidate_features: Optional[List[str]] = None,
    perturb_strategy: str = "train_mean",
    prefilter_top_k: Optional[int] = None,
) -> List[OneStepCFResult]:
    """
    一步反事实推理（对每个观测特征各做一次）。

    通俗流程（只改一个特征，其它不变）：
        1. 在决策时刻 t，把某个特征改成「典型值/零」→ 得到反事实观测 s'_t
        2. 策略模型 π 预测：若态势这样，会做什么动作 a'
        3. 转移模型 T 预测：下一时刻态势 s'_{t+1}
        4. 奖励模型 R 预测：这一步能赚多少分 r'
        5. 和仿真记录里的真实 r_t、真实动作对比

    用来回答：「如果当时这个因素不一样，这一步决策/收益会不会变？」

    需要 SurrogateBundle（π+T+R），比 local 多两个模型，比 multi_step 只往前看 1 步。
    """
    t = ctx.t_query
    record = ctx.record
    agent_id = ctx.agent_id
    feature_names = record.get_flat_feature_names(agent_id)

    if t >= record.total_steps - 1:
        return []

    factual_next = record.get_obs_vector(t + 1, agent_id) or []
    rewards = getattr(record, "rewards", [])
    factual_reward = float(rewards[t]) if t < len(rewards) else 0.0

    ref_value: Optional[List[Any]] = None
    if perturb_strategy == "train_mean":
        ref_value = bundle.obs_feature_means or None
        if not ref_value:
            perturb_strategy = "zero"

    if candidate_features is None:
        candidate_features = feature_names
    if prefilter_top_k is None:
        prefilter_top_k = _multi_prefilter_topk_from_env()
    if prefilter_top_k and prefilter_top_k > 0:
        candidate_features = prefilter_features_for_cf(
            ctx=ctx,
            policy=bundle.policy,
            candidate_features=list(candidate_features),
            feature_names=feature_names,
            perturb_strategy=perturb_strategy,
            ref_value=ref_value,
            top_k=prefilter_top_k,
        )
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}

    results: List[OneStepCFResult] = []
    for feat in candidate_features:
        cf_obs = perturb_obs_features(
            obs_vector=ctx.obs_t,
            feature_names=feature_names,
            candidate_features=[feat],
            strategy=perturb_strategy,
            ref_value=ref_value,
            feature_to_idx=feature_to_idx,
        )
        cf_action = bundle.policy.predict(cf_obs)
        cf_next = bundle.transition.predict(cf_obs, cf_action)
        cf_reward = bundle.reward.predict(cf_obs, cf_action, cf_next)

        action_changed = cf_action != ctx.action_t
        reward_delta = cf_reward - factual_reward

        results.append(
            OneStepCFResult(
                candidate_feature=feat,
                original_action=ctx.action_t,
                original_next_obs=[float(v) for v in factual_next],
                original_reward=factual_reward,
                cf_action=cf_action,
                cf_next_obs=cf_next,
                cf_reward=cf_reward,
                action_changed=action_changed,
                reward_delta=reward_delta,
                cf_obs=cf_obs,
            )
        )

    results.sort(key=lambda r: (-abs(r.reward_delta), not r.action_changed))
    return results


def _clamp_horizon(horizon: int) -> int:
    """
    把用户请求的滚动步数限制在 3～5 步之间。

    步数太少看不出走势，太多代理模型误差会累积，所以产品上限定为 5。
    """
    h = int(horizon)
    return max(3, min(5, h))


def _action_label_at(record: InferenceRecord, agent_id: int, t: int) -> Any:
    """
    读取仿真记录在时刻 t 的动作，转成统一的字符串标签。

    方便和策略模型输出的动作字符串做「是否相同」的比较。
    """
    d = record.get_decision_at(t, agent_id)
    if d is None:
        return ""
    return str(sorted(d.content.items()))


def _factual_horizon_metrics(
    record: InferenceRecord,
    agent_id: int,
    t_start: int,
    horizon: int,
) -> tuple[List[Any], float, List[float]]:
    """
    从仿真真值里截取「事实轨迹」指标（不跑代理模型）。

    返回三件事：
        - 随后 horizon 步的真实动作列表
        - 这几步真实奖励加起来（累计分）
        - 最后一帧的真实观测向量（末期态势）

    多步反事实里，反事实轨迹要和这套「事实基线」比高低。
    """
    actions: List[Any] = []
    cum_reward = 0.0
    steps = 0
    for i in range(horizon):
        t = t_start + i
        if t >= record.total_steps:
            break
        actions.append(_action_label_at(record, agent_id, t))
        if t < len(record.rewards):
            cum_reward += float(record.rewards[t])
        steps += 1
    t_final = min(t_start + max(steps - 1, 0), record.total_steps - 1)
    final_obs = record.get_obs_vector(t_final, agent_id) or []
    return actions, cum_reward, [float(v) for v in final_obs]


def _surrogate_rollout(
    bundle: SurrogateBundle,
    obs_start: List[Any],
    horizon: int,
) -> tuple[List[Any], float, List[float]]:
    """
    用三个近似模型「假装往后推演」horizon 步（反事实世界）。

    每一步固定套路：
        当前态势 obs → π 选动作 → T 算下一态势 → R 算这一步得分
    把 horizon 步的得分相加，得到反事实累计奖励。

    注意：这是 learned surrogate，不是重新跑仿真器。
    """
    obs = list(obs_start)
    actions: List[Any] = []
    cum_reward = 0.0
    for _ in range(horizon):
        action = bundle.policy.predict(obs)
        next_obs = bundle.transition.predict(obs, action)
        reward = bundle.reward.predict(obs, action, next_obs)
        actions.append(action)
        cum_reward += float(reward)
        obs = next_obs
    return actions, cum_reward, [float(v) for v in obs]


def _multi_prefilter_topk_from_env() -> int:
    """
    TODO(remove before release): 多步反事实预筛选特征数。
    0 表示不预筛选；默认 12。
    """
    raw = os.environ.get("ANALYSIS_CF_MULTI_PREFILTER_TOPK", "").strip()
    if not raw:
        return 12
    try:
        v = int(raw)
        return max(0, v)
    except Exception:
        return 12


def prefilter_features_for_cf(
    ctx: CFContext,
    policy: PolicySurrogate,
    candidate_features: List[str],
    feature_names: List[str],
    perturb_strategy: str,
    ref_value: Optional[List[Any]],
    top_k: int,
) -> List[str]:
    """
    对候选观测特征做轻量预筛选（策略概率 L1 差值）。

    供 local/one_step/multi_step 共用，减少高维特征全量检验开销。

    参数:
        ctx: 反事实推理上下文。
        policy: 已拟合的策略近似模型。
        candidate_features: 待筛选的候选特征名列表。
        feature_names: 完整展平特征名列表。
        perturb_strategy: 单特征扰动策略。
        ref_value: 扰动参考值（train_mean 等策略用）。
        top_k: 保留的特征数量上限。

    返回:
        按 L1 概率差降序选取的前 top_k 个特征名。
    """
    if top_k <= 0 or len(candidate_features) <= top_k:
        return candidate_features
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}
    base_proba = policy.predict_proba(ctx.obs_t)
    scored: List[tuple[str, float]] = []
    for feat in candidate_features:
        cf_obs = perturb_obs_features(
            obs_vector=ctx.obs_t,
            feature_names=feature_names,
            candidate_features=[feat],
            strategy=perturb_strategy,
            ref_value=ref_value,
            feature_to_idx=feature_to_idx,
        )
        cf_proba = policy.predict_proba(cf_obs)
        scored.append((feat, _l1_prob_delta(base_proba, cf_proba)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scored[:top_k]]


@dataclass
class MultiStepCFResult:
    """
    多步反事实里「检验某一个特征」的完整结果（小白版）。

    字段含义：
        candidate_feature          : 本轮被改动的那个观测特征名
        horizon                    : 实际向前滚了几步（3～5）
        original_action_seq        : 仿真记录里随后几步的真实动作
        original_cumulative_reward : 仿真记录里随后几步真实奖励之和
        original_final_obs         : 事实轨迹最后一帧观测
        cf_action_seq              : 扰动后代理模型滚出的动作序列
        cf_cumulative_reward       : 反事实轨迹累计奖励
        cf_final_obs               : 反事实轨迹最后一帧观测
        reward_delta               : 反事实累计 − 事实累计（解释「短期走势」）
        action_changed             : 扰动后「第一步」动作是否和真实不同
        cf_obs                     : 扰动后的起始观测（调试用）
    """
    candidate_feature: str
    horizon: int
    original_action_seq: List[Any]
    original_cumulative_reward: float
    original_final_obs: List[float]
    cf_action_seq: List[Any]
    cf_cumulative_reward: float
    cf_final_obs: List[float]
    reward_delta: float
    action_changed: bool
    cf_obs: List[Any]


def multi_step_counterfactual(
    ctx: CFContext,
    bundle: SurrogateBundle,
    *,
    horizon: int = 5,
    candidate_features: Optional[List[str]] = None,
    perturb_strategy: str = "train_mean",
    prefilter_top_k: Optional[int] = None,
) -> List[MultiStepCFResult]:
    """
    多步反事实推理（对每个观测特征各做一次，每次只改一个特征）。

    和一步反事实的差别（小白记忆）：
        - 一步：只看「改完之后下一步」赚多少分
        - 多步：改完之后连续往前「假装演」3～5 步，看累计赚多少分

    通俗流程：
        1. 事实基线：从仿真记录读出 t 起共 H 步的真实奖励总和
        2. 对每个特征 f：
           a. 只把 f 改成典型值 → 反事实起始态势 s'_t
           b. 用 π/T/R 连滚 H 步 → 得到反事实累计奖励
           c. 比较累计奖励差、第一步动作是否变化

    参数：
        horizon : 希望滚几步，会被限制在 3～5，且不能超过仿真剩余步数

    用来回答：「如果当时这个因素不一样，接下来一小段局势/收益走势会不会变？」
    """
    record = ctx.record
    agent_id = ctx.agent_id
    t = ctx.t_query
    feature_names = record.get_flat_feature_names(agent_id)

    horizon = _clamp_horizon(horizon)
    available = record.total_steps - t
    if available < 3:
        raise ValueError(
            f"t_query={t} 距仿真结束不足 3 步（剩余 {available} 步），无法进行多步反事实。"
        )
    effective_h = min(horizon, available)

    ref_value: Optional[List[Any]] = None
    if perturb_strategy == "train_mean":
        ref_value = bundle.obs_feature_means or None
        if not ref_value:
            perturb_strategy = "zero"

    if candidate_features is None:
        candidate_features = feature_names
    if prefilter_top_k is None:
        prefilter_top_k = _multi_prefilter_topk_from_env()
    candidate_features = prefilter_features_for_cf(
        ctx=ctx,
        policy=bundle.policy,
        candidate_features=list(candidate_features),
        feature_names=feature_names,
        perturb_strategy=perturb_strategy,
        ref_value=ref_value,
        top_k=prefilter_top_k,
    )
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}

    factual_actions, factual_cum, factual_final = _factual_horizon_metrics(
        record, agent_id, t, effective_h
    )

    results: List[MultiStepCFResult] = []
    for feat in candidate_features:
        cf_obs = perturb_obs_features(
            obs_vector=ctx.obs_t,
            feature_names=feature_names,
            candidate_features=[feat],
            strategy=perturb_strategy,
            ref_value=ref_value,
            feature_to_idx=feature_to_idx,
        )
        cf_actions, cf_cum, cf_final = _surrogate_rollout(bundle, cf_obs, effective_h)
        cf_first = cf_actions[0] if cf_actions else ""
        action_changed = bool(cf_first and cf_first != ctx.action_t)
        reward_delta = cf_cum - factual_cum

        results.append(
            MultiStepCFResult(
                candidate_feature=feat,
                horizon=effective_h,
                original_action_seq=list(factual_actions),
                original_cumulative_reward=factual_cum,
                original_final_obs=factual_final,
                cf_action_seq=cf_actions,
                cf_cumulative_reward=cf_cum,
                cf_final_obs=cf_final,
                reward_delta=reward_delta,
                action_changed=action_changed,
                cf_obs=cf_obs,
            )
        )

    results.sort(key=lambda r: (-abs(r.reward_delta), not r.action_changed))
    return results
