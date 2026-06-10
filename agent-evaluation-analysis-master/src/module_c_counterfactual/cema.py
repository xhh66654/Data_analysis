"""
CEMA 顶层编排器（基于观测空间的反事实推理版本）。

薄封装：训练 SurrogateBundle，调度 local / one_step / multi_step 或 K 采样路径。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from src.module_c_counterfactual.inference_record import InferenceRecord
from src.module_c_counterfactual.rollback import ObservationRollback
from src.module_c_counterfactual.surrogate_bundle import SurrogateBundle
from src.module_c_counterfactual.counterfactual import (
    CFContext,
    local_counterfactual,
    one_step_counterfactual,
    multi_step_counterfactual,
)
from src.module_c_counterfactual.cf_dataset import generate_cf_dataset
from src.module_c_counterfactual.causal_effect import mechanistic_effect, teleological_effect
from src.module_c_counterfactual.explain_nl import (
    render_cf_explanation,
    render_one_step_explanation,
    render_multi_step_explanation,
    render_k_sampling_explanation,
)

CFLevel = Literal["local", "one_step", "multi_step"]


@dataclass
class CEMAResult:
    """
    CEMA 反事实解释的统一输出结构。

    聚合机械性/目的性文本、反事实层级详情及扩展元数据。
    """

    query_agent_id: int
    t_query: int
    original_action: Any
    level: CFLevel
    detail: Any
    mechanistic_text: str = ""
    teleological_text: str = ""
    extra: Optional[Dict[str, Any]] = None


class CEMA:
    """基于观测空间的反事实效应模型（Counterfactual Effect-size Model for Agents）。"""

    def __init__(
        self,
        record: InferenceRecord,
        records_for_training: Optional[List[InferenceRecord]] = None,
        policy_max_depth: int = 5,
        policy_min_samples_leaf: int = 5,
        multi_step_horizon: int = 5,
        perturb_strategy: str = "train_mean",
    ) -> None:
        """
        初始化 CEMA 编排器。

        参数:
            record: 待解释的推理记录。
            records_for_training: 用于训练代理模型的记录列表；默认仅用 record。
            policy_max_depth: 策略决策树最大深度。
            policy_min_samples_leaf: 策略决策树叶节点最小样本数。
            multi_step_horizon: 多步反事实默认滚动步数。
            perturb_strategy: 特征扰动策略（如 train_mean、zero）。
        """
        self.record = record
        self.records_for_training = records_for_training or [record]
        self.policy_max_depth = policy_max_depth
        self.policy_min_samples_leaf = policy_min_samples_leaf
        self.multi_step_horizon = multi_step_horizon
        self.perturb_strategy = perturb_strategy
        self._rollback = ObservationRollback(record)
        self._bundle: Optional[SurrogateBundle] = None
        self._fitted_agent: Optional[int] = None

    def fit_models(self, agent_id: int) -> "CEMA":
        """
        为指定智能体训练 SurrogateBundle（π/T/R）。

        参数:
            agent_id: 目标智能体编号。

        返回:
            自身实例（支持链式调用）。
        """
        self._bundle = SurrogateBundle.fit(
            self.records_for_training,
            agent_id,
            policy_max_depth=self.policy_max_depth,
            policy_min_samples_leaf=self.policy_min_samples_leaf,
        )
        self._fitted_agent = agent_id
        return self

    def _require_bundle(self) -> SurrogateBundle:
        """
        获取已训练的 SurrogateBundle。

        返回:
            已拟合的 bundle。

        抛出:
            RuntimeError: 尚未调用 fit_models 时。
        """
        if self._bundle is None or self._fitted_agent is None:
            raise RuntimeError("请先调用 fit_models(agent_id)。")
        return self._bundle

    def explain_at(
        self,
        agent_id: int,
        t_query: int,
        level: CFLevel = "one_step",
        candidate_features: Optional[List[str]] = None,
        *,
        use_k_sampling: bool = False,
        k_samples: int = 100,
    ) -> CEMAResult:
        """
        在指定时间步执行反事实解释。

        参数:
            agent_id: 被解释智能体编号。
            t_query: 查询时间步（0-based）。
            level: 反事实层级（local / one_step / multi_step）。
            candidate_features: 候选扰动特征列表；None 表示全部特征。
            use_k_sampling: 是否使用 K 采样 + 表 2 效应量路径。
            k_samples: K 采样次数。

        返回:
            CEMAResult 解释结果。
        """
        ctx = self._rollback.build_context(agent_id, t_query)
        return self._run(
            ctx,
            level,
            candidate_features,
            use_k_sampling=use_k_sampling,
            k_samples=k_samples,
        )

    def explain_by_action(
        self,
        agent_id: int,
        action: Any,
        occurrence: int = 0,
        level: CFLevel = "one_step",
        candidate_features: Optional[List[str]] = None,
        *,
        decision_content: Optional[Dict[str, Any]] = None,
        use_k_sampling: bool = False,
        k_samples: int = 100,
    ) -> Optional[CEMAResult]:
        """
        按动作标签或决策内容定位并执行反事实解释。

        参数:
            agent_id: 被解释智能体编号。
            action: 动作标签字符串（与记录中 holistic 标签匹配）。
            occurrence: 同一动作第几次出现（0-based）。
            level: 反事实层级。
            candidate_features: 候选扰动特征列表。
            decision_content: 前端传入的完整决策 dict（优先于 action 定位）。
            use_k_sampling: 是否使用 K 采样路径。
            k_samples: K 采样次数。

        返回:
            CEMAResult；定位失败时返回 None。
        """
        if decision_content is not None:
            ctx = self._rollback.from_frontend_input(
                agent_id, decision_content, query_step=None
            )
        else:
            ctx = None
            snapshots = self.record.list_decision_snapshots(agent_id, limit=200)
            matches = [t for t, label in snapshots if str(label) == str(action)]
            if occurrence < len(matches):
                ctx = self._rollback.build_context(agent_id, matches[occurrence])
        if ctx is None:
            return None
        return self._run(
            ctx,
            level,
            candidate_features,
            use_k_sampling=use_k_sampling,
            k_samples=k_samples,
        )

    def _run(
        self,
        ctx: CFContext,
        level: CFLevel,
        candidate_features: Optional[List[str]],
        *,
        use_k_sampling: bool = False,
        k_samples: int = 100,
    ) -> CEMAResult:
        """
        根据反事实上下文执行指定层级的解释流程。

        参数:
            ctx: 反事实推理上下文。
            level: 反事实层级。
            candidate_features: 候选扰动特征列表。
            use_k_sampling: 是否走 K 采样 + 因果效应量路径。
            k_samples: K 采样次数。

        返回:
            CEMAResult 解释结果。
        """
        bundle = self._require_bundle()
        agent_id = ctx.agent_id
        flat_names = ctx.record.get_flat_feature_names(agent_id)

        if use_k_sampling and level in ("one_step", "multi_step"):
            reward_mode = "step" if level == "one_step" else "cumulative"
            horizon = 1 if level == "one_step" else self.multi_step_horizon
            samples = generate_cf_dataset(
                ctx,
                bundle,
                K=k_samples,
                horizon=horizon,
                reward_mode=reward_mode,
                seed=ctx.t_query,
            )
            mech = mechanistic_effect(samples, flat_names)
            tele = teleological_effect(samples)
            expl = render_k_sampling_explanation(
                mechanistic_factors=mech,
                teleological_factors=tele,
                action_t=ctx.action_t,
                k_meta={"K": k_samples, "horizon": horizon, "reward_mode": reward_mode},
            )
            return CEMAResult(
                query_agent_id=agent_id,
                t_query=ctx.t_query,
                original_action=ctx.action_t,
                level=level,
                detail=samples,
                mechanistic_text=expl["mechanistic"],
                teleological_text=expl["teleological"],
                extra=expl,
            )

        ref_value = bundle.obs_feature_means if self.perturb_strategy == "train_mean" else None
        if level == "local":
            detail = local_counterfactual(
                ctx,
                bundle.policy,
                candidate_features=candidate_features,
                perturb_strategy=self.perturb_strategy,
                ref_value=ref_value,
            )
            expl = render_cf_explanation(
                results=detail,
                obs_t=ctx.obs_t,
                feature_names=flat_names,
                action_t=ctx.action_t,
                top_k=5,
            )
            mech_text = expl["mechanistic"]
            tele_text = expl["teleological"]
        elif level == "one_step":
            detail = one_step_counterfactual(
                ctx,
                bundle,
                candidate_features=candidate_features,
                perturb_strategy=self.perturb_strategy,
            )
            rewards = getattr(ctx.record, "rewards", [])
            orig_r = float(rewards[ctx.t_query]) if ctx.t_query < len(rewards) else 0.0
            expl = render_one_step_explanation(
                results=detail,
                obs_t=ctx.obs_t,
                feature_names=flat_names,
                action_t=ctx.action_t,
                original_reward=orig_r,
                top_k=5,
                perturb_strategy=self.perturb_strategy,
            )
            mech_text = expl["mechanistic"]
            tele_text = expl["teleological"]
        else:
            detail = multi_step_counterfactual(
                ctx,
                bundle,
                horizon=self.multi_step_horizon,
                candidate_features=candidate_features,
                perturb_strategy=self.perturb_strategy,
            )
            expl = render_multi_step_explanation(
                results=detail,
                obs_t=ctx.obs_t,
                feature_names=flat_names,
                action_t=ctx.action_t,
                top_k=5,
                perturb_strategy=self.perturb_strategy,
            )
            mech_text = expl["mechanistic"]
            tele_text = expl["teleological"]

        return CEMAResult(
            query_agent_id=agent_id,
            t_query=ctx.t_query,
            original_action=ctx.action_t,
            level=level,
            detail=detail,
            mechanistic_text=mech_text,
            teleological_text=tele_text,
            extra=expl if isinstance(expl, dict) else None,
        )
