"""
模块 C：基于观测空间的反事实推理。

已启用：
    - local / one_step / multi_step 三档反事实（单特征扰动）
    - K 次代理采样 + 表 2 因果效应量（标量奖励，one_step/multi_step 默认）
    - CEMA 薄封装编排器
    - counterfactual_service 统一入口

数据约束：推理记录仅含每步标量 rewards[t]；向量奖励待上游支持。
"""
from .data_loader import (
    load_inference_record,
    load_inference_records,
    list_inference_task_ids,
    list_available_tasks,
)
from .inference_record import (
    InferenceRecord,
    AgentMeta,
    ActionItem,
    AgentDecision,
    StepDecision,
    AgentObservation,
    StepObservation,
)

try:
    from .rollback import ObservationRollback
except Exception:
    ObservationRollback = None  # type: ignore

try:
    from .policy_model import PolicySurrogate
except Exception:
    PolicySurrogate = None  # type: ignore

try:
    from .counterfactual import (
        CFContext,
        LocalCFResult,
        OneStepCFResult,
        MultiStepCFResult,
        perturb_obs_features,
        local_counterfactual,
        one_step_counterfactual,
        multi_step_counterfactual,
        prefilter_features_for_cf,
    )
except Exception:
    CFContext = LocalCFResult = OneStepCFResult = MultiStepCFResult = None  # type: ignore
    perturb_obs_features = local_counterfactual = one_step_counterfactual = None  # type: ignore
    multi_step_counterfactual = prefilter_features_for_cf = None  # type: ignore

try:
    from .surrogate_bundle import SurrogateBundle
    from .surrogate_cache import (
        clear_surrogate_bundle_cache,
        get_or_fit_surrogate_bundle,
        get_or_fit_policy_surrogate,
    )
except Exception:
    SurrogateBundle = None  # type: ignore
    clear_surrogate_bundle_cache = get_or_fit_surrogate_bundle = None  # type: ignore
    get_or_fit_policy_surrogate = None  # type: ignore

try:
    from .cf_dataset import CFSample, generate_cf_dataset
    from .causal_effect import CausalFactor, mechanistic_effect, teleological_effect
except Exception:
    CFSample = generate_cf_dataset = None  # type: ignore
    CausalFactor = mechanistic_effect = teleological_effect = None  # type: ignore

try:
    from .cema import CEMA, CEMAResult
except Exception:
    CEMA = CEMAResult = None  # type: ignore

try:
    from .explain_nl import (
        render_cf_explanation,
        render_one_step_explanation,
        render_multi_step_explanation,
        render_k_sampling_explanation,
        CF_SURROGATE_DISCLAIMER,
        CF_K_SAMPLING_DISCLAIMER,
    )
except Exception:
    render_cf_explanation = render_one_step_explanation = render_multi_step_explanation = None  # type: ignore
    render_k_sampling_explanation = None  # type: ignore
    CF_SURROGATE_DISCLAIMER = CF_K_SAMPLING_DISCLAIMER = ""  # type: ignore

__all__ = [
    "load_inference_record",
    "load_inference_records",
    "list_inference_task_ids",
    "list_available_tasks",
    "InferenceRecord",
    "AgentMeta",
    "ActionItem",
    "AgentDecision",
    "StepDecision",
    "AgentObservation",
    "StepObservation",
    "ObservationRollback",
    "PolicySurrogate",
    "CFContext",
    "LocalCFResult",
    "OneStepCFResult",
    "MultiStepCFResult",
    "perturb_obs_features",
    "local_counterfactual",
    "one_step_counterfactual",
    "multi_step_counterfactual",
    "prefilter_features_for_cf",
    "SurrogateBundle",
    "get_or_fit_surrogate_bundle",
    "get_or_fit_policy_surrogate",
    "clear_surrogate_bundle_cache",
    "CFSample",
    "generate_cf_dataset",
    "CausalFactor",
    "mechanistic_effect",
    "teleological_effect",
    "CEMA",
    "CEMAResult",
    "render_cf_explanation",
    "render_one_step_explanation",
    "render_multi_step_explanation",
    "render_k_sampling_explanation",
    "CF_SURROGATE_DISCLAIMER",
    "CF_K_SAMPLING_DISCLAIMER",
]
