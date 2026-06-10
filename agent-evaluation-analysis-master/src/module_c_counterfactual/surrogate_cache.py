"""
SurrogateBundle 进程内缓存 + 按 agent 磁盘 profile 增量。

查找链：memory → disk profile → fit。

环境变量：
    ANALYSIS_CF_BUNDLE_CACHE=0       关闭进程内缓存（默认开启）
    ANALYSIS_CF_SURROGATE_PROFILE=0  关闭磁盘 profile（默认开启）
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from src.module_c_counterfactual.agent_surrogate_profile import (
    AgentSurrogateProfile,
    bundle_joblib_path,
    extract_transition_rows,
    fingerprint_to_str,
    is_surrogate_profile_enabled,
    load_bundle,
    load_policy,
    load_profile,
    merge_reservoir,
    policy_joblib_path,
    profile_id_for_record,
    save_bundle,
    save_policy,
    save_profile,
    tree_params_match,
)
from src.module_a_rules.agent_profile import schema_fingerprint
from src.module_c_counterfactual.inference_record import InferenceRecord
from src.module_c_counterfactual.policy_model import PolicySurrogate
from src.module_c_counterfactual.surrogate_bundle import SurrogateBundle

# 进程内缓存：key -> SurrogateBundle / PolicySurrogate
_CACHE: Dict[Tuple, SurrogateBundle] = {}
_POLICY_CACHE: Dict[Tuple, PolicySurrogate] = {}


def is_surrogate_cache_enabled() -> bool:
    """
    是否启用 SurrogateBundle 进程内缓存。

    返回:
        环境变量 ANALYSIS_CF_BUNDLE_CACHE 非 0/false 时为 True。
    """
    flag = os.environ.get("ANALYSIS_CF_BUNDLE_CACHE", "1").strip().lower()
    return flag not in ("0", "false", "no", "off")


def records_fingerprint(records: List[InferenceRecord]) -> Tuple[Tuple[str, int], ...]:
    """
    用 (sim_id, total_steps) 列表标识训练数据版本。

    参数:
        records: 推理记录列表。

    返回:
        排序后的 (sim_id, total_steps) 元组。
    """
    return tuple(sorted((r.sim_id, int(r.total_steps)) for r in records))


def _policy_mode_from_env_for_cache() -> str:
    """读取环境变量中的策略模式（缓存键组成部分）。"""
    v = os.environ.get("ANALYSIS_CF_POLICY_MODE", "joint").strip().lower()
    if v in ("composed", "holistic_compose", "factorized", "compose"):
        return "composed"
    if v in ("per_item", "peritem", "item", "items"):
        return "per_item"
    if v in ("auto", "adaptive"):
        return "auto"
    if v in ("holistic", "whole", "full"):
        return "joint"
    return "joint"


def _transition_autotune_from_env_for_cache() -> bool:
    """读取 T 模型自动调参开关（缓存键组成部分）。"""
    v = os.environ.get("ANALYSIS_CF_T_AUTOTUNE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _transition_grouped_from_env_for_cache() -> bool:
    """读取 T 模型分维建模开关（缓存键组成部分）。"""
    v = os.environ.get("ANALYSIS_CF_T_GROUPED", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _strict_conservative_from_env_for_cache() -> bool:
    """读取严格保守模式开关（缓存键组成部分）。"""
    v = os.environ.get("ANALYSIS_STRICT_CONSERVATIVE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def build_tree_params(
    *,
    policy_max_depth: int,
    policy_min_samples_leaf: int,
) -> Dict[str, object]:
    """
    构造用于 profile 一致性比对的树训练参数字典。

    参数:
        policy_max_depth: 策略树最大深度。
        policy_min_samples_leaf: 策略树叶节点最小样本数。

    返回:
        含策略模式、预处理器、VIPER 等开关的参数字典。
    """
    pre_v = os.environ.get("ANALYSIS_CF_POLICY_PREPROCESS", "1").strip().lower()
    policy_preprocess = pre_v not in ("0", "false", "no", "off")
    viper_raw = os.environ.get("ANALYSIS_CF_POLICY_VIPER_ITERS", "2").strip()
    try:
        policy_viper_iters = max(0, int(viper_raw))
    except ValueError:
        policy_viper_iters = 2
    return {
        "policy_max_depth": int(policy_max_depth),
        "policy_min_samples_leaf": int(policy_min_samples_leaf),
        "policy_mode": _policy_mode_from_env_for_cache(),
        "policy_estimator": os.environ.get("ANALYSIS_CF_POLICY_ESTIMATOR", "tree"),
        "policy_preprocess": policy_preprocess,
        "policy_viper_iters": policy_viper_iters,
        "transition_autotune": _transition_autotune_from_env_for_cache(),
        "transition_grouped": _transition_grouped_from_env_for_cache(),
        "strict_conservative": _strict_conservative_from_env_for_cache(),
    }


def clear_surrogate_bundle_cache() -> int:
    """
    清空 SurrogateBundle 与 PolicySurrogate 的进程内缓存。

    返回:
        被清除的缓存条目总数。
    """
    n = len(_CACHE) + len(_POLICY_CACHE)
    _CACHE.clear()
    _POLICY_CACHE.clear()
    return n


def _persist_bundle_profile(
    records: List[InferenceRecord],
    agent_id: int,
    bundle: SurrogateBundle,
    tree_params: Dict[str, object],
    *,
    update_profile: bool,
    fp_str: str,
    existing: Optional[AgentSurrogateProfile],
) -> Tuple[bool, Optional[int]]:
    """
    将 SurrogateBundle 持久化到磁盘 profile。

    参数:
        records: 训练用推理记录列表。
        agent_id: 智能体编号。
        bundle: 拟合好的 bundle。
        tree_params: 树训练参数字典。
        update_profile: 是否写入磁盘。
        fp_str: 训练数据指纹 JSON 字符串。
        existing: 已有 profile（用于增量合并）。

    返回:
        (profile_hit, version) 元组。
    """
    if not update_profile or not is_surrogate_profile_enabled() or not records:
        return False, existing.version if existing else None

    record0 = records[0]
    pid = profile_id_for_record(agent_id, record0)
    new_rows = extract_transition_rows(records, agent_id)
    feat_names = bundle.feature_names or record0.get_flat_feature_names(agent_id)

    if existing is not None and tree_params_match(existing.tree_params, tree_params):
        reservoir = merge_reservoir(existing.reservoir, new_rows)
        seen = list(existing.seen_fingerprints)
        already_seen = fp_str in seen
        if not already_seen:
            seen.append(fp_str)
        version = existing.version if already_seen else existing.version + 1
        profile = AgentSurrogateProfile(
            profile_id=pid,
            agent_id=agent_id,
            schema_fingerprint=schema_fingerprint(record0, agent_id),
            feature_names=list(feat_names),
            policy_mode=str(tree_params["policy_mode"]),
            tree_params=dict(tree_params),
            n_transitions=len(reservoir),
            version=version,
            reservoir=reservoir,
            seen_fingerprints=seen,
        )
    else:
        reservoir = merge_reservoir([], new_rows)
        profile = AgentSurrogateProfile(
            profile_id=pid,
            agent_id=agent_id,
            schema_fingerprint=schema_fingerprint(record0, agent_id),
            feature_names=list(feat_names),
            policy_mode=str(tree_params["policy_mode"]),
            tree_params=dict(tree_params),
            n_transitions=len(reservoir),
            version=1,
            reservoir=reservoir,
            seen_fingerprints=[fp_str],
        )

    save_profile(profile)
    save_bundle(pid, bundle)
    return True, profile.version


def _persist_policy_profile(
    records: List[InferenceRecord],
    agent_id: int,
    policy: PolicySurrogate,
    tree_params: Dict[str, object],
    *,
    update_profile: bool,
    fp_str: str,
    existing: Optional[AgentSurrogateProfile],
) -> Tuple[bool, Optional[int]]:
    """
    将 PolicySurrogate 持久化到磁盘 profile。

    参数:
        records: 训练用推理记录列表。
        agent_id: 智能体编号。
        policy: 拟合好的策略模型。
        tree_params: 树训练参数字典。
        update_profile: 是否写入磁盘。
        fp_str: 训练数据指纹 JSON 字符串。
        existing: 已有 profile（用于增量合并）。

    返回:
        (profile_hit, version) 元组。
    """
    if not update_profile or not is_surrogate_profile_enabled() or not records:
        return False, existing.version if existing else None

    record0 = records[0]
    pid = profile_id_for_record(agent_id, record0)
    new_rows = extract_transition_rows(records, agent_id)
    feat_names = record0.get_flat_feature_names(agent_id)

    if existing is not None and tree_params_match(existing.tree_params, tree_params):
        reservoir = merge_reservoir(existing.reservoir, new_rows)
        seen = list(existing.seen_fingerprints)
        already_seen = fp_str in seen
        if not already_seen:
            seen.append(fp_str)
        version = existing.version if already_seen else existing.version + 1
        profile = AgentSurrogateProfile(
            profile_id=pid,
            agent_id=agent_id,
            schema_fingerprint=schema_fingerprint(record0, agent_id),
            feature_names=list(feat_names),
            policy_mode=str(tree_params["policy_mode"]),
            tree_params=dict(tree_params),
            n_transitions=len(reservoir),
            version=version,
            reservoir=reservoir,
            seen_fingerprints=seen,
        )
    else:
        reservoir = merge_reservoir([], new_rows)
        profile = AgentSurrogateProfile(
            profile_id=pid,
            agent_id=agent_id,
            schema_fingerprint=schema_fingerprint(record0, agent_id),
            feature_names=list(feat_names),
            policy_mode=str(tree_params["policy_mode"]),
            tree_params=dict(tree_params),
            n_transitions=len(reservoir),
            version=1,
            reservoir=reservoir,
            seen_fingerprints=[fp_str],
        )

    save_profile(profile)
    save_policy(pid, policy)
    return True, profile.version


def get_or_fit_policy_surrogate(
    records: List[InferenceRecord],
    agent_id: int,
    inference_task_id: str,
    *,
    policy_max_depth: int = 5,
    policy_min_samples_leaf: int = 5,
    use_cache: Optional[bool] = None,
    update_profile: bool = True,
) -> Tuple[PolicySurrogate, bool, bool, Optional[int]]:
    """
    获取或训练策略近似 π（local 反事实用）。

    查找链：memory → disk profile → fit。

    参数:
        records: 训练用推理记录列表。
        agent_id: 智能体编号。
        inference_task_id: 推理任务 id（缓存键）。
        policy_max_depth: 策略树最大深度。
        policy_min_samples_leaf: 策略树叶节点最小样本数。
        use_cache: 是否使用进程内缓存；None 时读环境变量。
        update_profile: 是否更新磁盘 profile。

    返回:
        (policy, cache_hit, profile_hit, profile_version) 四元组。
    """
    if use_cache is None:
        use_cache = is_surrogate_cache_enabled()

    fp = records_fingerprint(records)
    fp_str = fingerprint_to_str(fp)
    tree_params = build_tree_params(
        policy_max_depth=policy_max_depth,
        policy_min_samples_leaf=policy_min_samples_leaf,
    )

    key = (
        "policy_only",
        inference_task_id,
        int(agent_id),
        int(policy_max_depth),
        int(policy_min_samples_leaf),
        _policy_mode_from_env_for_cache(),
        fp,
    )

    if use_cache and key in _POLICY_CACHE:
        existing = (
            load_profile(profile_id_for_record(agent_id, records[0]))
            if records and is_surrogate_profile_enabled()
            else None
        )
        ver = existing.version if existing else None
        return _POLICY_CACHE[key], True, False, ver

    profile_hit = False
    profile_version: Optional[int] = None
    existing: Optional[AgentSurrogateProfile] = None
    pid = profile_id_for_record(agent_id, records[0]) if records else ""

    if is_surrogate_profile_enabled() and records:
        existing = load_profile(pid)
        if (
            existing is not None
            and tree_params_match(existing.tree_params, tree_params)
            and fp_str in existing.seen_fingerprints
            and policy_joblib_path(pid).is_file()
        ):
            loaded = load_policy(pid)
            if loaded is not None:
                if use_cache:
                    _POLICY_CACHE[key] = loaded
                return loaded, False, True, existing.version

    policy: PolicySurrogate
    if (
        existing is not None
        and tree_params_match(existing.tree_params, tree_params)
        and existing.reservoir
        and is_surrogate_profile_enabled()
    ):
        new_rows = extract_transition_rows(records, agent_id)
        merged = merge_reservoir(existing.reservoir, new_rows)
        policy = PolicySurrogate(
            max_depth=policy_max_depth,
            min_samples_leaf=policy_min_samples_leaf,
            mode=tree_params["policy_mode"],  # type: ignore[arg-type]
        )
        policy.fit_transition_rows(
            merged,
            feature_names=existing.feature_names,
            action_space=list(records[0].action_space),
        )
        profile_hit = True
    else:
        policy = PolicySurrogate(
            max_depth=policy_max_depth,
            min_samples_leaf=policy_min_samples_leaf,
        )
        policy.fit_records(records, agent_id)

    if use_cache:
        _POLICY_CACHE[key] = policy

    saved_hit, profile_version = _persist_policy_profile(
        records,
        agent_id,
        policy,
        tree_params,
        update_profile=update_profile,
        fp_str=fp_str,
        existing=existing if existing and tree_params_match(existing.tree_params, tree_params) else None,
    )
    if saved_hit:
        profile_hit = True

    return policy, False, profile_hit, profile_version


def get_or_fit_surrogate_bundle(
    records: List[InferenceRecord],
    agent_id: int,
    inference_task_id: str,
    *,
    policy_max_depth: int = 5,
    policy_min_samples_leaf: int = 5,
    use_cache: Optional[bool] = None,
    update_profile: bool = True,
) -> Tuple[SurrogateBundle, bool, bool, Optional[int]]:
    """
    获取或训练 SurrogateBundle（π/T/R 三模型）。

    查找链：memory → disk profile → fit。

    参数:
        records: 训练用推理记录列表。
        agent_id: 智能体编号。
        inference_task_id: 推理任务 id（缓存键）。
        policy_max_depth: 策略树最大深度。
        policy_min_samples_leaf: 策略树叶节点最小样本数。
        use_cache: 是否使用进程内缓存；None 时读环境变量。
        update_profile: 是否更新磁盘 profile。

    返回:
        (bundle, cache_hit, profile_hit, profile_version) 四元组。
    """
    if use_cache is None:
        use_cache = is_surrogate_cache_enabled()

    fp = records_fingerprint(records)
    fp_str = fingerprint_to_str(fp)
    tree_params = build_tree_params(
        policy_max_depth=policy_max_depth,
        policy_min_samples_leaf=policy_min_samples_leaf,
    )

    key = (
        inference_task_id,
        int(agent_id),
        int(policy_max_depth),
        int(policy_min_samples_leaf),
        _policy_mode_from_env_for_cache(),
        int(_transition_autotune_from_env_for_cache()),
        int(_transition_grouped_from_env_for_cache()),
        int(_strict_conservative_from_env_for_cache()),
        fp,
    )

    if use_cache and key in _CACHE:
        existing = (
            load_profile(profile_id_for_record(agent_id, records[0]))
            if records and is_surrogate_profile_enabled()
            else None
        )
        ver = existing.version if existing else None
        return _CACHE[key], True, False, ver

    profile_hit = False
    profile_version: Optional[int] = None
    existing: Optional[AgentSurrogateProfile] = None
    pid = profile_id_for_record(agent_id, records[0]) if records else ""

    if is_surrogate_profile_enabled() and records:
        existing = load_profile(pid)
        if (
            existing is not None
            and tree_params_match(existing.tree_params, tree_params)
            and fp_str in existing.seen_fingerprints
            and bundle_joblib_path(pid).is_file()
        ):
            loaded = load_bundle(pid)
            if loaded is not None:
                if use_cache:
                    _CACHE[key] = loaded
                return loaded, False, True, existing.version

    bundle: SurrogateBundle
    if (
        existing is not None
        and tree_params_match(existing.tree_params, tree_params)
        and existing.reservoir
        and is_surrogate_profile_enabled()
    ):
        new_rows = extract_transition_rows(records, agent_id)
        merged = merge_reservoir(existing.reservoir, new_rows)
        bundle = SurrogateBundle.fit_from_transition_rows(
            merged,
            agent_id,
            records[0],
            policy_max_depth=policy_max_depth,
            policy_min_samples_leaf=policy_min_samples_leaf,
        )
        profile_hit = True
    else:
        bundle = SurrogateBundle.fit(
            records,
            agent_id,
            policy_max_depth=policy_max_depth,
            policy_min_samples_leaf=policy_min_samples_leaf,
        )

    if use_cache:
        _CACHE[key] = bundle

    saved_hit, profile_version = _persist_bundle_profile(
        records,
        agent_id,
        bundle,
        tree_params,
        update_profile=update_profile,
        fp_str=fp_str,
        existing=existing if existing and tree_params_match(existing.tree_params, tree_params) else None,
    )
    if saved_hit:
        profile_hit = True

    return bundle, False, profile_hit, profile_version
