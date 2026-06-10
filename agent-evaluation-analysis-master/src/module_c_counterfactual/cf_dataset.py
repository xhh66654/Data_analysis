"""
反事实数据集生成（表 1）：在观测空间上对代理模型做 K 次采样。

不依赖环境重仿真；每步奖励仍为标量 reward_scalar。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np

from src.module_c_counterfactual.counterfactual import CFContext, _surrogate_rollout
from src.module_c_counterfactual.policy_model import PolicySurrogate
from src.module_c_counterfactual.surrogate_bundle import SurrogateBundle

RewardMode = Literal["step", "cumulative"]


@dataclass
class CFSample:
    """
    K 采样反事实数据集的单条样本（对应方案文档表 1）。

    属性:
        state_features: 扰动后的状态特征向量。
        reward_scalar: 标量奖励（单步或累计）。
        query_happened: 代理 rollout 首步动作是否与查询动作一致。
    """

    state_features: np.ndarray
    reward_scalar: float
    query_happened: bool


def sample_action_from_policy(policy: PolicySurrogate, obs: List[Any], rng: np.random.Generator) -> Any:
    """
    按策略模型类别概率分布随机采样动作。

    参数:
        policy: 已拟合的策略近似模型。
        obs: 当前观测向量。
        rng: numpy 随机数生成器。

    返回:
        采样得到的动作标签。
    """
    proba = policy.predict_proba(obs)
    if not proba:
        return policy.predict(obs)
    labels = list(proba.keys())
    weights = np.array([max(float(proba[k]), 0.0) for k in labels], dtype=float)
    s = weights.sum()
    if s <= 0:
        return policy.predict(obs)
    weights /= s
    idx = int(rng.choice(len(labels), p=weights))
    return labels[idx]


def _perturb_obs(
    obs: np.ndarray,
    rng: np.random.Generator,
    noise_scale: float,
) -> np.ndarray:
    """
    对观测向量各维施加相对幅度的高斯噪声。

    参数:
        obs: 原始观测数组。
        rng: 随机数生成器。
        noise_scale: 噪声相对幅度系数。

    返回:
        扰动后的观测副本。
    """
    out = obs.copy()
    for i in range(len(out)):
        scale = max(abs(float(out[i])), 1.0)
        out[i] = float(out[i]) + float(rng.normal(0.0, noise_scale * scale))
    return out


def generate_cf_dataset(
    ctx: CFContext,
    bundle: SurrogateBundle,
    *,
    K: int = 100,
    horizon: int = 1,
    reward_mode: RewardMode = "cumulative",
    noise_scale: float = 0.1,
    seed: Optional[int] = None,
) -> List[CFSample]:
    """
    在 t_query 的观测上扰动并代理 rollout，生成 K 条反事实样本。

    参数:
        ctx: 反事实推理上下文。
        bundle: 已拟合的 SurrogateBundle（π/T/R）。
        K: 采样次数（限制在 1～500）。
        horizon: cumulative 模式下 rollout 步数（one_step 用 1）。
        reward_mode: "step" 仅首步代理奖励；"cumulative" 为 H 步累计。
        noise_scale: 高斯扰动相对幅度。
        seed: 随机种子；None 表示非确定性。

    返回:
        CFSample 列表。
    """
    K = max(1, min(int(K), 500))
    rng = np.random.default_rng(seed)
    obs_base = np.asarray(ctx.obs_t, dtype=float)
    if obs_base.size == 0:
        return []

    h = max(1, int(horizon))
    if reward_mode == "cumulative":
        from src.module_c_counterfactual.counterfactual import _clamp_horizon

        h = _clamp_horizon(h) if h > 1 else h
        available = ctx.record.total_steps - ctx.t_query
        if available < 1:
            h = 1
        else:
            h = min(h, available)

    samples: List[CFSample] = []
    for _ in range(K):
        cf_obs_arr = _perturb_obs(obs_base, rng, noise_scale)
        cf_obs = [float(v) for v in cf_obs_arr]

        first_action = sample_action_from_policy(bundle.policy, cf_obs, rng)
        query_happened = bool(first_action == ctx.action_t)

        if reward_mode == "step":
            next_obs = bundle.transition.predict(cf_obs, first_action)
            reward_scalar = float(bundle.reward.predict(cf_obs, first_action, next_obs))
        else:
            _, cum_reward, _ = _surrogate_rollout(bundle, cf_obs, h)
            reward_scalar = float(cum_reward)

        samples.append(
            CFSample(
                state_features=cf_obs_arr,
                reward_scalar=reward_scalar,
                query_happened=query_happened,
            )
        )
    return samples
