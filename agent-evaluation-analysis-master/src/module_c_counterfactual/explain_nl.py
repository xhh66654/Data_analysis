"""
把反事实推理结果渲染为自然语言解释。

================================================================================
两类解释的含义（小白友好版）：
================================================================================

机械性解释（mechanistic）：
    "是什么状态因素导致了这个决策？"
    → 回答"是什么原因造成了这一结果"
    → 关注：哪些观测特征如果变了，决策就会不同
    → 例：因为【敌机距离较近（40km）】和【自身血量充足（0.9）】，智能体选择了发射导弹。

目的性解释（teleological）：
    "这个决策是为了达到什么目的？"
    → 回答"做这件事是为了什么"
    → 关注：这些特征的值处于哪个区间，以及这次决策与什么目标/收益相关
    → 例：选择发射导弹，是因为当前已进入攻击窗口（距离极低、威胁等级高），
          目标是消灭敌机获取高额奖励。

================================================================================
三类渲染入口（对应三种 cf_level）：
================================================================================
    render_cf_explanation()        → local（只看决策变不变）
    render_one_step_explanation()  → one_step（看一步奖励）
    render_multi_step_explanation()→ multi_step（看 3～5 步累计奖励）
    attach_natural_language_qa()   → 任意层级结果上再生成 nl_explanation

================================================================================
数据来源（局部反事实推理结果）：
================================================================================

local_counterfactual() 返回 List[LocalCFResult]，每条包含：
    - candidate_feature : 被扰动的特征名（展平复合键，如 "敌机距离.水平距离_km"）
    - original_action   : 真实动作
    - cf_action         : 反事实动作（扰动后决策树预测结果）
    - action_changed    : 该特征扰动后动作是否改变（True = 强解释因子）
    - cf_obs            : 扰动后的反事实观测向量

机械性解释 = 列出 action_changed=True 的特征（改变它会改变决策 → 是决策原因）
目的性解释 = 解读这些特征当前处于什么状态区间 + 这个状态指向了什么目标
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.module_c_counterfactual.counterfactual import LocalCFResult
    from src.module_a_rules.preprocess import Preprocessor


# ==============================================================================
# 核心渲染函数
# ==============================================================================

def render_cf_explanation(
    results: List["LocalCFResult"],
    obs_t: List[float],
    feature_names: List[str],
    action_t: Any,
    preprocessor: Optional["Preprocessor"] = None,
    top_k: int = 5,
) -> dict:
    """
    把「局部反事实」的数值结果翻译成中文报告（只用决策树，不看奖励模型）。

    小白理解：逐个问「如果只改这一个观测因素，决策会不会变？」
    本函数把答案整理成两段话 + key_features 表格。

    Parameters
    ----------
    results       : local_counterfactual() 的返回值，List[LocalCFResult]
    obs_t         : 该时间步真实的展平观测向量（与 feature_names 一一对应）
    feature_names : 展平后的特征名列表，如 ["自身状态.血量", "敌机距离.水平距离_km"]
    action_t      : 真实动作标签（字符串形式）
    preprocessor  : 预处理器（用于把特征值翻译成语义标签，如 "低" / "高"）
                    为 None 时直接显示数值
    top_k         : 最多展示几个关键因素，默认 5

    Returns
    -------
    {
        "mechanistic": "机械性解释字符串",
        "teleological": "目的性解释字符串",
        "key_features": [{"feature": ..., "value": ..., "label": ..., "changed": True}, ...],
        "original_action": "真实动作",
    }

    示例输出：
    {
        "mechanistic": "该决策主要由以下状态因素决定：
                          【敌机距离.水平距离_km = 40.0（极低）】：修改此特征后动作从"发射导弹"变为"追击"
                          【自身状态.血量 = 0.9（高）】：修改此特征后动作发生变化",
        "teleological": "智能体执行此决策的目的性解读：
                          当前状态处于进攻窗口（敌机距离极低，血量充足），
                          该决策指向：消灭敌机目标（以下因素支持此推断：...）",
        "key_features": [...],
        "original_action": "[('机动控制', '追击'), ('武器控制', '发射导弹')]",
    }
    """
    # 将 feature_names 做成 {名称: 列索引} 的快速查找表
    feat_idx_map = {name: i for i, name in enumerate(feature_names)}

    # ---- 按变化分数排序，分数越大越靠前 ----
    ordered = sorted(
        results,
        key=lambda r: (-float(getattr(r, "change_score", 0.0)), not r.action_changed),
    )[:top_k]

    # ---- 构造结构化特征列表 ----
    key_features = []
    for r in ordered:
        feat = r.candidate_feature
        idx = feat_idx_map.get(feat)
        raw_value = float(obs_t[idx]) if idx is not None else None

        # 获取语义标签（如 "低" / "极高"）
        if preprocessor is not None and raw_value is not None:
            label = preprocessor.discretize_label(feat, raw_value)
        else:
            label = f"{raw_value:.3f}" if raw_value is not None else "未知"

        key_features.append({
            "feature": feat,
            "value": raw_value,
            "label": label,
            "changed": r.action_changed,
            "change_score": float(getattr(r, "change_score", 0.0)),
            "change_score_mode": str(getattr(r, "change_score_mode", "action_change")),
            "cf_action": r.cf_action,
        })

    # ---- 格式化真实动作（去掉 Python repr 冗余） ----
    action_display = _format_action(action_t)

    # ---- 机械性解释 ----
    mechanistic = _build_mechanistic(key_features, action_display)

    # ---- 目的性解释 ----
    teleological = _build_teleological(key_features, action_display, results)

    return {
        "mechanistic": mechanistic,
        "teleological": teleological,
        "key_features": key_features,
        "original_action": action_display,
        "disclaimer": CF_SURROGATE_DISCLAIMER,
    }


# ==============================================================================
# 内部渲染函数
# ==============================================================================

CF_SURROGATE_DISCLAIMER = (
    "以上为基于历史数据训练的代理模型（π/T/R）推断，"
    "每次仅扰动一个观测特征；使用每步标量即时奖励；不代表仿真器重放结果。"
)

CF_K_SAMPLING_DISCLAIMER = (
    "以上为基于代理模型的 K 次随机扰动采样与表 2 效应量统计，"
    "奖励为标量（单步或累计）；非环境重仿真。"
)


def render_k_sampling_explanation(
    *,
    mechanistic_factors: List[Any],
    teleological_factors: List[Any],
    action_t: Any,
    k_meta: Optional[dict] = None,
    top_k: int = 5,
) -> dict:
    """
    将 K 采样 + 表 2 因果效应量渲染为自然语言解释。

    参数:
        mechanistic_factors: 机械论 CausalFactor 列表。
        teleological_factors: 目的论 CausalFactor 列表。
        action_t: 查询时刻的真实动作标签。
        k_meta: K 采样元数据（K、horizon、reward_mode 等）。
        top_k: 机械性因素最多展示条数。

    返回:
        含 mechanistic、teleological、key_features 等的解释字典。
    """
    from src.module_c_counterfactual.causal_effect import CausalFactor

    action_display = _format_action(action_t)
    k_meta = k_meta or {}
    K = k_meta.get("K", "?")
    horizon = k_meta.get("horizon", "")
    reward_mode = k_meta.get("reward_mode", "")

    mech_lines = [
        f"【K 采样·机械性解释】决策（{action_display}）的状态因果因素（K={K}）：",
        f"  · 方法：在查询时刻对态势加噪，代理 rollout 后拟合「是否发生查询动作」",
    ]
    mech_factors = [f for f in mechanistic_factors if isinstance(f, CausalFactor)][:top_k]
    if not mech_factors:
        mech_lines.append("  · 未能从样本中稳定识别主导特征（样本方差不足）。")
    else:
        for f in mech_factors:
            mech_lines.append(f"  · 【{f.name}】重要性 ≈ {f.effect:.4f}（rank {f.rank}）")

    tele_lines = [
        f"【K 采样·目的性解释】决策（{action_display}）的标量收益差异：",
        f"  · horizon={horizon}，reward_mode={reward_mode}",
    ]
    tele_factors = [f for f in teleological_factors if isinstance(f, CausalFactor)]
    if tele_factors:
        tf = tele_factors[0]
        sign = "升高" if tf.effect > 0 else ("降低" if tf.effect < 0 else "无明显差异")
        tele_lines.append(
            f"  · 查询动作发生 vs 未发生：标量奖励均值差 Δr ≈ {tf.effect:+.4f}（{sign}）"
        )
    else:
        tele_lines.append("  · 样本中查询动作发生/未发生分组不足，无法估计 Δr。")

    key_features = [
        {
            "feature": f.name,
            "effect": float(f.effect),
            "rank": int(f.rank),
            "source": "k_sampling_mechanistic",
        }
        for f in mech_factors
    ]

    return {
        "mechanistic": "\n".join(mech_lines),
        "teleological": "\n".join(tele_lines),
        "key_features": key_features,
        "mechanistic_factors": [
            {"name": f.name, "effect": f.effect, "rank": f.rank} for f in mech_factors
        ],
        "teleological_factors": [
            {"name": f.name, "effect": f.effect, "rank": f.rank} for f in tele_factors
        ],
        "teleological_effect_scalar": tele_factors[0].effect if tele_factors else None,
        "k_sampling_meta": dict(k_meta),
        "disclaimer": CF_K_SAMPLING_DISCLAIMER,
        "original_action": action_display,
    }


def _format_action(action_t: Any) -> str:
    """
    把动作从「程序内部格式」转成给人看的短句。

    支持 holistic JSON 标签与旧 tuple 字符串格式。
    """
    from src.module_c_counterfactual.agent_schema import format_holistic_action_label

    return format_holistic_action_label(action_t)


def _format_action_sequence(action_seq: List[Any]) -> List[str]:
    """把多步动作序列逐条格式化为可读字符串列表。"""
    return [_format_action(a) for a in (action_seq or [])]


def _build_mechanistic(key_features: list, action_display: str) -> str:
    """
    构造「局部反事实·机械性解释」正文。

    核心逻辑：改了某个特征后决策变了 → 这个特征就是「原因」之一。
    """
    changed_feats = [f for f in key_features if f["changed"]]
    unchanged_feats = [f for f in key_features if not f["changed"]]

    lines = [f"【机械性解释】该决策（{action_display}）的状态原因分析：\n"]

    if not changed_feats:
        lines.append("  · 未发现能够单独改变本次决策的关键状态特征。")
        lines.append("  · 该决策可能由多个特征联合决定，或策略对单特征扰动不敏感。")
    else:
        lines.append("  以下状态特征对本次决策有决定性影响（修改后决策会改变）：")
        for f in changed_feats:
            cf_display = _format_action(f["cf_action"])
            lines.append(
                f"  · 【{f['feature']}】当前值 = {f['value']:.3f}（{f['label']}）"
                f"  → 若修改此特征，决策将变为：{cf_display}"
            )

    if unchanged_feats:
        lines.append("\n  以下状态特征对本次决策影响较小（修改后决策不变）：")
        for f in unchanged_feats:
            lines.append(
                f"  · 【{f['feature']}】当前值 = {f['value']:.3f}（{f['label']}）"
            )

    return "\n".join(lines)


def _build_teleological(
    key_features: list,
    action_display: str,
    all_results: list,
) -> str:
    """
    构造「局部反事实·目的性解释」正文。

    核心逻辑：从特征处于「高/低/极高」等档位，推断当时处于什么战术阶段、
    智能体大概想达成什么（规则模板，不是 LLM 瞎编）。
    """
    changed_feats = [f for f in key_features if f["changed"]]
    n_changed = len(changed_feats)
    n_total = len(all_results)

    lines = [f"【目的性解释】该决策（{action_display}）的意图解读：\n"]

    # 根据有多少特征能改变决策，判断策略的确定性
    if n_total > 0:
        sensitivity_ratio = n_changed / n_total
        if sensitivity_ratio > 0.6:
            lines.append("  · 策略高度依赖当前观测状态，多个因素共同触发了本次决策。")
        elif sensitivity_ratio > 0.2:
            lines.append("  · 策略在当前状态下有一定的鲁棒性，少数关键因素主导了本次决策。")
        else:
            lines.append("  · 策略对当前观测高度确定，即使状态有所变化，决策也基本不变。")

    if changed_feats:
        lines.append("\n  关键状态因素的当前态势：")
        for f in changed_feats:
            # 根据语义标签推断态势含义
            intent_hint = _infer_intent(f["feature"], f["label"], f["value"])
            lines.append(f"  · 【{f['feature']}】处于 {f['label']} 水平（{f['value']:.3f}）{intent_hint}")

    lines.append(
        f"\n  综合以上态势，智能体执行【{action_display}】的目的性解读："
        f"\n  当前观测状态促使智能体采取了上述决策，"
        f"以应对{'当前威胁' if n_changed > 0 else '一般态势'}。"
    )

    return "\n".join(lines)


def _infer_intent(feature_name: str, label: str, value: float) -> str:
    """
    根据特征名和语义标签，生成简短的意图提示。

    这里用简单关键词匹配做初步推断，
    生产环境可替换为基于 LLM 或领域知识图谱的推断。
    """
    fn = feature_name.lower()

    # 距离相关
    if "距离" in fn or "dist" in fn:
        if label in ("极低", "低"):
            return "，目标已进入近战范围，适合发动攻击"
        elif label in ("高", "极高"):
            return "，目标距离较远，处于侦察/接近阶段"
        else:
            return "，目标处于中等距离，需继续机动接近"

    # 血量/生命值
    if "血量" in fn or "hp" in fn or "health" in fn:
        if label in ("极低", "低"):
            return "，自身状态危急，需优先规避"
        elif label in ("高", "极高"):
            return "，自身状态良好，具备主动进攻条件"
        else:
            return "，自身状态中等，需谨慎权衡攻防"

    # 威胁等级
    if "威胁" in fn or "threat" in fn:
        if label in ("高", "极高"):
            return "，当前受到较高威胁，触发防御策略"
        else:
            return "，当前威胁可控"

    # 信号强度
    if "强度" in fn or "signal" in fn or "dbm" in fn:
        if label in ("高", "极高"):
            return "，目标信号清晰，适合跟踪锁定"
        else:
            return "，目标信号微弱，需扩大扫描范围"

    # 电量/能源
    if "电量" in fn or "battery" in fn:
        if label in ("极低", "低"):
            return "，能源不足，需降低功耗"
        else:
            return "，能源充足，可维持高强度任务"

    # 覆盖率
    if "覆盖" in fn or "coverage" in fn:
        if label in ("低", "极低"):
            return "，目标区域侦察覆盖不足，需继续执行任务"
        else:
            return "，目标区域已充分覆盖"

    return ""  # 无法推断意图，返回空字符串


def render_one_step_explanation(
    results: List[Any],
    obs_t: List[float],
    feature_names: List[str],
    action_t: Any,
    original_reward: float,
    preprocessor: Optional["Preprocessor"] = None,
    top_k: int = 5,
    perturb_strategy: str = "train_mean",
) -> dict:
    """
    把「一步反事实」的数值结果翻译成中文报告（给前端/指挥员看）。

    输入：one_step_counterfactual() 返回的列表 + 当时的真实观测/动作。
    输出：mechanistic（机制性）、teleological（目的性）、key_features（结构化表）。

    小白理解：算法已经算完「改每个特征后一步奖励变多少」，
    这个函数负责排个序、写上「高/低/极高」这类话术，拼成两段文字。
    """
    feat_idx_map = {name: i for i, name in enumerate(feature_names)}
    ordered = sorted(results, key=lambda r: (-abs(r.reward_delta), not r.action_changed))[:top_k]

    key_features = []
    for r in ordered:
        feat = r.candidate_feature
        idx = feat_idx_map.get(feat)
        raw_value = float(obs_t[idx]) if idx is not None else None
        if preprocessor is not None and raw_value is not None:
            label = preprocessor.discretize_label(feat, raw_value)
        else:
            label = f"{raw_value:.3f}" if raw_value is not None else "未知"

        key_features.append({
            "feature": feat,
            "value": raw_value,
            "label": label,
            "changed": r.action_changed,
            "reward_delta": float(r.reward_delta),
            "original_reward": float(r.original_reward),
            "cf_reward": float(r.cf_reward),
            "cf_action": r.cf_action,
            "perturb_strategy": perturb_strategy,
        })

    action_display = _format_action(action_t)
    mechanistic = _build_one_step_mechanistic(key_features, action_display, original_reward, perturb_strategy)
    teleological = _build_one_step_teleological(key_features, action_display, original_reward)

    return {
        "mechanistic": mechanistic,
        "teleological": teleological,
        "key_features": key_features,
        "original_action": action_display,
        "original_reward": float(original_reward),
        "disclaimer": CF_SURROGATE_DISCLAIMER,
    }


def _build_one_step_mechanistic(
    key_features: list,
    action_display: str,
    original_reward: float,
    perturb_strategy: str,
) -> str:
    """拼装「一步反事实·机械性解释」正文：列举哪些特征一改，动作/一步奖励就变。"""
    lines = [
        f"【一步反事实·机械性解释】决策（{action_display}）的单步因果链分析：",
        f"  · 扰动策略：{perturb_strategy}（将候选特征替换为训练集典型值）",
        f"  · 真实一步奖励 r_t = {original_reward:.4f}",
        "  · 近似链路：扰动 s'_t → π → a' → T → s'_{t+1} → R → r'（代理模型，非重仿真）\n",
    ]

    impactful = [f for f in key_features if f["changed"] or abs(f["reward_delta"]) > 1e-6]
    if not impactful:
        lines.append("  · 未发现能显著改变动作或一步奖励的关键特征。")
        return "\n".join(lines)

    lines.append("  以下特征扰动后，动作和/或一步奖励发生变化：")
    for f in impactful:
        cf_display = _format_action(f["cf_action"])
        sign = "升高" if f["reward_delta"] > 0 else ("降低" if f["reward_delta"] < 0 else "不变")
        lines.append(
            f"  · 【{f['feature']}】当前 = {f['value']:.3f}（{f['label']}）"
            f"  → 动作变为 {cf_display}；"
            f" 预测奖励 {f['cf_reward']:.4f}（较真实 {sign} {abs(f['reward_delta']):.4f}）"
        )
    return "\n".join(lines)


def _build_one_step_teleological(
    key_features: list,
    action_display: str,
    original_reward: float,
) -> str:
    """拼装「一步反事实·目的性解释」：从短期收益角度说明这次决策值不值得。"""
    best = max(key_features, key=lambda f: abs(f["reward_delta"]), default=None)
    lines = [
        f"【一步反事实·目的性解释】决策（{action_display}）的短期收益解读：",
        f"  · 当前步真实奖励：{original_reward:.4f}",
    ]
    if best is None:
        lines.append("  · 单特征扰动对近似一步奖励影响不明显。")
        return "\n".join(lines)

    if best["reward_delta"] > 0:
        lines.append(
            f"  · 若【{best['feature']}】处于不同水平，近似模型认为一步收益可能更高"
            f"（Δr ≈ +{best['reward_delta']:.4f}），说明该态势因素与当前决策的短期收益相关。"
        )
    elif best["reward_delta"] < 0:
        lines.append(
            f"  · 【{best['feature']}】的当前取值有利于维持较高的一步收益；"
            f" 若偏离典型值，近似奖励可能下降（Δr ≈ {best['reward_delta']:.4f}）。"
        )
    else:
        lines.append("  · 在可检验的特征中，一步奖励对单维扰动不敏感，决策可能由多因素联合驱动。")

    lines.append(
        "\n  注：以上为代理模型（π/T/R）推断，用于解释性分析，不代表环境重仿真结果。"
    )
    return "\n".join(lines)


def render_multi_step_explanation(
    results: List[Any],
    obs_t: List[float],
    feature_names: List[str],
    action_t: Any,
    preprocessor: Optional["Preprocessor"] = None,
    top_k: int = 5,
    perturb_strategy: str = "train_mean",
) -> dict:
    """
    把「多步反事实」的数值结果翻译成中文报告。

    和 render_one_step_explanation 类似，但强调：
        - 滚动 horizon 步（3～5）
        - 比较的是「累计奖励」而不是单步 r_t

    输出里会多 horizon、original_cumulative_reward 字段，供前端展示曲线摘要。
    """
    feat_idx_map = {name: i for i, name in enumerate(feature_names)}
    ordered = sorted(results, key=lambda r: (-abs(r.reward_delta), not r.action_changed))[:top_k]
    best = ordered[0] if ordered else None
    horizon = best.horizon if best else 0
    factual_cum = best.original_cumulative_reward if best else 0.0
    factual_action_seq = _format_action_sequence(best.original_action_seq) if best else []
    factual_final = list(best.original_final_obs) if best else []

    key_features = []
    for r in ordered:
        feat = r.candidate_feature
        idx = feat_idx_map.get(feat)
        raw_value = float(obs_t[idx]) if idx is not None else None
        if preprocessor is not None and raw_value is not None:
            label = preprocessor.discretize_label(feat, raw_value)
        else:
            label = f"{raw_value:.3f}" if raw_value is not None else "未知"

        key_features.append({
            "feature": feat,
            "value": raw_value,
            "label": label,
            "changed": r.action_changed,
            "reward_delta": float(r.reward_delta),
            "original_cumulative_reward": float(r.original_cumulative_reward),
            "cf_cumulative_reward": float(r.cf_cumulative_reward),
            "horizon": int(r.horizon),
            "cf_action": r.cf_action_seq[0] if r.cf_action_seq else "",
            "original_action_seq": _format_action_sequence(r.original_action_seq),
            "cf_action_seq": _format_action_sequence(r.cf_action_seq),
            "perturb_strategy": perturb_strategy,
        })

    action_display = _format_action(action_t)
    mechanistic = _build_multi_step_mechanistic(key_features, action_display, horizon, perturb_strategy)
    teleological = _build_multi_step_teleological(key_features, action_display, factual_cum, horizon)

    cf_action_seq_top = (
        _format_action_sequence(best.cf_action_seq) if best else []
    )

    return {
        "mechanistic": mechanistic,
        "teleological": teleological,
        "key_features": key_features,
        "original_action": action_display,
        "horizon": horizon,
        "original_cumulative_reward": float(factual_cum),
        "original_action_seq": factual_action_seq,
        "cf_action_seq": cf_action_seq_top,
        "top_feature": best.candidate_feature if best else None,
        "original_final_obs": factual_final,
        "cf_final_obs": [float(v) for v in best.cf_final_obs] if best else [],
        "disclaimer": CF_SURROGATE_DISCLAIMER,
    }


def _build_multi_step_mechanistic(
    key_features: list,
    action_display: str,
    horizon: int,
    perturb_strategy: str,
) -> str:
    """拼装「多步反事实·机械性解释」：哪个特征一改，首步决策和后续累计分跟着变。"""
    lines = [
        f"【多步反事实·机械性解释】决策（{action_display}）的 {horizon} 步因果链分析：",
        f"  · 扰动策略：{perturb_strategy}（每次只改一个特征为训练集典型值）",
        f"  · 事实轨迹：取仿真记录随后 {horizon} 步真实奖励累计",
        "  · 反事实轨迹：在 t 时刻扰动单特征后，用代理 π/T/R 向前滚动同一步数\n",
    ]
    impactful = [f for f in key_features if f["changed"] or abs(f["reward_delta"]) > 1e-6]
    if not impactful:
        lines.append("  · 未发现能显著改变首步决策或累计奖励的特征。")
        return "\n".join(lines)

    lines.append("  以下单特征扰动后，首步决策和/或后续累计奖励发生变化：")
    for f in impactful:
        cf_display = _format_action(f.get("cf_action", ""))
        sign = "升高" if f["reward_delta"] > 0 else ("降低" if f["reward_delta"] < 0 else "不变")
        lines.append(
            f"  · 【{f['feature']}】当前 = {f['value']:.3f}（{f['label']}）"
            f"  → 首步可能变为 {cf_display}；"
            f"  {f['horizon']} 步累计奖励 {f['cf_cumulative_reward']:.4f}"
            f"（较真实累计 {sign} {abs(f['reward_delta']):.4f}）"
        )
    return "\n".join(lines)


def _build_multi_step_teleological(
    key_features: list,
    action_display: str,
    factual_cum: float,
    horizon: int,
) -> str:
    """拼装「多步反事实·目的性解释」：从随后几步累计收益解读战术意图。"""
    best = max(key_features, key=lambda f: abs(f["reward_delta"]), default=None)
    lines = [
        f"【多步反事实·目的性解释】决策（{action_display}）的 {horizon} 步短期收益解读：",
        f"  · 真实后续 {horizon} 步累计奖励（仿真记录）：{factual_cum:.4f}",
    ]
    if best is None:
        lines.append("  · 单特征扰动对代理滚动累计奖励影响不明显。")
        return "\n".join(lines)

    if best["reward_delta"] > 0:
        lines.append(
            f"  · 若【{best['feature']}】处于不同水平，代理模型认为随后 {horizon} 步累计收益可能更高"
            f"（Δ累计 ≈ +{best['reward_delta']:.4f}），说明该因素与当前决策的短期走势相关。"
        )
    elif best["reward_delta"] < 0:
        lines.append(
            f"  · 【{best['feature']}】的当前取值有利于维持较高的 {horizon} 步累计收益；"
            f" 若偏离典型值，代理累计奖励可能下降（Δ累计 ≈ {best['reward_delta']:.4f}）。"
        )
    else:
        lines.append("  · 在可检验的特征中，累计奖励对单维扰动不敏感。")

    lines.append(
        "\n  注：以上为代理模型多步 rollout 推断，用于解释性分析，不代表环境重仿真结果。"
    )
    return "\n".join(lines)


# ==============================================================================
# 自然语言问答式解释（对齐文档：为什么…？ / 回答 / 或者回答）
# ==============================================================================

def format_decision_content(decision_content: Dict[str, Any]) -> str:
    """
    把前端传来的「要解释的那次决策」转成一行中文。

    支持只解释一个动作项，也支持多个动作组合：
        {"机动控制": "规避", "武器控制": "不发射"}
        → "机动控制=规避、武器控制=不发射"
    """
    if not decision_content:
        return ""
    items = sorted(decision_content.items(), key=lambda kv: str(kv[0]))
    return "、".join(f"{k}={v}" for k, v in items)


def _format_action_subset(action_t: Any, decision_content: Optional[Dict[str, Any]]) -> str:
    """
    格式化动作，但只保留用户关心的那几个动作项。

    避免解释「机动=规避」时，正文里却出现雷达、武器等无关维度。
    """
    if not decision_content:
        return _format_action(action_t)
    keys = set(decision_content.keys())
    if not keys:
        return _format_action(action_t)
    try:
        raw = action_t
        pairs = eval(action_t) if isinstance(action_t, str) else raw  # noqa: S307
        if isinstance(pairs, list):
            subset = [(k, v) for k, v in pairs if k in keys]
            if subset:
                return "、".join(f"{k}={v}" for k, v in sorted(subset, key=lambda x: str(x[0])))
    except Exception:
        pass
    return format_decision_content(decision_content)


def _humanize_feature_name(feature: str) -> str:
    """把「敌机距离.水平距离_km」这类字段名改成更口语的短语。"""
    if not feature:
        return "未知因素"
    name = str(feature).strip()
    group_map = {
        "自身状态": "本机",
        "敌机距离": "与敌机距离",
        "敌机状态": "敌机",
    }
    if "." in name:
        g, f = name.split(".", 1)
        g = group_map.get(g, g)
        return f"{g}{f}"
    return group_map.get(name, name)


def _feature_to_goal_phrase(feature: str) -> str:
    """根据特征类型，猜一个「战术目的」说法，用于目的性问答里的 (+0.88) 描述。"""
    fn = feature.lower()
    if "距离" in fn or "dist" in fn:
        return "提高对主要威胁目标的打击效能"
    if "威胁" in fn:
        return "降低我方暴露与被动挨打风险"
    if "血量" in fn or "hp" in fn:
        return "保持本机生存与持续作战能力"
    if "速度" in fn or "高度" in fn:
        return "优化占位与攻击窗口"
    if "雷达" in fn or "信号" in fn:
        return "改善探测与锁定条件"
    return f"改善与「{_humanize_feature_name(feature)}」相关的战术收益"


def _normalize_impact_scores(key_features: List[dict], cf_level: str) -> List[tuple]:
    """
    把各特征的贡献换算成 0～1 的「影响强度」，用于 nl_explanation 里的 (+0.88)。

    - local：看 change_score（动作变没变）
    - one_step / multi_step：看 reward_delta（奖励差多大）
    """
    scored: List[tuple] = []
    if cf_level in ("one_step", "multi_step"):
        deltas = [
            (f, float(f.get("reward_delta", 0)))
            for f in key_features
            if abs(float(f.get("reward_delta", 0))) > 1e-8
        ]
        if not deltas:
            return []
        max_abs = max(abs(d) for _, d in deltas) or 1.0
        for f, d in deltas:
            strength = round(abs(d) / max_abs, 2)
            if d < 0:
                strength = -strength
            goal = _feature_to_goal_phrase(f.get("feature", ""))
            scored.append((goal, strength))
    else:
        for f in key_features:
            if not f.get("changed"):
                continue
            score = float(f.get("change_score", 1.0))
            goal = _feature_to_goal_phrase(f.get("feature", ""))
            scored.append((goal, round(min(max(score, 0.0), 1.0), 2)))
    scored.sort(key=lambda x: -abs(x[1]))
    return scored[:5]


def render_natural_language_qa(
    *,
    key_features: List[dict],
    action_display: str,
    agent_id: int,
    decision_content: Optional[Dict[str, Any]] = None,
    cf_level: str = "local",
    t_query: Optional[int] = None,
    original_reward: Optional[float] = None,
) -> dict:
    """
    生成「问答式」因果解释（前端主展示字段 nl_explanation）。

    固定三段结构：
        1. 问题：为什么智能体在某步做出某决策？
        2. 回答（目的性）：为了哪些战术收益（带 +0.xx 强度）
        3. 或者回答（机制性）：当时态势怎样，所以采取该决策

    解释对象始终是用户传入的 decision_content（可含多个动作项）。
    """
    decision_content = decision_content or {}
    step_hint = f"在第 {t_query} 步" if t_query is not None else "在该时刻"
    # 解释对象 = 用户输入的 decision_content（可为多动作组合）；无输入时才退回整步动作
    explained = format_decision_content(decision_content) or action_display

    if decision_content:
        if len(decision_content) > 1:
            question = (
                f"为什么智能体 {agent_id} {step_hint}做出"
                f"「{explained}」这一组合决策？"
            )
        else:
            question = f"为什么智能体 {agent_id} {step_hint}做出「{explained}」？"
    else:
        question = f"为什么智能体 {agent_id} {step_hint}做出如下决策：{action_display}？"

    impacts = _normalize_impact_scores(key_features, cf_level)
    if impacts:
        tele_parts = []
        for goal, strength in impacts[:3]:
            sign = "+" if strength >= 0 else ""
            tele_parts.append(f"{goal}（{sign}{abs(strength):.2f}）")
        answer_teleological = "因为" + "，同时".join(tele_parts) + "。"
    else:
        answer_teleological = (
            f"因为在当前态势下，采取「{explained}」更有利于达成战术目标，"
            f"并在近似评估中维持较稳定的一步收益。"
        )
        if original_reward is not None:
            answer_teleological += f"（该步记录奖励约 {float(original_reward):.4f}）"

    mech_clauses: List[str] = []
    pool = [f for f in key_features if f.get("changed")] or key_features[:5]
    for f in pool:
        fname = _humanize_feature_name(f.get("feature", ""))
        label = f.get("label", f.get("value", "未知"))
        clause = f"{fname}为{label}"
        if f.get("changed") and f.get("cf_action"):
            alt = _format_action_subset(f["cf_action"], decision_content)
            if alt != explained:
                clause += f"（若态势改变，相关决策可能变为 {alt}）"
        mech_clauses.append(clause)

    if not mech_clauses:
        for f in key_features[:4]:
            fname = _humanize_feature_name(f.get("feature", ""))
            mech_clauses.append(f"{fname}为{f.get('label', f.get('value', '未知'))}")

    answer_mechanistic = (
        "因为"
        + "，".join(mech_clauses[:6])
        + f"，所以采取「{explained}」。"
    )

    nl_explanation = (
        f"{question}\n\n"
        f"回答：{answer_teleological}\n\n"
        f"或者回答：{answer_mechanistic}"
    )

    return {
        "explained_decision": explained,
        "nl_question": question,
        "nl_answer_teleological": answer_teleological,
        "nl_answer_mechanistic": answer_mechanistic,
        "nl_explanation": nl_explanation,
        "teleological_factors": [
            {"factor": g, "impact_strength": f"{('+' if s >= 0 else '')}{s:.2f}"}
            for g, s in impacts
        ],
        "mechanistic_factors": [
            {"behavior": explained, "reason": r}
            for r in mech_clauses
        ],
    }


def attach_natural_language_qa(
    explanation: dict,
    *,
    agent_id: int,
    decision_content: Optional[Dict[str, Any]] = None,
    cf_level: str = "local",
    t_query: Optional[int] = None,
) -> dict:
    """
    在已有解释字典上「再贴一层」问答字段。

    典型调用链：
        render_*_explanation(...) → attach_natural_language_qa(...) → 返回给 counterfactual_service

    不会改动 key_features / mechanistic 等原字段，只是 merge nl_* 相关键。
    """
    qa = render_natural_language_qa(
        key_features=explanation.get("key_features") or [],
        action_display=explanation.get("original_action", ""),
        agent_id=agent_id,
        decision_content=decision_content,
        cf_level=cf_level,
        t_query=t_query,
        original_reward=(
            explanation.get("original_reward")
            or explanation.get("original_cumulative_reward")
        ),
    )
    out = dict(explanation)
    out.update(qa)
    return out
