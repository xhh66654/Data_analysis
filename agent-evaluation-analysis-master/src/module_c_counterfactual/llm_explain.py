"""
用预训练语言模型润色反事实解释（可选）。

默认仍使用规则模板；设置 ANALYSIS_LLM_EXPLAIN=1 且模型/API 可用时，
在**不改动** key_features 等结构化结果的前提下，生成更易读的 mechanistic / teleological 文本。

支持两种后端：
  - transformers：本地 HuggingFace 模型（推荐 Qwen2.5-1.5B-Instruct）
  - openai_compatible：OpenAI 兼容 HTTP API（DashScope、vLLM 等）
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_DIR = _PROJECT_ROOT / "data" / "models" / "Qwen2.5-1.5B-Instruct"
_DEFAULT_HF_REPO = "Qwen/Qwen2.5-1.5B-Instruct"

_MODEL_CACHE: Dict[str, Any] = {"backend": None, "model": None, "tokenizer": None}

_SYSTEM_PROMPT = """你是空战智能体决策解释撰写员，读者是指挥员和业务分析员（非算法工程师）。

你会收到「反事实分析事实稿」，请改写成通俗、好读的中文说明。

输出必须严格包含以下四个标题（按顺序），每个标题下用短段落或「·」条目，避免长句堆砌：

【一句话结论】
用 1 句话说明：智能体这一步为什么这样决策（先给结论，不要铺垫术语）。

【机械性解释】
说明：若当时某些态势不一样，决策或一步收益可能会怎么变。
· 每条只写 1 个因素；格式建议：「【因素通俗名】当前为……（程度）；若改为典型值，决策可能变为……」
· 优先写「会改变决策」的因素；若只有收益变化也简要说明。
· 把特征名翻译成业务语言（如「敌机距离.水平距离_km」→「与敌机的水平距离」），不要照抄带点号的字段名。

【目的性解释】
用 2～4 句话说明：在当前态势下，该决策的战术意图、风险与短期收益含义（白话，避免「代理模型」「特征向量」等词）。

【综合摘要】
3～5 条 bullet，给值班人员快速浏览用。

硬性要求：
1. 只能使用事实稿中的数据，禁止编造。
2. 文末用一句话注明：以上为基于历史数据训练的近似模型推断，不等于仿真重放结果。
3. 禁止输出 JSON、代码块、英文技术缩写；动作写成「机动=规避、武器=发射」这种格式。

最后必须追加「问答式因果解释」三段（供前端主展示，与上文四段分开）：

【问题】
一句以「为什么」开头的问句；问句中的决策对象必须且只能是用户传入的 decision_content
（若含多个动作维度，用「A=…、B=…」组合表述，不要改写成「完整动作」）。

【回答-目的性】
一段以「因为」开头的目的性回答：说明战术收益/目标，影响强度用（+0.xx）标注，数值只能来自事实稿。

【回答-机制性】
一段以「因为」开头的机制性回答：列举当时关键态势（距离、威胁、本机状态等），末句点明「所以采取……」。"""


def is_llm_explain_enabled() -> bool:
    """
    是否通过环境变量启用 LLM 解释润色。

    返回:
        ANALYSIS_LLM_EXPLAIN 为 1/true/yes/on 时返回 True。
    """
    flag = os.environ.get("ANALYSIS_LLM_EXPLAIN", "").strip().lower()
    return flag in ("1", "true", "yes", "on")


def is_local_llm_model_ready(model_path: Optional[str] = None) -> bool:
    """
    检查本地 HuggingFace 模型权重是否已就绪。

    参数:
        model_path: 模型目录；None 时使用 get_llm_config() 中的路径。

    返回:
        存在 .safetensors 或 model*.bin 时为 True；仅有 config/tokenizer 时为 False。
    """
    path = Path(model_path or get_llm_config()["model_path"])
    if not path.is_dir():
        return False
    if any(path.glob("*.safetensors")):
        return True
    if any(path.glob("model*.bin")):
        return True
    for child in path.iterdir():
        if child.is_dir() and (
            any(child.glob("*.safetensors")) or any(child.glob("model*.bin"))
        ):
            return True
    return False


def resolve_explain_with_llm(enabled: Optional[bool]) -> bool:
    """
    解析是否启用 LLM 润色。

    参数:
        enabled: 显式开关；None 时回退到环境变量 ANALYSIS_LLM_EXPLAIN。

    返回:
        是否启用 LLM 润色。
    """
    if enabled is not None:
        return bool(enabled)
    return is_llm_explain_enabled()


def get_llm_config() -> Dict[str, Any]:
    """
    读取环境变量中的 LLM 配置。

    返回:
        含 backend、model_path、hf_repo、max_new_tokens、OpenAI 兼容 API 等键的字典。
    """
    backend = os.environ.get("ANALYSIS_LLM_BACKEND", "transformers").strip().lower()
    model_path = os.environ.get("ANALYSIS_LLM_MODEL_PATH", str(_DEFAULT_MODEL_DIR))
    hf_repo = os.environ.get("ANALYSIS_LLM_HF_REPO", _DEFAULT_HF_REPO)
    max_new_tokens = int(os.environ.get("ANALYSIS_LLM_MAX_TOKENS", "768"))
    return {
        "backend": backend,
        "model_path": model_path,
        "hf_repo": hf_repo,
        "max_new_tokens": max_new_tokens,
        "openai_base": os.environ.get("OPENAI_API_BASE", "").strip(),
        "openai_key": os.environ.get("OPENAI_API_KEY", "").strip(),
        "openai_model": os.environ.get("ANALYSIS_LLM_MODEL_NAME", "qwen2.5-1.5b-instruct").strip(),
    }


def _format_action(action_t: Any) -> str:
    """将动作标签格式化为可读中文短句（LLM 事实稿用）。"""
    from src.module_c_counterfactual.agent_schema import format_holistic_action_label

    text = format_holistic_action_label(action_t)
    return text.replace("  ", "；")


def _humanize_feature_name(feature: str) -> str:
    """
    把展平特征键改成更易读的短语。

    参数:
        feature: 如「敌机距离.水平距离_km」的复合键。

    返回:
        业务友好的中文短语。
    """
    if not feature:
        return "未知因素"
    name = str(feature).strip()
    mapping = {
        "自身状态": "本机状态",
        "敌机距离": "与敌机距离",
        "敌机状态": "敌机态势",
        "水平距离_km": "水平距离",
        "高度差_km": "高度差",
        "速度_马赫": "速度",
        "威胁等级": "威胁程度",
        "燃油剩余率": "燃油",
    }
    if "." in name:
        group, field = name.split(".", 1)
        group = mapping.get(group, group)
        field = mapping.get(field, field)
        return f"{group}·{field}"
    return mapping.get(name, name)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """把可能为 None 的值安全转成 float（LLM 事实稿用）。"""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_value(value: Any, label: Any) -> str:
    """
    将特征数值与语义标签格式化为 LLM 事实稿中的可读文本。

    参数:
        value: 原始数值。
        label: 离散化语义标签（如「高」「极低」）。

    返回:
        格式化后的字符串。
    """
    if label is not None and str(label) not in ("未知", ""):
        if value is not None:
            try:
                return f"{label}（约 {float(value):.3f}）"
            except (TypeError, ValueError):
                return str(label)
        return str(label)
    if value is None:
        return "未知"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def build_readable_narrative(facts: Dict[str, Any]) -> str:
    """
    把结构化事实改写成给 LLM 的「事实稿」叙述体。

    参数:
        facts: build_fact_bundle 生成的结构化事实字典。

    返回:
        面向 LLM 的多段中文叙述文本。
    """
    lines: List[str] = []
    cf_level = facts.get("cf_level", "local")
    if cf_level == "one_step":
        level_cn = "一步反事实（单特征扰动 + 1 步奖励）"
    elif cf_level == "multi_step":
        level_cn = "多步反事实（单特征扰动 + 3～5 步累计奖励）"
    else:
        level_cn = "局部反事实（主要看决策是否改变）"

    lines.append("=== 分析背景 ===")
    lines.append(f"推理任务：{facts.get('inference_task_id')}，仿真局：{facts.get('sim_id')}，智能体编号：{facts.get('agent_id')}")
    lines.append(f"分析时刻：第 {facts.get('t_query')} 步；分析类型：{level_cn}")
    lines.append(f"训练数据：合并了 {facts.get('n_training_records', 0)} 局仿真记录后训练的近似模型。")
    dc = facts.get("decision_content") or {}
    from src.module_c_counterfactual.explain_nl import format_decision_content

    explained = format_decision_content(dc) if dc else _format_action(facts.get("original_action"))
    lines.append(
        "本次需要解释的用户指定决策（decision_content，可含多个动作维度，只解释这些）："
        f"{json.dumps(dc, ensure_ascii=False)}"
    )
    lines.append(f"解释对象（自然语言）：{explained}")

    lines.append("\n=== 该步完整动作（参考，勿偏离用户对 decision_content 的关注点）===")
    lines.append(_format_action(facts.get("original_action")))

    if cf_level == "multi_step":
        h = facts.get("horizon")
        if facts.get("original_cumulative_reward") is not None:
            lines.append(
                f"随后 {h} 步真实累计奖励（仿真记录）："
                f"{_safe_float(facts.get('original_cumulative_reward')):.4f}"
            )
        if facts.get("original_action_seq"):
            lines.append(f"事实动作序列：{facts.get('original_action_seq')}")
    elif facts.get("original_reward") is not None:
        lines.append(f"该步真实一步奖励（仿真记录）：{_safe_float(facts.get('original_reward')):.4f}")

    lines.append("\n=== 关键因素检验（逐条单因素扰动）===")
    for i, f in enumerate(facts.get("key_feature_facts") or [], start=1):
        fname = _humanize_feature_name(f.get("feature", ""))
        val_txt = _format_value(f.get("current_value"), f.get("level_label"))
        lines.append(f"\n{i}. {fname}")
        lines.append(f"   当前：{val_txt}")

        if f.get("action_changed"):
            lines.append("   影响类型：改变决策（强影响因素）")
        else:
            lines.append("   影响类型：未改变决策")

        if f.get("cf_action_if_perturbed"):
            lines.append(f"   若改为典型值后，模型预测决策：{f['cf_action_if_perturbed']}")

        if "reward_delta" in f:
            delta = _safe_float(f.get("reward_delta"))
            direction = "升高" if delta > 0 else ("降低" if delta < 0 else "基本不变")
            if cf_level == "multi_step":
                orig_cum = _safe_float(
                    f.get("original_cumulative_reward"),
                    _safe_float(facts.get("original_cumulative_reward")),
                )
                cf_cum = _safe_float(f.get("cf_cumulative_reward"), orig_cum + delta)
                h = f.get("horizon") or facts.get("horizon") or "?"
                lines.append(
                    f"   {h} 步累计奖励：事实 {orig_cum:.4f} → 反事实约 {cf_cum:.4f}"
                    f"（{direction} {abs(delta):.4f}）"
                )
                seq = f.get("cf_action_seq")
                if seq:
                    lines.append(f"   反事实动作序列：{seq}")
            else:
                orig_r = _safe_float(
                    f.get("original_reward"),
                    _safe_float(facts.get("original_reward")),
                )
                cf_r = _safe_float(f.get("cf_reward"), orig_r + delta)
                lines.append(
                    f"   一步奖励预测：由 {orig_r:.4f} 变为约 {cf_r:.4f}"
                    f"（{direction} {abs(delta):.4f}）"
                )

    lines.append("\n=== 算法模板参考（可改写，勿照抄技术符号）===")
    mech = (facts.get("template_mechanistic") or "")[:1200]
    teleo = (facts.get("template_teleological") or "")[:800]
    if mech:
        lines.append("[机械性模板]\n" + mech)
    if teleo:
        lines.append("[目的性模板]\n" + teleo)

    return "\n".join(lines)


def build_fact_bundle(
    explanation: Dict[str, Any],
    *,
    cf_level: str,
    inference_task_id: str,
    sim_id: str,
    agent_id: int,
    t_query: int,
    decision_content: Dict[str, Any],
    perturb_strategy: str,
    n_training_records: int = 0,
) -> Dict[str, Any]:
    """
    把模板解释与 key_features 整理成给 LLM 的事实包（减少幻觉）。

    参数:
        explanation: render_*_explanation 返回的解释字典。
        cf_level: 反事实层级（local/one_step/multi_step）。
        inference_task_id: 推理任务 id。
        sim_id: 仿真局 id。
        agent_id: 智能体编号。
        t_query: 查询时间步。
        decision_content: 用户指定要解释的决策内容。
        perturb_strategy: 特征扰动策略名称。
        n_training_records: 合并训练的仿真局数量。

    返回:
        含 narrative_for_llm 等键的事实包字典。
    """
    key_features = explanation.get("key_features") or []
    facts: List[Dict[str, Any]] = []
    for f in key_features[:8]:
        raw_feat = f.get("feature")
        item: Dict[str, Any] = {
            "feature": raw_feat,
            "feature_readable": _humanize_feature_name(raw_feat or ""),
            "current_value": f.get("value"),
            "level_label": f.get("label"),
            "action_changed": f.get("changed"),
        }
        if "reward_delta" in f:
            item["reward_delta"] = f.get("reward_delta")
            if cf_level == "multi_step":
                item["original_cumulative_reward"] = f.get("original_cumulative_reward")
                item["cf_cumulative_reward"] = f.get("cf_cumulative_reward")
                item["horizon"] = f.get("horizon")
                if f.get("cf_action_seq"):
                    item["cf_action_seq"] = f.get("cf_action_seq")
            else:
                item["cf_reward"] = f.get("cf_reward")
                item["original_reward"] = f.get("original_reward")
        if f.get("cf_action"):
            item["cf_action_if_perturbed"] = _format_action(f.get("cf_action"))
        facts.append(item)

    bundle = {
        "inference_task_id": inference_task_id,
        "sim_id": sim_id,
        "agent_id": agent_id,
        "t_query": t_query,
        "decision_content": decision_content,
        "cf_level": cf_level,
        "perturb_strategy": perturb_strategy,
        "n_training_records": n_training_records,
        "original_action": explanation.get("original_action"),
        "original_reward": explanation.get("original_reward"),
        "original_cumulative_reward": explanation.get("original_cumulative_reward"),
        "horizon": explanation.get("horizon"),
        "original_action_seq": explanation.get("original_action_seq"),
        "cf_action_seq": explanation.get("cf_action_seq"),
        "key_feature_facts": facts,
        "template_mechanistic": explanation.get("mechanistic", ""),
        "template_teleological": explanation.get("teleological", ""),
        "template_nl_explanation": explanation.get("nl_explanation", ""),
    }
    bundle["narrative_for_llm"] = build_readable_narrative(bundle)
    return bundle


def enhance_cf_explanation(
    explanation: Dict[str, Any],
    *,
    cf_level: str,
    inference_task_id: str,
    sim_id: str,
    agent_id: int,
    t_query: int,
    decision_content: Dict[str, Any],
    perturb_strategy: str,
    n_training_records: int = 0,
    enabled: Optional[bool] = None,
    llm_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    在 explanation 上增加 LLM 润色字段；失败时保留模板文本。

    参数:
        explanation: 规则模板生成的解释字典。
        cf_level: 反事实层级。
        inference_task_id: 推理任务 id。
        sim_id: 仿真局 id。
        agent_id: 智能体编号。
        t_query: 查询时间步。
        decision_content: 用户指定决策内容。
        perturb_strategy: 扰动策略名称。
        n_training_records: 训练仿真局数量。
        enabled: 显式 LLM 开关；None 时读环境变量。
        llm_config: 可选 LLM 配置覆盖项。

    返回:
        合并了 mechanistic/teleological/summary/nl_* 等字段的解释字典。
    """
    out = dict(explanation)
    out["mechanistic_raw"] = out.get("mechanistic", "")
    out["teleological_raw"] = out.get("teleological", "")
    out["explanation_backend"] = "template"

    use_llm = resolve_explain_with_llm(enabled)
    if not use_llm:
        return out

    cfg = {**get_llm_config(), **(llm_config or {})}

    if cfg.get("backend", "transformers") == "transformers" and not is_local_llm_model_ready(
        cfg.get("model_path")
    ):
        out["llm_error"] = (
            f"本地模型权重未就绪：{cfg.get('model_path')} 下缺少 *.safetensors。"
            "请运行：py scripts/download_explain_model_modelscope.py"
        )
        return out
    facts = build_fact_bundle(
        explanation,
        cf_level=cf_level,
        inference_task_id=inference_task_id,
        sim_id=sim_id,
        agent_id=agent_id,
        t_query=t_query,
        decision_content=decision_content,
        perturb_strategy=perturb_strategy,
        n_training_records=n_training_records,
    )
    user_prompt = (
        "请将下面「事实稿」改写成面向指挥员的通俗说明（遵守系统提示中的四个标题）。\n\n"
        + facts.get("narrative_for_llm", "")
    )

    try:
        if cfg["backend"] == "openai_compatible":
            text, backend_tag = _generate_openai_compatible(cfg, user_prompt)
        else:
            text, backend_tag = _generate_transformers(cfg, user_prompt)

        headline, mech, teleo, summary = _parse_four_sections(text)
        parsed_any = bool(headline or mech or teleo or summary)
        if headline:
            out["headline"] = headline
        if mech:
            out["mechanistic"] = _cleanup_readable_text(mech)
        if teleo:
            out["teleological"] = _cleanup_readable_text(teleo)
        if summary:
            out["summary"] = _cleanup_readable_text(summary)
        if not parsed_any:
            out["llm_raw"] = text
        qa = _parse_qa_sections(text)
        if qa.get("nl_question"):
            out["nl_question"] = qa["nl_question"]
        if qa.get("nl_answer_teleological"):
            out["nl_answer_teleological"] = qa["nl_answer_teleological"]
        if qa.get("nl_answer_mechanistic"):
            out["nl_answer_mechanistic"] = qa["nl_answer_mechanistic"]
        if qa.get("nl_explanation"):
            out["nl_explanation"] = qa["nl_explanation"]
        out["explanation_backend"] = backend_tag
    except Exception as exc:
        out["llm_error"] = str(exc)

    return out


def _cleanup_readable_text(text: str) -> str:
    """去掉多余 markdown 符号，统一空白。"""
    t = text.strip()
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t


def _strip_section_title(body: str) -> str:
    """去掉段首的【标题】或 ### 标题行。"""
    body = body.strip()
    body = re.sub(r"^【[^】]+】\s*", "", body)
    body = re.sub(r"^#{1,3}\s*[^\n]+\n?", "", body)
    return body.strip()


def _parse_four_sections(text: str) -> Tuple[str, str, str, str]:
    """从模型输出中切分四个段落（支持【】与 Markdown ### 标题）。"""
    text = text.strip()
    if not text:
        return "", "", "", ""

    section_patterns: List[Tuple[str, List[str]]] = [
        ("headline", [r"【一句话结论】", r"#{1,3}\s*一句话结论"]),
        ("mech", [r"【机械性解释】", r"#{1,3}\s*机械性解释"]),
        ("teleo", [r"【目的性解释】", r"#{1,3}\s*目的性解释"]),
        ("sum", [r"【综合摘要】", r"#{1,3}\s*综合摘要"]),
    ]
    positions: List[Tuple[int, str]] = []
    for tag, pats in section_patterns:
        for pat in pats:
            m = re.search(pat, text)
            if m:
                positions.append((m.start(), tag))
                break

    if len(positions) < 2:
        return "", "", "", ""

    positions.sort(key=lambda x: x[0])
    chunks: Dict[str, str] = {}
    for i, (start, tag) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        body = _strip_section_title(text[start:end])
        chunks[tag] = body

    return (
        chunks.get("headline", ""),
        chunks.get("mech", ""),
        chunks.get("teleo", ""),
        chunks.get("sum", ""),
    )


def _parse_three_sections(text: str) -> Tuple[str, str, str]:
    """兼容旧三段解析。"""
    headline, mech, teleo, summary = _parse_four_sections(text)
    if summary and not teleo:
        teleo = summary
    return mech, teleo, summary


def _parse_qa_sections(text: str) -> Dict[str, str]:
    """解析文档风格的问答式解释字段（支持【】与「问题：/回答-目的性：」）。"""
    block = text
    qa_hdr = re.search(r"(?:【问答式因果解释】|#{1,3}\s*问答式[^\n]*)", text)
    if qa_hdr:
        block = text[qa_hdr.end() :]

    def _pick(patterns: List[str]) -> str:
        """
        按顺序尝试正则模式，提取并清洗第一个匹配片段。

        参数:
            patterns: 正则表达式字符串列表，按优先级排列。

        返回:
            清洗后的匹配文本；无匹配时返回空字符串。
        """
        for pat in patterns:
            m = re.search(pat, block, re.DOTALL)
            if m:
                return _cleanup_readable_text(m.group(1))
        return ""

    question = _pick([
        r"【问题】\s*(.+?)(?=\n(?:【|#{1,3}\s*|回答[-－]|或者回答)|\Z)",
        r"问题[：:]\s*(.+?)(?=\n(?:回答[-－]|或者回答)|\Z)",
    ])
    tele = _pick([
        r"【回答-目的性】\s*(.+?)(?=\n(?:【|#{1,3}\s*|回答[-－]|或者回答)|\Z)",
        r"回答[-－]目的性[：:]\s*(.+?)(?=\n(?:回答[-－]机制|或者回答)|\Z)",
    ])
    mech = _pick([
        r"【回答-机制性】\s*(.+?)(?=\n(?:【|#{1,3}\s*)|\Z)",
        r"回答[-－]机制性[：:]\s*(.+?)(?=\n(?:【|#{1,3}\s*)|\Z)",
        r"或者回答[：:]\s*(.+?)(?=\n(?:【|#{1,3}\s*)|\Z)",
    ])

    if not (question or tele or mech):
        return {}

    chunks: Dict[str, str] = {}
    if question:
        chunks["nl_question"] = question if question.endswith("？") else question + "？"
    if tele:
        chunks["nl_answer_teleological"] = tele
    if mech:
        chunks["nl_answer_mechanistic"] = mech
    if question and tele and mech:
        chunks["nl_explanation"] = (
            f"{chunks['nl_question']}\n\n回答：{tele}\n\n或者回答：{mech}"
        )
    return chunks


def _resolve_model_path(cfg: Dict[str, Any]) -> str:
    """
    解析本地模型目录或回退到 HuggingFace 仓库 id。

    参数:
        cfg: get_llm_config() 返回的配置字典。

    返回:
        可用于 from_pretrained 的路径或仓库名。
    """
    path = Path(cfg["model_path"])
    if path.is_dir() and any(path.iterdir()):
        return str(path)
    return cfg["hf_repo"]


def _generate_transformers(cfg: Dict[str, Any], user_prompt: str) -> Tuple[str, str]:
    """
    使用本地 HuggingFace transformers 后端生成润色文本。

    参数:
        cfg: LLM 配置字典。
        user_prompt: 用户提示（含事实稿）。

    返回:
        (生成文本, backend_tag) 元组，backend_tag 为 "llm_transformers"。
    """
    AutoModelForCausalLM, AutoTokenizer = _import_transformers()
    model_id = _resolve_model_path(cfg)

    if _MODEL_CACHE.get("model_id") != model_id:
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        use_cuda = torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32
        load_kw: Dict[str, Any] = {
            "dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if use_cuda:
            load_kw["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
        if not use_cuda:
            model = model.to("cpu")
            model.eval()
        _MODEL_CACHE.update(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            backend="transformers",
        )

    model = _MODEL_CACHE["model"]
    tokenizer = _MODEL_CACHE["tokenizer"]
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = _SYSTEM_PROMPT + "\n\n" + user_prompt + "\n\n助手："

    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=cfg["max_new_tokens"],
            do_sample=True,
            temperature=0.28,
            top_p=0.88,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip(), "llm_transformers"


def _generate_openai_compatible(cfg: Dict[str, Any], user_prompt: str) -> Tuple[str, str]:
    """
    使用 OpenAI 兼容 HTTP API 后端生成润色文本。

    参数:
        cfg: 含 openai_base、openai_key、openai_model 的配置字典。
        user_prompt: 用户提示（含事实稿）。

    返回:
        (生成文本, backend_tag) 元组，backend_tag 为 "llm_openai"。

    抛出:
        RuntimeError: API 配置缺失或 HTTP 错误时。
    """
    import urllib.error
    import urllib.request

    base = cfg["openai_base"].rstrip("/")
    key = cfg["openai_key"]
    if not base or not key:
        raise RuntimeError(
            "openai_compatible 后端需要设置 OPENAI_API_BASE 与 OPENAI_API_KEY。"
        )

    url = f"{base}/chat/completions"
    payload = {
        "model": cfg["openai_model"],
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.35,
        "max_tokens": cfg["max_new_tokens"],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API HTTP {e.code}: {detail}") from e

    text = body["choices"][0]["message"]["content"]
    return str(text).strip(), "llm_openai"


def _import_transformers():
    """
    延迟导入 transformers 库。

    返回:
        (AutoModelForCausalLM, AutoTokenizer) 元组。

    抛出:
        RuntimeError: 未安装 transformers 时。
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "未安装 transformers。请执行：pip install transformers accelerate sentencepiece\n"
            "并运行：py scripts/download_explain_model.py"
        ) from e
    return AutoModelForCausalLM, AutoTokenizer
