"""
智能体 schema：装备个体、观测空间、动作空间的固定契约与展平/校验工具。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.module_c_counterfactual.inference_record import ActionItem, InferenceRecord

DEFAULT_UNIT = "__default__"


@dataclass(frozen=True)
class AgentSchema:
    """智能体定稿后的观测/动作/装备个体结构（同模板按个体复制）。"""

    agent_id: int
    equipment_units: Tuple[str, ...]
    observation_space: Tuple[str, ...]
    action_item_names: Tuple[str, ...]

    @property
    def is_multi_unit(self) -> bool:
        """是否为多装备个体智能体（非单 __default__ 单元）。"""
        return not (len(self.equipment_units) == 1 and self.equipment_units[0] == DEFAULT_UNIT)

    @classmethod
    def from_record(cls, record: "InferenceRecord", agent_id: int) -> "AgentSchema":
        """
        从推理记录中提取指定智能体的 schema。

        参数:
            record: 推理数据记录。
            agent_id: 智能体编号。

        返回:
            对应的 AgentSchema 实例。

        抛出:
            ValueError: agent_id 不在记录中。
        """
        meta = record.get_agent_meta(agent_id)
        if meta is None:
            raise ValueError(f"agent_id={agent_id} 不在记录 {record.sim_id} 中")
        units = tuple(meta.equipment_units) if meta.equipment_units else (DEFAULT_UNIT,)
        return cls(
            agent_id=int(agent_id),
            equipment_units=units,
            observation_space=tuple(record.observation_space),
            action_item_names=tuple(item.name for item in record.action_items),
        )

    def fingerprint_payload(self) -> Dict[str, Any]:
        """
        生成用于 schema 一致性比对的指纹字典。

        返回:
            含 equipment_units、observation_space、action_items 的字典。
        """
        return {
            "equipment_units": list(self.equipment_units),
            "observation_space": list(self.observation_space),
            "action_items": list(self.action_item_names),
        }


def is_multi_unit_payload(payload: Dict[str, Any], equipment_units: List[str]) -> bool:
    """判断 dict 是否按装备个体外层嵌套。"""
    if len(equipment_units) == 1 and equipment_units[0] == DEFAULT_UNIT:
        return False
    if not payload:
        return False
    unit_set = set(equipment_units)
    keys = set(payload.keys())
    if not keys <= unit_set:
        return False
    return all(isinstance(v, dict) for v in payload.values())


def _ensure_unit_obs_map(
    obs_values: Dict[str, Any],
    equipment_units: Tuple[str, ...],
) -> Dict[str, Dict[str, Any]]:
    """
    将观测 dict 规范化为 unit_id → 观测子字典 的映射。

    参数:
        obs_values: 原始观测值（扁平或多 unit 嵌套）。
        equipment_units: 装备个体 id 元组。

    返回:
        按 unit 分组的观测字典。

    抛出:
        ValueError: 多 unit 智能体缺少 unit 外层结构。
    """
    if is_multi_unit_payload(obs_values, list(equipment_units)):
        return {u: dict(obs_values.get(u, {})) for u in equipment_units}
    if len(equipment_units) == 1 and equipment_units[0] == DEFAULT_UNIT:
        return {DEFAULT_UNIT: dict(obs_values)}
    raise ValueError(
        f"多装备个体智能体需要 unit 外层观测，期望 keys={list(equipment_units)}，"
        f"实际 keys={list(obs_values.keys())}"
    )


def _ensure_unit_decision_map(
    content: Dict[str, Any],
    equipment_units: Tuple[str, ...],
    action_item_names: Tuple[str, ...],
) -> Dict[str, Dict[str, Any]]:
    """
    将决策 dict 规范化为 unit_id → 动作项字典 的映射。

    参数:
        content: 原始决策内容（扁平或多 unit 嵌套）。
        equipment_units: 装备个体 id 元组。
        action_item_names: 动作项名称元组。

    返回:
        按 unit 分组的决策字典。

    抛出:
        ValueError: 结构无法解析或与 schema 不匹配。
    """
    if is_multi_unit_payload(content, list(equipment_units)):
        out: Dict[str, Dict[str, Any]] = {}
        for u in equipment_units:
            unit_dec = content.get(u, {})
            if not isinstance(unit_dec, dict):
                raise ValueError(f"装备个体 {u} 的决策必须是 dict")
            out[u] = dict(unit_dec)
        return out
    if len(equipment_units) == 1 and equipment_units[0] == DEFAULT_UNIT:
        return {DEFAULT_UNIT: dict(content)}
    # 扁平但配置了多 unit：若顶层键全是动作项名，视为单 unit 误用
    if set(content.keys()) <= set(action_item_names):
        raise ValueError(
            f"多装备个体智能体需要 unit 外层决策，期望 units={list(equipment_units)}"
        )
    raise ValueError(f"无法解析决策结构，keys={list(content.keys())}")


def flatten_obs(
    obs_values: Dict[str, Any],
    schema: AgentSchema,
) -> Tuple[List[float], List[str]]:
    """展平观测：单 unit 为「观测项.子字段」，多 unit 为「unit.观测项.子字段」。"""
    unit_map = _ensure_unit_obs_map(obs_values, schema.equipment_units)
    vec: List[float] = []
    names: List[str] = []
    for unit_id in schema.equipment_units:
        unit_obs = unit_map.get(unit_id, {})
        prefix = f"{unit_id}." if schema.is_multi_unit else ""
        for obs_item in schema.observation_space:
            sub_dict = unit_obs.get(obs_item, {})
            if not isinstance(sub_dict, dict):
                continue
            for sub_key in sorted(sub_dict.keys()):
                vec.append(float(sub_dict[sub_key]))
                names.append(f"{prefix}{obs_item}.{sub_key}")
    return vec, names


def _canonicalize(obj: Any) -> Any:
    """递归规范化 JSON 对象（键排序），用于稳定标签序列化。"""
    if isinstance(obj, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    return obj


def canonical_decision_label(content: Dict[str, Any], schema: AgentSchema) -> str:
    """
    生成全动作空间的稳定 canonical 标签字符串。

    参数:
        content: 决策内容 dict。
        schema: 智能体 schema。

    返回:
        与 holistic_decision_label 相同的 JSON 格式标签。
    """
    return holistic_decision_label(content, schema)


def holistic_decision_label(content: Dict[str, Any], schema: AgentSchema) -> str:
    """
    一步决策整体标签：将该步完整 decision_content 视为**一个**动作类。

    训练语义：在状态 X 下预测整体决策 Y（非拆成多个 action_item 维度）。
    多装备个体时 Y 为嵌套 JSON；单个体时为扁平 JSON 包在 __default__ 下。
    """
    unit_map = _ensure_unit_decision_map(
        content, schema.equipment_units, schema.action_item_names
    )
    return json.dumps(_canonicalize(unit_map), ensure_ascii=False, sort_keys=True)


def discover_holistic_action_space(labels: Iterable[Any]) -> List[str]:
    """从训练样本中归纳该智能体的整体决策类集合（每个唯一标签 = 一种整体动作）。"""
    return sorted({str(v) for v in labels})


def deep_equal_decision(a: Dict[str, Any], b: Dict[str, Any], schema: AgentSchema) -> bool:
    """
    判断两份决策内容在 schema 下是否语义相等。

    参数:
        a: 第一份决策 dict。
        b: 第二份决策 dict。
        schema: 智能体 schema。

    返回:
        规范化 JSON 字符串一致则为 True。
    """
    try:
        ua = _ensure_unit_decision_map(a, schema.equipment_units, schema.action_item_names)
        ub = _ensure_unit_decision_map(b, schema.equipment_units, schema.action_item_names)
    except ValueError:
        return False
    sa = json.dumps(_canonicalize(ua), ensure_ascii=False, sort_keys=True)
    sb = json.dumps(_canonicalize(ub), ensure_ascii=False, sort_keys=True)
    return sa == sb


def get_action_value_from_content(
    content: Dict[str, Any],
    schema: AgentSchema,
    action_item: str,
    unit_id: Optional[str] = None,
) -> Optional[Any]:
    """
    从决策内容中读取指定动作项的值。

    参数:
        content: 决策 dict。
        schema: 智能体 schema。
        action_item: 动作项名称。
        unit_id: 可选装备个体 id；多 unit 且未指定时返回各 unit 值的元组字符串。

    返回:
        动作值；缺失时返回 None。
    """
    unit_map = _ensure_unit_decision_map(
        content, schema.equipment_units, schema.action_item_names
    )
    if unit_id is not None:
        return unit_map.get(unit_id, {}).get(action_item)
    if schema.is_multi_unit:
        vals = tuple(unit_map.get(u, {}).get(action_item) for u in schema.equipment_units)
        if any(v is None for v in vals):
            return None
        return str(vals)
    return unit_map.get(DEFAULT_UNIT, {}).get(action_item)


def action_label_from_content(
    content: Dict[str, Any],
    schema: AgentSchema,
    action_item: Optional[str] = None,
    unit_id: Optional[str] = None,
) -> Optional[Any]:
    """
    从决策内容生成动作标签（整体或单动作项）。

    参数:
        content: 决策 dict。
        schema: 智能体 schema。
        action_item: 为 None 时返回 holistic 整体标签。
        unit_id: 可选装备个体 id。

    返回:
        holistic JSON 字符串或单动作项值。
    """
    if action_item is None:
        return holistic_decision_label(content, schema)
    return get_action_value_from_content(content, schema, action_item, unit_id=unit_id)


def validate_step_obs(obs_values: Dict[str, Any], schema: AgentSchema) -> None:
    """
    校验单步观测是否覆盖 schema 要求的全部观测项。

    参数:
        obs_values: 观测值 dict。
        schema: 智能体 schema。

    抛出:
        ValueError: 缺少必需观测项时。
    """
    unit_map = _ensure_unit_obs_map(obs_values, schema.equipment_units)
    missing: List[str] = []
    for unit_id in schema.equipment_units:
        unit_obs = unit_map.get(unit_id, {})
        for obs_item in schema.observation_space:
            if obs_item not in unit_obs:
                missing.append(f"{unit_id}.{obs_item}")
    if missing:
        raise ValueError(f"观测不完整，缺少: {missing[:5]}{'...' if len(missing) > 5 else ''}")


def validate_step_decision(content: Dict[str, Any], schema: AgentSchema) -> None:
    """
    校验单步决策是否覆盖 schema 要求的全部动作项。

    参数:
        content: 决策 dict。
        schema: 智能体 schema。

    抛出:
        ValueError: 缺少必需动作项时。
    """
    unit_map = _ensure_unit_decision_map(
        content, schema.equipment_units, schema.action_item_names
    )
    missing: List[str] = []
    for unit_id in schema.equipment_units:
        unit_dec = unit_map.get(unit_id, {})
        for name in schema.action_item_names:
            if name not in unit_dec:
                missing.append(f"{unit_id}.{name}")
    if missing:
        raise ValueError(f"决策不完整，缺少: {missing[:5]}{'...' if len(missing) > 5 else ''}")


def format_holistic_action_label(label: Any) -> str:
    """将 holistic JSON 标签或旧 tuple 字符串格式化为可读文本。"""
    if not isinstance(label, str):
        return str(label)
    s = label.strip()
    if s.startswith("{"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                parts: List[str] = []
                for unit_id, unit_dec in sorted(obj.items(), key=lambda kv: str(kv[0])):
                    if not isinstance(unit_dec, dict):
                        continue
                    unit_prefix = f"{unit_id}:" if unit_id != DEFAULT_UNIT else ""
                    for k, v in sorted(unit_dec.items(), key=lambda kv: str(kv[0])):
                        if unit_prefix:
                            parts.append(f"{unit_prefix}{k}={v}")
                        else:
                            parts.append(f"{k}={v}")
                if parts:
                    return "  ".join(parts)
        except json.JSONDecodeError:
            pass
    try:
        pairs = eval(s)  # noqa: S307 — 仅处理项目内部可信数据
        if isinstance(pairs, list):
            return "  ".join(f"{k}={v}" for k, v in pairs)
    except Exception:
        pass
    return s


def assert_same_agent_schema(records: List["InferenceRecord"], agent_id: int) -> AgentSchema:
    """合并训练前断言多条记录的 agent schema 一致。"""
    if not records:
        raise ValueError("records 为空")
    ref = AgentSchema.from_record(records[0], agent_id)
    for rec in records[1:]:
        cur = AgentSchema.from_record(rec, agent_id)
        if cur.fingerprint_payload() != ref.fingerprint_payload():
            raise ValueError(
                f"agent_id={agent_id} 的 schema 不一致，无法合并训练。\n"
                f"  期望: {ref.fingerprint_payload()}\n"
                f"  实际: {cur.fingerprint_payload()}"
            )
    return ref
