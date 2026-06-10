"""AgentSchema 与多装备个体展平/标签测试。"""
import json

import pytest

from src.module_c_counterfactual.agent_schema import (
    AgentSchema,
    canonical_decision_label,
    deep_equal_decision,
    discover_holistic_action_space,
    flatten_obs,
    holistic_decision_label,
    is_multi_unit_payload,
)
from src.module_c_counterfactual.data_loader import load_inference_records


def _flat_schema() -> AgentSchema:
    """构造单装备（扁平）观测/动作的测试用 AgentSchema。"""
    return AgentSchema(
        agent_id=1,
        equipment_units=("__default__",),
        observation_space=("自身状态", "敌机距离"),
        action_item_names=("机动控制", "武器控制"),
    )


def _multi_schema() -> AgentSchema:
    """构造双装备个体嵌套观测的测试用 AgentSchema。"""
    return AgentSchema(
        agent_id=1,
        equipment_units=("alpha_1", "alpha_2"),
        observation_space=("自身状态",),
        action_item_names=("机动控制",),
    )


def test_flatten_single_unit():
    """单装备观测展平后应得到正确数值向量与特征名。"""
    obs = {
        "自身状态": {"血量": 0.8, "高度_km": 8.0},
        "敌机距离": {"水平距离_km": 40.0},
    }
    vec, names = flatten_obs(obs, _flat_schema())
    assert vec == [0.8, 8.0, 40.0]
    assert names == ["自身状态.血量", "自身状态.高度_km", "敌机距离.水平距离_km"]


def test_flatten_multi_unit():
    """多装备个体观测展平后应按装备前缀拼接特征。"""
    obs = {
        "alpha_1": {"自身状态": {"血量": 0.8}},
        "alpha_2": {"自身状态": {"血量": 0.9}},
    }
    vec, names = flatten_obs(obs, _multi_schema())
    assert vec == [0.8, 0.9]
    assert names == ["alpha_1.自身状态.血量", "alpha_2.自身状态.血量"]


def test_holistic_decision_is_one_class_per_step():
    """一步完整 decision 映射为一个整体动作类（非拆维度）。"""
    schema = _flat_schema()
    content = {"机动控制": "追击", "武器控制": "不发射"}
    label = holistic_decision_label(content, schema)
    assert label == canonical_decision_label(content, schema)
    parsed = json.loads(label)
    assert parsed["__default__"]["机动控制"] == "追击"


def test_discover_holistic_action_space():
    """整体动作空间发现应去重并保持首次出现顺序。"""
    space = discover_holistic_action_space(["A", "B", "A", "C"])
    assert space == ["A", "B", "C"]


def test_canonical_and_deep_equal():
    """规范标签与深度相等判断应忽略装备键顺序差异。"""
    schema = _multi_schema()
    a = {"alpha_1": {"机动控制": "追击"}, "alpha_2": {"机动控制": "保持"}}
    b = {"alpha_2": {"机动控制": "保持"}, "alpha_1": {"机动控制": "追击"}}
    assert deep_equal_decision(a, b, schema)
    label = canonical_decision_label(a, schema)
    parsed = json.loads(label)
    assert parsed["alpha_1"]["机动控制"] == "追击"


def test_is_multi_unit_payload():
    """多装备载荷检测应区分扁平观测与按装备嵌套的观测。"""
    assert not is_multi_unit_payload({"自身状态": {}}, ["__default__"])
    assert is_multi_unit_payload(
        {"alpha_1": {"自身状态": {}}, "alpha_2": {"自身状态": {}}},
        ["alpha_1", "alpha_2"],
    )


@pytest.mark.skipif(
    "INF_A_MULTI" not in __import__(
        "src.module_c_counterfactual.data_loader", fromlist=["list_inference_task_ids"]
    ).list_inference_task_ids(),
    reason="需先运行 generate_mock_data 生成 INF_A_MULTI",
)
def test_inf_a_multi_record_schema():
    """INF_A_MULTI 记录应暴露多装备 schema 与带装备前缀的特征名。"""
    records = load_inference_records("INF_A_MULTI")
    assert records
    schema = records[0].get_agent_schema(1)
    assert schema.is_multi_unit
    assert list(schema.equipment_units) == ["alpha_1", "alpha_2"]
    names = records[0].get_flat_feature_names(1)
    assert any(n.startswith("alpha_1.") for n in names)
    assert any(n.startswith("alpha_2.") for n in names)
