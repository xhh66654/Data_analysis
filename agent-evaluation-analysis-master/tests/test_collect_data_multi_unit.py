"""多装备个体 collect_data 测试。"""
import pytest

from src.module_a_rules.collect_data import collect_from_records
from src.module_c_counterfactual.data_loader import load_inference_records, list_inference_task_ids


@pytest.mark.skipif(
    "INF_A_MULTI" not in list_inference_task_ids(),
    reason="需先运行 generate_mock_data 生成 INF_A_MULTI",
)
def test_collect_multi_unit_full_action():
    """多装备任务的整体决策标签应为 JSON 字符串且特征含装备前缀。"""
    records = load_inference_records("INF_A_MULTI")
    X, y, rewards, fn = collect_from_records(records, agent_id=1, action_item=None)
    assert len(y) > 0
    assert X.shape[1] == len(fn)
    assert any(f.startswith("alpha_1.") for f in fn)
    assert all(isinstance(lbl, str) and lbl.startswith("{") for lbl in y)


def test_collect_flat_still_works():
    """扁平场景 INF_A_001 的 collect 仍应正常工作且无装备前缀特征。"""
    records = load_inference_records("INF_A_001")
    X, y, _, fn = collect_from_records(records, agent_id=1, action_item=None)
    assert len(y) > 0
    assert not any(f.startswith("alpha_1.") for f in fn)
