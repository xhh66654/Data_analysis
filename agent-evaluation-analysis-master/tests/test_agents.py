"""智能体单元测试（DQN 保存/加载与动作合法性，当前为占位）。"""
import pytest


@pytest.mark.skip(reason="DQN 未实现")
def test_dqn_act_returns_valid_action():
    """``act()`` 返回的动作索引必须在 ``[0, n_actions)`` 范围内。"""
    # TODO
    pass


@pytest.mark.skip(reason="DQN 未实现")
def test_dqn_save_load_roundtrip(tmp_path):
    """模型保存后再加载，``act`` 推理结果应保持一致。

    参数:
        tmp_path: 写入临时权重文件的目录。
    """
    # TODO
    pass
