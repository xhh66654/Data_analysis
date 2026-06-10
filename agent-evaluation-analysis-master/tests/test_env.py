"""环境单元测试（AirCombatEnv 快照与 step 行为，当前为占位）。"""
import pytest


@pytest.mark.skip(reason="环境未实现")
def test_env_reset_returns_obs():
    """``AirCombatEnv.reset`` 应返回正确形状的观测向量。"""
    # TODO
    pass


@pytest.mark.skip(reason="环境未实现")
def test_env_step_changes_state():
    """连续 ``step`` 调用应改变飞机位置等环境状态。"""
    # TODO
    pass


@pytest.mark.skip(reason="快照未实现")
def test_snapshot_restore_is_identical():
    """``capture → perturb → restore`` 应能完全恢复仿真原状态。"""
    # TODO
    pass
