"""
decision_tree — 离线 VIPER 决策树：轨迹 → Q_hat → 规则 + 树图。

主入口：run_pipeline.py（RUN_CONFIG）
流程：trajectory_io → fqe → l_hat → weights → viper_cart → rule_ensemble
"""

__all__ = ["__version__"]

__version__ = "0.2.0"
