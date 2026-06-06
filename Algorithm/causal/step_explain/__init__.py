"""
step_explain — 单机单步决策解释模块。

根据轨迹 CSV **现场训练 FQE**，对用户指定的一步输出归因与自然语言解释。

入口::

  python causal/step_explain/run_explain.py

API::

  from causal.step_explain import run_from_config, RUN_CONFIG
  result = run_from_config(RUN_CONFIG)
"""

__all__ = ["run_explain", "run_from_config", "ExplainQuery", "RUN_CONFIG"]


def __getattr__(name: str):
    if name in __all__:
        from .run_explain import ExplainQuery, RUN_CONFIG, run_explain, run_from_config

        return {
            "run_explain": run_explain,
            "run_from_config": run_from_config,
            "ExplainQuery": ExplainQuery,
            "RUN_CONFIG": RUN_CONFIG,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
