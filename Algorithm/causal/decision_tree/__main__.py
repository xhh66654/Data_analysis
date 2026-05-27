"""
包级入口：支持 python -m causal.decision_tree 启动命令行流水线。

等价于直接运行 cli.main()，将参数解析与阶段调度委托给 cli.py。
若需编辑 RUN_CONFIG 一键跑全流程，请使用 run_pipeline.py。
"""
from .cli import main

if __name__ == "__main__":
    main()
