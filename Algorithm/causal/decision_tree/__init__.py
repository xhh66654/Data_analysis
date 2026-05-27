"""
decision_tree 包：离线 VIPER 风格决策树流水线。

将仿真/专家轨迹 CSV 转为可部署的 IF-THEN 规则与决策树图，流程为：
  轨迹读入 → FQE(Q_hat) → l_hat → weights → VIPER+CART → 规则导出

子模块分工：
  trajectory_io  — CSV 读入与转移构造
  q_network      — Q 值 MLP 结构
  fqe            — Q_hat 训练与 checkpoint
  l_hat          — 逐行价值差距
  weights        — VIPER 抽样权重
  viper_cart     — 加权 CART 与规则/树图
  cli            — 命令行入口
  run_pipeline   — RUN_CONFIG 一键入口（推荐）
  tune_viper     — CART 超参网格搜索
  verify_phase_link — 阶段衔接校验

版本号见 __version__。
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
