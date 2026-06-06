# decision_tree

将轨迹 CSV 提炼为 **IF-THEN 规则** 与 **决策树 PDF**（主用途：展示，非 holdout 实验）。

十维动作、每步 1～3 维变化，且仍用 **一棵决策树（组合 action id）**、不改本目录代码时，见 **[十维动作-单树适配说明.md](./十维动作-单树适配说明.md)**。  
**推荐决策方案（轨迹模式 + 战术库合并，控制 N≈50～300）：** [十维动作-轨迹模式与战术库决策方案.md](./十维动作-轨迹模式与战术库决策方案.md)。

## 训练流程说明

端到端各阶段说明（轨迹 → FQE → l_hat → weights → VIPER → 规则/PDF）：**[决策树训练流程.md](./决策树训练流程.md)**。

## 运行

编辑 `run_pipeline.py` 顶部 **`RUN_CONFIG`**（至少 `trajectory_csv`），然后：

```bash
cd F:\cause_analysis\Algorithm
python causal/decision_tree/run_pipeline.py
```

或：`python -m causal.decision_tree.run_pipeline`

## 输出（`{轨迹目录}/fqe_out/`）

| 路径 | 说明 |
|------|------|
| `q_hat.pt` | FQE 训练的 Q 网络 |
| `l_hat.csv` / `weights.csv` | 价值差与 VIPER 抽样权重 |
| `viper_out/rules.txt` | IF-THEN 规则 |
| `viper_out/policy_tree.pdf` | 决策树图（类别分布为**真实标签百分比**；**仅叶节点**显示 class） |
| `data_flow_report.json` | 各阶段样本行数流转 |
| `final_result.json` | 全流程汇总 |

## 关键配置（RUN_CONFIG）

- **`pipeline_mode: "rules"`** — VIPER 用全表建树（默认）
- **`only_episode`** — 仅某一局提炼规则；FQE 仍用全表
- **`cart_class_weight: "balanced"`** — 训练时类均衡；PDF 仍显示真实分布
- **`oracle_relabel`** — `True` 用 Q 最优动作作标签；`False` 用 CSV `action`
- **`run_rule_ensemble`** — 多轮规则投票后取 Top-K

## 模块

| 文件 | 作用 |
|------|------|
| `run_pipeline.py` | 一键全流程 |
| `trajectory_io.py` | 读 CSV、构造转移 |
| `fqe.py` / `q_network.py` | Q_hat 训练 |
| `l_hat.py` / `weights.py` | 价值差、抽样权重 |
| `viper_cart.py` | VIPER + CART + 导出 |
| `rule_ensemble.py` | 多轮规则集成 |
| `data_flow.py` | 样本行数追踪 |
