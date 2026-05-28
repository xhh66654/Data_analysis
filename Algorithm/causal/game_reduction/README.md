# 博弈约简 + 溯因（独立模块）

在**不改动**仓库里 `MADDPG_file`、`decision_tree` 等既有代码的前提下，从 `training_trajectory.csv` 读取多智能体 JSON 轨迹，完成：

1. **联合表征**：默认联合状态为目标智能体 + 其余全体智能体的观测向量拼接（可用 `--neighbor-subset` 收窄）；联合动作为**全体**智能体连续动作向量拼接（与仿真一步决策对齐）。
2. **集中式联合 Q**：MLP \(Q_\theta([s_{\text{joint}}], [a_{\text{joint}}])\)，离线 SARSA 式 TD 回归（reward 可取目标体或团队之和）。
3. **反事实邻居影响**：对每个非目标智能体，将其在联合状态中对应观测块、在联合动作中对应动作块置零，再与前向 \(Q\) 比较；`\(|Q - Q_{\text{cf}}|\)` 均值作为影响度，`mean_signed_effect = Q - Q_cf`。
4. **邻居约简**：主智能体由 `--target-agent` 指定。`reduced_kept_agents` / `reduced_dropped_agents` 的规则：
   - 默认 **`--min-abs-influence -1`**（未启用阈值）：只按 **`--top-k`** 取影响度降序的前 \(K\) 个邻居（旧逻辑）；`K=0` 则只见主智能体。
   - **`--min-abs-influence THR`**（\( \mathrm{THR}\ge 0\)）：影响度从高到低逐个检查，**一旦出现** `mean_abs_effect < THR`，则**该邻居及所有更弱者**都不再保留；再配合 **`--top-k`**：`top-k>0` 表示在阈值截断后最多再保留 \(K\) 个，`top-k<=0` 表示不设人数上限、只服从阈值。**后续平均场与报告只基于保留集，“剩下的”（被剔除的邻居）不参与。**
5. **平均场（Mean Field）**：在 Top-K **保留邻居**上做异构分组（`--group-map-json`：`agent_id -> 群体名`），每一步对各群体内成员的 **观测、动作向量取算术均值** \(\bar{s}_G,\bar{a}_G\)，再与目标的 \(s_{\text{tgt}},a_{\text{tgt}}\) 拼成一行低维特征，把「一对一多智能体高维耦合」近似为「**目标 versus 少量群体均值场**」。可选 `--mf-include-obs-std` 在每群体后追加观测维标准差（离散度近似）。不写群体映射则所有保留邻居并入 `--mf-default-pool` 单群体。
6. **结构因果 SCM（神经网络结构方程近似）**：在平均场导出后 **`--train-scm`**。结局 \(Y=\) **`a_target` 连续向量**；父向量 \(=\) **`s_target` + 各群体 \(\bar s,\bar a\)**（以及可选 **`s_std`**）+ **`--scm-env-padding` 个 env 占位零维**。**不包含** `a_target` 作为输入。训练 MLP 后用 **整块父变量消融**得到 `scm_causal_edges.json` 中的 \(\Delta\)MSE **边强度近似**。
7. **反事实溯因（SCM 后）**：**`--cf-abduce`** 读取 `mean_field_features.csv` 与 `scm_model.pt`，对父变量块做 **乘法尺度**干预（移除/削弱平均场信息），重新前向 SCM 得比较 **预测动作向量**变化；可选 **`--cf-behavior-json`**，用离散行为原型向量做 **负 L2 softmax 伪概率**，便于对齐「逃逸/压制」类叙事。**非**真实意图分类器，仅存启发式。


## 依赖

与本仓库其余部分一致：`numpy`、`pandas`、`torch`。仅读取 CSV，不要求安装 PettingZoo 跑环境。

## 运行

在项目根目录 `Algorithm/` 下（按你本机替换 CSV 路径）::

```powershell
python causal/game_reduction/run_pipeline.py ^
  --csv F:/cause_analysis/Algorithm/causal/MADDPG_file/results/simple_spread_v3/MADDPG_10/training_trajectory.csv ^
  --target-agent agent_0 --top-k 2 --epochs 20 --device cpu --max-rows 8000 ^
  --train-scm --scm-epochs 20 ^
  --cf-abduce --cf-rows 0,5 --cf-scales 0.0,0.5,1.0 ^
  --cf-behavior-json F:/cause_analysis/Algorithm/causal/game_reduction/behavior_prototypes.example.json
```

- `--target-agent`：主/KOP 智能体。
- `--top-k`：硬上限或未启用阈值时直接取前 K 个邻居；与 `--min-abs-influence>=0` 并用时参见 README 条目 4。
- `--min-abs-influence THR`：阈值截断（见上）；默认 `-1` 关闭。
- `--train-scm`：在平均场 CSV 就绪后训练结构方程并得到 `scm_causal_edges.json`（勿与 `--no-mean-field` 同用）。
- `--scm-env-padding`：`env` 列不足时用零占位维数。
- `--cf-abduce`：在 `scm_model.pt` 就绪后输出 `counterfactual_abduction.json`。
- `--cf-behavior-json`：行为原型（见 `behavior_prototypes.example.json`），用于伪概率叙事。
- `--max-rows`：调试时截取前 \(N\) 行；正式发布可去掉。
- `--reward-mode target|team_sum`：Bellman 用的标量奖励。
- `--neighbor-subset agent_2,agent_4`：联合状态中仅包含「目标 + 这些邻居」的 obs；动作仍为全体拼接。

输出（默认写在 CSV **同目录**下 `game_reduction_out/`）：

- `joint_q.pt` — 权重与维度元数据
- `game_reduction_report.json` — 全流程摘要；启用 SCM 时含 `scm` 字段
- `mean_field_features.csv` / `mean_field_schema.json` — 默认写出（除非 `--no-mean-field`）
- `scm_model.pt` — 结构方程网络（`--train-scm` 时）
- `scm_causal_edges.json` — 父变量块消融得到的边强度排序
- `counterfactual_abduction.json` — 反事实干预与重推理（`--cf-abduce`）

### 模块化入口

若 `PYTHONPATH` 包含仓库根::

```bash
python -m causal.game_reduction.run_pipeline --csv ...
```

## 模块一览

| 文件 | 作用 |
|------|------|
| `trajectory_maddpg.py` | 读 CSV、解析 JSON、`JointTransitionBatch` |
| `joint_q_network.py` | 拼接 (s,a) 的联合 Q MLP |
| `train_joint_q.py` | TD 训练、保存/加载 `joint_q.pt` |
| `scm_counterfactual_abduction.py` | 块尺度干预 + 重推理 + 可选行为伪概率 |
| `mean_field.py` | 群体平均场特征 |
| `scm_learning.py` | 神经网络结构方程 + 消融边强度 |
| `neighbor_counterfactual.py` | 置零屏蔽与影响度聚合 |
| `run_pipeline.py` | CLI |

## 自然语言解释生成（可选）

反事实溯因完成后，可进一步生成**定量**或**战术级**的自然语言解释：

**阶段 1：机械式因果链**（已可用，无需配置）
```bash
python causal/game_reduction/causal_narrative_gen.py \
  game_reduction_out/counterfactual_abduction.json
```
输出：按影响大小排序的因素列表，包含 `causal_chain_ranking` 和 `mechanical_explanation` 字段。

**阶段 2：战术级解释**（待实现，需配置表）
```bash
python causal/game_reduction/tactical_narrative_gen.py \
  game_reduction_out/counterfactual_abduction.json \
  causal/game_reduction/scenarios/scenario_simple_spread.yaml \
  output_tactical.json
```
输出：符合领域认知的战术描述。需先编写 `scenarios/scenario_*.yaml` 配置。

详见 `NARRATIVE_GENERATION_SUMMARY.md` 与 `NARRATIVE_GENERATION_ROADMAP.md`。

## 说明

- CSV 中单格 JSON 字段含义仍以 `MADDPG_file/training_trajectory_fields.md` 为准；本模块**不负责**重写该文档。
- 联合 Q / 消融 / SCM 均为**离线近似**；因果边详见 `scm_causal_edges.json` 内说明，不可替代严格识别。
- 自然语言解释为可选，不与核心流水线绑定。
