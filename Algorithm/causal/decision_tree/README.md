<!-- 用途：decision_tree 模块总览、一键运行方式与各子命令说明。 -->
# decision_tree 离线流水线

决策树从数据到规则的逐步说明（含实例）：见 **[决策树生成的流程.md](./决策树生成的流程.md)**。  
超参数调优、FQE/Q_hat/CART 调参顺序与建议范围：见 **[调参文档.md](./调参文档.md)**。  
**系统性调优路线（阶段、基线、提 acc 方案）**：见 **[调优方案.md](./调优方案.md)**。  
决策树网格调参：`python -m causal.decision_tree.tune_viper -v` → 结果在 `{fqe_out}/viper_tune/`（见调参文档）。

## 一键运行（推荐）

编辑 `causal/decision_tree/run_pipeline.py` 顶部 **`RUN_CONFIG`**（填 `trajectory_csv`），然后：

```bash
cd F:\cause_analysis\Algorithm
python causal/decision_tree/run_pipeline.py
```

结束后在输出目录得到 `final_result.json`（汇总）、`viper_out/rules.txt`（规则），以及 **与 `DT/dt_auto_pipeline` 相同的决策树流程图**：默认 **`policy_tree_debug.dot` + `policy_tree.pdf`**；可选 **`--render-tree-png`** 或 **`--open-pdf`** 自动打开 PDF。

### 决策树流程图（与 `DT/dt_auto_pipeline.py` 相同）

1. `sklearn.tree.export_graphviz` → 写 **`policy_tree_debug.dot`**
2. 标签替换：`gini`→基尼系数、`samples`→样本数、`value`→类别分布
3. **graphviz** 渲染 **`policy_tree.pdf`**（方框流程图，`filled`+`rounded`）
4. 可选：同一 `.dot` 再渲染 **`policy_tree.png`**

| 方式 | 命令 / 配置 | 依赖 |
|------|-------------|------|
| 默认导出 PDF | VIPER 阶段默认开启（`render_tree_pdf=True`） | `pip install graphviz` + [系统 Graphviz](https://graphviz.org/download/) |
| 不生成 PDF | `--no-render-tree-pdf` | — |
| 额外 PNG | `--render-tree-png` | 同上（与 PDF 同风格） |
| 渲染后自动打开 | `--open-pdf`（同 DT 的 `--open-pdf`） | — |
| 仅有 `.dot` 再转图 | `python -m causal.decision_tree.render_tree_image --dot .../policy_tree_debug.dot --format png` | Graphviz |

主流程图：**`{output-dir}/viper_out/policy_tree.pdf`**。

## 第一阶段（已实现）：FQE 训练 `Q_hat`

```bash
cd F:\cause_analysis\Algorithm
python -m causal.decision_tree --csv F:\cause_analysis\Algorithm\causal\trajectories\trajectory_LLdV3_S0_1.csv --epochs 30 --target sarsa --device cuda
```

输出默认：`{csv 同目录}/fqe_out/q_hat.pt`（可用 `--output-dir` 指定）。

### 转移构造

- `s`：`s_0…s_7`；`s'`：`s_next_*`
- `a`：`action`；`r`：`reward`
- 若 `episode[i+1]==episode[i]` → `a_next[i]=action[i+1]`，否则 `done[i]=1`
- 末行 / `dw` / `truncated` 亦置 `done=1`

### 训练目标

- **SARSA（默认）**：`target = r + γ (1-done) Q(s', a')`
- **max-Q**：`target = r + γ (1-done) max_a' Q(s', a')`

## 第二阶段（已实现）：逐行 `l_hat`

FQE 结束后冻结 `Q_hat`，对 CSV **每一行**（与 `s_i`、`action[i]` 对齐）：

- `Q_hat_a{k}` = `Q_hat(s_i, k)`
- `V_hat` = `max_k Q_hat(s_i, k)`
- `Q_sa` = `Q_hat(s_i, a_i)`
- `l_hat` = `V_hat - Q_sa`（越大表示该步相对「最优动作」越有提升空间）

默认输出：`{output-dir}/l_hat.csv`（列：`episode`, `action`, `V_hat`, `Q_sa`, `l_hat`, `Q_hat_a0…`）

```bash
# 训练 + l_hat + weights（默认 --phase all）
python -m causal.decision_tree --csv ... --epochs 30 --device cuda

# 已有 q_hat.pt，只算 l_hat
python -m causal.decision_tree --phase l_hat --csv ... --checkpoint F:\...\fqe_out\q_hat.pt --device cpu

# 已有 l_hat.csv，只算 weights
python -m causal.decision_tree --phase weights --l-hat-csv F:\...\fqe_out\l_hat.csv
```

## 第三阶段（已实现）：`l_hat` → 采样权重 `weights`

只做一次（整表），供后续多轮 VIPER 固定使用：

- `w_raw_i = max(l_hat_i, 0) + eps`（默认 `eps=1e-6`）
- `weights_i = w_raw_i / sum(w_raw)` → 非负且和为 1，可用于 `np.random.choice(n, p=weights)`

默认输出：`{output-dir}/weights.csv`（列：`episode`, `action`, `l_hat`, `w_raw`, `weights`）

核对小例子（单行逻辑，与 `l_hat` 一致）：

- `q_all=[1,2,1.5,0.5]`，`a=1` → `l_hat=0` → `w_raw≈eps`（几乎不加重）
- 同状态 `a=3` → `l_hat=1.5` → `w_raw=1.5+eps`（相对更易被抽到）

## 第四～六阶段（已实现）：VIPER 重采样 → CART → IF-THEN 规则

**步骤 3** `idx = choice(N, size=M, replace=True, p=weights)` → `X', y'`（默认 `M=N`）  
**步骤 4** `DecisionTreeClassifier(max_depth=6).fit(X', y')`  
**步骤 5** 固定 `weights`，`N_round` 轮（默认 5）；每轮可对 `weights` 加少量噪声  
**步骤 6** DFS 规则 → `viper_out/rules.txt`、`rules.json`（对齐 `DT/dt_auto_pipeline.extract_decision_rules_if_then`）

```bash
# 全流程（含 VIPER）；导出 tree.json / .dot；加 --render-tree-png 另存 policy_tree.png
python -m causal.decision_tree --phase all --csv ... --n-round 5 --max-depth 6

# 仅有 weights.csv + 轨迹 CSV
python -m causal.decision_tree --phase viper --csv ... --weights-csv ...\weights.csv --n-round 5

# 导出 policy_tree.png（matplotlib，无需系统 Graphviz）
python -m causal.decision_tree --phase viper --csv ... --weights-csv ... --render-tree-png

# 尝试生成 policy_tree.pdf（需 pip install graphviz 且系统安装 Graphviz）
python -m causal.decision_tree --phase viper --csv ... --weights-csv ... --render-tree-pdf

# PNG + 自动用默认看图软件打开
python -m causal.decision_tree --phase viper ... --render-tree-png --show-tree-image
```

输出目录默认 `{output-dir}/viper_out/`：

| 文件 | 内容 |
|------|------|
| `rules.txt` | 每叶一条 `IF s_k <= t AND ... THEN 动作名` |
| `rules.json` | 规则列表 JSON |
| `viper_summary.json` | 每轮准确率；含 `tree` 字段指向下列文件 |
| `tree.json` | **嵌套决策树**（根节点起 `split`/`leaf`，含 `left`/`right`） |
| `tree_nodes.csv` | **节点表**（与 `DT/dt_auto_pipeline` 一致） |
| `policy_tree_debug.dot` | Graphviz 源文件（与 DT 的 `{prefix}_debug.dot` 一致） |
| `policy_tree.pdf` | **流程图主输出**（默认生成，与 `dt_auto_pipeline` 相同） |
| `policy_tree.png` | 可选；**`--render-tree-png`**（与 PDF 同风格） |
