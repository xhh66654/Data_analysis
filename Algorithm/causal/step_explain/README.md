# step_explain — 单机单步决策解释

## 功能概述

对轨迹 CSV（`trajectory_LLdV3_S0_5.csv` 格式）中**用户指定的某一步**，输出：

- **状态维约简**：识别哪几个状态语义块（位置/姿态/速度/目标…）对这次动作选择影响最大
- **动作备选比较**：当前动作的 Q 值与最优备选的差距
- **`narrative_zh`**：一段中文自然语言决策解释

不涉及多智能体、联合 Q、邻居 Top-K 等操作；邻居/对手信息已编码在 8 维观测状态内。

---

## 目录结构

```
step_explain/
├── __init__.py             # 包入口，导出 run_explain / ExplainQuery
├── __main__.py             # python -m causal.step_explain 入口
├── run_explain.py          # 主流程 + CLI + RUN_CONFIG
├── fqe_train.py            # 现场 FQE 训练（默认每次运行触发）
├── trajectory_loader.py    # CSV 加载与单步定位
├── model_quality_report.py # 训练 loss 面板
├── effect_validation.py    # 本步解释可信度验证
├── dim_reduce.py           # 状态块反事实归因
├── action_explain.py       # 动作 Q 值全量比较
├── narrative.py            # 中文段落 + explain.json 组装
├── state_block_map.yaml    # s_0~s_7 语义分块配置（可编辑）
└── README.md
```

---

## 依赖

复用 `causal/decision_tree` 的以下组件（无需重复安装）：
- `decision_tree.q_network.QHatNetwork`
- `decision_tree.fqe.load_q_hat`

外部依赖：`torch`, `numpy`, `pandas`, `pyyaml`

---

## 配置方式（推荐）

编辑 **`run_explain.py` 顶部的 `RUN_CONFIG`**，保存后直接运行。

**默认行为**：根据 `trajectory_csv` **现场训练 FQE**（写入 `output/q_hat.pt`），再解释指定一步；**不会**读取 `decision_tree` 的 `fqe_out/q_hat.pt`。

```python
RUN_CONFIG = {
    "trajectory_csv": r"F:\...\trajectory_LLdV3_S0_2.csv",
    "use_pretrained_q_hat": False,   # 保持 False = 每次重新训练
    "fqe_epochs": 5,
    "fqe_device": "cuda",
    "episode": 0,
    "global_step": 5,
    ...
}
```

仅调试时可设 `use_pretrained_q_hat=True` 并填写 `q_hat_path` 跳过训练。

```powershell
cd F:\cause_analysis\Algorithm
python causal/step_explain/run_explain.py
```

也可用命令行覆盖单项，例如 `--episode 1 --step 10`。

---

## 快速开始

### 1. 解释指定一步（CLI，内含 FQE 训练）

```bash
python causal/step_explain/run_explain.py
```

### 2. 带参数覆盖

```bash
python causal/step_explain/run_explain.py --episode 3 --step 12 --fqe-epochs 10
```

输出示例：
```
============================================================
【决策解释】
第 3 局 第 12 步，智能体选择了「动作2（转向左）」（动作编号 2），当前奖励为 -0.3614。
该动作 Q 估值为 -1.234，最优备选为「动作0（保持）」（Q = -0.891），差距为 0.343。
决策主要受以下状态因素影响：「本机姿态」（自身姿态角与角度偏差 (s_2, s_3)，影响程度 较高，ΔQ=-0.412）、「本机速度」（速度分量 (s_4, s_5)，影响程度 中等，ΔQ=+0.187）。
备选动作参考：「动作0（保持）」Q 值 -0.891（比当前选择高 0.343）；「动作1（前进）」Q 值 -1.056（比当前选择高 0.178）。
============================================================
完整结果已保存至: causal/step_explain/output/explain.json
```

### 3. Python API

```python
from causal.step_explain import run_from_config, RUN_CONFIG

result = run_from_config(RUN_CONFIG)
print(result["narrative_zh"])
print(result["validation"]["overall_percent"])  # 本步解释可信度 %
```

---

## 参数说明

| CLI 参数 | 说明 |
|----------|------|
| `--csv` | 轨迹 CSV（默认用 RUN_CONFIG） |
| `--fqe-epochs` | 现场训练 FQE 轮数 |
| `--use-pretrained` | 跳过训练，改用 `--q-hat` |
| `--episode` / `--step` / `--row` | 指定要解释的一步 |
| `--device` | 解释阶段推理设备 |

完整 FQE 参数见 `run_explain.py` 顶部 `RUN_CONFIG`（`fqe_*`、`enable_reward_norm` 等）。

---

## 状态块配置（state_block_map.yaml）

默认将 8 维状态分为 4 块，**请根据实际环境语义修改**：

```yaml
blocks:
  本机位置:
    dims: [0, 1]
    desc: "自身位置坐标 (s_0, s_1)"
  本机姿态:
    dims: [2, 3]
    desc: "自身姿态角与角度偏差 (s_2, s_3)"
  本机速度:
    dims: [4, 5]
    desc: "速度分量 (s_4, s_5)"
  目标状态:
    dims: [6, 7]
    desc: "目标/威胁离散标志 (s_6, s_7)"
```

---

## 与 decision_tree 的关系

```
trajectory CSV
        │
        ├─► decision_tree（可选）：FQE + VIPER 规则树 → fqe_out/
        │
        └─► step_explain（独立）：内置 FQE 训练 → output/q_hat.pt → 单步解释
```

二者共用 `decision_tree` 里的 **FQE 训练代码**（`train_q_hat`），但 `step_explain` **默认自己训 Q**，不依赖 `fqe_out` 里已有文件。

---

## 输出文件

```
output/
├── q_hat.pt                 # 本次现场训练的 Q 网络
├── explain.json             # 解释 + model_training + validation
├── model_training_report.json
└── validation_report.json
```

### explain.json 结构

```json
{
  "query": { "episode": 3, "global_step": 12, "row_index": 215 },
  "chosen_action": { "id": 2, "label": "动作2（转向左）", "q_value": -1.234, "rank": 3 },
  "best_action":   { "id": 0, "label": "动作0（保持）",   "q_value": -0.891 },
  "is_optimal": false,
  "margin": 0.343,
  "reward": -0.3614,
  "all_actions": [...],
  "block_importances": [
    { "block_name": "本机姿态", "delta_q": -0.412, "abs_delta": 0.412, ... },
    { "block_name": "本机速度", "delta_q":  0.187, "abs_delta": 0.187, ... }
  ],
  "narrative_zh": "..."
}
```
