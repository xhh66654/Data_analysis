# MADDPG_file 模块说明

本目录是 **多智能体强化学习（Multi-Agent RL）** 的实验代码，基于 **PettingZoo MPE** 环境训练 **MADDPG / MATD3** 等算法。

**与 `causal/decision_tree/` 无关**：decision tree 流水线读 LunarLander 轨迹 CSV；本目录是独立的算法实验分支，不生成 VIPER 用的轨迹文件。

---

## 1. 整体是干什么的

### 1.1 核心目标

在 **2D 粒子多智能体环境**（PettingZoo MPE）中：

1. 用 **MADDPG**（Multi-Agent DDPG）训练多个 agent 的协作/对抗策略  
2. 可选对比 **精简版 MADDPG**、**论文复现版**、**MATD3**  
3. 保存模型、TensorBoard 日志、回报曲线、评估 gif  
4. 支持多 seed 实验并绘制 mean±std 学习曲线  

### 1.2 算法思想（CTDE）

**集中训练、分散执行（Centralized Training, Decentralized Execution）**：

| 组件 | 输入 | 说明 |
|------|------|------|
| **Actor**（每个 agent 一个） | 仅自己的观测 `o_i` | 输出连续动作，部署时各 agent 独立决策 |
| **Critic**（每个 agent 一个） | **全局**所有 agent 的 obs + action | 训练时利用全局信息估计 Q 值 |

训练循环：环境交互 → 各 agent 独立 Replay Buffer → 采样 batch → 更新 Critic（MSE）→ 更新 Actor（最大化 Q）→ 软更新 target 网络。

### 1.3 仿真环境：PettingZoo MPE

**MPE（Multi-Particle Environment）** 不是空战仿真，而是俯视角 **2D 圆点粒子** 在平面移动的标准 benchmark。

默认环境 **`simple_spread_v3`**：

- N 个 agent（默认 `--N 5`）需分散覆盖 N 个地标  
- 连续动作（通常 2 维：平面力/速度）  
- 每 episode 最多 **25 步**（`max_cycles=25`）  
- 合作奖励：靠近地标、彼此远离  

代码通过 `get_env()` 动态加载：

```python
module = importlib.import_module(f'pettingzoo.mpe.{env_name}')
env = module.parallel_env(max_cycles=25, continuous_actions=True, N=env_agent_n)
```

支持的环境名（见各脚本注释）：

| 环境 | 任务类型 |
|------|----------|
| `simple_spread_v3` | 合作：分散占位（**默认**） |
| `simple_adversary_v3` | 对抗：保护地标 |
| `simple_tag_v3` | 追捕 |
| `simple_world_comm_v3` | 带通信的追捕 |

官方文档：<https://pettingzoo.farama.org/environments/mpe>

### 1.4 数据流（从训练到出图）

```
训练脚本 (MADDPG.py 等)
    ↓  PettingZoo MPE 交互
results/<env_name>/<算法名>_<序号>/
    ├── MADDPG.pth              # 各 agent 的 Actor 权重
    ├── MADDPG_seed_*_N_*.npy   # 每 agent 每 episode 回报
    ├── batch_size_obs_norm.pkl # 观测归一化统计（若启用）
    └── events.out.tfevents.*   # TensorBoard

MA_evaluate.py
    ↓  加载模型，无探索噪声评估
    ├── evaluate.png
    └── evaluate.gif

MA_plot_learning_curves.py
    ↓  聚合多 seed 的 .npy
learning_curves/<env_name>/
    ├── MADDPG.png
    └── MADDPG_<seed数>_seed.npy
```

---

## 2. 目录结构

```
MADDPG_file/
├── MADDPG.py                 # 【主入口】完整 MADDPG + 多种训练 trick
├── MADDPG_simple.py          # 精简 MADDPG（无 supplement）
├── MADDPG_reproduction.py    # 原论文两种 Actor 更新方式复现
├── MATD3_simple.py           # 多智能体 TD3
├── MA_evaluate.py            # 模型评估 + 渲染 gif
├── MA_plot_learning_curves.py# 多 seed 学习曲线
├── Buffer.py                 # Replay Buffer（含 PER 实现）
├── ATT.py                    # 注意力 Critic 模块（实验组件）
├── image_assist/             # 代码备份与辅助图片
├── note/                     # 论文复现实验笔记
├── results/                  # 训练产物（模型、日志、评估图）
└── learning_curves/          # 聚合后的学习曲线
```

---

## 3. Python 文件逐一说明

### 3.1 `MADDPG.py` — 主训练脚本（推荐入口）

**作用**：完整版 MADDPG 实现 + 训练主循环 + 环境封装 + 模型存取。

**主要类/函数**：

| 名称 | 说明 |
|------|------|
| `Actor` / `Critic` | 128-128 MLP；Actor 输出 `tanh` 连续动作 |
| `Agent` | 单 agent 的 actor/critic + target + optimizer |
| `MADDPG` | 多 agent 容器：`select_action`、`learn`、`save`、`load` |
| `get_env()` | 创建 PettingZoo MPE 并行环境，返回 `dim_info` |
| `make_dir()` | 在 `results/<env_name>/` 下自动编号创建实验目录 |
| `OUNoise` / `Normalization` | OU 探索噪声、观测归一化 |

**`supplement` 开关**（默认）：

```python
{
    'weight_decay': True,    # Critic Adam weight_decay=1e-3
    'OUNoise': True,         # OU 噪声探索（否则高斯噪声）
    'ObsNorm': False,        # 逐步 Obs 归一化
    'net_init': True,        # 特殊网络初始化
    'Batch_ObsNorm': True,   # batch 级 Obs 归一化（效果优于 ObsNorm）
}
```

**默认训练参数**：`simple_spread_v3`，`N=5`，`seed=100`，600 episodes，`batch_size=256`，`gamma=0.95`。

**运行示例**：

```bash
cd causal/MADDPG_file
python MADDPG.py --env_name simple_spread_v3 --N 5 --seed 42 --device cpu \
  --trajectory_format main_like
```

**轨迹 CSV 默认开启**；若不想写大文件，加 **`--no_save_trajectory`**。各列与 JSON 内字段释义见 **`training_trajectory_fields.md`**。  
- **`default`**（省略 `--trajectory_format`）：**每个环境 step 一行**；列 `episode`、`global_step`（从 0 计数）、`episode_step`，其后 **每个 agent 一列 JSON**（内含该 agent 一步的 `agent_id`、`obs_dim`、`action_dim`、`obs`、`action`、`reward`、`terminated`、`truncated`、`done`、`next_obs`）。→ **`training_trajectory.csv`**  
- **`json_per_step`**：与 **default 列结构完全相同**，另存 **`training_trajectory_agents_json.csv`**。  
- **`main_like`**：**每个 agent 每步一行**，扁平宽表（`s_*` / `s_next_*` / `dw` / `done`，对齐 `main.py` 风格）。→ **`training_trajectory_main_like.csv`**

说明：当前 **`default` = 一行一步 + 每 agent 一格 JSON**。若仍需旧版「每 agent 一行 + 整张扁平 `obs_*`」导出，需在历史提交中恢复或自行改脚本。

示例（JSON 分列，便于入库或下游按 agent 解析）：

```bash
python MADDPG.py --env_name simple_spread_v3 --N 5 --seed 42 --device cuda \
  --trajectory_format json_per_step
```


**动作映射注意**：PettingZoo MPE 期望动作 `[0,1]`，网络输出 `[-1,1]`，代码中会 `(action + 1) / 2` 转换。

---

### 3.2 `MADDPG_simple.py` — 精简版

**作用**：去掉 `MADDPG.py` 中所有 supplement trick，结构最干净，便于对照实验。

**与完整版差异**：

- 无 `weight_decay` / `OUNoise` / `net_init` / `Batch_ObsNorm`  
- 默认用 **高斯噪声** 探索  
- 注释说明：等价于 `MADDPG_reproduction` 的 `actor_learn_way=0`（ensemble 确定性策略）

**适用**：快速验证环境/算法是否正常；作为 ablation 的 baseline。

---

### 3.3 `MADDPG_reproduction.py` — 原论文复现

**作用**：复现 MADDPG 论文中 **两种 Actor 梯度更新方式**，用于和 OpenAI 原版 TensorFlow 代码对照。

| `actor_learn_way` | 名称 | 说明 |
|-------------------|------|------|
| `'0'` | **Ensemble** | Actor 输出确定性 `tanh` 动作 + 高斯探索（**主流复现方式，默认**） |
| `'1'` | **Approximate** | Actor 输出高斯分布参数，从分布采样（论文近似法，复现效果较差） |

**额外组件**：`DiagGaussianPd` 对角高斯分布封装。

**默认**：`N=3`，`seed=10`，CPU 训练。

**实验结论**见 `note/原论文复现记录.md`：方式 0 稳定；方式 1 易卡住且可能出现 NaN。

---

### 3.4 `MATD3_simple.py` — 多智能体 TD3

**作用**：在 MADDPG 框架上加入 **TD3** 三项改进，缓解 Q 过估计。

| TD3 机制 | `realize` 键 | 说明 |
|----------|--------------|------|
| 双 Q 网络 | `clip_double` | 两个 Critic，取 `min(Q1, Q2)` |
| 目标策略平滑 | `policy_noise` | target action 加 clipped 噪声 |
| 延迟策略更新 | `twin_delay` | 每 `policy_freq` 步才更新 Actor |

**默认**：`realize={'clip_double':True, 'policy_noise':True, 'twin_delay':True}`，`policy_name='MATD3_simple'`。

模型仍保存为 `MADDPG.pth`（历史命名），回报文件为 `MATD3_simple_seed_*.npy`。

---

### 3.5 `MA_evaluate.py` — 评估与可视化

**作用**：加载已训练模型，在环境中 **无探索噪声** 跑多 episode，输出曲线和 gif。

**依赖**：`from MADDPG import MADDPG, get_env`（评估完整版 MADDPG；simple 版需改 import）。

**输出**（写入对应 `results/.../folder_name/`）：

- `evaluate.png` — 各 agent 回报 + 指数平滑曲线  
- `evaluate.gif` — 随机选一个 episode 的渲染帧  

**关键参数**：

```bash
python MA_evaluate.py \
  --env_name simple_spread_v3 \
  --folder_name MADDPG_1 \
  --N 5 \
  --max_episodes 100 \
  --supplement "{'weight_decay':True,'OUNoise':True,...}"  # 须与训练一致
```

**注意**：若训练用了 `Batch_ObsNorm`，评估需加载同目录下的 `batch_size_obs_norm.pkl`。

---

### 3.6 `MA_plot_learning_curves.py` — 学习曲线聚合

**作用**：从 `results/<env_name>/` 读取多个 seed 的 `.npy` 回报，绘制 **均值曲线 + 标准差阴影**，保存到 `learning_curves/`。

**流程**：

1. 按文件夹名模式 `{policy_trick}_{i}` 读取各 seed 结果  
2. 多 agent 回报按 agent 维求平均  
3. 指数平滑（`smooth_rate=0.9`）  
4. 保存 `.png` 和聚合后的 `.npy`  

**运行示例**：

```bash
python MA_plot_learning_curves.py --env_name simple_spread_v3 --policy_name MADDPG --seed_num 3
```

可选 `--is_compare True` 对比多种算法（需事先准备好各算法的 `_seed.npy`）。

---

### 3.7 `Buffer.py` — 经验回放

**作用**：每个 agent 独立的 Replay Buffer，被所有训练脚本共用。

| 类 | 说明 |
|----|------|
| `Buffer` | 标准均匀采样 FIFO buffer：`obs, action, reward, next_obs, done` |
| `PER_Buffer` | 优先级经验回放（Priority Experience Replay），当前主流程 **未接入** MADDPG 训练 |

**设计细节**：区分 `act_dim`（存储维度）与 `action_dim`（动作空间维度），兼容离散/连续；MPE 连续动作为后者。

---

### 3.8 `ATT.py` — 注意力 Critic（实验模块）

**作用**：实现论文 *Modelling the Dynamic Joint Policy of Teammates with Attention Multi-agent DDPG* 中的注意力 Critic，用 attention 权重近似队友联合策略。

**主要类**：

| 类 | 说明 |
|----|------|
| `Attention_ATT` / `Attention_ATT_2` | 多头注意力编码-解码 |
| `MLPNetworkWithAttention` | 固定 3 agent 的示例 Q 网络 |
| `ATT_critic` / `ATT_critic_raw` | 可接入 MADDPG 的 Critic 变体 |

**现状**：独立模块文件；`results/` 下存在 `MADDPG_simple_ATT_*` 实验目录，说明曾做过 ATT 对照，但 **当前 `MADDPG.py` / `MADDPG_simple.py` 未 import 本文件**。扩展 ATT 需自行替换 Critic 并改 `policy_name`/文件夹命名。

---

## 4. 非 Python 文件与子目录

### 4.1 `image_assist/`

| 文件 | 说明 |
|------|------|
| `MADDPG copy.py` | `MADDPG.py` 的备份副本，含论文链接、参考超参、算法特点等长注释 |
| `image.png` ~ `image-3.png` | 文档/说明用截图 |

### 4.2 `note/原论文复现记录.md`

MADDPG 论文 **ensemble vs approximate** 两种 Actor 更新方式的对比实验记录，含 seed 0/1 下的曲线截图与结论：

- 方式 0（ensemble）效果更好，与 GitHub 主流复现一致  
- 方式 1（approx）易陷入次优，可能出现 NaN action  
- 讨论了动作 L2 惩罚 `(action**2).mean() * 1e-3` 的影响  

### 4.3 `results/`

训练自动生成的实验目录，命名规则：`{policy_name}_{trick后缀}_{自增编号}`。

典型内容：

| 文件 | 说明 |
|------|------|
| `MADDPG.pth` | `dict[agent_id -> actor_state_dict]` |
| `MADDPG_seed_{seed}_N_{N}.npy` | shape `(num_agents, num_episodes)` 的训练回报 |
| `training_episodes.csv` | 每 episode 一行汇总（默认可关 `--no_save_episode_csv`） |
| `training_trajectory.csv` | **默认写出**（`trajectory_format=default`）：**每环境步一行**；`episode`/`global_step`/`episode_step` + **每 agent 一列 JSON**（含 `obs`、`action`、`reward`、`terminated`、`truncated`、`done`、`next_obs`）；关闭：`--no_save_trajectory` |
| `training_trajectory_agents_json.csv` | **`--trajectory_format json_per_step`**：与 **`default` 同结构** 的副本文件名 |
| `training_trajectory_main_like.csv` | **`main_like`**：每 agent 每步一行，`s_*` / `dw` / `done` 扁平列 |
| `batch_size_obs_norm.pkl` | Batch 观测归一化的 mean/std |
| `events.out.tfevents.*` | TensorBoard 日志 |
| `evaluate.png` / `evaluate.gif` | `MA_evaluate.py` 产出 |

### 4.4 `learning_curves/`

`MA_plot_learning_curves.py` 输出的聚合曲线，如 `simple_spread_v3/MADDPG.png`。

---

## 5. 各训练脚本对比

| 脚本 | 算法 | Trick | 典型用途 |
|------|------|-------|----------|
| `MADDPG.py` | MADDPG | OU 噪声、Batch ObsNorm、weight_decay、net_init | **日常训练与出结果** |
| `MADDPG_simple.py` | MADDPG | 无 | Baseline / 快速调试 |
| `MADDPG_reproduction.py` | MADDPG | 论文 way0/way1 | 复现论文、方法对比 |
| `MATD3_simple.py` | MATD3 | 双 Q + 目标噪声 + 延迟更新 | 对比 TD3 是否优于 MADDPG |

---

## 6. 依赖环境

```
torch
numpy
gymnasium
pettingzoo[mpe]
tensorboard
matplotlib
imageio
pickle (标准库)
```

安装 MPE 环境（示例）：

```bash
pip install "pettingzoo[mpe]" gymnasium torch tensorboard matplotlib imageio
```

---

## 7. 推荐使用流程

```bash
# 1. 训练（完整版）
cd causal/MADDPG_file
python MADDPG.py --env_name simple_spread_v3 --N 5 --seed 0 --max_episodes 600

# 2. 换 seed 重复（如 0 / 10 / 100），得到 MADDPG_1, MADDPG_2, ...

# 3. 评估并出 gif
python MA_evaluate.py --folder_name MADDPG_1 --N 5

# 4. 画多 seed 学习曲线
python MA_plot_learning_curves.py --env_name simple_spread_v3 --policy_name MADDPG --seed_num 3

# 5. TensorBoard（可选）
tensorboard --logdir results/simple_spread_v3
```

---

## 8. 与仓库其他模块的关系

```
causal/
├── main.py, generate_trajectory*.py   → CartPole / LunarLander → trajectory_*.csv
├── decision_tree/                     → 读 CSV，FQE + VIPER + 规则（与本目录无关）
└── MADDPG_file/                       → PettingZoo MPE，多智能体 RL 实验（本目录）
```

**结论**：`MADDPG_file` 是独立的多智能体强化学习实验子项目，环境为 **PettingZoo 2D 粒子世界**，算法以 **MADDPG** 为核心，附带 **MATD3**、论文复现和 ATT 实验组件；产物在 `results/` 与 `learning_curves/`，不接入因果决策树流水线。

---

## 9. 参考

- MADDPG 论文：[Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)  
- OpenAI 参考实现：<https://github.com/openai/maddpg>  
- PettingZoo MPE：<https://pettingzoo.farama.org/environments/mpe>  
- 本目录复现笔记：`note/原论文复现记录.md`
