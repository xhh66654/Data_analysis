# 溯因分析项目 · 初学者导读

> 本目录（`Algorithm/doc/`）是**面向初学者**的项目架构解析文档。
> 如果你只是想"看懂这个仓库、知道从哪下手"，请按照本 README 的顺序阅读。
> 如需更详细的按文件逐个讲解，请看仓库根目录下的 `docs/` 文件夹（自动生成 + 深度说明）。

---

## 0. 一句话说清这个项目是做什么的

**这是一个"可解释强化学习（溯因/因果分析）"的研究型代码仓库**，主要做三件事：

1. **训练** 一个强化学习智能体（DQN / Duel DQN / Double DQN）在 Gym 环境（`CartPole-v1`、`LunarLander-v2`）里学会做决策；
2. **溯因**：给定某一条轨迹（state → action → reward），分析"为什么它在这一步选了这个动作"——通过**反事实对比**和**扰动/梯度归因**；
3. **决策树蒸馏**：把神经网络学到的行为策略提炼成一棵 **可读的 `if-then` 决策树**，导出 PDF / Excel / DOT，供人工查看。

额外还附带了一个第三方强化学习算法参考库 `FreeRL-main/`（只作学习参考，不是主线）。

---

## 1. 文档阅读推荐顺序（初学者路径）

请按如下顺序循序推进。预计**总耗时 4 - 8 小时**（含动手跑代码）。

| 阶段 | 你要做什么 | 对应文档 |
|---|---|---|
| ① 建立整体认知（30 分钟） | 读本文件 README，知道仓库有哪些目录、各自负责什么 | `Algorithm/doc/README.md`（本文） |
| ② 补齐 Python 语法（1 小时，如果你是 Java 背景） | 了解 `import`、`**kwargs`、`if __name__ == "__main__"` 等 Python 习惯 | `docs/01-Python与Java对照-难点说明.md` |
| ③ 看懂强化学习主流程（1 - 2 小时） | 跑通 `causal/main.py` 的 DQN 训练，理解"环境-智能体-回放池-网络更新"四件套 | 本文档第 3 章 + `docs/causal/*.py.md` |
| ④ 看懂"溯因/反事实"是怎么做的（1 - 2 小时） | 读 `casual-LLV2-1.py`、`test715.py` | `docs/causal/因果与强化学习-溯因分析包.md` |
| ⑤ 看懂"决策树蒸馏"做了什么（1 小时） | 跑 `DT/datatest8141.py`，打开生成的 `decision_tree*.pdf` | `docs/DT/决策树与数据处理模块说明.md` |
| ⑥ 选看参考算法（可选） | 只挑你关心的那一两个子目录看 | `docs/FreeRL-main/README-FreeRL导读.md` |

> 小建议：**不要一开始就试图"读懂每一行"**。先跑起来看输出，再回过头问"这一步是怎么实现的"。

---

## 2. 仓库目录地图（一张图讲清）

```
Algorithm/                   ← 你正在这里
├── causal/                  ← ★ 主线1：强化学习训练 + 溯因/反事实分析
│   ├── main.py              ← 训练入口（先跑这个！）
│   ├── DQN.py               ← DQN 智能体（网络、经验回放、训练）
│   ├── utils.py             ← 评估函数等工具
│   ├── generate_trajectory*.py  ← 用训练好的模型生成轨迹
│   ├── casual-LLV2-1.py     ← ★ 反事实分析脚本（溯因分析核心）
│   ├── test715.py           ← 扰动/梯度归因脚本
│   └── MADDPG_file/         ← 多智能体 RL（进阶，可暂时跳过）
│
├── DT/                      ← ★ 主线2：决策树蒸馏（可解释模块）
│   ├── datatest814*.py      ← 读 Excel → 训练决策树 → 导出 PDF/节点表
│   ├── calculate_stats.py   ← 数据归一化统计
│   └── total.py             ← 数据合并
│
├── model/                   ← 存放训练出的 .pth 权重文件
│                              命名：{算法}_{环境缩写}_{步数千}.pth
│                              如：DuelDDQN_LLdV2_500.pth
│
├── data/                    ← 训练/轨迹数据（Excel）
├── tree/                    ← 决策树可视化 PDF 归档
│
├── FreeRL-main/             ← 第三方参考算法库（不是主线）
│                              PPO / DDPG / SAC / MAPPO ... 各一个子目录
│
├── decision_tree*.pdf       ← DT 模块生成的可视化结果
├── decision_tree_nodes*.xlsx/csv   ← 决策树节点详情导出
├── normalization_params.csv ← 特征归一化参数
├── trajectory_*.txt         ← 保存的轨迹样本
├── code_summary.json        ← 全仓 Python 文件自动索引（工具生成）
│
├── test.py / test7.9.py / test7.10.py  ← 零散实验脚本（跳过）
└── doc/                     ← 你正在读的文档目录
```

**记住三条主线即可：**
- **`causal/`** = 训练 + 溯因
- **`DT/`** = 把学到的策略变成"看得懂的树"
- **`model/` + `data/` + `tree/` + 各种 pdf/xlsx** = 产物与数据

---

## 3. 怎么读懂"主线1：强化学习训练"

### 3.1 运行一次看结果（最关键的一步）

```powershell
# 进入 causal 目录（注意：Python 的 import 依赖当前工作目录）
cd E:\工作\溯因分析\Algorithm\causal

# 安装依赖（仅首次）
pip install gymnasium[box2d] torch numpy pandas scikit-learn matplotlib

# 用 CartPole 环境训练（最简单的环境，先跑这个）
python main.py --EnvIdex 0 --dvc cpu --Max_train_steps 50000
```

你会看到每 2000 步打印一次 `score`，随着训练推进分数不断上升。
训练产物会保存到 `../model/DuelDDQN_CPV1_*.pth`。

### 3.2 脑中建立"四件套"模型

`causal/main.py:85-117` 这段循环是强化学习的核心，它等价于：

```
while 还没训练够:
    重置环境 → 拿到初始状态 s
    while 本局游戏没结束:
        1. 智能体根据 s 选一个动作 a        ← DQN.py 中的 select_action
        2. 环境执行 a，返回 (s_next, r, done) ← gymnasium.env.step
        3. 把 (s, a, r, s_next, done) 存进经验池 ← replay_buffer.add
        4. 每 50 步，从经验池采样一批，反向传播更新网络 ← agent.train
        5. s = s_next，继续下一步
```

**这个"四件套"（环境 / 智能体 / 经验池 / 更新）是 RL 万变不离其宗的结构**。看懂这一段，后面所有 RL 代码你都能快速定位。

### 3.3 配套阅读

- `docs/causal/main.py.md` — `main.py` 的逐文件说明
- `docs/causal/DQN.py.md` — 网络结构与训练公式
- `docs/01-Python与Java对照-难点说明.md` 第 3、5、6 节（`**kwargs`、PyTorch、Gymnasium）

---

## 4. 怎么读懂"主线1的溯因分析"

当 `main.py` 训练出一个模型后，真正的"溯因"发生在：

### 4.1 生成轨迹
`causal/generate_trajectory (1).py` / `generate_trajectory(2).py`
→ 加载 `.pth` 权重 → 在环境里跑 N 局 → 把每一步的 `(state, action, reward)` 存到 `trajectory_*.txt`。

### 4.2 反事实分析
`causal/casual-LLV2-1.py` 的核心思路（伪代码）：

```
对轨迹中关键的一步 t：
    真实动作 a_t 获得的后续回报 = R_real
    强行替换成动作 a_t' ≠ a_t，用模型继续 rollout
    得到 R_counterfactual
    对比：如果 R_real >> R_counterfactual，说明"选 a_t 而不是 a_t' 是关键决策"
```

这就是"溯因"——**不是问"结果是什么"，而是问"如果当时不这么做会怎样"**。

### 4.3 扰动/梯度归因
`causal/test715.py` 的思路：
- 对输入状态的每一维加小扰动 → 看 Q 值变化多大 → 变化越大的维度越"重要"。
- 或用 `torch.autograd` 直接对输入求梯度。

### 4.4 配套阅读

- `docs/causal/因果与强化学习-溯因分析包.md`（最重要）
- `docs/03-算法模块归类-反事实到决策树流程.md`（把概念串起来）

---

## 5. 怎么读懂"主线2：决策树蒸馏"

这是把"黑盒神经网络"变"白盒 if-then 规则"的关键环节。

### 5.1 跑通一次

```powershell
cd E:\工作\溯因分析\Algorithm\DT
python datatest8141.py
```

它会：
1. 读 `data.xlsx` 或 `../data/training_data_*.xlsx`（状态 + 智能体输出的动作）；
2. 用 `sklearn.tree.DecisionTreeClassifier` 训练一棵树；
3. 导出 `decision_tree.pdf`（可视化）+ `decision_tree_nodes3.xlsx`（每个节点的分裂阈值、类别分布）。

**打开生成的 PDF 直接看图**——你会看到类似：

```
if  pole_angle ≤ 0.05:
    if  cart_velocity ≤ 1.2:
        动作 = 0（向左推）
    else:
        动作 = 1（向右推）
...
```

这就是"神经网络学到的策略"被提炼成了**人类可读的规则**。

### 5.2 配套阅读

- `docs/DT/决策树与数据处理模块说明.md`
- `docs/DT/datatest8141.py.md`

---

## 6. 初学者最常见的 5 个坑

1. **中文路径 + Windows PowerShell**：脚本里若出现 `C:/Users/01/Desktop/...` 写死路径（如 `test.py`），请手动改成自己的路径。
2. **`import` 失败**：`causal/main.py` 里 `from DQN import DQN_agent` 要求你 **`cd causal`** 再运行，不要在 `Algorithm/` 根目录运行。
3. **`.pth` 模型与网络结构强耦合**：如果你改了 `DQN.py` 的网络层数/宽度，旧的 `.pth` 就加载不进来了。
4. **`LunarLander-v2` 依赖 `box2d`**：`pip install gymnasium[box2d]`，否则只能跑 `CartPole`。
5. **`FreeRL-main/` 不是主线**：它只是第三方参考库，看不懂可以完全跳过，不影响理解主项目。

---

## 7. 推荐的"1 周学习计划"

| 日期 | 任务 |
|---|---|
| Day 1 | 读本文 + `docs/00-项目整体文档索引.md`；装好环境 |
| Day 2 | 跑通 `causal/main.py`（CartPole）；读 `DQN.py` 前半部分 |
| Day 3 | 读完 `DQN.py` + `utils.py`；理解"四件套"循环 |
| Day 4 | 跑 `generate_trajectory*.py`；看生成的 `trajectory_*.txt` |
| Day 5 | 读 `casual-LLV2-1.py`；跑一次反事实对比 |
| Day 6 | 跑 `DT/datatest8141.py`；打开 PDF 对比"树 vs 网络" |
| Day 7 | 读 `docs/03-算法模块归类-...md`，把所有概念串起来写一段自己的总结 |

---

## 8. 延伸：想深入时再看的文档

- 每个 `.py` 文件的自动化说明：`docs/<子目录>/<文件名>.py.md`
- 代码结构索引（程序可读）：`Algorithm/code_summary.json`
- 多智能体 RL（进阶）：`causal/MADDPG_file/` + `docs/causal/MADDPG_file/`
- 参考算法：`FreeRL-main/FreeRL-main/README.md`

---

**祝学习顺利。** 看懂本项目 = 看懂"**强化学习 + 可解释性**"这条研究主线的一个完整最小闭环。
