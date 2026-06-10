# 空战仿真溯因分析系统 — 使用手册

> 面向非技术人员的完整使用指南，包含系统原理、环境配置、数据准备、运行方法。

---

## 目录

1. [系统是做什么的？](#1-系统是做什么的)
2. [目录结构说明](#2-目录结构说明)
3. [环境配置（第一次使用必做）](#3-环境配置第一次使用必做)
4. [数据准备](#4-数据准备)
5. [服务A：规则抽取（explain_a）](#5-服务a规则抽取explain_a)
6. [服务B：反事实推理（explain_c）](#6-服务b反事实推理explain_c)
7. [Python API 调用方式](#7-python-api-调用方式)
8. [参数说明速查表](#8-参数说明速查表)
9. [常见问题 FAQ](#9-常见问题-faq)
10. [代码文件索引](#10-代码文件索引)
11. [连接真实数据库](#11-连接真实数据库)

---

## 1. 系统是做什么的？

这套系统用于解释空战智能体（AI 飞机/雷达/侦察机）的决策行为，回答两类问题：

### 问题A：规律是什么？

> "这个智能体在什么情况下会选择追击？在什么情况下会规避？"

系统会从推理数据中训练一棵**决策树**，然后提取出人类可读的 **IF-THEN 规则**，例如：

```
规则1：如果 飞行状态.高度_km <= 较低（7.51）
          且 机动状态.燃油剩余 <= 低（0.59）
       则 → 追击
       （支持度: 12步，置信度: 1.00）
```

### 问题B：为什么这一步这么选？

> "t=3 时刻，智能体选择了规避而不是追击，是因为什么？"

系统会进行**反事实推理**：逐一修改每个观测特征，检验"如果当时那个特征值不同，决策会变吗？"，最终给出两类解释：

- **机械性解释**：哪些特征是关键原因（修改它们决策就变）
- **目的性解释**：当前处于什么态势，这个决策指向什么目标

---

## 2. 目录结构说明

```
analysis/
├── main.py                          ← 命令行入口（你主要用这个）
├── config.yaml                      ← 全局配置（一般不需要改）
├── scripts/
│   └── generate_mock_data.py        ← 生成测试数据（第一次必须运行）
├── data/
│   └── mock_records/
│       ├── inference_task.json      ← 任务信息表（150个任务）
│       └── inference_step.json      ← 步骤流水表（约3000条记录）
├── output/                          ← 运行结果输出目录（自动创建）
│   ├── rule_tree_INF_A_001_agent1_联合动作.png  ← 决策树图片
│   └── counterfactual_INF_A_001_SIM_A_0001.json ← 反事实推理结果
└── src/
    ├── service.py                   ← 两个服务的主入口
    ├── module_a_rules/              ← 服务A：规则抽取模块
    │   ├── collect_data.py          ← 从记录里提取训练样本
    │   ├── preprocess.py            ← 特征归一化 + 自动分箱
    │   ├── viper.py                 ← VIPER 算法：训练决策树
    │   ├── extract_rules.py         ← 从决策树提取 IF-THEN 规则
    │   └── merge_rules.py           ← 合并冗余规则
    ├── module_c_counterfactual/     ← 服务B：反事实推理模块
    │   ├── inference_record.py      ← 核心数据结构定义
    │   ├── data_loader.py           ← 数据加载（本地JSON/Doris数据库）
    │   ├── policy_model.py          ← 策略近似决策树
    │   ├── counterfactual.py        ← 局部反事实推理算法
    │   ├── rollback.py              ← 定位决策时间步
    │   └── explain_nl.py            ← 生成自然语言解释
    └── viz/
        └── tree_plot.py             ← 决策树可视化（PDF/PNG）
```

---

## 3. 环境配置（第一次使用必做）

### 3.1 安装 Python

需要 Python 3.9 或以上版本。

检查是否已安装（在命令行中运行）：
```
py --version
```

如果看到 `Python 3.x.x` 说明已安装。否则请去 [python.org](https://www.python.org/downloads/) 下载安装。

### 3.2 安装依赖包

在项目根目录（`E:\工作\analysis\`）打开命令行，运行：

```
pip install numpy scikit-learn matplotlib
```

可选安装（用于生成 PDF 格式决策树，不安装会自动降级为 PNG）：
```
pip install graphviz
```
> 注意：安装 Python graphviz 包后，还需要安装系统级 Graphviz 工具：
> 下载地址：https://graphviz.org/download/ （Windows 下载 .exe 安装包，安装时勾选"Add to PATH"）

### 3.3 验证安装

```
py -c "import numpy, sklearn, matplotlib; print('依赖安装成功')"
```

---

## 4. 数据准备

### 4.1 生成测试数据（推荐新手先用这个）

项目内置了 3 种装备类型的模拟数据生成脚本：
- **歼-20**（INF_A_001 ~ INF_A_005）：8个观测特征，4个动作项
- **雷达站**（INF_B_001 ~ INF_B_005）：6个观测特征，3个动作项
- **侦察机**（INF_C_001 ~ INF_C_005）：9个观测特征，2个动作项

在项目根目录运行：
```
py scripts/generate_mock_data.py
```

成功后会看到：
```
已生成 150 个任务，共 2993 步骤记录
文件路径：data/mock_records/inference_task.json
         data/mock_records/inference_step.json
```

### 4.2 数据结构说明

系统使用两张数据库表（本地用 JSON 文件模拟）：

**表1：inference_task.json**（任务信息，每行一个任务）

| 字段 | 含义 | 示例 |
|------|------|------|
| task_id | 推理任务 id（inference_task_id） | "INF_A_001" |
| sim_id | 仿真 id（sim_id） | "SIM_A_0001" |
| agents_json | 智能体列表（JSON） | [{"agent_id":1,"agent_name":"Alpha","equipment_type":"歼-20"}] |
| observation_space | 观测项名称列表（JSON） | ["自身状态","敌机距离","机动状态"] |
| action_items_json | 动作项定义（JSON） | [{"name":"机动控制","possible_values":["追击","规避","保持"]}] |
| total_steps | 该任务总步数 | 20 |

**表2：inference_step.json**（步骤流水，每行一步一个智能体）

| 字段 | 含义 | 示例 |
|------|------|------|
| task_id | 关联推理任务 id（inference_task_id） | "INF_A_001" |
| sim_id | 仿真 id（sim_id） | "SIM_A_0001" |
| step | 时间步（从0开始） | 3 |
| agent_id | 智能体编号 | 1 |
| decision_json | 该步决策内容（JSON） | {"机动控制":"追击","武器控制":"发射导弹"} |
| obs_json | 该步观测内容（JSON，嵌套字典） | {"自身状态":{"血量":0.8,"高度_km":7.5},...} |
| reward | 该步奖励值 | 0.5 |

### 4.3 接入真实数据库

如果要连接 Doris 数据库，请修改 `src/module_c_counterfactual/data_loader.py`：

```python
# 第66行：改为 False
MOCK_MODE = False

# 第72-79行：填写真实数据库配置
DORIS_CONFIG = {
    "host": "你的数据库IP",
    "port": 9030,
    "user": "root",
    "password": "你的密码",
    "database": "simulation_db",
}
```

详见本文档第 [11节](#11-连接真实数据库)。

---

## 5. 服务A：规则抽取（explain_a，联合动作单树）

### 5.1 作用

从推理任务数据中，训练一棵 CART 决策树，提取 IF-THEN 规则集，并生成决策树图片。

### 5.2 基本用法

打开命令行，切换到项目根目录（`E:\工作\analysis\`），运行：

```
py main.py --mode explain_a --inference_task_id INF_A_001 --agent_id 1
```

**参数说明：**

| 参数 | 是否必填 | 含义 | 示例 |
|------|----------|------|------|
| `--inference_task_id` | 必填 | 推理任务编号 | `INF_A_001` |
| `--agent_id` | 可选（默认1） | 智能体编号 | `1` |
| `--out` | 可选 | 决策树图片保存路径（不含扩展名） | `output/my_tree` |

### 5.3 更多示例

```
# 指定输出路径
py main.py --mode explain_a --inference_task_id INF_A_001 --agent_id 1 --out output/my_tree
```

### 5.4 输出说明

运行后控制台会打印：

```
[基于规则的策略提取] inference_task_id=INF_A_001  agent_id=1  label=联合动作（完整决策内容）
  正在训练决策树……

  训练完成：
    标签       = 联合动作
    样本数       = 132
    决策树准确率 = 88.66%          ← 决策树模拟策略的准确程度
    规则条数     = 3
    决策树图片   = output/rule_tree_INF_A_001_agent1_联合动作.png

============================================================
【规则集】
============================================================
共提取到 3 条规则，展示前 3 条：

【规则 1】
IF 飞行状态.高度_km <= 较低（7.51）
     AND 机动状态.燃油剩余 <= 低（0.59）
THEN [('机动控制','追击'), ('武器控制','不发射'), ...]
（支持度: 12，置信度: 1.00）
--------------------------------------------------
【规则 2】
IF 飞行状态.高度_km > 较低（7.51）
THEN [('机动控制','追击'), ('武器控制','不发射'), ...]
（支持度: 8，置信度: 0.60）
--------------------------------------------------
...
```

**关键指标解读：**

- **决策树准确率**：决策树"模仿"智能体策略的准确程度，越高越好（一般 80%+ 可信）
- **支持度**：该规则覆盖了多少个训练样本，越高越可靠
- **置信度**：满足该规则条件的样本中，真的选了这个动作的比例（1.00=100%确定）

---

## 6. 服务B：反事实推理（explain_c）

### 6.1 作用

给定一条具体决策，解释"为什么智能体在这一步做了这个选择"，输出：
- **nl_explanation**（推荐主展示）：问答式因果解释
- **机械性解释**：哪些观测特征是关键原因
- **目的性解释**：当前态势如何，这个决策指向什么目标

反事实分三档（`--cf_level`）：
- `local`：只看决策变不变（最快）
- `one_step`：再看一步奖励
- `multi_step`：再看随后 3～5 步累计奖励（详见 [MULTI_STEP_CF.md](MULTI_STEP_CF.md)）

### 6.2 基本用法

```
py main.py --mode explain_c --inference_task_id INF_A_001 --sim_id SIM_A_0001 --agent_id 1 --decision 机动控制=规避
```

**参数说明：**

| 参数 | 是否必填 | 含义 | 示例 |
|------|----------|------|------|
| `--inference_task_id` | 必填 | 推理任务编号 | `INF_A_001` |
| `--sim_id` | 必填 | 仿真 id（用于定位一局推理数据） | `SIM_A_0001` |
| `--agent_id` | 必填 | 智能体编号 | `1` |
| `--decision` | 必填 | 具体决策内容，格式"动作项=值" | `机动控制=规避` |
| `--cf_level` | 可选 | `local` / `one_step` / `multi_step` | `local` |
| `--horizon` | 可选 | 多步滚动步数（仅 multi_step），3～5 | `5` |
| `--out` | 可选 | 结果 JSON 保存路径 | `output/cf_result.json` |

### 6.3 更多示例

```
# 解释武器控制决策
py main.py --mode explain_c --inference_task_id INF_A_001 --sim_id SIM_A_0001 --agent_id 1 --decision 武器控制=发射导弹

# 多动作项联合决策（逗号分隔）
py main.py --mode explain_c --inference_task_id INF_A_001 --sim_id SIM_A_0001 --agent_id 1 --decision 机动控制=规避,武器控制=不发射

# 保存结果到文件
py main.py --mode explain_c --inference_task_id INF_A_001 --sim_id SIM_A_0001 --agent_id 1 --decision 机动控制=规避 --out output/result.json

# 多步反事实（3～5 步累计奖励）
py main.py --mode explain_c --inference_task_id INF_A_001 --sim_id SIM_A_0001 --agent_id 1 --decision 机动控制=规避 --cf_level multi_step --horizon 4
```

### 6.4 输出说明

运行后控制台会打印：

```
[反事实推理] inference_task_id=INF_A_001  sim_id=SIM_A_0001  agent_id=1
  目标决策：{'机动控制': '规避'}
  正在推理……

  推理完成：
    定位时间步   = t=0              ← 在推理记录第0步找到了这条决策
    真实动作     = 机动控制=规避 武器控制=不发射 ...
    关键特征数   = 5 / 8           ← 8个特征中有5个是关键影响因素

============================================================
【机械性解释】该决策的状态原因分析：

  以下状态特征对本次决策有决定性影响（修改后决策会改变）：
  · 【自身状态.血量】当前值 = 1.000（高）  → 若修改此特征，决策将变为：机动控制=追击 ...
  · 【自身状态.速度_马赫】当前值 = 1.105（高）  → 若修改此特征，决策将变为：...
  · 【飞行状态.高度_km】当前值 = 7.550（高）  → 若修改此特征，决策将变为：...

============================================================
【目的性解释】该决策的意图解读：

  · 策略在当前状态下有一定的鲁棒性，少数关键因素主导了本次决策。

  关键状态因素的当前态势：
  · 【自身状态.血量】处于 高 水平（1.000），自身状态良好，具备主动进攻条件
  · 【飞行状态.高度_km】处于 高 水平（7.550），高度较高，影响机动性能

  综合以上态势，智能体执行【机动控制=规避 ...】的目的性解读：
  当前观测状态促使智能体采取了上述决策，以应对当前威胁。
```

---

## 7. Python API 调用方式

如果你要在自己的 Python 程序中调用本系统（而不是命令行），可以这样做：

### 7.1 规则抽取 API

```python
from src.service import rule_extraction_service

# 调用规则抽取服务
result = rule_extraction_service(
    agent_id=1,
    inference_task_id="INF_A_001",
    n_iters=5,               # VIPER 迭代轮数，越多越准但越慢
    max_depth=6,             # 决策树最大深度，越深规则越细
)

# 访问返回结果
print(result["rules_text"])      # 规则集文本（可直接展示）
print(result["pdf_path"])        # 决策树图片路径
print(result["accuracy"])        # 决策树准确率，如 0.8866
print(result["n_rules"])         # 规则条数
print(result["n_samples"])       # 训练样本数
print(result["feature_names"])   # 特征名列表
print(result["label_name"])      # "联合动作"

# 访问结构化规则对象（可进一步处理）
for rule in result["rules"]:
    print(f"动作: {rule.action}, 支持度: {rule.support}, 置信度: {rule.confidence:.2f}")
    for cond in rule.conditions:
        print(f"  特征{cond.feature_idx} {cond.op} {cond.threshold:.4f}")
```

### 7.2 反事实推理 API

```python
from src.service import counterfactual_service

# 调用反事实推理服务
result = counterfactual_service(
    agent_id=1,
    inference_task_id="INF_A_001",
    sim_id="SIM_A_0001",
    decision_content={"机动控制": "规避"},  # 要解释的具体决策
    top_k=5,             # 最多展示几个关键特征
    max_depth=5,         # 策略近似决策树深度
)

# 访问返回结果
print(result["mechanistic"])              # 机械性解释文本
print(result["teleological"])             # 目的性解释文本
print(result["original_action"])          # 真实动作字符串
print(result["t_query"])                  # 定位到的时间步
print(result["n_key_features_changed"])   # 关键特征数量
print(result["n_features_total"])         # 检验的特征总数

# 访问结构化关键特征列表
for feat_info in result["key_features"]:
    print(feat_info["feature"])   # 特征名
    print(feat_info["value"])     # 当前值
    print(feat_info["label"])     # 语义标签（如"高"/"低"）
    print(feat_info["changed"])   # True=修改后决策变了（关键原因）
    print(feat_info["cf_action"]) # 修改后的反事实动作
```

### 7.3 列出所有可用推理任务

```python
from src.module_c_counterfactual.data_loader import list_inference_task_ids

tasks = list_inference_task_ids()
print(tasks)
# ['INF_A_001', 'INF_A_002', ...]
```

### 7.4 直接加载推理数据

```python
from src.module_c_counterfactual.data_loader import load_inference_record

record = load_inference_record("INF_A_001")

print(record.task_id)            # "INF_A_001"
print(record.total_steps)        # 20（总步数）
print(record.observation_space)  # ["自身状态", "敌机距离", "机动状态"]
print(record.action_space)       # ["机动控制", "武器控制", ...]
print(record.agent_ids)          # [1]

# 查看第3步、智能体1的观测向量
obs_vec = record.get_obs_vector(t=3, agent_id=1)
print(obs_vec)  # [0.8, 7.5, 1.2, 40.0, 0.5, ...]

# 查看第3步、智能体1的特征名
feat_names = record.get_flat_feature_names(agent_id=1)
print(feat_names)  # ["自身状态.血量", "自身状态.高度_km", ...]

# 查看第3步、智能体1的决策
decision = record.get_decision_at(t=3, agent_id=1)
print(decision.content)  # {"机动控制": "追击", "武器控制": "不发射", ...}
```

---

## 8. 参数说明速查表

### 命令行参数

| 参数 | 适用模式 | 含义 | 默认值 |
|------|---------|------|--------|
| `--mode` | 所有 | 运行模式：`explain_a`（规则抽取）或 `explain_c`（反事实推理） | 无（必填） |
| `--inference_task_id` | 两者 | 推理任务 id，如 `INF_A_001` | 无（必填） |
| `--sim_id` | explain_c | 仿真 id（用于定位一局推理数据） | 无（必填） |
| `--agent_id` | 两者 | 智能体编号 | `1` |
| `--decision` | explain_c | 具体决策内容，格式 `动作项=值` | 无（必填） |
| `--out` | 两者 | 输出路径（规则抽取：图片前缀；反事实推理：JSON 路径） | 自动生成到 `output/` |

### 各场景可用的 inference_task_id 与决策字段

| 装备类型 | inference_task_id 范围 | decision_content 可能包含的动作项 |
|----------|-------------|-----------------|
| 歼-20 | INF_A_001 ~ INF_A_005 | 机动控制、武器控制、雷达开关控制、雷达方向控制 |
| 雷达站 | INF_B_001 ~ INF_B_005 | 扫描模式、发射功率控制、目标优先级选择 |
| 侦察机 | INF_C_001 ~ INF_C_005 | 飞行模式、侦察任务控制 |

---

## 9. 常见问题 FAQ

### Q1：运行时报错 "模拟数据文件不存在"

**原因**：还没有生成测试数据。

**解决**：
```
py scripts/generate_mock_data.py
```

---

### Q2：运行时报错 "找不到匹配决策步"

**原因**：`--decision` 里填的决策值在该任务记录里不存在。

**解决**：先查看该任务有哪些可用决策值：
```python
from src.module_c_counterfactual.data_loader import load_inference_record

record = load_inference_record("INF_A_001")
# 查看第1步的决策内容
decision = record.get_decision_at(t=0, agent_id=1)
print(decision.content)   # 看看有哪些动作项和值
```

---

### Q3：决策树图片中文字符显示为方块

**原因**：matplotlib 默认字体不支持中文。

**解决方案（任选一种）**：

方案1（推荐）：安装 graphviz 生成 PDF（PDF 不依赖字体）
```
pip install graphviz
# 再安装系统 Graphviz：https://graphviz.org/download/
```

方案2：在 `src/viz/tree_plot.py` 第174行后添加字体设置：
```python
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False
```

---

### Q4：`agent_id` 应该填什么？

每个任务只有一个智能体，`agent_id` 固定为 `1`（默认值）。
如果自己的数据有多个智能体，运行以下代码查看：
```python
from src.module_c_counterfactual.data_loader import load_inference_record
record = load_inference_record("INF_A_001")
print(record.agent_ids)  # 例如 [1, 2, 3]
```

---

### Q5：我想批量分析多个任务，怎么做？

```python
from src.module_c_counterfactual.data_loader import list_inference_task_ids
from src.service import rule_extraction_service

# 批量分析若干推理任务（联合动作单树）
for inference_task_id in list_inference_task_ids()[:10]:
    try:
        result = rule_extraction_service(
            agent_id=1,
            inference_task_id=inference_task_id,
        )
        print(f"{inference_task_id}: 准确率={result['accuracy']:.2%}, 规则数={result['n_rules']}")
    except Exception as e:
        print(f"{inference_task_id}: 失败 - {e}")
```

---

### Q6：准确率太低（低于 60%）怎么办？

尝试调整决策树参数：
```python
result = rule_extraction_service(
    agent_id=1,
    inference_task_id="INF_A_001",
    max_depth=8,             # 加深决策树（默认6）
    min_samples_leaf=1,      # 减小叶节点最小样本数（默认2）
    n_iters=10,              # 增加迭代轮数（默认5）
)
```

---

## 10. 代码文件索引

以下是各核心文件的功能和关键函数说明，方便开发者查阅：

### src/module_c_counterfactual/inference_record.py

**功能**：定义所有数据结构（推理记录的"模型"）

| 类/函数 | 作用 |
|---------|------|
| `AgentMeta` | 智能体元信息（id、名称、装备类型） |
| `ActionItem` | 动作项定义（名称 + 可选值列表） |
| `AgentDecision` | 某步某智能体的决策内容 |
| `AgentObservation` | 某步某智能体的观测内容（嵌套字典） |
| `AgentObservation.get_flat_vector()` | 把嵌套观测展平成一维数组 |
| `AgentObservation.get_flat_feature_names()` | 返回展平后的特征名（格式：观测项.子字段） |
| `InferenceRecord` | 一次完整仿真的推理数据（主容器） |
| `InferenceRecord.locate_decision_step()` | 根据决策内容定位时间步 |

---

### src/module_c_counterfactual/data_loader.py

**功能**：从数据库（或本地 JSON）加载推理数据

| 函数 | 作用 |
|------|------|
| `load_inference_record(task_id)` | 加载指定任务的推理记录 |
| `list_available_tasks()` | 列出所有可用任务的 id |
| `MOCK_MODE` | True=本地 JSON，False=Doris 数据库 |

---

### src/module_a_rules/collect_data.py

**功能**：从推理记录中提取 (观测, 动作, 奖励) 训练样本

| 函数 | 作用 |
|------|------|
| `collect_from_record(record, agent_id, action_item)` | 提取单条记录的训练样本 |
| `collect_from_records(records, agent_id, action_item)` | 提取多条记录的合并样本 |
| `compute_return_to_go(rewards, gamma)` | 计算 return-to-go 权重 |

---

### src/module_a_rules/preprocess.py

**功能**：特征归一化 + 自动分箱（把数值转成语义标签）

| 类/方法 | 作用 |
|---------|------|
| `Preprocessor` | 预处理器类 |
| `Preprocessor.fit(X)` | 从训练数据学习归一化参数和分箱边界 |
| `Preprocessor.transform(X)` | 对数据做 z-score 归一化 |
| `Preprocessor.fit_transform(X)` | 先 fit 再 transform 的快捷方式 |
| `Preprocessor.discretize_label(feat_name, value)` | 把数值转成语义标签（如"低"/"高"） |
| `Preprocessor.denormalize_threshold(feat_idx, thresh)` | 把归一化阈值还原为原始单位 |
| `Preprocessor.get_bin_summary()` | 查看所有特征的分箱配置 |

---

### src/module_a_rules/viper.py

**功能**：VIPER 算法训练决策树（迭代加权）

| 类/方法 | 作用 |
|---------|------|
| `VIPERData` | VIPER 训练器 |
| `VIPERData.from_record(record, agent_id, action_item)` | 从单条记录构建 |
| `VIPERData.from_records(records, agent_id, action_item)` | 从多条记录构建 |
| `VIPERData.run(n_iters, penalty_factor)` | 执行迭代训练，返回最佳决策树 |
| `VIPERResult` | 训练结果（决策树 + 预处理器 + 历史记录） |

---

### src/module_a_rules/extract_rules.py

**功能**：从决策树 DFS 提取 IF-THEN 规则

| 类/函数 | 作用 |
|---------|------|
| `Rule` | 一条规则（条件列表 + 动作 + 支持度 + 置信度） |
| `RuleCondition` | 一个条件（特征索引 + 运算符 + 阈值） |
| `extract_rules_from_tree(tree, preprocessor)` | 从决策树提取所有规则 |
| `rules_to_text(rules, preprocessor, top_k)` | 把规则列表转成可读文本 |

---

### src/module_a_rules/merge_rules.py

**功能**：合并冗余规则，减少规则数量

| 函数 | 作用 |
|------|------|
| `merge_rules(rules)` | 对规则集进行两轮合并（子集合并 + 区间合并） |
| `rules_coverage(rules, X, y)` | 评估规则集在数据上的准确率 |

---

### src/module_c_counterfactual/policy_model.py

**功能**：训练策略近似决策树（用于反事实推理）

| 类/方法 | 作用 |
|---------|------|
| `PolicySurrogate` | 策略近似决策树模型 |
| `PolicySurrogate.fit(record, agent_id)` | 从推理记录训练决策树 |
| `PolicySurrogate.predict(obs_vector)` | 预测给定观测的动作 |
| `PolicySurrogate.feature_importances()` | 返回特征重要性 |

---

### src/module_c_counterfactual/counterfactual.py

**功能**：局部反事实推理算法

| 类/函数 | 作用 |
|---------|------|
| `CFContext` | 反事实推理上下文（包含真实观测和真实动作） |
| `LocalCFResult` | 单个特征的反事实检验结果 |
| `perturb_obs_features()` | 对观测特征进行扰动（生成反事实观测） |
| `local_counterfactual(ctx, model)` | 局部反事实推理主函数 |

---

### src/module_c_counterfactual/rollback.py

**功能**：根据前端输入定位决策时间步，构建 CFContext

| 类/方法 | 作用 |
|---------|------|
| `ObservationRollback` | 决策回溯控制器 |
| `ObservationRollback.from_frontend_input()` | 根据决策内容定位时间步，返回 CFContext |
| `ObservationRollback.build_context()` | 根据时间步构建 CFContext |

---

### src/module_c_counterfactual/explain_nl.py

**功能**：把反事实推理结果渲染为自然语言解释

| 函数 | 作用 |
|------|------|
| `render_cf_explanation()` | 生成机械性解释 + 目的性解释 |

---

### src/service.py

**功能**：两个服务的统一入口（推荐从这里调用）

| 函数 | 作用 |
|------|------|
| `rule_extraction_service()` | 服务A：规则抽取 |
| `counterfactual_service()` | 服务B：反事实推理 |

---

## 11. 连接真实数据库

当需要接入生产环境的 Doris 数据库时，请按以下步骤操作：

### 11.1 安装 PyMySQL 驱动

```
pip install pymysql
```

### 11.2 修改配置

编辑 `src/module_c_counterfactual/data_loader.py`：

```python
# 第66行：改为 False（切换到真实数据库模式）
MOCK_MODE = False

# 第72-79行：填写你的数据库连接信息
DORIS_CONFIG = {
    "host": "192.168.1.100",   # 数据库服务器 IP
    "port": 9030,               # Doris 默认端口
    "user": "admin",            # 数据库用户名
    "password": "your_pass",    # 数据库密码
    "database": "simulation_db", # 数据库名
    "charset": "utf8mb4",
}
```

### 11.3 创建数据库表

在 Doris 中创建以下两张表：

```sql
-- 任务信息表
CREATE TABLE inference_task (
    task_id           VARCHAR(64)  NOT NULL,
    sim_id            VARCHAR(64),
    agents_json       TEXT,          -- JSON 格式智能体列表
    observation_space TEXT,          -- JSON 格式观测项列表
    action_items_json TEXT,          -- JSON 格式动作项定义
    total_steps       INT
) UNIQUE KEY(task_id)
DISTRIBUTED BY HASH(task_id) BUCKETS 1;

-- 步骤流水表
CREATE TABLE inference_step (
    task_id       VARCHAR(64)  NOT NULL,
    step          INT          NOT NULL,
    agent_id      INT          NOT NULL,
    decision_json TEXT,                -- JSON 格式决策字典
    obs_json      TEXT,                -- JSON 格式嵌套观测字典
    reward        DOUBLE
) DUPLICATE KEY(task_id, step, agent_id)
DISTRIBUTED BY HASH(task_id) BUCKETS 8;
```

### 11.4 实现真实查询逻辑

打开 `src/module_c_counterfactual/data_loader.py`，找到 `_load_from_doris` 函数（约第243行），
按照注释中的 TODO 提示，用 PyMySQL 实现查询逻辑（模板代码已在注释中提供）。

---

*文档版本：v1.0 | 最后更新：2026-05-29*
