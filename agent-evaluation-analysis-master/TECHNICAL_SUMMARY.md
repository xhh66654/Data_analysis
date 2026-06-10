    # 空战仿真溯因分析系统 — 技术总结文档

> 本文档面向其他模型或开发者阅读，系统性说明两个核心模块（规则抽取 Module A、反事实推理 Module C）的实现流程、数据来源、输入输出、算法细节与论文来源。

---

## 目录

1. [系统概览](#1-系统概览)
2. [数据来源与数据结构](#2-数据来源与数据结构)
3. [Module A：规则抽取](#3-module-a规则抽取)
4. [Module C：反事实推理](#4-module-c反事实推理)
5. [两模块的共用组件](#5-两模块的共用组件)
6. [调用接口总结](#6-调用接口总结)
7. [论文来源与算法溯源](#7-论文来源与算法溯源)
8. [关键设计约束与决策](#8-关键设计约束与决策)

---

## 1. 系统概览

本系统是一个**多智能体空战仿真场景下的溯因分析系统**，用于解释强化学习智能体（如歼-20、雷达站、侦察机）在某一时刻为什么做出某个决策。

系统提供两种解释能力：

| 模块 | 功能 | 核心算法 |
|------|------|----------|
| Module A：规则抽取 | 从历史推理轨迹提取 IF-THEN 决策规则，解释智能体整体策略 | VIPER + CART 决策树 + DFS 规则提取 |
| Module C：反事实推理 | 对某一具体决策步解释「为何如此决策」；支持 local / one_step / multi_step 三档 | 决策树 + 可选 π/T/R 代理 |

系统入口：
- **Python API**：`from src.service import rule_extraction_service, counterfactual_service`
- **命令行**：`py main.py --mode explain_a / explain_c ...`

---

## 2. 数据来源与数据结构

### 2.1 数据来源

| 模式 | 说明 |
|------|------|
| **MOCK 模式**（开发/测试） | 读取本地 JSON 文件，目录：`data/mock_records/`，`MOCK_MODE=True` |
| **生产模式** | 查询 Doris 数据库，以 `inference_task_id` 拉取该任务下全部推理数据 |

**业务语义**：

- **推理任务**（`inference_task_id` / 表字段 `task_id`）：用户配置并运行的一次推理任务，可产生多条推理数据
- **推理数据**（`sim_id`）：一次完整仿真/对局的记录，`inference_task` 表每行对应一条推理数据

Mock 数据由 `scripts/generate_mock_data.py` 生成，覆盖三类场景：

| 场景 | inference_task_id 范围 | 每任务仿真局数 | 智能体 | 可用动作项 |
|------|------------------------|---------------|--------|-----------|
| 歼-20 战斗机 | INF_A_001 ~ INF_A_005 | 3 局 | Alpha编队、Bravo编队 | 机动控制 / 武器控制 / 雷达开关控制 / 雷达方向控制 |
| 雷达站 | INF_B_001 ~ INF_B_005 | 2 局 | 雷达站_A、雷达站_B | 功率调节 / 扫描模式 / 发射干扰 |
| 侦察机 | INF_C_001 ~ INF_C_005 | 2 局 | 侦察机_X | 飞行模式 / 传感器模式 |

另有边界测试任务 `INF_A_SINGLE`（仅 1 局仿真）。共 16 个推理任务、36 条推理数据。

**数据库表结构**：

| 表 | 说明 |
|----|------|
| `inference_task.json` | 推理数据基础信息，每行 = 一条推理数据（`task_id` + `sim_id` 唯一标识一局） |
| `inference_step.json` | 步骤流水，字段含 `task_id`、`sim_id`、`step`、`agent_id`、`decision_json`、`obs_json`、`reward` |

### 2.2 核心数据结构：InferenceRecord

`src/module_c_counterfactual/inference_record.py`

```
InferenceRecord                         # 一条推理数据 = 一局完整仿真
├── task_id: str                        # 关联推理任务 id，e.g. "INF_A_001"
├── sim_id: str                         # 推理数据 id / 仿真 id，e.g. "SIM_A_0001"
├── agents: List[AgentMeta]             # 智能体元信息（agent_id, agent_name, equipment_type）
├── observation_space: List[str]        # 观测项名称列表，e.g. ["自身状态", "敌机距离"]
├── action_items: List[ActionItem]      # 动作项定义（name, possible_values, is_continuous）
├── decisions: List[StepDecision]       # 每步所有智能体的决策
├── observations: List[StepObservation] # 每步所有智能体的观测（嵌套字典）
├── total_steps: int                    # 总决策步数
├── rewards: List[float]                # 每步全局奖励
└── 辅助方法与属性：
    ├── agent_ids                       → List[int]（所有智能体 id）
    ├── get_obs_vector(t, agent_id)     → List[float]（展平后的观测向量）
    ├── get_action_value(t, agent_id, action_item) → str
    ├── get_flat_feature_names(agent_id) → List[str]（展平特征名）
    └── get_decision_at(t, agent_id)    → AgentDecision
```

### 2.3 观测展平与一步决策契约

**一步决策**：`obs_json` = 该智能体**完整观测空间**；`decision_json` = 该智能体**完整动作空间**。

智能体定稿后 `observation_space`、`action_items`、`equipment_units`（装备个体 id 列表）固定不变。

**单装备个体**（`equipment_units=["__default__"]`）：扁平 JSON，特征名 `观测项.子字段`。

**多装备个体**（同模板复制）：嵌套 JSON `unit_id -> 观测项 -> 子字段`，特征名 `unit_id.观测项.子字段`。

```
单个体：
  "自身状态": {"血量": 0.8, ...}  →  自身状态.血量

多个体：
  "alpha_1": {"自身状态": {"血量": 0.8}}, "alpha_2": {...}
  →  alpha_1.自身状态.血量, alpha_2.自身状态.血量, ...
```

动作标签默认使用 `canonical_decision_label`（全动作空间稳定 JSON 字符串）。

> **重要约束**：观测/动作字段完全动态，不得硬编码特征名、动作项名或分箱边界。

---

## 3. Module A：规则抽取

### 3.1 功能概述

给定**智能体 id**（`agent_id`）与**推理任务 id**（`inference_task_id`），加载该推理任务下的全部推理数据，提取指定智能体在各局中的全部历史推理步骤，使用**全动作空间标签**（`canonical_decision_label`，完整 `decision_content`）训练一棵 CART 决策树来近似该智能体策略，然后从这棵树中提取 IF-THEN 规则集，以人类可读文本输出。可选 `action_item` / `unit_id` 训练单维度或单装备个体子树。

### 3.1.1 输入参数语义

| 参数 | 类型 | 必填 | 含义 |
|------|------|------|------|
| `agent_id` | int | 是 | 要分析的智能体 id |
| `inference_task_id` | str | 是 | 推理任务 id，用于拉取该任务下全部推理数据 |
`agent_id` 与 `inference_task_id` 是溯源 id；系统据此定位数据，再按智能体过滤并合并多局样本进行联合动作分析。

### 3.2 完整处理流程

```
输入：agent_id, inference_task_id
         ↓
[步骤1] data_loader.load_inference_records(inference_task_id)
         → 加载该推理任务下全部 InferenceRecord（多条，每局一条）
         ↓
[步骤2] collect_data.collect_from_records(records, agent_id, action_item=None)
         → 遍历各局每步，展平观测向量 → X_raw (T×F)，跨局合并
         → 取完整 decision_content 的联合标签 → y (T,)
         → 取全局奖励 → rewards (T,)
         → 返回特征名列表 feature_names
         ↓
[步骤3] collect_data.compute_return_to_go(rewards)
         → 计算每步 return-to-go：RTG[t] = Σ_{k≥t} γ^(k-t) · r[k]
         → 归一化到 (0,1] + 1e-6 偏移，作为样本权重
         ↓
[步骤4] preprocess.Preprocessor.fit_transform(X_raw)
         → Z-score 归一化（μ/σ）
         → 自动分位数分箱（默认3分位 → 4个区间）
         → 生成语义标签（极低/低/高/极高等）
         ↓
[步骤5] viper.VIPERData.run(n_iters=5)
         → VIPER 迭代加权训练：
           初始权重 = RTG
           循环 n_iters 次：
             1. 用当前样本权重拟合 CART 决策树
             2. 计算加权准确率
             3. 对预测错误样本乘 penalty_factor（默认2.0）放大权重
             4. 重新归一化权重
           → 保留加权准确率最高的树
         ↓
[步骤6] extract_rules.extract_rules_from_tree(best_tree, preprocessor)
         → DFS 深度优先遍历决策树
         → 每条根→叶路径 = 一条 IF-THEN 规则
         → 规则包含：条件列表、预测动作、支持度、置信度
         → 条件阈值反归一化 + 语义标签（如"≤ 低（7.51）"）
         ↓
[步骤7] merge_rules.merge_rules(raw_rules)
         → 子集合并：若规则A的条件集合是规则B的真子集且动作相同，删除B
         → 区间合并：同动作相邻规则若仅阈值不同，合并为更宽松的规则
         → 按支持度从高到低排序
         ↓
[步骤8] viz.tree_plot.export_tree_pdf(...)
         → 用 matplotlib 导出决策树可视化 PNG
         → 配置中文字体（Microsoft YaHei / SimHei / FangSong）
         ↓
输出：rules_text + rules + accuracy + pdf_path + n_records + sim_ids + ...
```

### 3.3 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_iters` | 5 | VIPER 迭代轮数 |
| `max_depth` | 6 | 决策树最大深度 |
| `min_samples_leaf` | 2 | 叶节点最少样本数 |
| `gamma` | 1.0 | RTG 折扣因子（当前为1，不折扣） |
| `penalty_factor` | 2.0 | VIPER 错误样本惩罚放大系数 |
| `n_quantiles` | 3 | 分箱分位数数量（3 → 4区间） |

### 3.4 输出格式示例

```
【规则 1】
IF 自身状态.高度_km <= 低（7.51）
     AND 敌机状态.燃油剩余率 <= 低（0.59）
THEN 追击
（支持度: 1，置信度: 1.00）
```

### 3.5 涉及文件

| 文件 | 职责 |
|------|------|
| `src/module_a_rules/collect_data.py` | 从 InferenceRecord 提取训练样本、计算 RTG |
| `src/module_a_rules/preprocess.py` | 归一化预处理器 + 自动分位数分箱 |
| `src/module_a_rules/viper.py` | VIPER 迭代加权决策树训练 |
| `src/module_a_rules/extract_rules.py` | DFS 规则提取，Rule/RuleCondition 数据类 |
| `src/module_a_rules/merge_rules.py` | 子集合并 + 区间合并 + 覆盖率评估 |
| `src/viz/tree_plot.py` | 决策树 PNG 可视化 |

---

## 4. Module C：反事实推理

### 4.1 功能概述

给定一条具体决策（`inference_task_id` + `sim_id` + `agent_id` + `decision_content`），定位到决策时刻 `t_query`，逐一**单特征扰动**观测，找出关键状态因素，并生成：

- **机械性解释**：哪些态势因素推动了这次决策
- **目的性解释**：这次决策指向什么战术收益/走势
- **nl_explanation**（推荐主展示）：问答式「为什么 / 回答 / 或者回答」

**三档反事实（`cf_level`）均已实现：**

| cf_level | 模型需求 | 向前看多远 | 主要对比指标 |
|----------|----------|------------|--------------|
| `local` | 仅 π（决策树） | 0（只看当前步决策） | 动作是否改变 |
| `one_step` | π + T + R | 1 步 | 单步奖励 `r_t` |
| `multi_step` | π + T + R | 3～5 步（`horizon`） | 后续 H 步**累计奖励** |

说明：一步/多步使用 `SurrogateBundle` 在**全任务**记录上训练代理模型；解释时按 `sim_id` 定位单局。多步为代理 rollout，非环境重仿真。

### 4.2 完整处理流程（`counterfactual_service`）

```
输入：inference_task_id, sim_id, agent_id, decision_content, cf_level, horizon(可选)
         ↓
[1] load_inference_record(task, sim)  → 定位用单局
[2] load_inference_records(task)      → 训练代理用全任务多局
[3] ObservationRollback → CFContext(t_query, obs_t, action_t)
[4] SurrogateBundle.fit(records_all, agent_id)   # one_step / multi_step
    或仅 PolicySurrogate（local，在单局/全任务上训练策略树）
         ↓
[5] 反事实（每个观测特征各扰动一次）：
    local       → local_counterfactual
    one_step    → one_step_counterfactual（π→T→R，比 r_t）
    multi_step  → multi_step_counterfactual（π→T→R 滚 H=3~5 步，比累计奖励）
         ↓
[6] fit_preprocessor_with_profile(records_all) → 与 Module A 共用 per-agent 语义标尺
[7] render_*_explanation + attach_natural_language_qa → nl_explanation
[8] 可选 enhance_cf_explanation（本地 LLM 润色）
         ↓
输出：nl_explanation, key_features, mechanistic, teleological, ...
      multi_step 额外：horizon, original_cumulative_reward,
                      original_action_seq, cf_action_seq, disclaimer
      profile 元数据：agent_profile_version, surrogate_profile_hit,
                      surrogate_profile_version, surrogate_cache_hit
```

### 4.2.1 三层 per-agent 架构

| 层 | 存储 | 作用 |
|----|------|------|
| **π/T/R 实例** | 进程内缓存 + 可选磁盘 joblib | 每 `(agent_id, schema)` 独立拟合；`local` 仅 π |
| **AgentPreprocessorProfile** | `output/agent_profiles/` | 稳定语义标尺（远/近/高/低）；与 Module A 共用 |
| **AgentSurrogateProfile** | `output/agent_surrogate_profiles/` | 转移 reservoir + 跨 task 增量重训 |

- **增量含义**：reservoir 合并 + 决策树重训，**非**神经网络权重热更新。
- **查找链**：memory cache → disk profile（`seen_fingerprints` 命中则直接 load joblib）→ merge reservoir refit → 写回。
- **环境变量**：`ANALYSIS_CF_SURROGATE_PROFILE=0` 关闭磁盘 profile；`AGENT_SURROGATE_MAX_RESERVOIR` 默认 2000。

### 4.2.2 多步反事实（`multi_step`）要点

- **事实基线**：仿真记录中 `t_query .. t_query+H-1` 的真实奖励之和 + 真实动作序列。
- **反事实**：仅扰动 `t_query` 的一个特征 → 从 `s'_t` 起用 π/T/R **自回归**滚 H 步 → 累计代理奖励。
- **对比**：`reward_delta = cf_cumulative - factual_cumulative`；`action_changed` 看首步动作是否变化。
- **K 采样（可选）**：`use_k_sampling=True`（one_step/multi_step 默认）在代理模型上做 K 次扰动 + 表 2 效应量；标量奖励 only。
- **未实现**：联合多特征扰动、环境重仿真、向量奖励（数据侧未提供）。

### 4.3 action_t 标签格式

`action_t` 是该时间步智能体**所有动作项**的联合字符串标签：

```python
action_t = str(sorted(dec.content.items()))
# e.g. "[('机动控制', '追击'), ('武器控制', '不发射'), ('雷达开关控制', '开'), ('雷达方向控制', '前方')]"
```

PolicySurrogate 预测返回的也是同格式字符串，二者对比是否相等来判断 `action_changed`。

### 4.4 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_depth` | 5 | 策略近似树最大深度 |
| `min_samples_leaf` | 5 | 策略近似树叶节点最少样本数 |
| `top_k` | 5 | 展示几个关键特征 |
| `cf_level` | `"local"` | `local` / `one_step` / `multi_step` |
| `horizon` | — | 仅 multi_step，3～5，默认 5 |
| 扰动策略 | 默认 `train_mean` | 可用 `zero`；`bidirectional_perturb` 对 local 做双向敏感性 |
| `use_k_sampling` | one/multi 默认 True | K 次代理采样 + 表2；`local` 默认 False |
| `k_samples` | 100 | K 采样次数上限 500 |
| `alternative_decision_content` | — | 可选，输出 A vs B 动作标签对照 |
| `update_agent_profile` | True | 是否更新 `output/agent_profiles/` |
| `update_surrogate_profile` | True | 是否更新 `output/agent_surrogate_profiles/` |

### 4.5 输出示例

**机械性解释：**
```
【机械性解释】该决策（机动控制=追击  武器控制=不发射  ...）的状态原因分析：

  以下状态特征对本次决策有决定性影响（修改后决策会改变）：
  · 【自身状态.高度_km】当前值 = 7.249（低）  → 若修改此特征，决策将变为：机动控制=规避  ...
  · 【敌机距离.水平距离_km】当前值 = 91.651（极高）  → 若修改此特征，决策将变为：...
```

**目的性解释：**
```
【目的性解释】该决策（机动控制=追击  ...）的意图解读：

  · 策略在当前状态下有一定的鲁棒性，少数关键因素主导了本次决策。

  关键状态因素的当前态势：
  · 【敌机距离.水平距离_km】处于 极高 水平（91.651），目标距离较远，处于侦察/接近阶段

  综合以上态势，智能体执行【...】的目的性解读：
  当前观测状态促使智能体采取了上述决策，以应对当前威胁。
```

### 4.6 涉及文件

| 文件 | 职责 |
|------|------|
| `src/module_c_counterfactual/inference_record.py` | 核心数据结构，观测/动作/决策的容器与辅助方法 |
| `src/module_c_counterfactual/data_loader.py` | 数据入口，`MOCK_MODE` 开关，加载 JSON 或 Doris |
| `src/module_c_counterfactual/rollback.py` | 前端输入 → CFContext，定位 t_query |
| `src/module_c_counterfactual/policy_model.py` | PolicySurrogate：CART 决策树策略近似 |
| `src/module_c_counterfactual/surrogate_bundle.py` | π+T+R 三模型统一训练（one_step / multi_step） |
| `src/module_c_counterfactual/counterfactual.py` | local / one_step / multi_step 反事实核心 |
| `src/module_c_counterfactual/cf_dataset.py` | K 采样数据集 `generate_cf_dataset` |
| `src/module_c_counterfactual/causal_effect.py` | 表 2 机械/目的效应量（标量） |
| `src/module_c_counterfactual/cema.py` | CEMA 薄封装编排 |
| `src/module_c_counterfactual/surrogate_cache.py` | 进程内缓存 + 磁盘 profile 查找链 |
| `src/module_c_counterfactual/agent_surrogate_profile.py` | Surrogate 转移 reservoir 与 joblib 持久化 |
| `src/module_c_counterfactual/explain_nl.py` | 解释渲染 + nl_explanation 问答 |
| `src/service.py` | `counterfactual_service` 对外入口 |

---

## 5. 两模块的共用组件

### 5.1 RTG（Return-to-Go）样本权重

定义：

```
RTG[t] = r[t] + γ·r[t+1] + γ²·r[t+2] + ...   （γ=1.0）
归一化到 (0,1] + 1e-6 偏移
```

用途：
- Module A 的 VIPER 训练：初始样本权重，越靠近高回报时刻的决策权重越大
- Module C 的 PolicySurrogate 训练：同上

实现：`src/module_a_rules/collect_data.py::compute_return_to_go()`

### 5.2 Preprocessor（特征预处理器）

- **归一化**：Z-score，`(x - μ) / σ`，σ=0 时置1
- **自动分箱**：3分位数 → 4个区间 → 语义标签（极低/低/高/极高）
- **反归一化**：`thresh * σ + μ`，用于规则文本中展示原始值
- **动态**：所有分箱边界从训练数据分位数自动推断，无硬编码

实现：`src/module_a_rules/preprocess.py::Preprocessor`

### 5.3 数据加载器

`src/module_c_counterfactual/data_loader.py`

| 函数 | 说明 |
|------|------|
| `load_inference_records(inference_task_id)` | 加载推理任务下全部推理数据（规则抽取主入口） |
| `load_inference_record(task_id)` | 加载第一条推理数据（兼容旧接口） |
| `list_inference_task_ids()` | 列出可用推理任务 id（去重） |

- `MOCK_MODE=True`：从 `data/mock_records/inference_task.json` + `inference_step.json` 读取
- `MOCK_MODE=False`：查询 Doris（按 `inference_task_id` 查多行，step 表按 `task_id` + `sim_id` 过滤）

---

## 6. 调用接口总结

### 6.1 Python API

```python
from src.service import rule_extraction_service, counterfactual_service

# Module A：规则抽取（联合动作，单棵树）
result = rule_extraction_service(
    agent_id=1,
    inference_task_id="INF_A_001",
)
# result["rules_text"]         → 文本规则集
# result["pdf_path"]           → 决策树 PNG 路径
# result["accuracy"]           → 决策树加权准确率
# result["n_rules"]            → 规则数量
# result["n_records"]          → 合并了几条推理数据
# result["sim_ids"]            → 参与训练的仿真 id 列表
# result["rules"]              → List[Rule] 结构化规则

# Module C：反事实推理（按 inference_task_id + sim_id 精确定位）
result = counterfactual_service(
    agent_id=1,
    inference_task_id="INF_A_001",
    sim_id="SIM_A_0001",
    decision_content={"机动控制": "规避"},
)
# result["mechanistic"]          → 机械性解释文本
# result["teleological"]         → 目的性解释文本
# result["key_features"]         → 关键特征列表
# result["t_query"]              → 定位到的时间步
# result["n_key_features_changed"] → 关键因素数量
```

### 6.2 命令行

```bash
# Module A：规则抽取
py main.py --mode explain_a --inference_task_id INF_A_001 --agent_id 1

# Module C：反事实推理
py main.py --mode explain_c --inference_task_id INF_A_001 --sim_id SIM_A_0001 --agent_id 1 --decision 机动控制=规避
```

### 6.3 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MOCK_MODE` | `True` | `True` 读本地 JSON，`False` 连接 Doris |
| `ANALYSIS_OUTPUT_DIR` | `./output` | 输出文件目录 |
| `DORIS_HOST` / `DORIS_PORT` / `DORIS_USER` / `DORIS_PASSWORD` | — | Doris 数据库连接参数（生产模式） |

---

## 7. 论文来源与算法溯源

### 7.1 VIPER（Verifiable Imitation learning via Policy Extraction for Reinforcement）

> **论文**：Bastani, O., Pu, Y., & Solar-Lezama, A. (2018).
> *Verifiable Reinforcement Learning via Policy Extraction.*
> NeurIPS 2018. [arXiv:1805.08328](https://arxiv.org/abs/1805.08328)

**原始算法**：
- 在线与策略（on-policy）数据收集：通过真实环境运行神经网络策略，采样高回报轨迹
- 用神经网络的动作分布（概率标注）作为教师标签
- 用 DAgger 风格的迭代数据聚合训练 CART 决策树
- 保证决策树策略的可验证性（有界次优性定理）

**本系统的近似实现**（`src/module_a_rules/viper.py`）：
- **无真实环境**：改用历史推理记录（离线数据）作为数据来源
- **无神经网络教师**：直接用历史动作标签（硬标注）替代软概率标注
- **保留核心思想**：用 return-to-go 作为样本权重代替 DAgger 的重要性采样
- **迭代加权**：对预测错误的样本放大权重（penalty_factor=2.0），多轮迭代

与原论文的偏差是工程化妥协：生产环境无法重跑仿真环境，离线数据不包含神经网络输出的软标签。

### 7.2 CART 决策树

> **论文**：Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984).
> *Classification and Regression Trees.* Wadsworth.

实现：`sklearn.tree.DecisionTreeClassifier`，Gini 不纯度准则。

### 7.3 局部反事实解释

> **论文**：Wachter, S., Mittelstadt, B., & Russell, C. (2017).
> *Counterfactual Explanations Without Opening the Black Box.*
> Harvard Journal of Law & Technology, 31(2). [arXiv:1711.00399](https://arxiv.org/abs/1711.00399)

**原始方法**：最小化特征变动量，寻找能改变模型决策的最近反事实样本。

**本系统实现**（`src/module_c_counterfactual/counterfactual.py`）：
- 采用**单特征归零扰动**（而非优化搜索），逐一将每个特征置零
- 对比扰动前后决策树预测是否变化，判断特征是否为关键因素
- 这是反事实解释的简化版：牺牲最小扰动保证，换取计算效率和可解释性

### 7.4 机械性解释 vs 目的性解释框架

> **参考**：Woodward, J. (2003). *Making Things Happen: A Theory of Causal Explanation.*
> Oxford University Press.

以及：

> Miller, T. (2019). *Explanation in Artificial Intelligence: Insights from the Social Sciences.*
> Artificial Intelligence, 267, 1-38.

本系统将解释分为两类：
- **机械性（Mechanistic）**：回答"是什么状态因素导致了这个决策"——因果前件分析
- **目的性（Teleological）**：回答"做这个决策是为了什么"——从意图/目标角度解读

### 7.5 Return-to-Go 样本权重

> **参考**：Chen, L., et al. (2021).
> *Decision Transformer: Reinforcement Learning via Sequence Modeling.*
> NeurIPS 2021. [arXiv:2106.01345](https://arxiv.org/abs/2106.01345)

RTG 的概念来自 Decision Transformer，本系统借用 RTG 作为样本重要性权重，使决策树训练更关注高回报轨迹中的决策。

---

## 8. 关键设计约束与决策

| 约束 | 说明 |
|------|------|
| **完全动态的观测/动作空间** | 不同场景（歼-20/雷达站/侦察机）字段完全不同，系统不硬编码任何特征名或动作项名 |
| **展平键格式** | 统一为 `"观测项.子字段"`，子字段按字母序排列，全项目一致 |
| **action_label 格式** | `str(sorted(dec.content.items()))`，PolicySurrogate 和 rollback 三处保持一致 |
| **叶节点判断** | 使用 `TREE_LEAF = -1`（非 `TREE_UNDEFINED = -2`），这是 sklearn 的正确常量 |
| **不依赖真实环境** | Module A 和 Module C 均基于 `InferenceRecord` 工作，无需 `AirCombatEnv` |
| **PNG 替代 PDF** | 不强制要求 graphviz，使用 matplotlib + 中文字体配置输出 PNG |
| **Module C 当前阶段** | 三档反事实 + K 采样表2（标量）+ CEMA 薄封装；仅 `rewards[t]` 标量；向量奖励待上游 |
| **K 采样** | `use_k_sampling`, `k_samples`, `k_noise_scale`, `k_seed`；`local` 默认 `use_k_sampling=False` |
| **networkx 可选** | `src/viz/__init__.py` 用 `try/except` 保护，缺失不影响运行 |

---

*文档更新时间：2026-06-01。项目工作目录：`E:\工作\analysis\`。*
