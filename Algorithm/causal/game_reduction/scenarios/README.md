# 场景配置编写指南

## 快速开始

本目录存放各类环境的战术级解释配置文件。每个环境一个 YAML 文件。

### 文件命名约定

```
scenario_<environment_name>.yaml

示例：
  scenario_simple_spread.yaml      # simple_spread 环境
  scenario_predator_prey.yaml      # predator_prey 环境
  scenario_navigation.yaml         # 导航任务环境
```

---

## 配置文件结构详解

### 1. 基本信息

```yaml
scenario_name: "简单分散任务（Simple Spread v3）"
description: "多智能体目标分散任务环境。..."
```

- `scenario_name`：便于在输出中识别
- `description`：环境和任务的简短说明

---

### 2. 块语义映射 (`blocks_meaning`)

**从 SCM 技术块名映射到战术概念**。

块名来源：`scm_causal_edges.json` 中的 `parent_block` 字段。

```yaml
blocks_meaning:
  s_target:
    zh_name: "己方智能体当前状态"              # 战术名
    zh_description: "目标智能体自身的位置、速度..."  # 详细说明
    interpretation_hint: "反映己方当前的处境..."    # 在推理中的作用
```

**编写建议**：
1. 列出所有来自 `scm_causal_edges.json` 的父块
2. 为每个块定义**简洁的**战术名称（3~8 字）
3. 说明该块代表的是"威胁"、"支援"、还是"环境"

---

### 3. 行为标签语义 (`behavior_semantics`)

**从原型标签映射到战术行为**。

标签来源：`behavior_prototypes.json` 中的键名。

```yaml
behavior_semantics:
  escape-ish:
    zh_label: "撤退规避"             # 战术标签
    zh_description: "倾向于远离..."    # 具体含义
    intensity_order: 1               # 强度等级（0=被动, 3=激进）
```

**编写建议**：
1. 对每个行为原型，给出**一个 2~4 字的标签**
2. `intensity_order` 用于在多行为时排序重要性
3. 选择符合领域认知的标签（军事、运动、等等）

---

### 4. 强度等级 (`intensity_descriptions`)

**L2 变化量 → 强度等级 → 描述词**。

```yaml
intensity_descriptions:
  l2_thresholds: [0.0, 0.3, 0.6, 1.0]
  levels:
    0:
      name: "无关"
      zh_phrase: ["无关", "无影响", "可忽略"]
      emphasis: "该因素在此时刻的影响可以忽略"
```

**编写建议**：
1. `l2_thresholds` 定义 4 个强度等级的分界点
   - `[0.0, 0.3, 0.6, 1.0]`：推荐默认值
   - 可根据实际数据分布调整（查看历史 L2 变化的分布）
2. 每个等级提供 2~3 个同义词，便于随机选择

---

### 5. 强度词库 (`intensity_phrases`)

**快速映射表，用于文本生成时查表**。

```yaml
intensity_phrases:
  "无关": ["无关", "无影响", "可忽略"]
  "轻微": ["轻微", "有限", "初步影响"]
  "中等": ["显著", "明显", "相当影响"]
  "强烈": ["决定性", "主导", "关键因素"]
```

**编写建议**：
- 与 `intensity_descriptions.levels` 保持一致
- 提供的词语应可直接代入文本而无歧义

---

### 6. 模板库 (`templates`)

**组织好的文本模板，支持变量填充**。

```yaml
templates:
  single_factor:
    strong_positive:
      - "{behavior_zh}倾向明显，主要受{block_zh}驱动；"
      - "由于{block_zh}强烈，目标采取{behavior_zh}；"
    
    strong_negative:
      - "目标避免{behavior_zh}，因为{block_zh}并不支持；"
  
  multi_factor:
    two_factors_coordinated:
      - "目标决策同时受{factor1_zh}和{factor2_zh}的推动；"
```

**支持的变量**：
- `{block_zh}`：块的中文名
- `{behavior_zh}`：行为的中文标签
- `{intensity_zh}`：强度等级描述词
- `{delta_p}`：行为概率变化（自动计算）
- `{factor1_zh}`, `{factor2_zh}`：多因子情况下的因素名

**编写建议**：
1. 为常见情景设计模板（单因子、双因子、三因子等）
2. 提供多个模板选项（生成时随机选择，增加多样性）
3. 确保模板使用的变量都是可用的

---

### 7. 行为组合规则（可选）

```yaml
behavior_combinations:
  escape_and_hold:
    zh_interpretation: "采取保留姿态，准备躲避"
```

**编写建议**：
- 仅在**特定行为组合有特殊含义**时使用
- 通常可以忽略

---

## 验证检查清单

在提交配置文件前，请检查：

- [ ] **块映射完整**
  - 列出 `scm_causal_edges.json` 中的所有 `parent_block`
  - 每个块都有 `zh_name`, `zh_description`, `interpretation_hint`

- [ ] **行为标签完整**
  - 列出 `behavior_prototypes.json` 中的所有键
  - 每个标签都有 `zh_label`, `intensity_order`

- [ ] **强度阈值合理**
  ```python
  # 快速检查：统计过去 100 行的 l2_pred_change 分布
  # 阈值应接近 25%, 50%, 75% 分位数
  import numpy as np
  
  # 这样设置使四个等级大致均匀分布
  percentiles = [0, 25, 50, 75, 100]
  l2_changes = [...]  # 历史数据
  thresholds = [np.percentile(l2_changes, p) for p in percentiles]
  ```

- [ ] **模板语法正确**
  - 所有变量都用 `{variable_name}` 格式
  - 检查是否有拼写错误

- [ ] **中文表述自然**
  - 读一遍生成的解释是否流畅
  - 是否符合领域常识

- [ ] **手工测试**
  ```bash
  python tactical_narrative_gen.py \
    counterfactual_abduction.json \
    scenarios/scenario_simple_spread.yaml \
    output.json
  
  # 查看前 3 行的输出
  jq '.rows[0:3] | .[].tactical_narratives' output.json
  ```

---

## 常见问题

### Q1：怎样定义块的"战术名"？

**A**：思考块代表的**战术概念**，而非技术细节。

**不好的例子**：
```yaml
zh_name: "s_mean__pooled_neighbors 均值"  # 太技术性
```

**好的例子**：
```yaml
zh_name: "邻居群体平均态势"  # 战术性强，易理解
```

### Q2：`intensity_order` 应该怎么设置？

**A**：用于在有多个行为变化时排序重要性。

```yaml
# 被动→低强度→中等→高强度
hold-neutral:  intensity_order: 0   # 最被动
escape-ish:    intensity_order: 1   # 低强度
maneuver:      intensity_order: 2   # 中等
press-forward: intensity_order: 3   # 最激进
```

### Q3：强度阈值 `l2_thresholds` 怎样定？

**A**：基于实际数据的分位数。

```bash
# 1. 提取所有历史的 l2_pred_change 值
# 2. 计算 25%, 50%, 75% 分位数
# 3. 四舍五入后作为阈值

例如，历史数据中：
  0-0.25 的 25% 时刻     → 等级 0（无关）
  0.25-0.5 的 25% 时刻   → 等级 1（轻微）
  0.5-0.75 的 25% 时刻   → 等级 2（中等）
  0.75-1.0 的 25% 时刻   → 等级 3（强烈）
```

### Q4：模板里可以添加更多变量吗？

**A**：可以，但需要同步修改代码中的填充逻辑。推荐**先用现有变量**，后续再扩展。

### Q5：如何在多种环境间复用配置？

**A**：相似环境可以部分复用。

```yaml
# scenario_prey_predator.yaml 可以复用 simple_spread 的结构
# 只需修改：
#   - scenario_name / description
#   - behavior_semantics（捕食者/猎物行为不同）
#   - 部分模板词汇（"捕猎" vs "分散"）
```

---

## 下一步

配置文件完成后：

1. **测试生成**：运行 `tactical_narrative_gen.py` 生成前 5 行
2. **人工审验**：检查生成的解释是否符合直觉
3. **迭代改进**：
   - 如果解释过于技术性 → 调整块名和行为标签
   - 如果解释不自然 → 补充或调整模板
   - 如果强度分类不合理 → 调整 `l2_thresholds`
4. **集成上线**：将配置文件路径传给 `run_pipeline.py --narrative-scenario`

---

## 配置版本管理

建议在配置中添加版本号和修改记录：

```yaml
# 文件头部添加
metadata:
  version: "1.0"
  last_updated: "2026-05-26"
  author: "你的名字"
  notes: |
    v1.0: 初版，基于 simple_spread 100 个样本标注
    - 3 个父块映射
    - 4 个行为标签
    - 默认强度阈值
```

---

## 获取帮助

- 查看 `scenario_template.yaml` 了解完整的字段说明
- 查看 `scenario_simple_spread.yaml` 了解一个完整的实例
- 运行 `tactical_narrative_gen.py --help` 查看命令行选项
