# 多步反事实说明（multi_step）

面向指挥员/前端：在「一步反事实」基础上，多看 **随后 3～5 步** 的收益走势（代理推断，非重仿真）。

---

## 何时使用

| 场景 | 推荐 cf_level |
|------|----------------|
| 只关心「当时为何选这个动作」 | `local` |
| 关心「改一个因素，这一步分数会不会变」 | `one_step` |
| 关心「改一个因素，**接下来几步**走势会不会变」 | **`multi_step`** |

---

## 调用示例

### Python

```python
from src.service import counterfactual_service

result = counterfactual_service(
    agent_id=1,
    inference_task_id="INF_A_001",
    sim_id="SIM_A_0001",
    decision_content={"机动控制": "规避"},  # 可多个动作项
    cf_level="multi_step",
    horizon=4,                    # 3～5，默认 5
    perturb_strategy="train_mean",
    explain_with_llm=False,       # True 时用本地 Qwen 润色（可选）
)

# 主展示
print(result["nl_explanation"])

# 轨迹摘要（影响最大的那个特征）
print(result["top_feature"])
print(result["original_action_seq"])   # 仿真记录：随后 H 步真实动作
print(result["cf_action_seq"])         # 扰动 top 特征后：代理滚出的 H 步动作
print(result["original_cumulative_reward"])
print(result["disclaimer"])
```

### 命令行

```powershell
cd E:\工作\analysis
.\.venv\Scripts\python.exe main.py --mode explain_c `
  --inference_task_id INF_A_001 --sim_id SIM_A_0001 --agent_id 1 `
  --decision 机动控制=规避 --cf_level multi_step --horizon 4
```

### 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -s tests/test_module_c.py::test_counterfactual_service_multi_step_smoke -q
```

---

## 返回字段（multi_step 特有）

| 字段 | 含义 |
|------|------|
| `horizon` | 实际滚动步数（3～5，且不超过仿真剩余步数） |
| `original_cumulative_reward` | 事实：随后 H 步真实奖励之和 |
| `original_action_seq` | 事实：随后 H 步真实动作（可读字符串列表） |
| `cf_action_seq` | 反事实：对「影响最大」特征扰动后，代理滚出的 H 步动作 |
| `top_feature` | 按 \|reward_delta\| 排序后的首要特征名 |
| `key_features[].reward_delta` | 该特征扰动后的累计奖励差 |
| `key_features[].cf_action_seq` | 每个候选特征各自的反事实动作序列 |
| `disclaimer` | 代理推断免责声明 |

---

## 算法直觉（小白版）

1. 在决策时刻 **只改一个观测因素**（其余保持真实）。
2. **事实线**：仿真里已经发生的后几步，把奖励加起来。
3. **反事实线**：从改过的态势出发，用学到的「决策→下一态势→得分」模型连猜 H 步，把得分加起来。
4. 两者相减 → 解释「这个因素对短期走势有多重要」。

---

## 代理模型缓存（加速重复请求）

同一 `inference_task_id` + `agent_id` + 决策树参数下，**第二次**起解释会复用已训练的 π/T/R，返回里 `surrogate_cache_hit=true`。

```powershell
# 关闭缓存（调试训练逻辑时）
$env:ANALYSIS_CF_BUNDLE_CACHE = "0"
```

换 mock 数据（sim 列表或步数变化）后缓存会自动失效并重新训练。

---

## 限制（必读）

- 每次仍只扰动 **一个** 特征（联合多特征未实现）。
- 反事实线全程用 **代理模型** 自回归，误差会随步数累积；`horizon` 限制在 3～5。
- 其他智能体不单独建模，其影响体现在观测向量中。
- 前端请优先展示 `nl_explanation`，不要强依赖 LLM 改写后的 `mechanistic` 长文。
