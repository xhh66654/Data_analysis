# 反事实解释：预训练模型润色说明

## 作用

反事实推理的 **结构化结果**（`key_features`、准确率、训练样本数等）仍由算法产生，不变。

可选开启 **LLM 润色**，把模板化的 `mechanistic` / `teleological` 改写成更易读的中文，并增加 `summary` 综合摘要。

未开启或模型不可用时，行为与原来完全一致（`explanation_backend=template`）。

---

## 推荐模型

| 用途 | 模型 | 大小 | 存放路径 |
|------|------|------|----------|
| 本地中文润色（推荐） | `Qwen/Qwen2.5-1.5B-Instruct` | ~3GB | `data/models/Qwen2.5-1.5B-Instruct/` |

更小可选：`Qwen/Qwen2.5-0.5B-Instruct`（约 1GB，质量略低）。

---

## 一、下载本地模型（三种方式，国内优先 B）

| 方式 | 速度（国内） | 命令 |
|------|----------------|------|
| **A. 魔搭 ModelScope** | 最快（推荐） | `py scripts/download_explain_model_modelscope.py` |
| **B. HF 镜像** | 较快 | `$env:HF_ENDPOINT="https://hf-mirror.com"; py scripts/download_explain_model.py` |
| **C. HF 加速传输** | 视网络而定 | `pip install hf_transfer` + `$env:HF_HUB_ENABLE_HF_TRANSFER="1"` 再跑 A/B |

### 1. 安装依赖（请用项目虚拟环境）

```powershell
cd E:\工作\analysis
.\.venv\Scripts\pip install transformers accelerate sentencepiece
.\.venv\Scripts\python scripts\verify_llm_model.py
```

（项目已含 `torch`。若 `verify_llm_model.py` 显示「权重就绪: True」即可用。）

### 2A. 【推荐】魔搭下载

```powershell
pip install modelscope
py scripts/download_explain_model_modelscope.py
```

### 2B. HuggingFace 镜像下载

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
py scripts/download_explain_model.py
```

### 2C. 启用 hf_transfer（多线程，可与镜像叠加）

```powershell
pip install hf_transfer
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
$env:HF_ENDPOINT = "https://hf-mirror.com"
py scripts/download_explain_model.py
```

默认下载到：`data/models/Qwen2.5-1.5B-Instruct/`

自定义路径：

```powershell
$env:ANALYSIS_LLM_MODEL_PATH = "D:\models\Qwen2.5-1.5B-Instruct"
$env:ANALYSIS_LLM_HF_REPO = "Qwen/Qwen2.5-1.5B-Instruct"
py scripts/download_explain_model.py
```

---

## 二、启用润色

### 环境变量（PowerShell）

```powershell
$env:ANALYSIS_LLM_EXPLAIN = "1"
$env:ANALYSIS_LLM_BACKEND = "transformers"
$env:ANALYSIS_LLM_MODEL_PATH = "data\models\Qwen2.5-1.5B-Instruct"
# 可选：生成长度
$env:ANALYSIS_LLM_MAX_TOKENS = "512"
```

### Python API

```python
import os
os.environ["ANALYSIS_LLM_EXPLAIN"] = "1"

from src.service import counterfactual_service

result = counterfactual_service(
    agent_id=1,
    inference_task_id="INF_A_001",
    sim_id="SIM_A_0001",
    decision_content={"机动控制": "规避"},
    cf_level="one_step",
    explain_with_llm=True,
)

print(result["summary"])              # 综合摘要（LLM）
print(result["mechanistic"])          # 润色后机械性解释
print(result["mechanistic_raw"])      # 模板原文备份
print(result["explanation_backend"])  # llm_transformers | template
```

### 命令行

```powershell
$env:ANALYSIS_LLM_EXPLAIN = "1"
py main.py --mode explain_c --inference_task_id INF_A_001 --sim_id SIM_A_0001 --agent_id 1 --decision 机动控制=规避 --cf_level one_step --llm_explain
```

---

## 三、使用 API 后端（不下载本地模型）

适用于已部署的 OpenAI 兼容服务（如阿里云 DashScope、自建 vLLM）。

```powershell
$env:ANALYSIS_LLM_EXPLAIN = "1"
$env:ANALYSIS_LLM_BACKEND = "openai_compatible"
$env:OPENAI_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:OPENAI_API_KEY = "你的密钥"
$env:ANALYSIS_LLM_MODEL_NAME = "qwen-plus"
```

---

## 四、返回字段说明

| 字段 | 说明 |
|------|------|
| `headline` | 一句话结论（适合列表/卡片标题） |
| `mechanistic` | 通俗版机械性解释 |
| `teleological` | 通俗版目的性解释 |
| `summary` | 综合摘要（条目） |
| `mechanistic_raw` | 算法模板原文 |
| `teleological_raw` | 算法模板原文 |
| `explanation_backend` | `template` / `llm_transformers` / `llm_openai` |
| `llm_error` | 润色失败时的错误信息（仍保留模板解释） |

`key_features` 等结构化字段 **不被 LLM 修改**。

可读性增强（已默认启用）：事实稿叙述体、特征通俗化、四段输出结构。可调 `$env:ANALYSIS_LLM_MAX_TOKENS="1024"`。效果更好可换 **qwen-plus** API 或更大本地模型。

---

## 五、硬件建议

| 环境 | 说明 |
|------|------|
| CPU | 可用，单次润色约数十秒 |
| NVIDIA GPU + CUDA | 推荐，明显更快 |
| 内存 | 1.5B 模型建议 ≥ 8GB 可用内存 |

---

## 六、注意事项

1. LLM 仅根据「事实稿」撰写，提示词要求禁止编造特征与数值。
2. 解释中仍会标明基于 **代理模型** 推断，非环境重仿真。
3. 首次加载模型较慢，同一进程内会缓存模型实例。
