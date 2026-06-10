#!/usr/bin/env python
"""
下载反事实解释润色用的本地中文指令模型（默认 Qwen2.5-1.5B-Instruct）。

用法（在项目根目录）：
    py scripts/download_explain_model.py

可选环境变量：
    ANALYSIS_LLM_HF_REPO     默认 Qwen/Qwen2.5-1.5B-Instruct
    ANALYSIS_LLM_MODEL_PATH  默认 data/models/Qwen2.5-1.5B-Instruct
    HF_ENDPOINT              国内可设 https://hf-mirror.com

下载完成后启用润色：
    set ANALYSIS_LLM_EXPLAIN=1
    set ANALYSIS_LLM_BACKEND=transformers
    set ANALYSIS_LLM_MODEL_PATH=data/models/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_REPO = os.environ.get("ANALYSIS_LLM_HF_REPO", "Qwen/Qwen2.5-1.5B-Instruct")
DEFAULT_DIR = Path(
    os.environ.get(
        "ANALYSIS_LLM_MODEL_PATH",
        str(ROOT / "data" / "models" / "Qwen2.5-1.5B-Instruct"),
    )
)


def main() -> None:
    """从 HuggingFace 下载解释润色用本地中文指令模型到默认目录。"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("请先安装：pip install huggingface_hub")
        sys.exit(1)

    DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ.get("HF_ENDPOINT", "(默认 huggingface.co)")
    print(f"正在下载 {DEFAULT_REPO} → {DEFAULT_DIR.resolve()}")
    print(f"HF_ENDPOINT = {endpoint}")
    print(
        "说明：进度条按「整文件」计数，前几个是小文件很快完成；\n"
        "      最大的 model.safetensors（约 3GB）下载时可能长时间停在 0/10，属正常现象。\n"
        "      国内若一直不动，请先 Ctrl+C，再执行：\n"
        "        $env:HF_ENDPOINT = 'https://hf-mirror.com'\n"
        "        py scripts/download_explain_model.py\n"
    )

    snapshot_download(
        repo_id=DEFAULT_REPO,
        local_dir=str(DEFAULT_DIR),
        resume_download=True,
        max_workers=4,
    )

    print("\n下载完成。启用方式：")
    print("  set ANALYSIS_LLM_EXPLAIN=1")
    print("  set ANALYSIS_LLM_BACKEND=transformers")
    print(f"  set ANALYSIS_LLM_MODEL_PATH={DEFAULT_DIR}")


if __name__ == "__main__":
    main()
