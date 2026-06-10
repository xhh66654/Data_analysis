#!/usr/bin/env python
"""
【国内推荐】通过魔搭 ModelScope 下载解释润色模型，通常比直连 HuggingFace 更快。

用法（项目根目录）：
    pip install modelscope
    py scripts/download_explain_model_modelscope.py

下载到与 transformers 相同的目录，可直接用于 ANALYSIS_LLM_MODEL_PATH。
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MS_MODEL = os.environ.get("ANALYSIS_MS_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
DEFAULT_DIR = Path(
    os.environ.get(
        "ANALYSIS_LLM_MODEL_PATH",
        str(ROOT / "data" / "models" / "Qwen2.5-1.5B-Instruct"),
    )
)
CACHE_ROOT = ROOT / "data" / "models" / "_modelscope_cache"


def main() -> None:
    """从魔搭 ModelScope 下载解释润色模型并同步到项目模型目录。"""
    try:
        from modelscope import snapshot_download as ms_download
    except ImportError:
        print("请先安装：pip install modelscope")
        sys.exit(1)

    DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"魔搭下载：{DEFAULT_MS_MODEL}")
    print(f"目标目录：{DEFAULT_DIR.resolve()}\n")

    model_path = Path(ms_download(DEFAULT_MS_MODEL, cache_dir=str(CACHE_ROOT)))
    print(f"魔搭缓存路径：{model_path}\n正在复制到目标目录…")

    n_files = 0
    for src in model_path.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(model_path)
        dest = DEFAULT_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dest)
        n_files += 1

    has_weights = any(DEFAULT_DIR.glob("*.safetensors")) or any(
        DEFAULT_DIR.glob("model*.bin")
    )
    print(f"\n已同步 {n_files} 个文件。")
    if has_weights:
        print(f"权重 OK：{list(DEFAULT_DIR.glob('*.safetensors'))[:3]}")
        print("\n启用润色：")
        print("  $env:ANALYSIS_LLM_EXPLAIN = '1'")
        print("  $env:ANALYSIS_LLM_BACKEND = 'transformers'")
        print(f"  $env:ANALYSIS_LLM_MODEL_PATH = '{DEFAULT_DIR}'")
    else:
        print("未检测到 .safetensors，请检查网络或模型 ID。")
        sys.exit(1)


if __name__ == "__main__":
    main()
