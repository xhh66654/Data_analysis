#!/usr/bin/env python
"""检查解释润色模型是否下载完整。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.module_c_counterfactual.llm_explain import get_llm_config, is_local_llm_model_ready


def main() -> None:
    """检查本地 LLM 权重文件是否完整并就绪可用。"""
    cfg = get_llm_config()
    path = Path(cfg["model_path"])
    print(f"模型目录: {path.resolve()}")
    print(f"目录存在: {path.is_dir()}")
    if path.is_dir():
        safes = list(path.glob("*.safetensors"))
        bins = list(path.glob("model*.bin"))
        print(f"safetensors: {[f.name for f in safes]}")
        print(f"bin: {[f.name for f in bins]}")
        print(f"其他文件: {[f.name for f in path.iterdir() if f.is_file()][:15]}")
    ready = is_local_llm_model_ready()
    print(f"\n权重就绪: {ready}")
    if ready:
        print("可启用: $env:ANALYSIS_LLM_EXPLAIN='1' 或 explain_with_llm=True")
        print("启用: $env:ANALYSIS_LLM_EXPLAIN='1' 或 explain_with_llm=True")
    else:
        print("未完成。请运行: py scripts/download_explain_model_modelscope.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
