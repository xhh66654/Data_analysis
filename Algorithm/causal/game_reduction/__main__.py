"""支持 python -m causal.game_reduction （需 PYTHONPATH 含仓库根目录）。"""
from __future__ import annotations

from .run_pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
