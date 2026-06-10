"""
通用工具包。

提供配置加载、日志、随机种子等跨模块复用的基础能力。
"""
from .config import load_config, get
from .logger import get_logger
from .seeding import set_seed

__all__ = ["load_config", "get", "get_logger", "set_seed"]
