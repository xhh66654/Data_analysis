"""
统一日志工具。

用法：
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("hello")
"""
import logging
import sys
from pathlib import Path
from typing import Optional


_INITIALIZED = False


def _setup_root(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """
    初始化根日志记录器（仅执行一次）。

    参数
    ----
    level : int
        日志级别，默认 ``logging.INFO``。
    log_file : str | None
        可选日志文件路径；指定时额外写入该文件。
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    fmt = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(level)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    root.addHandler(sh)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """
    获取命名日志记录器。

    首次调用时会自动完成根日志器初始化。

    参数
    ----
    name : str
        日志器名称，通常传入 ``__name__``。

    返回
    ----
    logging.Logger
        对应名称的日志记录器实例。
    """
    _setup_root()
    return logging.getLogger(name)
