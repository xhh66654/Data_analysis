"""
配置加载工具。

从 YAML 文件读取项目配置，并支持点分路径访问嵌套字段。
"""
from pathlib import Path
from typing import Any, Dict, Union
import yaml

def load_config(path: Union[str, Path] = "config.yaml") -> Dict[str, Any]:
    """
    读取 YAML 配置文件。

    参数
    ----
    path : str | Path
        配置文件路径，默认为项目根目录下的 ``config.yaml``。

    返回
    ----
    Dict[str, Any]
        解析后的配置字典。
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get(cfg: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """
    按点分路径从嵌套字典中取值。

    示例：``get(cfg, 'env.n_blue')`` 等价于 ``cfg['env']['n_blue']``。

    参数
    ----
    cfg : Dict[str, Any]
        配置字典。
    dotted_key : str
        点分键路径，如 ``"env.n_blue"``。
    default : Any
        键不存在或中间节点非字典时的默认返回值。

    返回
    ----
    Any
        查找到的值，或 ``default``。
    """
    cur: Any = cfg
    for k in dotted_key.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
