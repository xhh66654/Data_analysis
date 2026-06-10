"""因果图可视化工具（模块 B 使用）。"""
from __future__ import annotations

from typing import Any

import networkx as nx


def plot_causal_graph(
    graph: nx.DiGraph,
    weights: dict | None = None,
    out_path: str | None = None,
    ax: Any = None,
) -> None:
    """
    绘制因果有向无环图（DAG）。

    若提供边权重，则按权重大小映射边的粗细。

    参数
    ----
    graph : nx.DiGraph
        待绘制的因果图，节点为变量、边为因果方向。
    weights : dict | None
        可选边权重字典，键为 ``(源节点, 目标节点)`` 元组。
    out_path : str | None
        可选输出图片路径；为 ``None`` 时仅显示不保存。
    ax : Any
        可选 matplotlib Axes；为 ``None`` 时自动创建。
    """
    # TODO: networkx draw + matplotlib
    raise NotImplementedError
