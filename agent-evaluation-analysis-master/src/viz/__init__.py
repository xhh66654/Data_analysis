"""
可视化工具包。

提供决策树导出、因果图绘制、环境渲染及 Markdown 报告生成等功能。
可选依赖缺失时对应符号置为 ``None``，不影响包的整体导入。
"""
# 使用 try/except 保护导入，避免缺少可选依赖（networkx 等）时影响整个包
try:
    from .env_render import render_frame, animate_trajectory
except ImportError:
    render_frame = None          # type: ignore
    animate_trajectory = None    # type: ignore

from .tree_plot import export_tree_pdf

try:
    from .causal_graph_plot import plot_causal_graph
except ImportError:
    plot_causal_graph = None     # type: ignore

try:
    from .report import write_markdown_report
except ImportError:
    write_markdown_report = None  # type: ignore

__all__ = [
    "render_frame",
    "animate_trajectory",
    "export_tree_pdf",
    "plot_causal_graph",
    "write_markdown_report",
]
