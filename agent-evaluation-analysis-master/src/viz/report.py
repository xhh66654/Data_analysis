"""解释报告生成：将规则、反事实与因果图等内容合并为 Markdown 文档。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def write_markdown_report(
    out_path: str | Path,
    title: str,
    sections: Dict[str, str],
) -> None:
    """
    将若干章节内容写入一份 Markdown 报告文件。

    参数
    ----
    out_path : str | Path
        报告输出路径。
    title : str
        报告一级标题。
    sections : Dict[str, str]
        章节字典，键为二级标题，值为对应 Markdown 正文片段。
    """
    # TODO
    raise NotImplementedError
