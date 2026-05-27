"""
将 VIPER 已导出的 Graphviz .dot 决策树文件渲染为位图或矢量图。

本模块不参与训练，仅做后处理：读取 viper_out/policy_tree_debug.dot（或任意 .dot），
调用 Python graphviz 包调用系统 dot 可执行文件，生成 PNG / PDF / SVG。

依赖：
  pip install graphviz
  系统安装 Graphviz 并将 dot 加入 PATH

示例：
  python -m causal.decision_tree.render_tree_image --dot fqe_out/viper_out/policy_tree_debug.dot --format png
  python -m causal.decision_tree.render_tree_image --dot policy_tree.dot --format png -o my_tree.png --show

run_pipeline.py 中 render_tree_pdf/render_tree_png=True 时会在 VIPER 阶段内联渲染；
本模块供已有 .dot 文件单独补渲时使用。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def render_dot_to_image(
    dot_path: Path,
    *,
    fmt: str,
    output_path: Path | None,
    show: bool,
) -> Path:
    dot_path = Path(dot_path).resolve()
    if not dot_path.is_file():
        raise FileNotFoundError(f"未找到 .dot 文件: {dot_path}")

    try:
        import graphviz as gv_mod
    except ImportError as exc:
        raise RuntimeError("请先 pip install graphviz") from exc

    try:
        src = gv_mod.Source.from_file(str(dot_path), encoding="utf-8")
    except TypeError:
        src = gv_mod.Source.from_file(str(dot_path))
    # 与 VIPER 导出一致：中文标签字体
    dot_body = src.source.replace("helvetica", "Microsoft YaHei")
    src = gv_mod.Source(dot_body, encoding="utf-8")

    if output_path is None or not str(output_path).strip():
        out_base = dot_path.with_suffix("")
    else:
        output_path = Path(output_path).resolve()
        out_base = output_path.parent / output_path.stem

    rendered_raw = src.render(filename=str(out_base), format=fmt, cleanup=True)
    rendered = Path(rendered_raw)
    logger.info("已生成: %s", rendered)

    if show:
        from causal.decision_tree.viper_cart import open_image_file

        open_image_file(rendered)

    return rendered


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="将 decision_tree 导出的 .dot 转成图片")
    p.add_argument("--dot", type=str, required=True, help="policy_tree.dot 路径")
    p.add_argument(
        "--format",
        type=str,
        default="png",
        choices=("png", "pdf", "svg"),
        help="输出格式（默认 png）",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="输出文件路径（可带后缀）；默认与 .dot 同目录同名",
    )
    p.add_argument("--show", action="store_true", help="渲染后用系统默认程序打开")
    args = p.parse_args(argv)

    dot_p = Path(args.dot)
    out_p = Path(args.output) if args.output.strip() else None
    try:
        render_dot_to_image(dot_p, fmt=args.format, output_path=out_p, show=args.show)
    except Exception as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
