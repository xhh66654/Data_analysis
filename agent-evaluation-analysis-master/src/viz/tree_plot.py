"""
决策树可视化：默认导出 PDF（矢量，节点间距更大，适合宽树）。

节点语义（与规则抽取一致）：
- 非叶节点：显示分裂条件 + 样本数 + 主导动作（多数类，中文）
- 叶节点：无分裂条件，只显示样本数 + 主导动作 + 置信度
- 不展示 sklearn 默认的长 value 向量（易被误认为“矩阵”）

渲染优先级：
1. 自定义 DOT + 系统 dot（矢量 PDF）
2. Python graphviz 包 + dot
3. matplotlib plot_tree（中文环境兜底）
"""
from __future__ import annotations

import ast
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, List, Optional, TYPE_CHECKING

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import _tree

if TYPE_CHECKING:
    from src.module_a_rules.preprocess import Preprocessor

DEFAULT_DISPLAY_MAX_DEPTH = 6
MAX_DISPLAY_DEPTH_CAP = 8
DEFAULT_OUTPUT_FORMAT = "pdf"  # pdf | png


def export_tree_pdf(
    tree: DecisionTreeClassifier,
    out_path: str,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    preprocessor: Optional["Preprocessor"] = None,
    title: str = "",
    display_max_depth: int = DEFAULT_DISPLAY_MAX_DEPTH,
) -> str:
    """
    导出决策树可视化文件（默认 PDF 矢量格式）。

    按优先级尝试 matplotlib、系统 dot、Python graphviz 包等后端；
    同时生成 ``_legend.txt`` 图例说明文件。

    参数
    ----
    tree : DecisionTreeClassifier
        已训练的 sklearn 决策树分类器。
    out_path : str
        输出路径（可含或不含 ``.pdf`` / ``.png`` 扩展名）。
    feature_names : List[str]
        特征名称列表，与训练时顺序一致。
    class_names : List[str] | None
        类别名称列表；为 ``None`` 时使用 ``tree.classes_``。
    preprocessor : Preprocessor | None
        可选预处理器，用于将归一化阈值反解为原始量纲及语义标签。
    title : str
        图标题。
    display_max_depth : int
        展示的最大树深度，默认 6，上限 8。

    返回
    ----
    str
        实际写入的渲染文件绝对/相对路径。

    环境变量
    --------
    TREE_VIZ_MAX_DEPTH : 展示层数（默认 6，上限 8）
    TREE_VIZ_FORMAT    : ``pdf`` | ``png``（默认 pdf）
    TREE_VIZ_BACKEND   : ``auto`` | ``dot`` | ``matplotlib``（含中文时 Windows 优先 matplotlib）
    """
    out_base = _normalize_out_base(out_path)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    out_fmt = os.environ.get("TREE_VIZ_FORMAT", DEFAULT_OUTPUT_FORMAT).lower().strip()
    if out_fmt not in ("pdf", "png"):
        out_fmt = DEFAULT_OUTPUT_FORMAT

    if class_names is None:
        class_names = [str(c) for c in tree.classes_]
    else:
        class_names = [str(c) for c in class_names]

    display_max_depth = max(1, min(int(display_max_depth), MAX_DISPLAY_DEPTH_CAP))
    show_depth = min(display_max_depth, tree.get_depth())

    display_feature_names = _truncate_labels(feature_names, max_len=28)
    readable_class_names = [_format_action_label(c) for c in class_names]

    subtitle = (
        f"{title}\\n展示深度 {show_depth}/{tree.get_depth()} 层（完整规则见 rules_text）"
        if title
        else f"展示深度 {show_depth}/{tree.get_depth()} 层"
    )

    backend = os.environ.get("TREE_VIZ_BACKEND", "auto").lower().strip()
    use_matplotlib_first = backend == "matplotlib" or (
        backend == "auto" and _prefer_matplotlib_backend(display_feature_names, readable_class_names)
    )

    if use_matplotlib_first:
        try:
            fallback = out_base.with_suffix(".pdf" if out_fmt == "pdf" else ".png")
            result = _export_matplotlib(
                tree,
                str(fallback),
                display_feature_names,
                readable_class_names,
                class_names,
                subtitle.replace("\\n", "\n"),
                display_max_depth=show_depth,
                file_format=out_fmt,
                preprocessor=preprocessor,
            )
            _write_viz_legend(result, display_feature_names, class_names, readable_class_names)
            _cleanup_orphan_artifacts(out_base)
            return result
        except Exception as e:
            print(f"[tree_plot] matplotlib 导出失败（{e}），尝试 graphviz/dot…")

    dot_data = _build_custom_dot_data(
        tree,
        display_feature_names,
        class_names,
        subtitle,
        max_depth=show_depth,
        preprocessor=preprocessor,
    )

    if _dot_available():
        try:
            result = _render_dot_file(dot_data, out_base, out_fmt)
            _write_viz_legend(result, display_feature_names, class_names, readable_class_names)
            _cleanup_orphan_artifacts(out_base)
            return result
        except Exception as e:
            print(f"[tree_plot] dot 渲染失败（{e}），尝试 graphviz 包…")

    if _graphviz_python_available():
        try:
            result = _export_graphviz_python(dot_data, str(out_base), out_fmt)
            _write_viz_legend(result, display_feature_names, class_names, readable_class_names)
            _cleanup_orphan_artifacts(out_base)
            return result
        except Exception as e:
            print(f"[tree_plot] graphviz 包导出失败（{e}），降级为 matplotlib…")

    fallback = out_base.with_suffix(".pdf" if out_fmt == "pdf" else ".png")
    result = _export_matplotlib(
        tree,
        str(fallback),
        display_feature_names,
        readable_class_names,
        class_names,
        subtitle.replace("\\n", "\n"),
        display_max_depth=show_depth,
        file_format=out_fmt,
        preprocessor=preprocessor,
    )
    _write_viz_legend(result, display_feature_names, class_names, readable_class_names)
    _cleanup_orphan_artifacts(out_base)
    return result


def _prefer_matplotlib_backend(
    feature_names: List[str],
    class_names: List[str],
) -> bool:
    """
    判断是否应优先使用 matplotlib 后端。

    Windows 平台且标签含非 ASCII 字符（如中文）时返回 ``True``，
    因 matplotlib 对 PDF 中文字体支持更稳定。

    参数
    ----
    feature_names : List[str]
        特征名列表。
    class_names : List[str]
        类别名列表。

    返回
    ----
    bool
        是否优先 matplotlib。
    """
    if platform.system().lower() != "windows":
        return False
    text = " ".join(feature_names) + " " + " ".join(class_names)
    try:
        text.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def _normalize_out_base(out_path: str) -> Path:
    """
    规范化输出路径，去除 ``.png`` / ``.pdf`` 扩展名作为基名。

    参数
    ----
    out_path : str
        用户传入的输出路径。

    返回
    ----
    Path
        不含图片扩展名的基路径。
    """
    p = Path(out_path)
    if p.suffix.lower() in (".png", ".pdf"):
        return p.with_suffix("")
    return p


def _cleanup_orphan_artifacts(out_base: Path) -> None:
    """
    清理渲染过程中产生的无扩展名残留文件及 ``.dot`` 旁路文件。

    参数
    ----
    out_base : Path
        输出基路径（无扩展名）。
    """
    orphan = Path(out_base)
    if orphan.exists() and orphan.is_file() and orphan.suffix == "":
        try:
            orphan.unlink()
        except OSError:
            pass
    dot_sidecar = out_base.with_suffix(".dot")
    if dot_sidecar.exists():
        try:
            dot_sidecar.unlink()
        except OSError:
            pass


def _dot_available() -> bool:
    """
    检测系统 PATH 中是否可用 Graphviz 的 ``dot`` 命令。

    返回
    ----
    bool
        ``dot`` 可执行文件存在时为 ``True``。
    """
    return shutil.which("dot") is not None


def _graphviz_python_available() -> bool:
    """
    检测 Python ``graphviz`` 包及系统 ``dot`` 是否均可用。

    返回
    ----
    bool
        两者均可用时为 ``True``。
    """
    try:
        import graphviz  # noqa: F401
    except ImportError:
        return False
    return _dot_available()


def _dot_escape(text: str) -> str:
    """
    转义 DOT 格式字符串中的反斜杠与双引号。

    参数
    ----
    text : str
        原始文本。

    返回
    ----
    str
        DOT 安全字符串。
    """
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _spacing_for_tree(n_leaves: int) -> tuple[float, float]:
    """
    根据可见叶节点数估算 Graphviz 的节点间距与层级间距。

    参数
    ----
    n_leaves : int
        可见叶节点数量。

    返回
    ----
    tuple[float, float]
        ``(nodesep, ranksep)`` 间距参数。
    """
    if n_leaves <= 8:
        return 0.9, 1.2
    if n_leaves <= 16:
        return 1.1, 1.5
    if n_leaves <= 32:
        return 1.3, 1.8
    return 1.6, 2.2


def _inject_graphviz_style(dot_body: str, title: str, n_leaves: int) -> str:
    """
    为 DOT 节点/边定义包裹全局样式、字体与间距配置。

    参数
    ----
    dot_body : str
        节点与边的 DOT 主体内容。
    title : str
        图标题（显示在顶部）。
    n_leaves : int
        可见叶节点数，用于计算间距。

    返回
    ----
    str
        完整的 DOT 源字符串。
    """
    nodesep, ranksep = _spacing_for_tree(n_leaves)
    return f'''digraph Tree {{
    charset="UTF-8";
    labelloc="t";
    label="{_dot_escape(title)}";
    fontsize=14;
    fontname="Microsoft YaHei";
    bgcolor="white";
    graph [rankdir=TB, pad=0.6, nodesep={nodesep}, ranksep={ranksep},
           splines=polyline, overlap=false, concentrate=false, bgcolor="white"];
    node [shape=box, style="rounded,filled", fontname="Microsoft YaHei", fontsize=10,
          color="#334155", penwidth=1.0, margin="0.28,0.14", width=0, height=0];
    edge [fontname="Microsoft YaHei", fontsize=9, color="#64748b", arrowsize=0.7];
{dot_body}
}}
'''


def _build_custom_dot_data(
    tree: DecisionTreeClassifier,
    feature_names: List[str],
    class_names: List[str],
    title: str,
    max_depth: int,
    preprocessor: Optional[Any] = None,
) -> str:
    """
    手工构造决策树 DOT 源数据。

    节点标签与规则抽取模块一致：分裂条件、样本数、主导动作及置信度。

    参数
    ----
    tree : DecisionTreeClassifier
        决策树模型。
    feature_names : List[str]
        特征名列表。
    class_names : List[str]
        类别名列表。
    title : str
        图标题。
    max_depth : int
        展示最大深度。
    preprocessor : Any | None
        可选预处理器，用于阈值反归一化。

    返回
    ----
    str
        完整 DOT 字符串。
    """
    t = tree.tree_
    visible = _visible_nodes(tree, max_depth)
    n_leaves = sum(
        1 for nid in visible
        if t.children_left[nid] == _tree.TREE_LEAF
        or t.children_left[nid] not in visible
    )
    n_leaves = max(n_leaves, 1)

    node_lines: List[str] = []
    edge_lines: List[str] = []

    for node_id in sorted(visible):
        depth = _node_depth(tree, node_id)
        label = _format_node_label(
            tree,
            node_id,
            feature_names,
            class_names,
            preprocessor,
            max_depth=max_depth,
            depth=depth,
        )
        node_lines.append(f'    {node_id} [label="{_dot_escape(label)}"];')

    for node_id in sorted(visible):
        left = int(t.children_left[node_id])
        right = int(t.children_right[node_id])
        if left == _tree.TREE_LEAF:
            continue
        if left in visible:
            edge_lines.append(
                f'    {node_id} -> {left} [labeldistance=2.5, labelangle=45, headlabel="是"];'
            )
        if right in visible:
            edge_lines.append(
                f'    {node_id} -> {right} [labeldistance=2.5, labelangle=-45, headlabel="否"];'
            )

    body = "\n".join(node_lines + edge_lines)
    return _inject_graphviz_style(body, title, n_leaves)


def _visible_nodes(tree: DecisionTreeClassifier, max_depth: int) -> set[int]:
    """
    收集展示深度内的所有节点 ID。

    语义与 sklearn ``export_graphviz(max_depth=...)`` 一致。

    参数
    ----
    tree : DecisionTreeClassifier
        决策树模型。
    max_depth : int
        从根节点起算的最大展示深度。

    返回
    ----
    set[int]
        可见节点 ID 集合。
    """
    t = tree.tree_
    visible: set[int] = set()
    stack: List[tuple[int, int]] = [(0, 0)]
    while stack:
        node_id, depth = stack.pop()
        visible.add(node_id)
        if depth >= max_depth:
            continue
        left = int(t.children_left[node_id])
        right = int(t.children_right[node_id])
        if left == _tree.TREE_LEAF:
            continue
        stack.append((right, depth + 1))
        stack.append((left, depth + 1))
    return visible


def _node_depth(tree: DecisionTreeClassifier, node_id: int) -> int:
    """
    计算指定节点相对根节点的深度。

    参数
    ----
    tree : DecisionTreeClassifier
        决策树模型。
    node_id : int
        目标节点 ID。

    返回
    ----
    int
        深度值，根节点为 0。
    """
    t = tree.tree_
    depth = 0
    cur = node_id
    while cur != 0:
        parent = _find_parent(tree, cur)
        if parent is None:
            break
        depth += 1
        cur = parent
    return depth


def _find_parent(tree: DecisionTreeClassifier, child_id: int) -> Optional[int]:
    """
    查找指定子节点的父节点 ID。

    参数
    ----
    tree : DecisionTreeClassifier
        决策树模型。
    child_id : int
        子节点 ID。

    返回
    ----
    int | None
        父节点 ID；未找到时返回 ``None``。
    """
    t = tree.tree_
    for nid in range(t.node_count):
        if int(t.children_left[nid]) == child_id or int(t.children_right[nid]) == child_id:
            return nid
    return None


def _format_node_label(
    tree: DecisionTreeClassifier,
    node_id: int,
    feature_names: List[str],
    class_names: List[str],
    preprocessor: Optional[Any],
    max_depth: int,
    depth: int,
) -> str:
    """
    格式化单个树节点的显示标签。

    分裂节点首行为判断条件；所有节点均含样本数、主导动作与置信度。

    参数
    ----
    tree : DecisionTreeClassifier
        决策树模型。
    node_id : int
        节点 ID。
    feature_names : List[str]
        特征名列表。
    class_names : List[str]
        类别名列表。
    preprocessor : Any | None
        可选预处理器。
    max_depth : int
        展示最大深度（用于折叠提示）。
    depth : int
        当前节点深度。

    返回
    ----
    str
        多行节点标签文本（``\\n`` 分隔）。
    """
    t = tree.tree_
    n_samples = int(t.n_node_samples[node_id])
    values = t.value[node_id][0]
    total = float(values.sum())
    best_idx = int(values.argmax())
    dominant = _format_action_label(class_names[best_idx])
    confidence = float(values[best_idx] / total) if total > 0 else 0.0

    lines: List[str] = []
    feat_idx = int(t.feature[node_id])
    left_child = int(t.children_left[node_id])
    is_split = feat_idx != _tree.TREE_UNDEFINED

    if is_split:
        lines.append(
            _format_split_condition(
                feat_idx,
                float(t.threshold[node_id]),
                feature_names,
                preprocessor,
            )
        )
        if depth >= max_depth and left_child != _tree.TREE_LEAF:
            lines.append("（更深层已折叠）")

    lines.append(f"samples = {n_samples}")
    lines.append(f"主导动作 = {dominant}")
    lines.append(f"置信度 = {confidence:.1%}")
    return "\\n".join(lines)


def _format_split_condition(
    feat_idx: int,
    norm_threshold: float,
    feature_names: List[str],
    preprocessor: Optional[Any],
) -> str:
    """
    将分裂特征索引与归一化阈值格式化为可读条件字符串。

    参数
    ----
    feat_idx : int
        分裂特征在 ``feature_names`` 中的索引。
    norm_threshold : float
        归一化后的分裂阈值。
    feature_names : List[str]
        特征名列表。
    preprocessor : Any | None
        可选预处理器，用于反归一化及语义离散化标签。

    返回
    ----
    str
        如 ``"敌机距离.水平距离_km <= 40.000（极低）"`` 的条件文本。
    """
    feat_name = feature_names[feat_idx] if 0 <= feat_idx < len(feature_names) else f"feature_{feat_idx}"
    if preprocessor is not None:
        try:
            raw_th = float(preprocessor.denormalize_threshold(feat_idx, norm_threshold))
            sem = str(preprocessor.discretize_label(str(preprocessor.get_feature_name(feat_idx)), raw_th))
            return f"{feat_name} <= {raw_th:.3f}（{sem}）"
        except Exception:
            pass
    return f"{feat_name} <= {norm_threshold:.3f}"


def _format_action_label(action: object) -> str:
    """
    将训练标签格式化为可读中文动作描述。

    支持 holistic JSON 决策组合及旧版 tuple 字符串格式；超长时截断。

    参数
    ----
    action : object
        原始类别标签（任意可序列化对象）。

    返回
    ----
    str
        格式化后的中文动作标签，最长 48 字符。
    """
    from src.module_c_counterfactual.agent_schema import format_holistic_action_label

    s = format_holistic_action_label(action)
    if len(s) > 48:
        return s[:47] + "…"
    return s


def _render_dot_file(dot_data: str, out_base: Path, fmt: str) -> str:
    """
    调用系统 ``dot`` 命令将 DOT 数据渲染为图片文件。

    参数
    ----
    dot_data : str
        完整 DOT 源字符串。
    out_base : Path
        输出基路径（无扩展名）。
    fmt : str
        输出格式，``"pdf"`` 或 ``"png"``。

    返回
    ----
    str
        实际写入的文件路径。

    异常
    ----
    RuntimeError
        ``dot`` 命令执行失败时抛出。
    """
    out_file = _choose_writable_output_path(out_base, fmt)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".dot",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(dot_data)
        dot_path = f.name
    try:
        dot_fmt = f"{fmt}:cairo" if fmt == "pdf" else fmt
        proc = subprocess.run(
            ["dot", f"-T{dot_fmt}", dot_path, "-o", str(out_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "dot failed")
        return str(out_file)
    finally:
        Path(dot_path).unlink(missing_ok=True)


def _export_graphviz_python(dot_data: str, out_path: str, fmt: str) -> str:
    """
    使用 Python ``graphviz`` 包渲染 DOT 数据。

    参数
    ----
    dot_data : str
        完整 DOT 源字符串。
    out_path : str
        输出路径（可含扩展名）。
    fmt : str
        输出格式，``"pdf"`` 或 ``"png"``。

    返回
    ----
    str
        渲染后的文件路径。
    """
    import graphviz

    source = graphviz.Source(dot_data)
    base = Path(out_path)
    target = _choose_writable_output_path(base, fmt).with_suffix("")
    rendered = source.render(filename=str(target), format=fmt, cleanup=True)
    suffix = f".{fmt}"
    if str(rendered).endswith(suffix):
        return str(rendered)
    return str(out_path) + suffix


def _truncate_labels(names: List[str], max_len: int = 24) -> List[str]:
    """
    截断过长标签，末尾以省略号替代。

    参数
    ----
    names : List[str]
        原始标签列表。
    max_len : int
        单条标签最大字符数。

    返回
    ----
    List[str]
        截断后的标签列表。
    """
    return [
        n if len(n) <= max_len else n[: max_len - 1] + "…"
        for n in map(str, names)
    ]


def _estimate_visible_leaves(tree: DecisionTreeClassifier, max_depth: int) -> int:
    """
    估算展示深度内的可见叶节点数量。

    参数
    ----
    tree : DecisionTreeClassifier
        决策树模型。
    max_depth : int
        展示最大深度。

    返回
    ----
    int
        可见叶节点数，至少为 1。
    """
    visible = _visible_nodes(tree, max_depth)
    t = tree.tree_
    leaves = 0
    for nid in visible:
        left = int(t.children_left[nid])
        if left == _tree.TREE_LEAF or left not in visible:
            leaves += 1
    return max(leaves, 1)


def _style_matplotlib_tree(ax) -> None:
    """
    为 matplotlib 绘制的决策树应用统一视觉样式。

    参数
    ----
    ax : matplotlib.axes.Axes
        决策树绑定的坐标轴对象。
    """
    ax.set_facecolor("#f8fafc")
    for coll in ax.collections:
        try:
            coll.set_edgecolor("#475569")
            coll.set_linewidth(0.9)
        except Exception:
            pass
    for patch in ax.patches:
        try:
            patch.set_linewidth(0.9)
            patch.set_edgecolor("#475569")
        except Exception:
            pass


def _export_matplotlib(
    tree: DecisionTreeClassifier,
    out_path: str,
    feature_names: List[str],
    readable_class_names: List[str],
    raw_class_names: List[str],
    title: str,
    display_max_depth: int,
    file_format: str,
    preprocessor: Optional[Any] = None,
) -> str:
    """
    使用 sklearn ``plot_tree`` + matplotlib 导出决策树图。

    根据可见叶节点数自适应画布尺寸，并配置中文字体。

    参数
    ----
    tree : DecisionTreeClassifier
        决策树模型。
    out_path : str
        输出文件路径。
    feature_names : List[str]
        特征名列表（已截断）。
    readable_class_names : List[str]
        可读中文类别名。
    raw_class_names : List[str]
        原始训练标签列表。
    title : str
        图标题。
    display_max_depth : int
        展示最大深度。
    file_format : str
        文件格式，``"pdf"`` 或 ``"png"``。
    preprocessor : Any | None
        可选预处理器（当前 matplotlib 路径未直接使用）。

    返回
    ----
    str
        写入的文件路径。
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "FangSong", "Arial Unicode MS"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#f8fafc",
        "pdf.fonttype": 42,
    })

    depth_show = min(display_max_depth, tree.get_depth())
    n_leaves = _estimate_visible_leaves(tree, display_max_depth)
    fig_w = max(32.0, min(96.0, n_leaves * 5.0))
    fig_h = max(14.0, min(36.0, 2.8 * depth_show + 6.0))
    font_size = max(7, min(10, 12 - depth_show // 2))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=readable_class_names,
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=font_size,
        impurity=False,
        proportion=True,
        precision=2,
        max_depth=display_max_depth,
    )
    _style_matplotlib_tree(ax)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", color="#1e293b", y=0.998)
    note = (
        "说明：叶节点无分裂条件；节点 class 为当前样本最多动作类型。"
        f" 共 {len(raw_class_names)} 类。"
    )
    fig.text(0.01, 0.01, note, fontsize=9, color="#475569")
    fig.subplots_adjust(top=0.92 if title else 0.98, left=0.02, right=0.98, bottom=0.04)
    plt.tight_layout(pad=2.0)

    final_path = str(_choose_writable_output_path(Path(out_path).with_suffix(""), file_format))
    Path(final_path).parent.mkdir(parents=True, exist_ok=True)
    dpi = 120 if file_format == "pdf" else 200
    plt.savefig(
        final_path,
        format=file_format,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    return final_path


def _choose_writable_output_path(out_base: Path, fmt: str) -> Path:
    """
    选择可写的输出文件路径；若目标被占用则追加时间戳后缀。

    参数
    ----
    out_base : Path
        输出基路径（无扩展名）。
    fmt : str
        文件扩展名（不含点），如 ``"pdf"``。

    返回
    ----
    Path
        最终可写的输出路径。
    """
    cand = out_base.with_suffix(f".{fmt}")
    if _can_open_for_write(cand):
        return cand
    ts = int(time.time() * 1000)
    return out_base.with_name(f"{out_base.name}_{ts}").with_suffix(f".{fmt}")


def _can_open_for_write(path: Path) -> bool:
    """
    检测路径是否可追加写入（文件未被锁定等）。

    参数
    ----
    path : Path
        待检测的文件路径。

    返回
    ----
    bool
        可写时为 ``True``。
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "ab"):
            pass
        return True
    except OSError:
        return False


def _write_viz_legend(
    rendered_path: str,
    feature_names: List[str],
    raw_class_names: List[str],
    readable_class_names: List[str],
) -> None:
    """
    在渲染文件旁生成 ``_legend.txt`` 图例说明文件。

    参数
    ----
    rendered_path : str
        已渲染的决策树图片路径。
    feature_names : List[str]
        完整特征名列表。
    raw_class_names : List[str]
        原始训练标签列表。
    readable_class_names : List[str]
        可读中文类别名列表。
    """
    out = Path(rendered_path)
    legend_path = out.with_name(f"{out.stem}_legend.txt")
    lines = [
        "决策树图例说明",
        "",
        "1) 非叶节点第一行是判断条件；叶节点没有判断条件（这是正常的）。",
        "2) 主导动作 = 该节点训练样本中出现次数最多的动作类型（与规则 THEN 一致）。",
        "3) 图中不再展示 value 长数组，避免误读为矩阵。",
        "",
        "[特征名]",
        *feature_names,
        "",
        "[动作类型（训练标签）]",
    ]
    for i, raw in enumerate(raw_class_names):
        short = readable_class_names[i] if i < len(readable_class_names) else str(raw)
        lines.append(f"- {short}")
        if short != str(raw):
            lines.append(f"  原始: {raw}")
    legend_path.write_text("\n".join(lines), encoding="utf-8")
