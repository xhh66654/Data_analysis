"""
VIPER 加权模仿 + CART 决策树：流水线最终阶段（规则与树图导出）。

核心流程（对应 VIPER 论文思路）：
  1. 从轨迹 CSV 取 X=s_0…s_7、y=action（标签始终来自 CSV，不用 Q 网络直接决策）
  2. 读取 weights.csv，多轮有放回按 weights 重采样得到 D'
  3. 在 D' 上训练 sklearn DecisionTreeClassifier（CART）
  4. 在全量数据上评估 acc_full 与重采样集 acc_resampled，可选选 acc_full 最优轮
  5. 将最优树导出为 IF-THEN 规则、tree.json、tree_nodes.csv、Graphviz PDF/DOT

主要类型与函数：
  ViperConfig / ViperRunResult — 配置与多轮汇总结果
  run_viper_loop()             — VIPER 外循环（重采样→训树→评估）
  run_viper_from_files()       — 从 csv + weights.csv 一键跑 VIPER 并写 viper_out/
  extract_rules()              — 从 CART 提取可读 IF-THEN 规则（调用 DT/dt_auto_pipeline）
  export_tree_artifacts()      — 导出 tree.json、DOT、PDF、可选 PNG

依赖：sklearn；树图渲染复用项目根目录 DT/dt_auto_pipeline.py 的 Graphviz 逻辑。
"""
from __future__ import annotations

import copy
import importlib.util
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from .trajectory_io import ACTION_COL, STATE_COLS, load_trajectory_csv
from .weights import WEIGHTS_COL, sample_row_indices

logger = logging.getLogger(__name__)


def _load_dt_auto_pipeline():
    dt_path = Path(__file__).resolve().parents[2] / "DT" / "dt_auto_pipeline.py"
    if not dt_path.is_file():
        raise FileNotFoundError(f"未找到 DT 规则模块: {dt_path}")
    spec = importlib.util.spec_from_file_location("dt_auto_pipeline", dt_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {dt_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """X: s_0…s_7；y: action（与轨迹 CSV 行对齐）。"""
    x = df[STATE_COLS].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    y = pd.to_numeric(df[ACTION_COL], errors="coerce").values.astype(np.int64)
    if np.isnan(x).any():
        raise ValueError("特征列含 NaN")
    if np.isnan(y).any():
        raise ValueError("动作列含 NaN")
    return x, y


def load_weights_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    if WEIGHTS_COL not in df.columns:
        raise ValueError(f"weights CSV 缺少列 {WEIGHTS_COL!r}")
    w = pd.to_numeric(df[WEIGHTS_COL], errors="coerce").values.astype(np.float64)
    if np.isnan(w).any():
        raise ValueError("weights 含 NaN")
    return w


def perturb_weights(
    weights: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """每轮对 weights 加少量乘性噪声后重新归一化，避免每轮抽到完全相同的 D'。"""
    if noise_std <= 0:
        return np.asarray(weights, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).copy()
    w *= 1.0 + rng.normal(0.0, noise_std, size=w.size)
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    if s <= 0 or not np.isfinite(s):
        logger.warning("扰动后 weights 无效，回退为原始 weights")
        w = np.asarray(weights, dtype=np.float64)
        s = float(w.sum())
    return w / s


def resample_xy(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    n_samples: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    步骤 3：idx = choice(N, size=M, replace=True, p=weights)
    返回 X', y' 及所用行索引 idx。
    """
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError("X 与 y 行数不一致")
    m = n if n_samples is None else int(n_samples)
    if m < 1:
        raise ValueError(f"M 须 >= 1，得到 {m}")
    gen = rng if rng is not None else np.random.default_rng()
    idx = sample_row_indices(weights, m, rng=gen, replace=True)
    return x[idx], y[idx], idx


def train_cart(
    x_prime: np.ndarray,
    y_prime: np.ndarray,
    *,
    max_depth: int = 6,
    min_samples_leaf: int = 1,
    min_samples_split: int = 2,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """步骤 4：CART 拟合 (s → action)。"""
    kw: dict = {"max_depth": max_depth, "random_state": random_state}
    if min_samples_leaf > 1:
        kw["min_samples_leaf"] = int(min_samples_leaf)
    if min_samples_split > 2:
        kw["min_samples_split"] = int(min_samples_split)
    model = DecisionTreeClassifier(**kw)
    model.fit(x_prime, y_prime)
    return model


@dataclass
class ViperRoundResult:
    round_index: int
    train_accuracy_resampled: float
    full_data_accuracy: float
    n_resampled: int


@dataclass
class ViperRunResult:
    model: DecisionTreeClassifier
    rounds: list[ViperRoundResult] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    selected_round: int = 1
    selected_acc_full: float = 0.0
    selected_acc_resampled: float = 0.0


@dataclass
class ViperConfig:
    n_round: int = 5
    max_depth: int = 6
    min_samples_leaf: int = 1
    min_samples_split: int = 2
    random_state: int = 42
    resample_size: int | None = None  # None → M = N
    weight_noise_std: float = 0.01
    weighted_sampling: bool = True  # True=按 weights 加权；False=均匀 1/n
    pick_best_by_full_acc: bool = True
    class_mapping: dict | None = None
    export_tree: bool = True
    # 与 DT/dt_auto_pipeline 一致：Graphviz 导出 .dot 并渲染 PDF 流程图
    render_tree_pdf: bool = True
    # 从同一 .dot 再渲染 PNG（样式与 PDF 一致）；失败时可回退 matplotlib
    render_tree_png: bool = False
    tree_image_dpi: int = 150
    # 对应 dt 的 --open-pdf：渲染成功后用系统默认程序打开 PDF（无 PDF 则尝试 PNG）
    open_tree_pdf: bool = False
    show_tree_image: bool = False  # 同 open_tree_pdf，保留兼容


def run_viper_loop(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    cfg: ViperConfig,
) -> ViperRunResult:
    """
    步骤 5：固定 weights，多轮「重采样 → CART」。
    默认按全量准确率 acc_full 选取最优轮（pick_best_by_full_acc=True）。
    """
    if cfg.n_round < 1:
        raise ValueError(f"n_round 须 >= 1，得到 {cfg.n_round}")
    n = x.shape[0]
    if weights.shape[0] != n:
        raise ValueError("weights 长度须与样本数 N 一致")

    rng = np.random.default_rng(cfg.random_state)
    m = cfg.resample_size if cfg.resample_size is not None else n
    rounds: list[ViperRoundResult] = []
    best_model: DecisionTreeClassifier | None = None
    best_round = 1
    best_full = -1.0
    best_resampled = 0.0
    last_model: DecisionTreeClassifier | None = None

    for r in range(1, cfg.n_round + 1):
        w_round = perturb_weights(weights, cfg.weight_noise_std, rng)
        xp, yp, _idx = resample_xy(x, y, w_round, n_samples=m, rng=rng)
        model = train_cart(
            xp,
            yp,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            min_samples_split=cfg.min_samples_split,
            random_state=cfg.random_state + r,
        )
        acc_resampled = float(accuracy_score(yp, model.predict(xp)))
        acc_full = float(accuracy_score(y, model.predict(x)))
        rounds.append(
            ViperRoundResult(
                round_index=r,
                train_accuracy_resampled=acc_resampled,
                full_data_accuracy=acc_full,
                n_resampled=m,
            )
        )
        logger.info(
            "VIPER round %d/%d acc_resampled=%.4f acc_full=%.4f",
            r,
            cfg.n_round,
            acc_resampled,
            acc_full,
        )
        last_model = model
        if cfg.pick_best_by_full_acc:
            if acc_full > best_full + 1e-9 or (
                abs(acc_full - best_full) <= 1e-9 and acc_resampled > best_resampled
            ):
                best_full = acc_full
                best_resampled = acc_resampled
                best_round = r
                best_model = copy.deepcopy(model)
        else:
            best_model = model
            best_full = acc_full
            best_resampled = acc_resampled
            best_round = r

    assert last_model is not None
    if cfg.pick_best_by_full_acc:
        assert best_model is not None
        selected = best_model
        logger.info(
            "VIPER 选用第 %d 轮（acc_full=%.4f acc_resampled=%.4f）",
            best_round,
            best_full,
            best_resampled,
        )
    else:
        selected = last_model
        best_round = rounds[-1].round_index
        best_full = rounds[-1].full_data_accuracy
        best_resampled = rounds[-1].train_accuracy_resampled

    return ViperRunResult(
        model=selected,
        rounds=rounds,
        feature_names=list(STATE_COLS),
        selected_round=best_round,
        selected_acc_full=best_full,
        selected_acc_resampled=best_resampled,
    )


def _to_name_fn(class_mapping: dict | None = None):
    dtp = _load_dt_auto_pipeline()
    mapping = class_mapping if class_mapping is not None else dtp.CLASS_MAPPING
    return lambda c: dtp.to_display_name(c, mapping)


def extract_rules(
    model: DecisionTreeClassifier,
    feature_names: list[str] | None = None,
    class_mapping: dict | None = None,
) -> list[str]:
    """步骤 6：DFS 提取 IF-THEN 规则（对齐 DT/dt_auto_pipeline）。"""
    dtp = _load_dt_auto_pipeline()
    fn = feature_names if feature_names is not None else list(STATE_COLS)
    to_name = _to_name_fn(class_mapping)
    return dtp.extract_decision_rules_if_then(model, list(fn), to_name)


def export_tree_nodes_df(
    model: DecisionTreeClassifier,
    feature_names: list[str] | None = None,
    class_mapping: dict | None = None,
) -> pd.DataFrame:
    """导出决策树节点表（与 DT/dt_auto_pipeline 一致）。"""
    dtp = _load_dt_auto_pipeline()
    fn = feature_names if feature_names is not None else list(STATE_COLS)
    to_name = _to_name_fn(class_mapping)
    return dtp.extract_tree_structure_strong(
        model,
        feature_names=list(fn),
        model_classes=np.array(model.classes_),
        to_name=to_name,
        weighted_counts=False,
        print_count_check=False,
    )


def build_nested_tree_json(
    model: DecisionTreeClassifier,
    feature_names: list[str] | None = None,
    class_mapping: dict | None = None,
) -> dict:
    """将 sklearn 决策树构造成嵌套 JSON 树（便于前端或反事实模块读取）。"""
    from sklearn.tree import _tree

    tree = model.tree_
    fn = feature_names if feature_names is not None else list(STATE_COLS)
    to_name = _to_name_fn(class_mapping)
    classes = list(model.classes_)

    def _leaf_dist(node_id: int) -> dict[str, int]:
        dist = tree.value[node_id][0]
        return {to_name(classes[j]): int(dist[j]) for j in range(len(classes))}

    def _walk(node_id: int) -> dict:
        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])
        if left == right:
            dist = _leaf_dist(node_id)
            dist_raw = tree.value[node_id][0]
            pred_i = int(np.argmax(dist_raw))
            pred_name = to_name(classes[pred_i])
            return {
                "type": "leaf",
                "node_id": node_id,
                "prediction": pred_name,
                "class_counts": dist,
                "n_samples": int(tree.n_node_samples[node_id]),
            }
        fi = int(tree.feature[node_id])
        thr = float(tree.threshold[node_id])
        fname = fn[fi] if fi != _tree.TREE_UNDEFINED else "?"
        return {
            "type": "split",
            "node_id": node_id,
            "feature": fname,
            "threshold": thr,
            "n_samples": int(tree.n_node_samples[node_id]),
            "left": _walk(left),
            "right": _walk(right),
            "left_branch": f"{fname} <= {thr}",
            "right_branch": f"{fname} > {thr}",
        }

    return {
        "root": _walk(0),
        "feature_names": list(fn),
        "n_nodes": int(tree.node_count),
        "n_leaves": int(model.get_n_leaves()),
        "max_depth": int(tree.max_depth),
    }


def export_tree_matplotlib_png(
    model: DecisionTreeClassifier,
    out_dir: Path,
    *,
    feature_names: list[str] | None = None,
    class_mapping: dict | None = None,
    prefix: str = "policy_tree",
    dpi: int = 150,
) -> Path | None:
    """用 sklearn.tree.plot_tree + matplotlib 保存 PNG（无需安装 Graphviz 可执行文件）。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree
    except ImportError as exc:
        logger.warning("无法导出 PNG（需 matplotlib）：%s", exc)
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = feature_names if feature_names is not None else list(STATE_COLS)
    to_name = _to_name_fn(class_mapping)
    class_names = [str(to_name(c)) for c in model.classes_]

    depth = max(1, int(model.get_depth()))
    fig_w = float(min(56.0, 10.0 + depth * 4.5))
    fig_h = float(min(42.0, 6.0 + depth * 3.0))

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fontsize = max(6, min(11, 14 - depth))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plot_tree(
        model,
        feature_names=list(fn),
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=fontsize,
        ax=ax,
        impurity=False,
        proportion=False,
    )
    fig.tight_layout()
    png_path = out_dir / f"{prefix}.png"
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    logger.info("已生成决策树 PNG: %s", png_path.resolve())
    return png_path


def open_image_file(path: Path) -> None:
    """用系统默认程序打开图片/PDF（便于本地查看）。"""
    import os
    import subprocess
    import sys

    p = Path(path)
    if not p.is_file():
        logger.warning("文件不存在，跳过打开: %s", p)
        return
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(p.resolve()))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(p)], check=False)
        else:
            subprocess.run(["xdg-open", str(p)], check=False)
    except Exception as exc:
        logger.warning("无法打开查看器: %s", exc)


def build_tree_dot_data(
    model: DecisionTreeClassifier,
    feature_names: list[str] | None = None,
    class_mapping: dict | None = None,
) -> tuple[str, list[str]]:
    """
    与 DT/dt_auto_pipeline.py 相同：sklearn export_graphviz + 中文标签替换。
    """
    from sklearn.tree import export_graphviz

    fn = feature_names if feature_names is not None else list(STATE_COLS)
    to_name = _to_name_fn(class_mapping)
    class_names = [str(to_name(c)) for c in model.classes_]

    dot_data = export_graphviz(
        model,
        feature_names=list(fn),
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
    )
    dot_data = (
        dot_data.replace("gini =", "基尼系数 =")
        .replace("samples =", "样本数 =")
        .replace("value =", "类别分布 =")
    )
    return dot_data, list(fn)


def export_tree_flowchart_dt_style(
    model: DecisionTreeClassifier,
    out_dir: Path,
    *,
    feature_names: list[str] | None = None,
    class_mapping: dict | None = None,
    prefix: str = "policy_tree",
    render_pdf: bool = True,
    render_png: bool = False,
    open_viewer: bool = False,
) -> dict[str, Path]:
    """
    决策树流程图展示，对齐 DT/dt_auto_pipeline.py：
      1. 写 {prefix}_debug.dot
      2. graphviz 渲染 {prefix}.pdf（filled/rounded 方框流程图）
      3. 可选再渲染 {prefix}.png（与 PDF 同源）
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dot_data, _fn = build_tree_dot_data(model, feature_names, class_mapping)
    dot_path = out_dir / f"{prefix}_debug.dot"
    dot_path.write_text(dot_data, encoding="utf-8")
    paths: dict[str, Path] = {"tree_dot": dot_path}

    dot_for_gv = dot_data.replace("helvetica", "Microsoft YaHei")
    pdf_base = out_dir / prefix
    png_base = out_dir / prefix

    def _render(fmt: str, base: Path, *, view: bool) -> Path | None:
        try:
            import graphviz as gv_mod

            graph = gv_mod.Source(dot_for_gv, encoding="utf-8")
            out = Path(graph.render(filename=str(base), view=view, format=fmt, cleanup=True))
            logger.info("已生成决策树 %s: %s", fmt.upper(), out.resolve())
            return out
        except ModuleNotFoundError:
            logger.warning(
                "[WARN] 未安装 Python 包 graphviz（pip install graphviz），已跳过 %s 渲染；仍可使用 .dot。",
                fmt.upper(),
            )
        except Exception as exc:
            logger.warning(
                "[WARN] 无法渲染 %s（通常系统未安装 Graphviz 或未加入 PATH）：%s\n"
                "       已保留 %s，可用在线工具或其它方式生成。",
                fmt.upper(),
                exc,
                dot_path.name,
            )
        return None

    if render_pdf:
        pdf = _render("pdf", pdf_base, view=open_viewer)
        if pdf is not None:
            paths["tree_pdf"] = pdf

    if render_png:
        # 与 PDF 同一套 Graphviz 流程图，仅格式不同
        png = _render("png", png_base, view=False)
        if png is not None:
            paths["tree_png"] = png

    return paths


def export_tree_graphviz(
    model: DecisionTreeClassifier,
    out_dir: Path,
    *,
    feature_names: list[str] | None = None,
    class_mapping: dict | None = None,
    prefix: str = "policy_tree",
    render_pdf: bool = True,
    render_png: bool = False,
    open_viewer: bool = False,
) -> dict[str, Path]:
    """兼容旧名；内部走 export_tree_flowchart_dt_style。"""
    return export_tree_flowchart_dt_style(
        model,
        out_dir,
        feature_names=feature_names,
        class_mapping=class_mapping,
        prefix=prefix,
        render_pdf=render_pdf,
        render_png=render_png,
        open_viewer=open_viewer,
    )


def export_tree_artifacts(
    model: DecisionTreeClassifier,
    out_dir: str | Path,
    *,
    feature_names: list[str] | None = None,
    class_mapping: dict | None = None,
    render_pdf: bool = True,
    render_png: bool = False,
    tree_image_dpi: int = 150,
    open_viewer: bool = False,
) -> dict[str, Path]:
    """导出节点表 / JSON / DT 风格 Graphviz 流程图（dot+pdf）；PNG 优先 Graphviz，失败再 matplotlib。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_nodes = export_tree_nodes_df(model, feature_names, class_mapping)
    nodes_csv = out_dir / "tree_nodes.csv"
    df_nodes.to_csv(nodes_csv, index=False, encoding="utf-8-sig")

    tree_json = build_nested_tree_json(model, feature_names, class_mapping)
    tree_json_path = out_dir / "tree.json"
    tree_json_path.write_text(
        json.dumps(tree_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    paths = {
        "tree_nodes_csv": nodes_csv,
        "tree_json": tree_json_path,
        **export_tree_flowchart_dt_style(
            model,
            out_dir,
            feature_names=feature_names,
            class_mapping=class_mapping,
            render_pdf=render_pdf,
            render_png=render_png,
            open_viewer=open_viewer,
        ),
    }
    if render_png and "tree_png" not in paths:
        png = export_tree_matplotlib_png(
            model,
            out_dir,
            feature_names=feature_names,
            class_mapping=class_mapping,
            dpi=tree_image_dpi,
        )
        if png is not None:
            paths["tree_png"] = png
    logger.info(
        "已导出决策树结构: nodes=%s json=%s",
        nodes_csv.resolve(),
        tree_json_path.resolve(),
    )
    return paths


def save_viper_outputs(
    result: ViperRunResult,
    out_dir: str | Path,
    *,
    class_mapping: dict | None = None,
    export_tree: bool = True,
    render_tree_pdf: bool = True,
    render_tree_png: bool = False,
    tree_image_dpi: int = 150,
    open_tree_pdf: bool = False,
    show_tree_image: bool = False,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rules = extract_rules(result.model, result.feature_names, class_mapping)
    result.rules = rules

    rules_txt = out_dir / "rules.txt"
    rules_txt.write_text("\n".join(rules) + ("\n" if rules else ""), encoding="utf-8")

    rules_json = out_dir / "rules.json"
    rules_json.write_text(
        json.dumps(rules, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "feature_names": result.feature_names,
        "n_rules": len(rules),
        "selected_round": result.selected_round,
        "selected_acc_full": result.selected_acc_full,
        "selected_acc_resampled": result.selected_acc_resampled,
        "cart": {
            "max_depth": int(result.model.get_depth()),
            "n_leaves": int(result.model.get_n_leaves()),
        },
        "rounds": [
            {
                "round": rd.round_index,
                "train_accuracy_resampled": rd.train_accuracy_resampled,
                "full_data_accuracy": rd.full_data_accuracy,
                "n_resampled": rd.n_resampled,
            }
            for rd in result.rounds
        ],
    }
    summary_path = out_dir / "viper_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    paths: dict[str, Path] = {
        "rules_txt": rules_txt,
        "rules_json": rules_json,
        "summary_json": summary_path,
    }

    if export_tree:
        open_view = bool(open_tree_pdf or show_tree_image)
        paths.update(
            export_tree_artifacts(
                result.model,
                out_dir,
                feature_names=result.feature_names,
                class_mapping=class_mapping,
                render_pdf=render_tree_pdf,
                render_png=render_tree_png,
                tree_image_dpi=tree_image_dpi,
                open_viewer=open_view,
            )
        )
        summary["tree"] = {
            "n_nodes": int(result.model.tree_.node_count),
            "n_leaves": int(result.model.get_n_leaves()),
            "tree_json": str(paths.get("tree_json", "")),
            "tree_nodes_csv": str(paths.get("tree_nodes_csv", "")),
            "tree_dot": str(paths.get("tree_dot", "")),
            "tree_pdf": str(paths.get("tree_pdf", "")),
            "tree_png": str(paths.get("tree_png", "")),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if open_view and "tree_pdf" not in paths:
            for key in ("tree_png",):
                pth = paths.get(key)
                if isinstance(pth, Path) and pth.is_file():
                    open_image_file(pth)
                    break

    logger.info("已保存 rules: %s (%d 条)", rules_txt.resolve(), len(rules))
    return paths


def run_viper_from_files(
    csv_path: str | Path,
    weights_path: str | Path,
    out_dir: str | Path,
    cfg: ViperConfig,
) -> ViperRunResult:
    df = load_trajectory_csv(str(csv_path))
    w_df = pd.read_csv(weights_path, encoding="utf-8-sig")
    if len(w_df) != len(df):
        raise ValueError(
            f"weights 行数 {len(w_df)} 与轨迹 CSV {len(df)} 不一致，请使用同一轨迹生成的 weights"
        )
    x, y = build_xy(df)
    weights = load_weights_array(weights_path)
    if not cfg.weighted_sampling:
        weights = np.full(len(weights), 1.0 / len(weights), dtype=np.float64)
    result = run_viper_loop(x, y, weights, cfg)
    save_viper_outputs(
        result,
        out_dir,
        class_mapping=cfg.class_mapping,
        export_tree=cfg.export_tree,
        render_tree_pdf=cfg.render_tree_pdf,
        render_tree_png=cfg.render_tree_png,
        tree_image_dpi=cfg.tree_image_dpi,
        open_tree_pdf=cfg.open_tree_pdf,
        show_tree_image=cfg.show_tree_image,
    )
    return result
