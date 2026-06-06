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
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.tree import DecisionTreeClassifier

from .trajectory_io import ACTION_COL, EPISODE_COL, STATE_COLS, load_trajectory_csv
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


def split_by_episode(
    episodes: np.ndarray,
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[DATA-CROP-06] 按 episode 整组划分 train/val/test，避免同一回合的相邻样本泄漏到不同集合。

    决策树仅在 train 索引子集上训练；val/test 不参与 CART.fit。
    返回三组「行索引」数组（对 df 行号）。val_frac/test_frac 为 0 时对应集合为空。
    """
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1.0:
        raise ValueError(f"val_frac+test_frac 须 <1 且非负，得到 {val_frac}+{test_frac}")
    ep = np.asarray(episodes)
    uniq = np.unique(ep)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_ep = len(uniq)
    n_test = int(round(n_ep * test_frac))
    n_val = int(round(n_ep * val_frac))
    test_eps = set(uniq[:n_test].tolist())
    val_eps = set(uniq[n_test : n_test + n_val].tolist())

    is_test = np.isin(ep, list(test_eps)) if test_eps else np.zeros(len(ep), dtype=bool)
    is_val = np.isin(ep, list(val_eps)) if val_eps else np.zeros(len(ep), dtype=bool)
    is_train = ~(is_test | is_val)
    train_idx = np.flatnonzero(is_train)
    val_idx = np.flatnonzero(is_val)
    test_idx = np.flatnonzero(is_test)
    logger.info(
        "按 episode 划分: episodes=%d → train=%d val=%d test=%d 行 (train_ep=%d val_ep=%d test_ep=%d)",
        n_ep,
        train_idx.size,
        val_idx.size,
        test_idx.size,
        n_ep - n_test - n_val,
        n_val,
        n_test,
    )
    return train_idx, val_idx, test_idx


def _selection_score(model: DecisionTreeClassifier, x: np.ndarray, y: np.ndarray, metric: str) -> float:
    pred = model.predict(x)
    if metric == "macro_f1":
        return float(f1_score(y, pred, average="macro", zero_division=0))
    return float(accuracy_score(y, pred))


def compute_metrics(
    model: DecisionTreeClassifier,
    x: np.ndarray,
    y: np.ndarray,
    *,
    class_mapping: dict | None = None,
) -> dict:
    """在给定数据集上评估模型：accuracy / balanced_accuracy / macro-F1 / 每类 + 混淆矩阵。"""
    if x.shape[0] == 0:
        return {"n": 0}
    pred = model.predict(x)
    labels = sorted(np.unique(np.concatenate([y, pred])).tolist())
    to_name = _to_name_fn(class_mapping)
    per_f1 = f1_score(y, pred, average=None, labels=labels, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    return {
        "n": int(x.shape[0]),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "per_class_f1": {str(to_name(c)): float(v) for c, v in zip(labels, per_f1)},
        "labels": [str(to_name(c)) for c in labels],
        "confusion_matrix": cm.astype(int).tolist(),
    }


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
    """[DATA-CROP-07] 按 weights 有放回抽取 n_samples 行 → 供 CART.fit 的 (xp, yp)。"""
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
    class_weight: str | dict | None = None,
    ccp_alpha: float = 0.0,
) -> DecisionTreeClassifier:
    """步骤 4：CART 拟合 (s → action)。

    class_weight: None | "balanced"，应对动作类别不均衡。
    ccp_alpha:    代价复杂度剪枝系数（>0 时启用），用准确率/树规模的帕累托剪枝
                  替代死扣 max_depth；越大树越小。
    """
    kw: dict = {"max_depth": max_depth, "random_state": random_state}
    if min_samples_leaf > 1:
        kw["min_samples_leaf"] = int(min_samples_leaf)
    if min_samples_split > 2:
        kw["min_samples_split"] = int(min_samples_split)
    if class_weight is not None:
        kw["class_weight"] = class_weight
    if ccp_alpha and ccp_alpha > 0:
        kw["ccp_alpha"] = float(ccp_alpha)
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
class ViperRoundFullResult:
    """每轮的完整结果（用于规则集成）"""
    round_index: int
    model: DecisionTreeClassifier
    acc_resampled: float
    acc_full: float
    rules: list[str]


@dataclass
class ViperRunResult:
    model: DecisionTreeClassifier
    rounds: list[ViperRoundResult] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    selected_round: int = 1
    selected_acc_full: float = 0.0
    selected_acc_resampled: float = 0.0
    # 每轮完整结果（用于规则集成）
    round_results: list[ViperRoundFullResult] = field(default_factory=list)
    # train/val/test 评估指标（由 run_viper_from_files 填充）
    metrics: dict = field(default_factory=dict)
    # 导出树图/节点表时用于统计「真实」类别分布（与 class_weight 训练权重无关）
    display_x: np.ndarray | None = None
    display_y: np.ndarray | None = None


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
    # 应对类别不均衡：None | "balanced"
    class_weight: str | dict | None = None
    # 代价复杂度剪枝；>0 时优先于 max_depth 控制树规模
    ccp_alpha: float = 0.0
    # 选轮指标："acc"=准确率（兼容旧行为）；"macro_f1"=宏平均 F1（不均衡更稳）
    selection_metric: str = "acc"
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
    *,
    x_eval: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
    data_flow: Any = None,
) -> ViperRunResult:
    """
    步骤 5：固定 weights，多轮「重采样 → CART」。

    选轮：在「评估集」(x_eval,y_eval) 上按 cfg.selection_metric 选最优轮。
          若未提供评估集，则回退到训练集 (x,y)（兼容旧行为，存在乐观偏差）。
    """
    if cfg.n_round < 1:
        raise ValueError(f"n_round 须 >= 1，得到 {cfg.n_round}")
    n = x.shape[0]
    if weights.shape[0] != n:
        raise ValueError("weights 长度须与样本数 N 一致")

    use_eval = x_eval is not None and y_eval is not None and x_eval.shape[0] > 0
    xe = x_eval if use_eval else x
    ye = y_eval if use_eval else y

    rng = np.random.default_rng(cfg.random_state)
    m = cfg.resample_size if cfg.resample_size is not None else n

    # [DATA-CROP-07] 每轮 CART.fit 使用 m 条有放回 bootstrap 样本（非 train 池唯一行数）
    if data_flow is not None:
        from .data_flow import record_viper_bootstrap

        record_viper_bootstrap(
            data_flow,
            n,
            m,
            resample_size=cfg.resample_size,
            n_round=cfg.n_round,
        )

    rounds: list[ViperRoundResult] = []
    round_results: list[ViperRoundFullResult] = []  # 用于规则集成
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
            class_weight=cfg.class_weight,
            ccp_alpha=cfg.ccp_alpha,
        )
        acc_resampled = float(accuracy_score(yp, model.predict(xp)))
        # acc_full 复用字段名，实际为「评估集」上的选轮指标（acc 或 macro_f1）
        acc_full = _selection_score(model, xe, ye, cfg.selection_metric)
        
        # 提取当前轮的规则（用于规则集成）
        round_rules = extract_rules(model, list(STATE_COLS), cfg.class_mapping)
        
        rounds.append(
            ViperRoundResult(
                round_index=r,
                train_accuracy_resampled=acc_resampled,
                full_data_accuracy=acc_full,
                n_resampled=m,
            )
        )
        
        # 保存每轮完整结果
        round_results.append(
            ViperRoundFullResult(
                round_index=r,
                model=copy.deepcopy(model),
                acc_resampled=acc_resampled,
                acc_full=acc_full,
                rules=round_rules,
            )
        )
        
        logger.info(
            "VIPER round %d/%d acc_resampled=%.4f eval_%s=%.4f",
            r,
            cfg.n_round,
            acc_resampled,
            cfg.selection_metric,
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
        round_results=round_results,
    )


def refit_final_tree_on_full_data(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    cfg: ViperConfig,
    *,
    selected_round: int,
    data_flow: Any = None,
) -> tuple[DecisionTreeClassifier, list[str], int]:
    """
    在 train/val 上完成 VIPER 选轮后，用**全表**行重训一棵 CART 作为最终导出树。

    这样 rules.txt / policy_tree.pdf 代表全部轨迹，而 metrics.json 仍反映划分集上的泛化评估。
    """
    n = int(x.shape[0])
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != n:
        raise ValueError("全量重训：weights 长度须与样本数一致")
    s = float(w.sum())
    w = w / s if s > 0 else np.full(n, 1.0 / n, dtype=np.float64)

    rng = np.random.default_rng(cfg.random_state + 99_991)
    m = int(cfg.resample_size) if cfg.resample_size is not None else n

    if data_flow is not None:
        from .data_flow import record_full_refit

        record_full_refit(data_flow, n, m, selected_round=selected_round)

    xp, yp, _idx = resample_xy(x, y, w, n_samples=m, rng=rng)
    model = train_cart(
        xp,
        yp,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        min_samples_split=cfg.min_samples_split,
        random_state=cfg.random_state + int(selected_round) + 10_000,
        class_weight=cfg.class_weight,
        ccp_alpha=cfg.ccp_alpha,
    )
    rules = extract_rules(model, list(STATE_COLS), cfg.class_mapping)
    logger.info(
        "全量重训最终决策树: 全表 n=%d bootstrap_m=%d（导出树覆盖 train 子集上的选轮模型）",
        n,
        m,
    )
    return model, rules, m


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


def true_class_counts_per_node(
    model: DecisionTreeClassifier,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """各节点上真实（未加权）类别计数，shape (n_nodes, n_classes)。

    使用 decision_path（非 apply）：apply 只返回叶节点，内部节点无样本计数。
    """
    x = np.asarray(x)
    y = np.asarray(y).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("display_x 与 display_y 行数须一致")
    classes = np.asarray(model.classes_)
    n_nodes = int(model.tree_.node_count)
    n_classes = len(classes)
    counts = np.zeros((n_nodes, n_classes), dtype=np.int64)
    label_idx = np.zeros(y.shape[0], dtype=np.int64)
    for j, c in enumerate(classes):
        label_idx[y == c] = j
    paths = model.decision_path(x)
    for node_id in range(n_nodes):
        mask = np.asarray(paths[:, node_id].todense()).ravel().astype(bool)
        if not np.any(mask):
            continue
        counts[node_id] = np.bincount(label_idx[mask], minlength=n_classes)
    return counts


def gini_from_counts(counts: np.ndarray) -> float:
    s = float(np.sum(counts))
    if s <= 0:
        return 0.0
    p = counts.astype(np.float64) / s
    return float(1.0 - np.sum(p * p))


def _format_count_list(counts: np.ndarray) -> str:
    return "[" + ", ".join(str(int(v)) for v in counts) + "]"


def _format_percent_list(counts: np.ndarray) -> str:
    """类别分布展示为百分比（与 model.classes_ 顺序一致）。"""
    s = float(np.sum(counts))
    if s <= 0:
        return "[" + ", ".join("0.0%" for _ in counts) + "]"
    pcts = 100.0 * counts.astype(np.float64) / s
    return "[" + ", ".join(f"{p:.1f}%" for p in pcts) + "]"


def _node_is_leaf(model: DecisionTreeClassifier, node_id: int) -> bool:
    t = model.tree_
    return int(t.children_left[node_id]) == int(t.children_right[node_id])


def _strip_class_from_label(body: str) -> str:
    """中间节点不展示 class（仅叶节点对应 IF-THEN 规则结论）。"""
    body = re.sub(r"<br/>class = [^<]+", "", body)
    body = re.sub(r"\\nclass = [^\\n]+", "", body)
    return body.rstrip("<br/>")


def export_tree_nodes_df(
    model: DecisionTreeClassifier,
    feature_names: list[str] | None = None,
    class_mapping: dict | None = None,
    *,
    display_x: np.ndarray | None = None,
    display_y: np.ndarray | None = None,
) -> pd.DataFrame:
    """导出决策树节点表；若提供 display_x/y，类别分布列为真实计数（非 class_weight 加权）。"""
    dtp = _load_dt_auto_pipeline()
    fn = feature_names if feature_names is not None else list(STATE_COLS)
    to_name = _to_name_fn(class_mapping)
    classes = np.array(model.classes_)
    df = dtp.extract_tree_structure_strong(
        model,
        feature_names=list(fn),
        model_classes=classes,
        to_name=to_name,
        weighted_counts=False,
        print_count_check=False,
    )
    if display_x is None or display_y is None:
        return df

    true_counts = true_class_counts_per_node(model, display_x, display_y)
    sanitize_col = dtp.sanitize_col

    for i in range(true_counts.shape[0]):
        cnt = true_counts[i]
        s = float(cnt.sum())
        dist = {}
        for j in range(len(classes)):
            name = to_name(classes[j])
            if s > 0:
                dist[name] = f"{100.0 * cnt[j] / s:.1f}%"
            else:
                dist[name] = "0.0%"
        df.at[i, "类别分布JSON"] = json.dumps(dist, ensure_ascii=False)
        df.at[i, "基尼系数"] = gini_from_counts(cnt)
        is_leaf = _node_is_leaf(model, i)
        if is_leaf and cnt.sum() > 0:
            pred_j = int(np.argmax(cnt))
            df.at[i, "预测类别"] = to_name(classes[pred_j])
        else:
            df.at[i, "预测类别"] = ""
        for j, cls in enumerate(classes):
            col = f"count_{sanitize_col(to_name(cls))}"
            if col in df.columns:
                df.at[i, col] = int(true_counts[i, j])
    if "_各类计数之和" in df.columns:
        count_cols = [c for c in df.columns if c.startswith("count_")]
        df["_各类计数之和"] = df[count_cols].sum(axis=1)
    return df


def build_nested_tree_json(
    model: DecisionTreeClassifier,
    feature_names: list[str] | None = None,
    class_mapping: dict | None = None,
    *,
    display_x: np.ndarray | None = None,
    display_y: np.ndarray | None = None,
) -> dict:
    """将 sklearn 决策树构造成嵌套 JSON 树（便于前端或反事实模块读取）。"""
    from sklearn.tree import _tree

    tree = model.tree_
    fn = feature_names if feature_names is not None else list(STATE_COLS)
    to_name = _to_name_fn(class_mapping)
    classes = list(model.classes_)
    true_counts: np.ndarray | None = None
    if display_x is not None and display_y is not None:
        true_counts = true_class_counts_per_node(model, display_x, display_y)

    def _dist(node_id: int) -> dict[str, int]:
        if true_counts is not None:
            return {
                to_name(classes[j]): int(true_counts[node_id, j])
                for j in range(len(classes))
            }
        dist = tree.value[node_id][0]
        return {to_name(classes[j]): int(dist[j]) for j in range(len(classes))}

    def _pred_name(node_id: int) -> str:
        if true_counts is not None:
            cnt = true_counts[node_id]
            pred_i = int(np.argmax(cnt)) if cnt.sum() > 0 else 0
        else:
            pred_i = int(np.argmax(tree.value[node_id][0]))
        return to_name(classes[pred_i])

    def _walk(node_id: int) -> dict:
        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])
        if left == right:
            return {
                "type": "leaf",
                "node_id": node_id,
                "prediction": _pred_name(node_id),
                "class_counts": _dist(node_id),
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
            "class_counts": _dist(node_id),
            "left": _walk(left),
            "right": _walk(right),
            "left_branch": f"{fname} <= {thr}",
            "right_branch": f"{fname} > {thr}",
        }

    out = {
        "root": _walk(0),
        "feature_names": list(fn),
        "n_nodes": int(tree.node_count),
        "n_leaves": int(model.get_n_leaves()),
        "max_depth": int(tree.max_depth),
    }
    if true_counts is not None:
        out["class_counts_note"] = "class_counts 为 apply(X) 后真实标签计数，非 class_weight 加权"
    return out


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


def _patch_dot_label_body(
    body: str,
    nid: int,
    true_counts: np.ndarray,
    class_names: list[str],
    *,
    is_leaf: bool,
) -> str:
    cnt = true_counts[nid]
    body = re.sub(
        r"value = \[[^\]]*\]",
        f"类别分布 = {_format_percent_list(cnt)}",
        body,
    )
    body = re.sub(r"gini = [\d.]+", f"基尼系数 = {gini_from_counts(cnt):.3f}", body)
    if is_leaf:
        pred_j = int(np.argmax(cnt)) if cnt.sum() > 0 else 0
        if re.search(r"class = ", body):
            body = re.sub(r"class = [^<>\n\\]+", f"class = {class_names[pred_j]}", body)
        else:
            sep = "<br/>" if "<br/>" in body else "\\n"
            body = f"{body}{sep}class = {class_names[pred_j]}"
    else:
        body = _strip_class_from_label(body)
    return body


def patch_dot_with_true_class_distribution(
    dot_data: str,
    model: DecisionTreeClassifier,
    x: np.ndarray,
    y: np.ndarray,
    class_mapping: dict | None = None,
) -> str:
    """将 Graphviz 节点中的 value/gini/class 替换为真实标签统计（训练仍可用 class_weight）。

    类别分布为百分比；仅叶节点展示 class（与 IF-THEN 规则结论一致）。
    """
    true_counts = true_class_counts_per_node(model, x, y)
    to_name = _to_name_fn(class_mapping)
    classes = list(model.classes_)
    class_names = [str(to_name(c)) for c in classes]
    re_quoted = re.compile(r'^(\d+) \[label="((?:[^"\\]|\\.)*)"\](.*)$')
    re_html = re.compile(r"^(\d+) \[label=<(.+)>, fillcolor=(.*)$")

    out_lines: list[str] = []
    for line in dot_data.splitlines():
        m = re_quoted.match(line)
        if m:
            nid = int(m.group(1))
            body = _patch_dot_label_body(
                m.group(2),
                nid,
                true_counts,
                class_names,
                is_leaf=_node_is_leaf(model, nid),
            )
            out_lines.append(f'{nid} [label="{body}"{m.group(3)}')
            continue
        m = re_html.match(line)
        if m:
            nid = int(m.group(1))
            body = _patch_dot_label_body(
                m.group(2),
                nid,
                true_counts,
                class_names,
                is_leaf=_node_is_leaf(model, nid),
            )
            out_lines.append(f"{nid} [label=<{body}>, fillcolor={m.group(3)}")
            continue
        out_lines.append(line)
    return "\n".join(out_lines) + ("\n" if dot_data.endswith("\n") else "")


def build_tree_dot_data(
    model: DecisionTreeClassifier,
    feature_names: list[str] | None = None,
    class_mapping: dict | None = None,
    *,
    display_x: np.ndarray | None = None,
    display_y: np.ndarray | None = None,
) -> tuple[str, list[str]]:
    """
    sklearn export_graphviz + 中文标签；若提供 display_x/y，类别分布/基尼/class 为真实标签统计。
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
    if display_x is not None and display_y is not None:
        dot_data = patch_dot_with_true_class_distribution(
            dot_data, model, display_x, display_y, class_mapping
        )
    else:
        dot_data = (
            dot_data.replace("gini =", "基尼系数 =")
            .replace("value =", "类别分布 =")
        )
    dot_data = dot_data.replace("samples =", "样本数 =")
    return dot_data, list(fn)


def export_tree_flowchart_dt_style(
    model: DecisionTreeClassifier,
    out_dir: Path,
    *,
    feature_names: list[str] | None = None,
    class_mapping: dict | None = None,
    display_x: np.ndarray | None = None,
    display_y: np.ndarray | None = None,
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

    dot_data, _fn = build_tree_dot_data(
        model,
        feature_names,
        class_mapping,
        display_x=display_x,
        display_y=display_y,
    )
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


def export_tree_artifacts(
    model: DecisionTreeClassifier,
    out_dir: str | Path,
    *,
    feature_names: list[str] | None = None,
    class_mapping: dict | None = None,
    display_x: np.ndarray | None = None,
    display_y: np.ndarray | None = None,
    render_pdf: bool = True,
    render_png: bool = False,
    tree_image_dpi: int = 150,
    open_viewer: bool = False,
) -> dict[str, Path]:
    """导出节点表 / JSON / DT 风格 Graphviz 流程图（dot+pdf）；PNG 优先 Graphviz，失败再 matplotlib。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_nodes = export_tree_nodes_df(
        model,
        feature_names,
        class_mapping,
        display_x=display_x,
        display_y=display_y,
    )
    nodes_csv = out_dir / "tree_nodes.csv"
    df_nodes.to_csv(nodes_csv, index=False, encoding="utf-8-sig")

    tree_json = build_nested_tree_json(
        model,
        feature_names,
        class_mapping,
        display_x=display_x,
        display_y=display_y,
    )
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
            display_x=display_x,
            display_y=display_y,
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
        "metrics": result.metrics,
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
        disp_x = getattr(result, "display_x", None)
        disp_y = getattr(result, "display_y", None)
        paths.update(
            export_tree_artifacts(
                result.model,
                out_dir,
                feature_names=result.feature_names,
                class_mapping=class_mapping,
                display_x=disp_x,
                display_y=disp_y,
                render_pdf=render_tree_pdf,
                render_png=render_tree_png,
                tree_image_dpi=tree_image_dpi,
                open_viewer=open_view,
            )
        )
        if disp_x is not None and disp_y is not None:
            tc = true_class_counts_per_node(result.model, disp_x, disp_y)
            to_name = _to_name_fn(class_mapping)
            classes = list(result.model.classes_)
            summary["label_distribution_true_root"] = {
                to_name(classes[j]): int(tc[0, j]) for j in range(len(classes))
            }
            summary["tree_display_note"] = (
                "PDF：类别分布为真实标签百分比；仅叶节点显示 class；"
                "CART 训练仍可使用 class_weight=balanced"
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
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    oracle_relabel: bool = False,
    l_hat_path: str | Path | None = None,
    data_flow: Any = None,
    refit_on_full_data: bool = False,
    only_episode: int | None = None,
    pipeline_mode: str = "rules",
) -> ViperRunResult:
    """
    端到端 VIPER：从轨迹行提炼 CART 规则。

    pipeline_mode="rules"（默认）：不 holdout，全部行参与建树 → 适合「大量数据→规则展示」。
    pipeline_mode="eval"：按 episode 划分 train/val/test，用于泛化指标。
    only_episode：仅指定局号的行参与规则/树（FQE 仍建议用全表 CSV 先算好 weights）。
    """
    df = load_trajectory_csv(str(csv_path))
    n_full = len(df)
    w_df = pd.read_csv(weights_path, encoding="utf-8-sig")
    if len(w_df) != len(df):
        raise ValueError(
            f"weights 行数 {len(w_df)} 与轨迹 CSV {len(df)} 不一致，请使用同一轨迹生成的 weights"
        )

    ep_mask: np.ndarray | None = None
    if only_episode is not None:
        ep = pd.to_numeric(df[EPISODE_COL], errors="coerce").values
        ep_mask = ep == int(only_episode)
        if not np.any(ep_mask):
            raise ValueError(f"only_episode={only_episode} 在轨迹中无匹配行")
        df = df.loc[ep_mask].reset_index(drop=True)
        w_df = w_df.loc[ep_mask].reset_index(drop=True)
        logger.info(
            "规则提炼范围: 仅 episode=%d → %d 行（全 CSV %d 行）",
            only_episode,
            len(df),
            n_full,
        )

    x, y = build_xy(df)

    # 可选：oracle 重标注（DAgger/VIPER 精神）——把行为动作替换为 Q 最优动作
    if oracle_relabel:
        if l_hat_path is None:
            raise ValueError("oracle_relabel=True 需提供 l_hat_path（含 a_star 列）")
        lh_df = pd.read_csv(l_hat_path, encoding="utf-8-sig")
        if "a_star" not in lh_df.columns:
            raise ValueError("l_hat.csv 缺少 a_star 列，请用新版 l_hat 重新生成")
        if len(lh_df) != n_full:
            raise ValueError("l_hat.csv 行数与轨迹 CSV 不一致")
        if ep_mask is not None:
            lh_df = lh_df.loc[ep_mask].reset_index(drop=True)
        if len(lh_df) != len(df):
            raise ValueError("l_hat.csv 行数与当前规则提炼子集不一致")
        y = pd.to_numeric(lh_df["a_star"], errors="coerce").values.astype(np.int64)
        logger.info("已启用 oracle 重标注：y ← argmax_a Q_hat(s,a)")

    if WEIGHTS_COL not in w_df.columns:
        raise ValueError(f"weights CSV 缺少列 {WEIGHTS_COL!r}")
    weights = pd.to_numeric(w_df[WEIGHTS_COL], errors="coerce").values.astype(np.float64)
    if np.isnan(weights).any():
        raise ValueError("weights 含 NaN")
    if not cfg.weighted_sampling:
        weights = np.full(len(weights), 1.0 / len(weights), dtype=np.float64)

    episodes = pd.to_numeric(df[EPISODE_COL], errors="coerce").values
    mode = str(pipeline_mode).strip().lower()
    if mode == "rules" or (val_frac <= 0 and test_frac <= 0):
        # 规则展示：全部行都用于建树，不拆 holdout
        n_rows = len(df)
        train_idx = np.arange(n_rows, dtype=np.int64)
        val_idx = np.array([], dtype=np.int64)
        test_idx = np.array([], dtype=np.int64)
        logger.info(
            "规则提炼模式: 全部 %d 行参与 VIPER（不划分 val/test）",
            n_rows,
        )
        if data_flow is not None:
            data_flow.record(
                "06",
                "规则模式-全量建树",
                n_in=n_rows,
                n_out=n_rows,
                reduces_rows=False,
                module="viper_cart.run_viper_from_files",
                note="pipeline_mode=rules；导出 rules/树图覆盖当前输入全部行",
            )
    else:
        train_idx, val_idx, test_idx = split_by_episode(
            episodes, val_frac=val_frac, test_frac=test_frac, seed=cfg.random_state
        )
        if data_flow is not None:
            from .data_flow import record_episode_split

            record_episode_split(
                data_flow,
                len(df),
                int(train_idx.size),
                int(val_idx.size),
                int(test_idx.size),
                val_frac=val_frac,
                test_frac=test_frac,
                n_episodes=int(pd.Series(episodes).nunique()),
            )

    x_tr, y_tr = x[train_idx], y[train_idx]
    w_tr = weights[train_idx].astype(np.float64)
    s = float(w_tr.sum())
    w_tr = w_tr / s if s > 0 else np.full(w_tr.size, 1.0 / w_tr.size)
    x_val, y_val = x[val_idx], y[val_idx]
    x_te, y_te = x[test_idx], y[test_idx]

    result = run_viper_loop(
        x_tr, y_tr, w_tr, cfg, x_eval=x_val, y_eval=y_val, data_flow=data_flow
    )

    split_model = result.model
    do_refit = bool(refit_on_full_data) and int(train_idx.size) < len(df)
    if do_refit:
        w_all = np.asarray(weights, dtype=np.float64)
        s_all = float(w_all.sum())
        w_all = w_all / s_all if s_all > 0 else np.full(len(df), 1.0 / len(df))
        result.model, result.rules, m_full = refit_final_tree_on_full_data(
            x,
            y,
            w_all,
            cfg,
            selected_round=result.selected_round,
            data_flow=data_flow,
        )
        logger.info(
            "已用全表 %d 行重训并替换导出模型（选轮仍在 train=%d / val 上完成）",
            len(df),
            int(train_idx.size),
        )

    # 独立集评估（指标仍基于划分集上选出的 split_model，避免乐观偏差）
    metrics = {
        "selection_metric": cfg.selection_metric,
        "oracle_relabel": bool(oracle_relabel),
        "refit_on_full_data": do_refit,
        "split": {"train": int(train_idx.size), "val": int(val_idx.size), "test": int(test_idx.size)},
        "train": compute_metrics(split_model, x_tr, y_tr, class_mapping=cfg.class_mapping),
        "val": compute_metrics(split_model, x_val, y_val, class_mapping=cfg.class_mapping),
        "test": compute_metrics(split_model, x_te, y_te, class_mapping=cfg.class_mapping),
    }
    if do_refit:
        metrics["exported_tree"] = {
            "trained_on": "full_data",
            "n_rows": int(len(df)),
            "bootstrap_m": int(m_full),
            "selected_round_hyperparams": int(result.selected_round),
            "full_data_fit": compute_metrics(
                result.model, x, y, class_mapping=cfg.class_mapping
            ),
        }
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    (out_dir_p / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result.metrics = metrics
    te = metrics["test"]
    if te.get("n", 0) > 0:
        logger.info(
            "测试集评估: acc=%.4f balanced_acc=%.4f macro_f1=%.4f (n=%d)",
            te["accuracy"],
            te["balanced_accuracy"],
            te["macro_f1"],
            te["n"],
        )

    result.display_x = x
    result.display_y = y

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
