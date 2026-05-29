"""
VIPER/CART 超参数网格搜索（跳过 FQE，复用已有 weights.csv）。

在 FQE→l_hat→weights 已完成的前提下，对 cart_max_depth、cart_min_samples_leaf
等 CART/VIPER 参数做笛卡尔积搜索。每组配置独立跑 n_round 轮 VIPER，
记录 acc_full、acc_resampled、规则条数等指标，并选出最优一组。

主要函数：
  run_viper_tune_grid() — 执行网格搜索并写入 viper_tune/
  _finalize_best_run()  — 将最优组结果复制到 viper_out/

结果目录（默认 {weights 父目录}/viper_tune/）：
  tune_results.json    — 全部组合 + best 指标（主文件）
  tune_results.csv     — 表格，便于 Excel 对照
  tune_summary.txt     — 文本摘要
  best_run_config.json — 建议回填到 run_pipeline.py 的 cart_* 参数
  runs/run_XX_*/       — 每组独立 metrics.json
  viper_out_best/      — 最优组的 rules.txt、PDF、tree.json 等

用法：
  python -m causal.decision_tree.tune_viper -v

与 run_pipeline.py 的关系：run_pipeline 中 run_viper_tune_grid=True 时会调用本模块。
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path

_ALGORITHM_ROOT = Path(__file__).resolve().parents[2]
if str(_ALGORITHM_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALGORITHM_ROOT))

from causal.decision_tree.trajectory_io import load_trajectory_csv
from causal.decision_tree.viper_cart import (
    ViperConfig,
    ViperRunResult,
    build_xy,
    extract_rules,
    load_weights_array,
    run_viper_loop,
    save_viper_outputs,
)

logger = logging.getLogger(__name__)

# 网格：depth × min_samples_leaf（可按需增删）
TUNE_GRID: list[dict] = [
    {"max_depth": 4, "min_samples_leaf": 1},
    {"max_depth": 4, "min_samples_leaf": 10},
    {"max_depth": 5, "min_samples_leaf": 1},
    {"max_depth": 5, "min_samples_leaf": 10},
    {"max_depth": 6, "min_samples_leaf": 1},
    {"max_depth": 6, "min_samples_leaf": 10},
    {"max_depth": 7, "min_samples_leaf": 1},
    {"max_depth": 7, "min_samples_leaf": 10},
    {"max_depth": 8, "min_samples_leaf": 1},
    {"max_depth": 8, "min_samples_leaf": 10},
    {"max_depth": 10, "min_samples_leaf": 1},
    {"max_depth": 10, "min_samples_leaf": 5},
    {"max_depth": 12, "min_samples_leaf": 1},
    {"max_depth": 12, "min_samples_leaf": 5},
]

DEFAULT_CSV = _ALGORITHM_ROOT / "causal/trajectories/trajectory_LLdV3_S0_5.csv"
DEFAULT_WEIGHTS = _ALGORITHM_ROOT / "causal/trajectories/fqe_out/weights.csv"

CSV_COLUMNS = [
    "grid_index",
    "max_depth",
    "min_samples_leaf",
    "acc_full",
    "acc_resampled",
    "acc_gap",
    "n_rules",
    "n_leaves",
    "n_nodes",
    "selected_round",
    "is_best",
]


def _run_tag(grid: dict) -> str:
    return f"d{grid['max_depth']}_leaf{grid.get('min_samples_leaf', 1)}"


def _run_one(
    x,
    y,
    weights,
    *,
    grid_index: int,
    seed: int,
    n_round: int,
    weight_noise: float,
    grid: dict,
) -> dict:
    cfg = ViperConfig(
        n_round=n_round,
        max_depth=int(grid["max_depth"]),
        min_samples_leaf=int(grid.get("min_samples_leaf", 1)),
        min_samples_split=int(grid.get("min_samples_split", 2)),
        random_state=seed,
        weight_noise_std=weight_noise,
        pick_best_by_full_acc=True,
        export_tree=False,
        render_tree_pdf=False,
    )
    result = run_viper_loop(x, y, weights, cfg)
    n_rules = len(extract_rules(result.model, result.feature_names))
    acc_full = float(result.selected_acc_full)
    acc_resampled = float(result.selected_acc_resampled)
    return {
        "grid_index": grid_index,
        "max_depth": cfg.max_depth,
        "min_samples_leaf": cfg.min_samples_leaf,
        "min_samples_split": cfg.min_samples_split,
        "n_round": n_round,
        "weight_noise": weight_noise,
        "selected_round": result.selected_round,
        "acc_full": acc_full,
        "acc_resampled": acc_resampled,
        "acc_gap": acc_resampled - acc_full,
        "n_rules": n_rules,
        "n_leaves": int(result.model.get_n_leaves()),
        "n_nodes": int(result.model.tree_.node_count),
        "rounds": [
            {
                "round": rd.round_index,
                "acc_full": rd.full_data_accuracy,
                "acc_resampled": rd.train_accuracy_resampled,
            }
            for rd in result.rounds
        ],
    }


def _pick_best(rows: list[dict]) -> dict:
    return max(rows, key=lambda r: (r["acc_full"], r["acc_resampled"]))


def _write_tune_results_csv(path: Path, rows: list[dict], best: dict) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = dict(r)
            row["is_best"] = r["grid_index"] == best["grid_index"]
            w.writerow(row)


def _write_tune_summary_txt(path: Path, rows: list[dict], best: dict, meta: dict) -> None:
    lines = [
        "VIPER/CART 网格调参结果",
        f"时间: {meta.get('finished_at', '')}",
        f"轨迹: {meta.get('csv', '')}",
        f"weights: {meta.get('weights_csv', '')}",
        f"n_round={meta.get('n_round')} weight_noise={meta.get('weight_noise')} seed={meta.get('seed')}",
        "",
        f"{'#':>3} {'depth':>5} {'leaf':>6} {'acc_full':>9} {'acc_res':>9} {'gap':>7} "
        f"{'rules':>6} {'leaves':>7} {'best':>5}",
        "-" * 72,
    ]
    for r in sorted(rows, key=lambda x: (-x["acc_full"], -x["acc_resampled"])):
        mark = "*" if r["grid_index"] == best["grid_index"] else ""
        lines.append(
            f"{r['grid_index']:3d} {r['max_depth']:5d} {r['min_samples_leaf']:6d} "
            f"{r['acc_full']:9.4f} {r['acc_resampled']:9.4f} {r['acc_gap']:7.4f} "
            f"{r['n_rules']:6d} {r['n_leaves']:7d} {mark:>5}"
        )
    lines.extend(
        [
            "-" * 72,
            f"最优 (*): depth={best['max_depth']} min_samples_leaf={best['min_samples_leaf']} "
            f"acc_full={best['acc_full']:.4f} n_rules={best['n_rules']}",
            "",
            "建议写入 run_pipeline.py RUN_CONFIG:",
            f'  "cart_max_depth": {best["max_depth"]},',
            f'  "cart_min_samples_leaf": {best["min_samples_leaf"]},',
            f'  "viper_n_round": {meta.get("n_round")},',
            f'  "weight_noise": {meta.get("weight_noise")},',
            f'  "viper_pick_best_round": True,',
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_best_run_config(path: Path, best: dict, meta: dict) -> None:
    payload = {
        "description": "由 tune_viper.py 自动生成，可复制到 run_pipeline.py 的 RUN_CONFIG",
        "viper_n_round": meta.get("n_round"),
        "viper_pick_best_round": True,
        "cart_max_depth": best["max_depth"],
        "cart_min_samples_leaf": best["min_samples_leaf"],
        "cart_min_samples_split": best.get("min_samples_split", 2),
        "weight_noise": meta.get("weight_noise"),
        "resample_size": 0,
        "metrics": {
            "acc_full": best["acc_full"],
            "acc_resampled": best["acc_resampled"],
            "n_rules": best["n_rules"],
            "n_leaves": best["n_leaves"],
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_all_results(
    out_dir: Path,
    rows: list[dict],
    best: dict,
    meta: dict,
) -> dict[str, Path]:
    """写入 viper_tune 目录下约定文件。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    for r in rows:
        run_dir = runs_dir / f"run_{r['grid_index']:02d}_{_run_tag(r)}"
        run_dir.mkdir(exist_ok=True)
        (run_dir / "metrics.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    for r in rows:
        r["is_best"] = r["grid_index"] == best["grid_index"]

    summary = {**meta, "best": best, "all": rows}
    paths: dict[str, Path] = {}

    p_json = out_dir / "tune_results.json"
    p_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["tune_results.json"] = p_json

    p_csv = out_dir / "tune_results.csv"
    _write_tune_results_csv(p_csv, rows, best)
    paths["tune_results.csv"] = p_csv

    p_txt = out_dir / "tune_summary.txt"
    _write_tune_summary_txt(p_txt, rows, best, meta)
    paths["tune_summary.txt"] = p_txt

    p_cfg = out_dir / "best_run_config.json"
    _write_best_run_config(p_cfg, best, meta)
    paths["best_run_config.json"] = p_cfg

    return paths


def _finalize_best_run(
    x,
    y,
    weights,
    best: dict,
    dest_dir: Path,
    *,
    seed: int,
    n_round: int,
    weight_noise: float,
    export_tree: bool,
    render_tree_pdf: bool,
    render_tree_png: bool,
    tree_image_dpi: int,
    open_tree_pdf: bool,
    resample_size: int | None = None,
) -> ViperRunResult:
    m = (
        None
        if resample_size is None or int(resample_size) <= 0
        else int(resample_size)
    )
    cfg = ViperConfig(
        n_round=n_round,
        max_depth=int(best["max_depth"]),
        min_samples_leaf=int(best["min_samples_leaf"]),
        min_samples_split=int(best.get("min_samples_split", 2)),
        random_state=seed,
        resample_size=m,
        weight_noise_std=weight_noise,
        pick_best_by_full_acc=True,
        export_tree=export_tree,
        render_tree_pdf=render_tree_pdf,
        render_tree_png=render_tree_png,
        tree_image_dpi=tree_image_dpi,
        open_tree_pdf=open_tree_pdf,
        show_tree_image=open_tree_pdf,
    )
    result = run_viper_loop(x, y, weights, cfg)
    dest_dir.mkdir(parents=True, exist_ok=True)
    save_viper_outputs(
        result,
        dest_dir,
        export_tree=export_tree,
        render_tree_pdf=render_tree_pdf,
        render_tree_png=render_tree_png,
        tree_image_dpi=tree_image_dpi,
        open_tree_pdf=open_tree_pdf,
        show_tree_image=open_tree_pdf,
    )
    (dest_dir / "tune_best_metrics.json").write_text(
        json.dumps(
            {
                "grid_index": best["grid_index"],
                "selected_round": result.selected_round,
                "acc_full": result.selected_acc_full,
                "acc_resampled": result.selected_acc_resampled,
                "n_rules": len(result.rules),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def _export_best_tree(
    x,
    y,
    weights,
    best: dict,
    out_dir: Path,
    *,
    seed: int,
    n_round: int,
    weight_noise: float,
    render_pdf: bool,
) -> Path:
    best_dir = out_dir / "viper_out_best"
    _finalize_best_run(
        x,
        y,
        weights,
        best,
        best_dir,
        seed=seed,
        n_round=n_round,
        weight_noise=weight_noise,
        export_tree=True,
        render_tree_pdf=render_pdf,
        render_tree_png=False,
        tree_image_dpi=150,
        open_tree_pdf=False,
        resample_size=None,
    )
    return best_dir


def run_viper_tune_grid(
    csv_path: str | Path,
    weights_path: str | Path,
    tune_out_dir: str | Path,
    *,
    n_round: int = 8,
    weight_noise: float = 0.02,
    seed: int = 42,
    export_best_to: str | Path | None = None,
    render_tree_pdf: bool = True,
    render_tree_png: bool = False,
    tree_image_dpi: int = 150,
    open_tree_pdf: bool = False,
    export_tree: bool = True,
    resample_size: int | None = None,
    print_each_grid: bool = False,
) -> tuple[dict, dict[str, Path], ViperRunResult]:
    """
    在已有轨迹与 weights 上跑 TUNE_GRID，写出 viper_tune/ 表格；可选用最优一组重训并导出到 export_best_to。

    供 run_pipeline.py「run_viper_tune_grid=True」调用。
    """
    csv_path_p = Path(csv_path).resolve()
    weights_path_p = Path(weights_path).resolve()
    tune_out_dir_p = Path(tune_out_dir).resolve()

    if not csv_path_p.is_file():
        raise FileNotFoundError(csv_path_p)
    if not weights_path_p.is_file():
        raise FileNotFoundError(weights_path_p)

    df = load_trajectory_csv(str(csv_path_p))
    x, y = build_xy(df)
    weights = load_weights_array(weights_path_p)
    if len(weights) != len(df):
        raise ValueError("weights 行数与 CSV 不一致")

    rows: list[dict] = []
    for i, grid in enumerate(TUNE_GRID, 1):
        tag = f"depth={grid['max_depth']} leaf={grid.get('min_samples_leaf', 1)}"
        logger.info("[%d/%d] %s", i, len(TUNE_GRID), tag)
        row = _run_one(
            x,
            y,
            weights,
            grid_index=i,
            seed=seed,
            n_round=n_round,
            weight_noise=weight_noise,
            grid=grid,
        )
        rows.append(row)
        if print_each_grid:
            print(
                f"  [{i}/{len(TUNE_GRID)}] {tag} -> acc_full={row['acc_full']:.4f} "
                f"rules={row['n_rules']} leaves={row['n_leaves']}"
            )

    best = _pick_best(rows)
    meta = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "csv": str(csv_path_p),
        "weights_csv": str(weights_path_p),
        "n_grid": len(TUNE_GRID),
        "n_round": n_round,
        "weight_noise": weight_noise,
        "seed": seed,
    }
    paths = _save_all_results(tune_out_dir_p, rows, best, meta)

    dest = Path(export_best_to).resolve() if export_best_to else None
    if dest is not None:
        viper_result = _finalize_best_run(
            x,
            y,
            weights,
            best,
            dest,
            seed=seed,
            n_round=n_round,
            weight_noise=weight_noise,
            export_tree=export_tree,
            render_tree_pdf=render_tree_pdf,
            render_tree_png=render_tree_png,
            tree_image_dpi=tree_image_dpi,
            open_tree_pdf=open_tree_pdf,
            resample_size=resample_size,
        )
    else:
        cfg = ViperConfig(
            n_round=n_round,
            max_depth=int(best["max_depth"]),
            min_samples_leaf=int(best["min_samples_leaf"]),
            min_samples_split=int(best.get("min_samples_split", 2)),
            random_state=seed,
            weight_noise_std=weight_noise,
            pick_best_by_full_acc=True,
            export_tree=False,
            render_tree_pdf=False,
            render_tree_png=False,
        )
        viper_result = run_viper_loop(x, y, weights, cfg)
        viper_result.rules = extract_rules(viper_result.model, viper_result.feature_names)

    return best, paths, viper_result


def main(argv: list[str] | None = None) -> int:
    p = ArgumentParser(description="VIPER/CART 网格调参（需已有 weights.csv）")
    p.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    p.add_argument("--weights-csv", type=str, default=str(DEFAULT_WEIGHTS))
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="结果目录；默认 {weights 父目录}/viper_tune",
    )
    p.add_argument("--n-round", type=int, default=8)
    p.add_argument("--weight-noise", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-export-best",
        action="store_true",
        help="不导出 viper_out_best/（默认会导出最优 rules/PDF）",
    )
    p.add_argument("--no-render-pdf", action="store_true", help="导出最优树时不渲染 PDF")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    csv_path = Path(args.csv).resolve()
    weights_path = Path(args.weights_csv).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)

    out_dir = (
        Path(args.out).resolve()
        if args.out.strip()
        else weights_path.parent / "viper_tune"
    )

    print(f"调参开始: {len(TUNE_GRID)} 组, n_round={args.n_round}")
    print(f"结果将写入: {out_dir.resolve()}")
    export_dest = None if args.no_export_best else out_dir / "viper_out_best"
    best, paths, _ = run_viper_tune_grid(
        csv_path,
        weights_path,
        out_dir,
        n_round=args.n_round,
        weight_noise=args.weight_noise,
        seed=args.seed,
        export_best_to=export_dest,
        render_tree_pdf=not args.no_render_pdf,
        render_tree_png=False,
        tree_image_dpi=150,
        open_tree_pdf=False,
        export_tree=True,
        resample_size=None,
        print_each_grid=True,
    )

    print("\n" + "=" * 60)
    print("调参完成，结果文件：")
    for name, pth in paths.items():
        print(f"  {name}: {pth}")
    print(f"  每组明细: {out_dir / 'runs'}/run_XX_*/metrics.json")
    print(
        f"\n最优: depth={best['max_depth']} min_samples_leaf={best['min_samples_leaf']} "
        f"acc_full={best['acc_full']:.4f} n_rules={best['n_rules']}"
    )

    if export_dest is not None:
        print(f"  viper_out_best/: {export_dest}")
        print("    rules.txt, tree.json, policy_tree.pdf, viper_summary.json, ...")

    print("=" * 60)
    print("可将 best_run_config.json 中的参数复制到 run_pipeline.py RUN_CONFIG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
