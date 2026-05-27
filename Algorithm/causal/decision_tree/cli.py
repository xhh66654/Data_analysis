"""
决策树离线流水线的命令行入口（与 run_pipeline.py 功能等价，参数化配置）。

通过 argparse 暴露各阶段参数，适合脚本化、CI 或只重跑某一阶段。
--phase 可选：all | fqe | l_hat | weights | viper。

典型用法：
  python -m causal.decision_tree --csv path/to/trajectory.csv --output-dir fqe_out
  python -m causal.decision_tree --phase viper --weights-csv fqe_out/weights.csv

各 phase 依赖的前置产物：
  fqe     → 需要轨迹 CSV；输出 q_hat.pt
  l_hat   → 需要 q_hat.pt；输出 l_hat.csv
  weights → 需要 l_hat.csv；输出 weights.csv
  viper   → 需要轨迹 CSV + weights.csv；输出 viper_out/
  all     → 依次执行上述四步

入口函数 main() 由 __main__.py 调用；run_pipeline.py 则直接 import 各子模块函数。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .fqe import FQETrainConfig, load_q_hat, save_q_hat, train_q_hat
from .l_hat import compute_l_hat, l_hat_dataframe, save_l_hat_csv
from .viper_cart import ViperConfig, run_viper_from_files
from .weights import run_weights_from_l_hat_csv
from .trajectory_io import ACTION_COL, STATE_COLS, build_transitions, load_trajectory_csv

logger = logging.getLogger(__name__)

_DEFAULT_CSV = (
    r"F:\cause_analysis\Algorithm\causal\trajectories\trajectory_LLdV3_S0_1.csv"
)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        stream=sys.stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="决策树离线流水线：FQE → l_hat → weights → VIPER CART → 规则",
    )
    p.add_argument("--csv", type=str, default=_DEFAULT_CSV, help="轨迹 CSV 绝对路径")
    p.add_argument("--output-dir", type=str, default="", help="输出目录；默认 csv 同目录下 fqe_out")
    p.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=("all", "fqe", "l_hat", "weights", "viper"),
        help="all=全流程；viper=仅重采样+CART+规则（需已有 weights.csv）",
    )
    p.add_argument(
        "--weights-csv",
        type=str,
        default="",
        help="weights.csv；phase=viper 时默认 {output-dir}/weights.csv",
    )
    p.add_argument("--n-round", type=int, default=5, help="VIPER 外循环轮数（建议 3～10）")
    p.add_argument("--max-depth", type=int, default=6, help="CART max_depth")
    p.add_argument(
        "--resample-size",
        type=int,
        default=0,
        help="重采样规模 M；0 表示 M=N（与轨迹行数相同）",
    )
    p.add_argument(
        "--weight-noise",
        type=float,
        default=0.02,
        help="每轮对 weights 的乘性噪声标准差；0 表示不扰动",
    )
    p.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="CART 叶节点最少样本数，>1 可剪枝",
    )
    p.add_argument(
        "--min-samples-split",
        type=int,
        default=2,
        help="CART 分裂最少样本数，>2 可剪枝",
    )
    p.add_argument(
        "--last-round-only",
        action="store_true",
        help="VIPER 使用最后一轮树，不按 acc_full 选优",
    )
    p.add_argument(
        "--l-hat-csv",
        type=str,
        default="",
        help="l_hat.csv 路径；phase=weights 时默认 {output-dir}/l_hat.csv",
    )
    p.add_argument(
        "--viper-weighted-sampling",
        dest="viper_weighted_sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="VIPER 抽样：启用=按 weights 加权；--no-viper-weighted-sampling=均匀 1/n",
    )
    p.add_argument("--eps", type=float, default=1e-6, help="w_raw = max(l_hat,0)+eps（仅加权模式）")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="q_hat.pt 路径；phase=l_hat 时必填；默认 {output-dir}/q_hat.pt",
    )
    p.add_argument("--l-hat-batch-size", type=int, default=4096, help="l_hat 前向批大小")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--target",
        type=str,
        default="sarsa",
        choices=("sarsa", "max_q"),
        help="bootstrap：sarsa（推荐）或 max_q",
    )
    p.add_argument("--use-target-network", action="store_true", help="启用 target 网络稳定 FQE")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--no-export-tree",
        action="store_true",
        help="VIPER 阶段不导出 tree.json / tree_nodes.csv / policy_tree.dot（仍导出 rules）",
    )
    p.add_argument(
        "--no-render-tree-pdf",
        action="store_true",
        help="不渲染 PDF 流程图（默认与 dt_auto_pipeline 一致会生成 policy_tree.pdf）",
    )
    p.add_argument(
        "--render-tree-png",
        action="store_true",
        help="额外从同一 .dot 渲染 policy_tree.png（Graphviz，与 PDF 同风格）",
    )
    p.add_argument("--tree-image-dpi", type=int, default=150, help="Graphviz 失败时 matplotlib 回退 PNG 的 DPI")
    p.add_argument(
        "--open-pdf",
        action="store_true",
        help="与 DT/dt_auto_pipeline --open-pdf 相同：PDF 渲染成功后用系统默认程序打开",
    )
    p.add_argument(
        "--show-tree-image",
        action="store_true",
        help="同 --open-pdf（兼容旧参数名）",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"未找到轨迹 CSV: {csv_path}")

    out_dir = Path(args.output_dir) if args.output_dir.strip() else csv_path.parent / "fqe_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_trajectory_csv(str(csv_path))
    ckpt_path = Path(args.checkpoint) if args.checkpoint.strip() else out_dir / "q_hat.pt"
    l_hat_path = Path(args.l_hat_csv) if args.l_hat_csv.strip() else out_dir / "l_hat.csv"
    weights_path = Path(args.weights_csv) if args.weights_csv.strip() else out_dir / "weights.csv"
    viper_out_dir = out_dir / "viper_out"

    if args.phase in ("all", "fqe"):
        trans = build_transitions(df)
        cfg = FQETrainConfig(
            gamma=args.gamma,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            hidden=args.hidden,
            device=args.device,
            target=args.target,
            use_target_network=args.use_target_network,
            seed=args.seed,
        )
        result = train_q_hat(trans, cfg)
        meta = {
            "csv": str(csv_path.resolve()),
            "gamma": cfg.gamma,
            "target": cfg.target,
            "state_dim": trans.s.shape[1],
            "n_actions": int(trans.a.max()) + 1,
            "hidden": cfg.hidden,
            "final_loss": result.final_loss,
            "loss_history": result.history,
        }
        save_q_hat(ckpt_path, result.q_net, meta)
        logger.info("第一阶段完成 final_loss=%.6f", result.final_loss)

    if args.phase in ("all", "l_hat"):
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"未找到 Q_hat 检查点: {ckpt_path}（先运行 --phase fqe 或指定 --checkpoint）"
            )
        states = df[STATE_COLS].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
        actions = pd.to_numeric(df[ACTION_COL], errors="coerce").values.astype(np.int64)
        q_net, _ = load_q_hat(ckpt_path, device=args.device)
        lh = compute_l_hat(
            q_net,
            states,
            actions,
            device=args.device,
            batch_size=args.l_hat_batch_size,
        )
        save_l_hat_csv(l_hat_path, l_hat_dataframe(df, lh))
        logger.info("第二阶段完成 l_hat -> %s", l_hat_path.resolve())

    if args.phase in ("all", "weights"):
        if not l_hat_path.is_file():
            raise FileNotFoundError(
                f"未找到 l_hat CSV: {l_hat_path}（先运行 --phase l_hat 或指定 --l-hat-csv）"
            )
        run_weights_from_l_hat_csv(
            l_hat_path,
            output_path=weights_path,
            eps=args.eps,
            weighted_sampling=bool(args.viper_weighted_sampling),
        )
        logger.info("第三阶段完成 weights -> %s", weights_path.resolve())

    if args.phase in ("all", "viper"):
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"未找到 weights CSV: {weights_path}（先运行 --phase weights 或指定 --weights-csv）"
            )
        m = args.resample_size if args.resample_size > 0 else None
        viper_cfg = ViperConfig(
            n_round=args.n_round,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            min_samples_split=args.min_samples_split,
            random_state=args.seed,
            resample_size=m,
            weight_noise_std=args.weight_noise,
            weighted_sampling=bool(args.viper_weighted_sampling),
            pick_best_by_full_acc=not args.last_round_only,
            export_tree=not args.no_export_tree,
            render_tree_pdf=not args.no_render_tree_pdf,
            render_tree_png=args.render_tree_png,
            tree_image_dpi=args.tree_image_dpi,
            open_tree_pdf=args.open_pdf or args.show_tree_image,
            show_tree_image=args.show_tree_image,
        )
        result = run_viper_from_files(csv_path, weights_path, viper_out_dir, viper_cfg)
        logger.info(
            "VIPER 完成 rounds=%d 选用轮=%d acc_full=%.4f rules=%d -> %s",
            len(result.rounds),
            result.selected_round,
            result.selected_acc_full,
            len(result.rules),
            viper_out_dir.resolve(),
        )


if __name__ == "__main__":
    main()
