#!/usr/bin/env python3
"""
博弈约简 + 溯因流水线入口（仅用本目录内模块）。

用法示例（请将 CSV 换为你的绝对路径）::

  cd <Algorithm 仓库根目录>
  python causal/game_reduction/run_pipeline.py ^
    --csv F:/cause_analysis/Algorithm/causal/MADDPG_file/results/simple_spread_v3/MADDPG_10/training_trajectory.csv ^
        --target-agent agent_0 --top-k 2 --epochs 25

或（PYTHONPATH 含仓库根目录时）::

  python -m causal.game_reduction.run_pipeline --csv ...

不修改仓库中其它文件。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

try:
    from .mean_field import build_mean_field_features, load_group_map_json
    from .neighbor_counterfactual import estimate_neighbor_effects, select_key_neighbors
    from .train_joint_q import JointQTrainConfig, save_joint_q, train_joint_q
    from .trajectory_maddpg import build_joint_batch, load_maddpg_trajectory_csv
except ImportError:  # 直接脚本启动：python causal/game_reduction/run_pipeline.py
    import sys
    _repo = Path(__file__).resolve().parents[2]
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
    from causal.game_reduction.mean_field import build_mean_field_features, load_group_map_json
    from causal.game_reduction.neighbor_counterfactual import estimate_neighbor_effects, select_key_neighbors
    from causal.game_reduction.train_joint_q import JointQTrainConfig, save_joint_q, train_joint_q
    from causal.game_reduction.trajectory_maddpg import build_joint_batch, load_maddpg_trajectory_csv

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="博弈约简：联合 Q + 邻居反事实 + Top-K")
    p.add_argument(
        "--csv",
        type=str,
        required=True,
        help="MADDPG training_trajectory.csv 的绝对路径（可随时更换）",
    )
    p.add_argument(
        "--target-agent",
        type=str,
        required=True,
        help="主/KOP 智能体 id（如 agent_0），后期可改用其它 id；全流程以该智能体为目标",
    )
    p.add_argument(
        "--neighbor-subset",
        type=str,
        default="",
        help="逗号分隔：联合状态中包含的邻居 obs；留空=除目标外全部",
    )
    p.add_argument(
        "--reward-mode",
        choices=("target", "team_sum"),
        default="target",
        help="Bellman 标量：target=目标体 reward；team_sum=全员 reward 和",
    )
    p.add_argument("--max-rows", type=int, default=0, help=">0 时只用前 N 行（调试）")
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--top-k",
        type=int,
        default=3,
        metavar="K",
        help="硬上限：最多保留多少个关键邻居。与 --min-abs-influence 并用时：阈值截断后最多再保留 K 个。"
        "<=0 且已启用阈值时：不设个数上限，只按阈值截断；<=0 且未设阈值：不保留邻居（只见主智能体）。",
    )
    p.add_argument(
        "--min-abs-influence",
        type=float,
        default=-1.0,
        metavar="THR",
        help=">=0 时启用：**从高到低**扫描影响度，一旦出现某邻居 mean_abs_effect < THR，"
        "则**该邻居及所有更弱者**都不再保留；阈值之前的继续保留。"
        "-1（默认）：关闭该规则，仅按 --top-k 保留前 K 个邻居（旧逻辑）。",
    )
    p.add_argument(
        "--ablation-mode",
        choices=("zero", "noise"),
        default="zero",
        help="反事实干预模式：zero=置零（传统），noise=加高斯噪声（更接近训练分布）",
    )
    p.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="噪声干预缩放系数，仅 noise 模式生效：1.0=与数据同方差",
    )
    p.add_argument(
        "--use-normalized-threshold",
        action="store_true",
        help="阈值基于归一化影响度 (|ΔQ|/σ_Q) 而非原始 |ΔQ|",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="输出目录；默认写到 CSV 同目录下的 game_reduction_out/",
    )
    p.add_argument(
        "--no-mean-field",
        action="store_true",
        help="跳过平均场特征输出（默认在 Top-K 后生成 mean_field_features.csv）",
    )
    p.add_argument(
        "--group-map-json",
        type=str,
        default="",
        help="异构群体映射 JSON 绝对路径：{\"agent_1\":\"enemy_attack\",...}；留空则全部保留邻居归入 --mf-default-pool 单群体",
    )
    p.add_argument(
        "--mf-default-pool",
        type=str,
        default="pooled_neighbors",
        help="未在 group-map 中出现的 Top-K 邻居统一归入该群体名（单池平均场）",
    )
    p.add_argument(
        "--mf-include-obs-std",
        action="store_true",
        help="每个群体在 s_mean,a_mean 之后再拼接该群体 obs 的逐维标准差（离散度/压迫度近似）",
    )
    p.add_argument(
        "--train-scm",
        action="store_true",
        help="平均场导出后训练神经结构方程 a_target=f(父变量块)并做分组消融得到边强度",
    )
    p.add_argument("--scm-epochs", type=int, default=25)
    p.add_argument("--scm-lr", type=float, default=1e-3)
    p.add_argument("--scm-hidden", type=int, default=128)
    p.add_argument(
        "--scm-env-padding",
        type=int,
        default=0,
        help="SCM 父向量末尾拼接的 env 占位零维个数（无外生环境列时占位，后续可自行填）",
    )
    p.add_argument(
        "--cf-abduce",
        action="store_true",
        help="SCM 完成后（或读取已有 scm_model.pt）自动做块缩放干预 + 重推理，输出溯因 JSON",
    )
    p.add_argument(
        "--cf-rows",
        type=str,
        default="0,1,2",
        help="平均场表中要解释的行号（逗号分隔，0-based）",
    )
    p.add_argument(
        "--cf-scales",
        type=str,
        default="0.0,0.5,1.0",
        help="对单个父块的乘法干预尺度：0=移除倾向，1=不变",
    )
    p.add_argument(
        "--cf-behavior-json",
        type=str,
        default="",
        help="可选：行为标签→动作原型向量 JSON，用于伪概率叙事（参见 behavior_prototypes.example.json）",
    )
    p.add_argument("--cf-behavior-temp", type=float, default=1.0)
    p.add_argument("--cf-max-blocks", type=int, default=6)
    p.add_argument("--cf-max-joint", type=int, default=10, help="两两整块置零的联合干预上限条数")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    csv_path = Path(args.csv).resolve()
    if not csv_path.is_file():
        logger.error("找不到 CSV：%s", csv_path)
        return 2

    if args.output_dir:
        out = Path(args.output_dir).resolve()
    else:
        out = csv_path.parent / "game_reduction_out"
    out.mkdir(parents=True, exist_ok=True)

    scm_session_model = None
    scm_session_meta = None
    mf_df_cached: pd.DataFrame | None = None
    mf_schema_cached: dict | None = None

    ns = [x.strip() for x in args.neighbor_subset.split(",") if x.strip()]
    neighbor_subset = ns if ns else None

    df = load_maddpg_trajectory_csv(csv_path)
    if args.max_rows > 0:
        df = df.iloc[: int(args.max_rows)].copy()

    batch = build_joint_batch(
        df,
        args.target_agent,
        reward_mode=args.reward_mode,
        neighbor_subset=neighbor_subset,
    )

    cfg = JointQTrainConfig(
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden=args.hidden,
        device=args.device,
        seed=args.seed,
        use_target_network=True,
    )

    tr = train_joint_q(
        batch.joint_obs_state,
        batch.joint_action_full,
        batch.reward_signal,
        batch.joint_next_obs_state,
        batch.joint_action_next,
        batch.done,
        cfg,
    )

    meta = {
        "csv": str(csv_path),
        "target_agent": args.target_agent,
        "state_agents": list(batch.state_agents),
        "agent_order_full": list(batch.agent_order_full),
        "obs_dim_per_agent": batch.obs_dim,
        "act_dim_per_agent": batch.act_dim,
        "state_dim": batch.joint_obs_state.shape[1],
        "action_dim": batch.joint_action_full.shape[1],
        "hidden": args.hidden,
        "gamma": args.gamma,
        "epochs": args.epochs,
        "reward_mode": args.reward_mode,
        "top_k_hard_cap": args.top_k,
        "min_abs_influence": args.min_abs_influence,
    }
    save_joint_q(out / "joint_q.pt", tr.q_net, meta)

    infl = estimate_neighbor_effects(
        tr.q_net,
        batch,
        args.target_agent,
        neighbors=None,
        device=args.device,
        ablation_mode=getattr(args, "ablation_mode", "zero"),
        noise_scale=getattr(args, "noise_scale", 1.0),
    )
    # 记录归一化影响度
    for row in infl:
        logger.info(
            "邻居 %s: |ΔQ|=%.4f  signed=%.4f  |ΔQ|/σ_Q=%.3fσ",
            row.neighbor, row.mean_abs_effect, row.mean_signed_effect, row.mean_abs_effect_normalized,
        )
    thr = args.min_abs_influence
    use_thr = thr >= 0.0
    use_norm = getattr(args, "use_normalized_threshold", False)
    sel = select_key_neighbors(
        infl,
        min_abs_effect=float(thr) if use_thr else None,
        max_count=args.top_k,
        value_attr="mean_abs_effect_normalized" if use_norm else "mean_abs_effect",
    )
    main_id = args.target_agent
    key_neighbor_ids_ordered = [row.neighbor for row in sel]
    reduced_kept_agents = [main_id] + key_neighbor_ids_ordered
    dropped = [aid for aid in batch.agent_order_full if aid not in set(reduced_kept_agents)]

    report = {
        "final_train_loss": tr.final_loss,
        "ablation_mode": getattr(args, "ablation_mode", "zero"),
        "q_value_stats": q_validation,
        "neighbor_influence": [asdict(x) for x in infl],
        "selected_neighbor_rows": [asdict(x) for x in sel],
        "reduction_semantics": {
            "kop_main_agent": main_id,
            "neighbor_selection": (
                f"threshold_trunc min_abs_influence={thr} + cap top_k={args.top_k}"
                if use_thr
                else f"top_k_only K={args.top_k}"
            ),
            "min_abs_influence_enabled": use_thr,
            "min_abs_influence": float(thr) if use_thr else None,
            "top_k_hard_cap": args.top_k,
            "ablation_mode": getattr(args, "ablation_mode", "zero"),
            "use_normalized_threshold": use_norm,
            "threshold_applied_to": "mean_abs_effect_normalized" if use_norm else "mean_abs_effect",
            "notes": "启用阈值时：影响度从高到低，一旦 < 阈值则该体及更弱者全部丢弃；"
            "top-k>0 时在截断结果上再的人数上限；top-k<=0 且启用阈值时不设人数上限。",
        },
        "reduced_kept_agents": reduced_kept_agents,
        "reduced_dropped_agents": dropped,
        "meta": meta,
    }

    if not args.no_mean_field:
        gmap_path = args.group_map_json.strip()
        gm = load_group_map_json(Path(gmap_path)) if gmap_path else {}
        mf_df, mf_schema = build_mean_field_features(
            df,
            target_agent=main_id,
            kept_neighbors=key_neighbor_ids_ordered,
            group_map=gm,
            default_pool_name=args.mf_default_pool,
            include_obs_std=args.mf_include_obs_std,
        )
        mf_csv = out / "mean_field_features.csv"
        mf_df.to_csv(mf_csv, index=False, encoding="utf-8-sig")
        schema_path = out / "mean_field_schema.json"
        mf_schema_with_cols = dict(mf_schema)
        mf_schema_with_cols["csv_columns"] = list(mf_df.columns)
        schema_path.write_text(
            json.dumps(mf_schema_with_cols, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        mf_df_cached = mf_df
        mf_schema_cached = mf_schema_with_cols
        nf = sum(1 for c in mf_df.columns if str(c).startswith("mf_"))
        report["mean_field"] = {
            "literature_approx": (
                "Q_i(s_i,a_i,a_1..a_N) ~ Q_i(s_i,a_i, {avg group states},{avg group actions}); "
                "实现为拼接 [s_target,a_target] 与各群体 [s_mean_G,a_mean_G,(std_G)]"
            ),
            "features_csv": str(mf_csv.resolve()),
            "schema_json": str(schema_path.resolve()),
            "mf_feature_columns_n": nf,
            "group_map_used": gm if gm else "(empty -> all kept neighbors -> single pool)",
        }
        logger.info("平均场特征 CSV: %s (dim=%s)", mf_csv.resolve(), nf)

        if args.train_scm:
            try:
                from .scm_learning import (
                    SCMTrainConfig,
                    build_scm_tensors,
                    edge_strength_via_ablation,
                    save_scm_checkpoint,
                    specs_to_slices,
                    train_structural_equation,
                )
            except ImportError:
                from causal.game_reduction.scm_learning import (
                    SCMTrainConfig,
                    build_scm_tensors,
                    edge_strength_via_ablation,
                    save_scm_checkpoint,
                    specs_to_slices,
                    train_structural_equation,
                )

            Xm, ym, scm_pre = build_scm_tensors(
                mf_df,
                mf_schema_with_cols,
                env_pad_dim=int(args.scm_env_padding),
            )
            scm_cfg = SCMTrainConfig(
                epochs=int(args.scm_epochs),
                lr=float(args.scm_lr),
                batch_size=min(int(args.batch_size), max(64, Xm.shape[0] // 8)),
                hidden=int(args.scm_hidden),
                device=args.device,
                seed=args.seed,
            )
            scm_res = train_structural_equation(Xm, ym, scm_pre, scm_cfg)
            sl_obj = specs_to_slices(scm_pre["parent_slices_spec"])
            edges, base_mse = edge_strength_via_ablation(
                scm_res.model,
                Xm,
                ym,
                sl_obj,
                device=args.device,
            )
            scm_ckpt = out / "scm_model.pt"
            save_scm_checkpoint(scm_ckpt, scm_res.model, scm_res.meta)
            scm_edges_path = out / "scm_causal_edges.json"
            scm_edges_blob = {
                "outcome": "a_target",
                "baseline_mse_on_all_rows": base_mse,
                "edges_sorted_by_delta_mse": edges,
                "notes": (
                    "置零消融得到 parent_block→Y 的强度启发式；"
                    "非等价于Pearl意义下ATE；时间先后由逐步轨迹隐含"
                ),
            }
            scm_edges_path.write_text(
                json.dumps(scm_edges_blob, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            report["scm"] = {
                "structural_equation": "a_target ≈ NN( concat(parents_without_a_target), env_pad )",
                "checkpoint": str(scm_ckpt.resolve()),
                "edges_json": str(scm_edges_path.resolve()),
                "train_mse": scm_res.final_train_mse,
                "val_mse": scm_res.val_mse,
                "parent_blocks": scm_pre["parent_block_order"],
            }
            scm_session_model = scm_res.model
            scm_session_meta = scm_res.meta

    elif args.train_scm:
        logger.warning("跳过 SCM：需要平均场 CSV（禁止使用 --no-mean-field）")

    if args.cf_abduce:
        mf_use = mf_df_cached
        sch_use = mf_schema_cached
        if mf_use is None:
            p_csv = out / "mean_field_features.csv"
            p_sc = out / "mean_field_schema.json"
            if p_csv.is_file() and p_sc.is_file():
                mf_use = pd.read_csv(p_csv, encoding="utf-8-sig")
                sch_use = json.loads(p_sc.read_text(encoding="utf-8"))
            else:
                mf_use = None
        scm_ck = out / "scm_model.pt"
        if mf_use is None:
            logger.warning("溯因跳过：无 mean_field_features.csv / schema")
        elif not scm_ck.is_file():
            logger.warning("溯因跳过：无 scm_model.pt（请先 --train-scm 或放入检查点）")
        else:
            try:
                from .scm_counterfactual_abduction import (
                    load_behavior_prototypes_json,
                    run_counterfactual_abduction,
                )
                from .scm_learning import load_scm_checkpoint
            except ImportError:
                from causal.game_reduction.scm_counterfactual_abduction import (
                    load_behavior_prototypes_json,
                    run_counterfactual_abduction,
                )
                from causal.game_reduction.scm_learning import load_scm_checkpoint

            if scm_session_model is not None:
                cf_model, cf_meta = scm_session_model, scm_session_meta
            else:
                cf_model, cf_meta = load_scm_checkpoint(scm_ck, args.device)

            row_list = []
            for tok in args.cf_rows.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                row_list.append(int(tok))
            row_list = [i for i in row_list if 0 <= i < len(mf_use)]
            if not row_list:
                row_list = [0]

            scale_list = []
            for tok in args.cf_scales.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                scale_list.append(float(tok))
            if not scale_list:
                scale_list = [0.0, 1.0]

            bjp = args.cf_behavior_json.strip()
            protos = load_behavior_prototypes_json(bjp if bjp else None)

            edges_path = out / "scm_causal_edges.json"
            cf_blob = run_counterfactual_abduction(
                mf_use,
                sch_use,
                cf_model,
                cf_meta,
                row_indices=row_list,
                scales=scale_list,
                device=args.device,
                behavior_prototypes=protos if protos else None,
                behavior_temperature=float(args.cf_behavior_temp),
                edge_json_path=edges_path if edges_path.is_file() else None,
                max_blocks_from_edges=int(args.cf_max_blocks),
                max_joint_pairs=int(args.cf_max_joint),
                env_pad_dim=int(args.scm_env_padding),
            )
            cf_out = out / "counterfactual_abduction.json"
            cf_out.write_text(json.dumps(cf_blob, ensure_ascii=False, indent=2), encoding="utf-8")
            report["counterfactual_abduction"] = {
                "artifact": str(cf_out.resolve()),
                "rows_used": row_list,
                "scales": scale_list,
            }

    logger.info(
        "博弈约简：主智能体=%s + 关键邻居(%s个)=%s → 总计参与 %s 个 | 剔除 %s",
        main_id,
        len(key_neighbor_ids_ordered),
        key_neighbor_ids_ordered,
        len(reduced_kept_agents),
        dropped if dropped else "（无）",
    )

    js = out / "game_reduction_report.json"
    js.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("报告已写入 %s", js.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
