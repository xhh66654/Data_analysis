#!/usr/bin/env python3
"""
博弈约简 + 溯因流水线一键入口。

本脚本集成联合 Q 训练 → 邻居约简 → 平均场特征 → SCM 学习 → 反事实溯因 → 因果链排序。

配置方式：修改下方 RUN_CONFIG 字典（至少填写 trajectory_csv）。

主要输出（默认 {轨迹目录}/game_reduction_out/）：
  joint_q.pt                      — 联合 Q 网络
  mean_field_features.csv         — 低维平均场特征
  scm_model.pt                    — 结构方程模型
  scm_causal_edges.json           — 因果边强度排序
  counterfactual_abduction.json   — 反事实干预结果 + 因果链排序
  game_reduction_report.json      — 全流程摘要

用法：
  1. 修改下方 RUN_CONFIG（至少填写 trajectory_csv）
  2. 在 Algorithm 目录执行：
       python causal/game_reduction/run_pipeline.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

# 保证从任意工作目录运行都能 import causal.*
_ALGORITHM_ROOT = Path(__file__).resolve().parents[2]
if str(_ALGORITHM_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALGORITHM_ROOT))

import torch

try:
    from .mean_field import build_mean_field_features, build_self_obs_features, load_group_map_json
    from .neighbor_counterfactual import estimate_neighbor_effects, select_key_neighbors
    from .train_joint_q import JointQTrainConfig, save_joint_q, train_joint_q
    from .trajectory_maddpg import build_joint_batch, load_maddpg_trajectory_csv
    from .causal_narrative_gen import enrich_counterfactual_with_narrative
except ImportError:
    from causal.game_reduction.mean_field import build_mean_field_features, build_self_obs_features, load_group_map_json
    from causal.game_reduction.neighbor_counterfactual import estimate_neighbor_effects, select_key_neighbors
    from causal.game_reduction.train_joint_q import JointQTrainConfig, save_joint_q, train_joint_q
    from causal.game_reduction.trajectory_maddpg import build_joint_batch, load_maddpg_trajectory_csv
    from causal.game_reduction.causal_narrative_gen import enrich_counterfactual_with_narrative

logger = logging.getLogger(__name__)


# =============================================================================
# 在此填写你的数据与参数（运行前只需改这一块）
# =============================================================================
RUN_CONFIG = {
    # ========== 输入/输出路径 ==========
    # 【必填】MADDPG 训练轨迹 CSV
    "trajectory_csv": r"F:\cause_analysis\Algorithm\causal\MADDPG_file\results\simple_spread_v3\MADDPG_10\training_trajectory.csv",
    
    # 输出目录（留空 "" → 自动设为 CSV 所在目录/game_reduction_out/）
    "output_dir": "",
    
    # ========== 数据与主智能体 ==========
    # 目标/关键智能体（如 "agent_0"）
    "target_agent": "agent_0",
    
    # 调试时截取前 N 行；0 表示用全部
    "max_rows": 3000,
    
    # ========== 邻居约简参数 ==========
    # 硬上限：最多保留 K 个关键邻居
    "top_k": 5,

    # 影响度阈值；-1 表示禁用，仅按 top_k；≥0 时启用阈值截断
    "min_abs_influence": 0.19,

    # 阈值是否基于归一化影响度 (|ΔQ|/σ_Q)；开启后 min_abs_influence 以 σ 为单位
    "use_normalized_threshold": True,

    # 反事实干预模式："zero"（置零，默认）或 "noise"（加高斯噪声，更接近训练分布）
    "ablation_mode": "noise",

    # 噪声干预的缩放系数（仅 noise 模式生效）：1.0=与数据同方差，0.5=半方差
    "noise_scale": 1.0,

    # 邻居子集（留空 "" 表示全部；"agent_1,agent_2" 表示仅这些邻居）
    "neighbor_subset": "",
    
    # ========== 联合 Q 训练参数 ==========
    # 训练轮数
    "epochs": 25,
    
    # 批大小
    "batch_size": 512,
    
    # 学习率
    "lr":5e-4,
    
    # 隐层宽度
    "hidden": 256,
    
    # 折扣因子
    "gamma": 0.95,
    
    # Bellman 奖励："target" 或 "team_sum"
    "reward_mode": "target",
    
    # 训练设备："cpu" 或 "cuda"
    "device": "cpu",
    
    # 随机种子
    "seed": 42,
    
    # ========== 平均场参数 ==========
    # 是否生成平均场特征 CSV（推荐保持 False 以启用）
    "no_mean_field": False,
    
    # 群体映射 JSON 路径（留空 "" 表示无；格式：{agent_id → 群体名}）
    "group_map_json": "",
    
    # 未在 group_map 中的邻居统一归入此群体名
    "mf_default_pool": "pooled_neighbors",
    
    # 是否在每个群体后追加观测标准差
    "mf_include_obs_std": False,

    # ========== 单智能体决策约简（邻居为空时自动启用） ==========
    # 自身观测的语义分块，格式 {"块名": [start, end), ...}，必须完整覆盖 [0, obs_dim)
    # 留空 {} 则邻居为空时退化为「单块 s_target」（行为克隆，无约简意义）
    # 示例（Simple Spread 30维，请按实际观测结构调整）:
    # "obs_blocks": {
    #     "s_self_pos":      [0, 2],
    #     "s_self_vel":      [2, 4],
    #     "s_landmarks":     [4, 16],
    #     "s_other_agents":  [16, 28],
    #     "s_comm":          [28, 30],
    # },
    "obs_blocks": {
        "s_self_vel":      [0, 2],
        "s_self_pos":      [2, 4],
        "s_landmarks":     [4, 14],
        "s_other_agents":  [14, 22],
        "s_comm":          [22, 30],
    },
    
    # ========== SCM 参数 ==========
    # 是否训练结构方程模型
    "train_scm": True,
    
    # SCM 训练轮数
    "scm_epochs": 25,
    
    # SCM 学习率
    "scm_lr": 1e-3,
    
    # SCM 隐层宽度
    "scm_hidden": 128,
    
    # SCM 末尾环境占位维度
    "scm_env_padding": 0,
    
    # ========== 反事实溯因参数 ==========
    # 是否进行反事实溯因
    "cf_abduce": True,
    
    # 要解释的行号（0-based，逗号分隔）
    "cf_rows": "0,1",
    
    # 干预尺度（逗号分隔；0=移除, 1=不变）
    "cf_scales": "0.0,1.0",
    
    # 行为原型 JSON 路径（用于伪概率；留空 "" 表示无）
    "cf_behavior_json": "",
    
    # 行为伪概率温度参数
    "cf_behavior_temp": 1.0,
    
    # 从因果边选取优先块的最多个数
    "cf_max_blocks": 6,
    
    # 联合干预（两两置零）的最多对数
    "cf_max_joint": 4,
    
    # ========== 自然语言生成参数 ==========
    # 是否补充阶段 1 的因果链排序（定量文本）
    "add_causal_narrative": True,
    
    # 因果链中最多保留几个因素
    "narrative_max_factors": 3,
}


def parse_args(argv: list[str] | None = None, use_config: bool = True) -> argparse.Namespace:
    """
    解析命令行参数或从 RUN_CONFIG 读取。
    
    Args:
        argv: 命令行参数（若为 None 则从 sys.argv[1:] 读取）
        use_config: 若为 True 且 sys.argv[1:] 为空，则从 RUN_CONFIG 构建参数
    """
    p = argparse.ArgumentParser(description="博弈约简：联合 Q + 邻居反事实 + Top-K")
    p.add_argument(
        "--csv",
        type=str,
        default="",
        help="MADDPG training_trajectory.csv 的绝对路径（若留空则从 RUN_CONFIG 读取）",
    )
    p.add_argument(
        "--target-agent",
        type=str,
        default="",
        help="主/KOP 智能体 id（如 agent_0）；若留空则从 RUN_CONFIG 读取",
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
    p.add_argument("--add-causal-narrative", action="store_true", default=None, help="补充阶段 1 因果链排序")
    p.add_argument("--narrative-max-factors", type=int, default=3)
    
    # 解析
    args = p.parse_args(argv)
    
    # 从 RUN_CONFIG 填充参数
    if use_config and (argv is None or len(argv) == 0):
        # 遍历 RUN_CONFIG 中的所有参数
        config_mapping = {
            "trajectory_csv": "csv",
            "target_agent": "target_agent",
            "output_dir": "output_dir",
            "neighbor_subset": "neighbor_subset",
            "reward_mode": "reward_mode",
            "max_rows": "max_rows",
            "gamma": "gamma",
            "epochs": "epochs",
            "batch_size": "batch_size",
            "lr": "lr",
            "hidden": "hidden",
            "device": "device",
            "seed": "seed",
            "top_k": "top_k",
            "min_abs_influence": "min_abs_influence",
            "ablation_mode": "ablation_mode",
            "noise_scale": "noise_scale",
            "use_normalized_threshold": "use_normalized_threshold",
            "no_mean_field": "no_mean_field",
            "group_map_json": "group_map_json",
            "mf_default_pool": "mf_default_pool",
            "mf_include_obs_std": "mf_include_obs_std",
            "obs_blocks": "obs_blocks",
            "train_scm": "train_scm",
            "scm_epochs": "scm_epochs",
            "scm_lr": "scm_lr",
            "scm_hidden": "scm_hidden",
            "scm_env_padding": "scm_env_padding",
            "cf_abduce": "cf_abduce",
            "cf_rows": "cf_rows",
            "cf_scales": "cf_scales",
            "cf_behavior_json": "cf_behavior_json",
            "cf_behavior_temp": "cf_behavior_temp",
            "cf_max_blocks": "cf_max_blocks",
            "cf_max_joint": "cf_max_joint",
            "add_causal_narrative": "add_causal_narrative",
            "narrative_max_factors": "narrative_max_factors",
        }
        
        for config_key, arg_attr in config_mapping.items():
            if config_key in RUN_CONFIG:
                config_val = RUN_CONFIG[config_key]
                # 直接用 RUN_CONFIG 的值覆盖（如果不是通过命令行指定的）
                setattr(args, arg_attr, config_val)
    
    return args


def validate_q_network(q_net, s_joint, a_joint, device='cpu'):
    """验证 Q 网络质量指标"""
    import torch
    
    q_net.eval()
    # 转为 tensor（如果还不是的话）
    if not isinstance(s_joint, torch.Tensor):
        s_joint = torch.from_numpy(s_joint).float()
    if not isinstance(a_joint, torch.Tensor):
        a_joint = torch.from_numpy(a_joint).float()
    
    with torch.no_grad():
        Q_vals = q_net.forward_sa(s_joint.to(device), a_joint.to(device))
    
    return {
        "q_value_range": [float(Q_vals.min()), float(Q_vals.max())],
        "q_value_mean": float(Q_vals.mean()),
        "q_value_std": float(Q_vals.std()),
        "q_value_median": float(Q_vals.median()),
        "q_value_percentiles": {
            "p5": float(Q_vals.quantile(0.05)),
            "p25": float(Q_vals.quantile(0.25)),
            "p75": float(Q_vals.quantile(0.75)),
            "p95": float(Q_vals.quantile(0.95)),
        },
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    
    # 如果没有命令行参数，使用 RUN_CONFIG；否则用命令行参数
    if argv is None or (isinstance(argv, list) and len(argv) == 0):
        # 直接从 RUN_CONFIG 加载（不走命令行解析）
        args = parse_args(argv=[], use_config=True)
    else:
        # 用命令行参数覆盖
        args = parse_args(argv=argv, use_config=False)


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

    # 验证 Q 网络质量
    q_validation = validate_q_network(tr.q_net, batch.joint_obs_state, batch.joint_action_full, args.device)
    logger.info(
        "Q 值范围 [%.3f, %.3f]，均值 %.3f，方差 %.3f",
        q_validation["q_value_range"][0],
        q_validation["q_value_range"][1],
        q_validation["q_value_mean"],
        q_validation["q_value_std"],
    )

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

        obs_blocks_cfg = getattr(args, "obs_blocks", None) or {}
        use_self_obs = (len(key_neighbor_ids_ordered) == 0) and (len(obs_blocks_cfg) > 0)

        if use_self_obs:
            logger.info(">>> 单智能体决策约简模式 <<<")
            logger.info("  触发原因: 邻居为空 (top_k=%s, min_abs_influence=%s)", args.top_k, args.min_abs_influence)
            logger.info("  观测分块: %s 个语义块 → %s", len(obs_blocks_cfg), list(obs_blocks_cfg.keys()))
            logger.info("  注意: 此模式仅做自身观测维度约简，不包含邻居交互信息，SCM 预测精度受限")
            logger.info("  适用场景: 识别冗余观测维度、观测空间压缩")
            mf_df, mf_schema = build_self_obs_features(
                df,
                target_agent=main_id,
                obs_blocks=obs_blocks_cfg,
            )
            report["mode"] = "single_agent_decision_reduction"
            report["obs_blocks_used"] = obs_blocks_cfg
            report["single_agent_note"] = (
                "SCM 仅以自身观测分块预测 a_target，不含邻居交互统计；"
                "边强度排序可用于识别冗余观测维度，但绝对预测质量受限"
            )
        else:
            if len(key_neighbor_ids_ordered) == 0:
                logger.warning("邻居为空且未配置 obs_blocks，退化为单块 s_target（无约简意义）")
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

            if use_self_obs:
                top_blocks = [e["parent_block"] for e in edges[:3]]
                weak_blocks = [e["parent_block"] for e in edges if e["relative_strength"] < 0.15]
                logger.info(
                    ">>> 决策约简结果: 核心因素=%s | 可裁维度=%s (共 %s 维) <<<",
                    top_blocks,
                    weak_blocks if weak_blocks else "（无）",
                    sum(obs_blocks_cfg.get(b, [0, 0])[1] - obs_blocks_cfg.get(b, [0, 0])[0] for b in weak_blocks),
                )

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

    # ========== 阶段 4：自动补充因果链排序（阶段 1 自然语言生成） ==========
    if args.cf_abduce and args.add_causal_narrative:
        logger.info("正在补充因果链排序...")
        cf_out_path = out / "counterfactual_abduction.json"
        if cf_out_path.is_file():
            try:
                enrich_counterfactual_with_narrative(
                    cf_out_path,
                    output_path=None,  # 原地覆盖
                    max_factors=int(args.narrative_max_factors)
                )
                logger.info("因果链排序已补充到 %s", cf_out_path.resolve())
            except Exception as e:
                logger.warning("补充因果链排序失败: %s", e)
        else:
            logger.warning("未找到反事实溯因文件，跳过因果链排序补充")

    # 将 Q 网络验证结果存入报告
    report["q_network_validation"] = q_validation

    js = out / "game_reduction_report.json"
    js.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("报告已写入 %s", js.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
