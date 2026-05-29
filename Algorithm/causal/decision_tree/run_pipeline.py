"""
VIPER 决策树离线全流程一键入口（推荐日常使用）。

本脚本是 decision_tree 包的主入口：读取轨迹 CSV，按顺序执行
  阶段1 FQE → 阶段2 l_hat → 阶段3 weights → 阶段4 VIPER+CART，
并在 output_dir 下写出 q_hat.pt、l_hat.csv、weights.csv、viper_out/ 规则与树图，
最后汇总为 final_result.json。可选调用 verify_phase_link 做端到端衔接校验。

配置方式：修改下方 RUN_CONFIG 字典（至少填写 trajectory_csv）。
也可设置 run_viper_tune_grid=True，自动调用 tune_viper 做 CART 网格搜索。

主要输出（默认 {轨迹目录}/fqe_out/）：
  q_hat.pt          — FQE 训练的 Q 网络 checkpoint
  l_hat.csv         — 逐行价值差距
  weights.csv       — VIPER 抽样权重
  viper_out/        — rules.txt、tree.json、policy_tree.pdf 等
  final_result.json — 全流程指标与路径汇总

用法：
  1. 修改下方 RUN_CONFIG（至少填写 trajectory_csv）
  2. 在 Algorithm 目录执行：
       python causal/decision_tree/run_pipeline.py
     或：
       python -m causal.decision_tree.run_pipeline

与 cli.py 的关系：本文件用 RUN_CONFIG 字典配置；cli.py 提供等价的命令行参数版本。
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# 保证从任意工作目录运行都能 import causal.*
_ALGORITHM_ROOT = Path(__file__).resolve().parents[2]
if str(_ALGORITHM_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALGORITHM_ROOT))

import numpy as np
import pandas as pd

from causal.decision_tree.fqe import FQETrainConfig, load_q_hat, save_q_hat, train_q_hat
from causal.decision_tree.l_hat import compute_l_hat, l_hat_dataframe, save_l_hat_csv
from causal.decision_tree.rule_ensemble import RuleEnsemble, ensemble_rules_from_rounds, save_ensemble_rules, rules_to_if_then_strings
from causal.decision_tree.trajectory_io import ACTION_COL, STATE_COLS, build_transitions, load_trajectory_csv, normalize_rewards, RewardNormConfig
from causal.decision_tree.viper_cart import ViperConfig, run_viper_from_files
from causal.decision_tree.weights import run_weights_from_l_hat_csv

logger = logging.getLogger(__name__)


# =============================================================================
# 在此填写你的数据与参数（运行前只需改这一块）
# =============================================================================
RUN_CONFIG = {
    # -------------------------------------------------------------------------
    # 输入 / 输出路径
    # -------------------------------------------------------------------------
    # 【必填】离线轨迹 CSV（通常由 causal/main.py 导出）
    # 必需列：s_0…s_7, s_next_0…s_next_7, action, reward, episode, dw, truncated
    "trajectory_csv": r"F:\cause_analysis\Algorithm\causal\trajectories\trajectory_LLdV3_S0_5.csv",
    # 全流程产物目录；留空 "" → 自动设为 {trajectory_csv 所在目录}/fqe_out/
    # 其中 viper_out/ 存放规则与决策树图
    "output_dir": "",
    # -------------------------------------------------------------------------
    # 阶段 1：FQE —— 训练 Q_hat 神经网络（影响最大，见 调参文档.md）
    # 输出：fqe_out/q_hat.pt；下游 l_hat / weights / VIPER 均依赖此 Q 估计
    # -------------------------------------------------------------------------
    # 训练轮数；loss 仍下降时可加大（大表可试 50～100）
    "fqe_epochs": 5,
    # 训练设备："cuda" | "cpu"（百万行建议 GPU）
    "fqe_device": "cuda",
    # Bootstrap 目标："sarsa"=用轨迹真实下一步动作 a'（默认，贴近行为策略）
    #              "max_q"=用 max_a' Q(s',a')（更乐观，l_hat 往往更大）
    "fqe_target": "sarsa",
    # 折扣因子 γ，用于 target = r + γ(1-done)·Q(s',·)；长回合可保持 0.99
    "fqe_gamma": 0.99,
    # Adam 学习率；loss 震荡时可降至 5e-4～1e-4
    "fqe_lr": 1e-3,#5e-2,
    # 每批样本数；显存允许可增至 512～1024 使更新更稳
    "fqe_batch_size": 1024,
    # 隐层宽度（网络：8 → hidden → hidden → n_actions）；过大易过拟合
    "fqe_hidden": 256,#128
    # 是否启用目标网络稳定 TD 目标（大表、loss 抖动时可改 True）
    "fqe_use_target_network": True,
    # -------------------------------------------------------------------------
    # 阶段 2：l_hat —— 冻结 Q_hat，逐行计算 V_hat - Q(s,a_实际)
    # 输出：fqe_out/l_hat.csv（仅推理，不改变数值）
    # -------------------------------------------------------------------------
    # 前向分块大小；只影响速度与显存，不影响 l_hat 结果
    "l_hat_batch_size": 256,
    # -------------------------------------------------------------------------
    # 阶段 3：weights —— w_raw=max(l_hat,0)+eps，再归一化为抽样概率
    # 输出：fqe_out/weights.csv（VIPER 多轮共用，不再重算）
    # -------------------------------------------------------------------------
    # 平滑项 ε；仅 viper_weighted_sampling=True 时生效
    "weights_eps": 0.1,
    # acc_full 优先：S0_5 上均匀抽样优于加权（~67% vs ~66% @ depth=5）
    "viper_weighted_sampling": False,
    # -------------------------------------------------------------------------
    # 阶段 4～6：VIPER + CART + 规则/树导出
    # 输出：fqe_out/viper_out/rules.txt、tree.json、policy_tree.pdf 等
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # 【策略：100轮浅树 + 规则集成提纯Top100】
    #        单轮: depth=5 → 32条规则
    #        100轮: 100 × 32 = 3200 规则实例
    #        集成后: 仅保留跨轮最稳定的前100条
    # -------------------------------------------------------------------------
    "run_viper_tune_grid": False,
    "viper_n_round": 100,
    "viper_pick_best_round": False,
    "cart_max_depth": 5,
    "cart_min_samples_leaf": 1,
    "cart_min_samples_split": 2,
    "resample_size": 0,
    "weight_noise": 0.05,
    # -------------------------------------------------------------------------
    # 动态权重更新（自适应重采样）
    # -------------------------------------------------------------------------
    # 是否启用动态权重更新（根据预测误差调整权重）
    "viper_adaptive_resample": False,
    # 错误样本权重增加比例
    "viper_adaptation_rate": 0.1,
    # 每多少轮进行一次权重更新
    "viper_adapt_interval": 3,
    # -------------------------------------------------------------------------
    # 奖励归一化（提升 FQE 训练稳定性）
    # -------------------------------------------------------------------------
    # 是否启用奖励归一化
    "enable_reward_norm": True,
    # 奖励裁剪范围
    "reward_clip_min": -10.0,
    "reward_clip_max": 10.0,
    # 是否按 episode 分别归一化（保留相对奖励结构）
    "reward_norm_per_episode": False,
    # -------------------------------------------------------------------------
    # 是否导出 tree.json、tree_nodes.csv、policy_tree_debug.dot
    "export_tree": True,
    # -------------------------------------------------------------------------
    # 规则集成：100轮投票 → 按置信度排序 → 取前100条
    # -------------------------------------------------------------------------
    "run_rule_ensemble": True,
    "ensemble_max_rules": 100,
    "ensemble_confidence_threshold": 0.0,
    "ensemble_min_support_rounds": 2,
    "ensemble_feature_pattern": True,
    # 是否用 Graphviz 渲染 PDF 流程图（需 pip install graphviz + 系统 Graphviz）
    "render_tree_pdf": True,
    # 是否再从同一 .dot 导出 PNG（与 PDF 同风格）
    "render_tree_png": False,
    # Graphviz 失败时 matplotlib 回退 PNG 的 DPI（仅 render_tree_png 回退路径用）
    "tree_image_dpi": 150,
    # 渲染成功后是否用系统默认程序打开 PDF
    "open_tree_pdf": False,
    # 与 open_tree_pdf 相同，保留旧参数名
    "show_tree_image": False,
    # -------------------------------------------------------------------------
    # 其它
    # -------------------------------------------------------------------------
    # 随机种子（FQE 训练、VIPER 抽样、CART random_state=seed+轮次）
    "seed": 45,
    # 流程结束后是否运行 verify_e2e 衔接校验
    "run_verify": True,
    # 是否打印 INFO 日志
    "verbose": True,
}


@dataclass
class PipelineResult:
    success: bool
    trajectory_csv: str
    output_dir: str
    n_samples: int
    q_hat_pt: str
    l_hat_csv: str
    weights_csv: str
    rules_txt: str
    rules_json: str
    viper_summary_json: str
    final_result_json: str
    fqe_final_loss: float
    viper_last_acc_full: float
    viper_last_acc_resampled: float
    n_rules: int
    rules_preview: list[str]
    tree_json: str
    tree_nodes_csv: str
    tree_dot: str
    tree_pdf: str
    tree_png: str


def _setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )


def run_full_pipeline(cfg: dict) -> PipelineResult:
    csv_path = Path(cfg["trajectory_csv"]).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"未找到轨迹 CSV: {csv_path}")

    out_dir = Path(cfg["output_dir"]).resolve() if str(cfg.get("output_dir", "")).strip() else csv_path.parent / "fqe_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "q_hat.pt"
    l_hat_path = out_dir / "l_hat.csv"
    weights_path = out_dir / "weights.csv"
    viper_out_dir = out_dir / "viper_out"
    final_json_path = out_dir / "final_result.json"

    seed = int(cfg.get("seed", 42))
    device = str(cfg.get("fqe_device", "cuda"))

    logger.info("读取轨迹: %s", csv_path)
    df = load_trajectory_csv(str(csv_path))
    n = len(df)

    # --- 奖励归一化（提升 FQE 训练稳定性）---
    if bool(cfg.get("enable_reward_norm", False)):
        reward_norm_cfg = RewardNormConfig(
            clip_range=(
                float(cfg.get("reward_clip_min", -10.0)),
                float(cfg.get("reward_clip_max", 10.0)),
            ),
            standardize=True,
            per_episode=bool(cfg.get("reward_norm_per_episode", False)),
        )
        df = normalize_rewards(df, reward_norm_cfg)

    # --- 阶段 1: FQE ---
    logger.info("阶段 1/4: FQE 训练 Q_hat (%d epochs, device=%s)", cfg.get("fqe_epochs", 30), device)
    trans = build_transitions(df)
    fqe_cfg = FQETrainConfig(
        gamma=float(cfg.get("fqe_gamma", 0.99)),
        lr=float(cfg.get("fqe_lr", 1e-3)),
        batch_size=int(cfg.get("fqe_batch_size", 256)),
        epochs=int(cfg.get("fqe_epochs", 30)),
        hidden=int(cfg.get("fqe_hidden", 256)),
        device=device,
        target=str(cfg.get("fqe_target", "sarsa")),
        use_target_network=bool(cfg.get("fqe_use_target_network", False)),
        seed=seed,
    )
    fqe_result = train_q_hat(trans, fqe_cfg)
    save_q_hat(
        ckpt_path,
        fqe_result.q_net,
        {
            "csv": str(csv_path),
            "gamma": fqe_cfg.gamma,
            "target": fqe_cfg.target,
            "state_dim": trans.s.shape[1],
            "n_actions": int(trans.a.max()) + 1,
            "hidden": fqe_cfg.hidden,
            "final_loss": fqe_result.final_loss,
            "loss_history": fqe_result.history,
        },
    )

    # --- 阶段 2: l_hat ---
    logger.info("阶段 2/4: 计算 l_hat")
    states = df[STATE_COLS].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    actions = pd.to_numeric(df[ACTION_COL], errors="coerce").values.astype(np.int64)
    q_net, _ = load_q_hat(ckpt_path, device=device)
    lh = compute_l_hat(
        q_net,
        states,
        actions,
        device=device,
        batch_size=int(cfg.get("l_hat_batch_size", 4096)),
    )
    save_l_hat_csv(l_hat_path, l_hat_dataframe(df, lh))

    # --- 阶段 3: weights ---
    weighted_sampling = bool(cfg.get("viper_weighted_sampling", True))
    sampling_label = "VIPER 加权" if weighted_sampling else "均匀"
    logger.info("阶段 3/4: 计算 weights（%s抽样）", sampling_label)
    run_weights_from_l_hat_csv(
        l_hat_path,
        output_path=weights_path,
        eps=float(cfg.get("weights_eps", 1e-6)),
        weighted_sampling=weighted_sampling,
    )

    # --- 阶段 4～6: VIPER + CART + 规则 ---
    m = int(cfg.get("resample_size", 0))
    n_round = int(cfg.get("viper_n_round", 8))
    weight_noise = float(cfg.get("weight_noise", 0.02))
    tune_best: dict | None = None
    viper_tune_dir = out_dir / "viper_tune"

    if bool(cfg.get("run_viper_tune_grid", False)):
        from causal.decision_tree.tune_viper import TUNE_GRID, run_viper_tune_grid

        logger.info("阶段 4/4: %s 抽样 + TUNE_GRID 网格调参 (%d 组) → viper_tune/", sampling_label, len(TUNE_GRID))
        tune_best, tune_paths, viper_result = run_viper_tune_grid(
            csv_path,
            weights_path,
            viper_tune_dir,
            n_round=n_round,
            weight_noise=weight_noise,
            seed=seed,
            export_best_to=viper_out_dir,
            render_tree_pdf=bool(cfg.get("render_tree_pdf", True)),
            render_tree_png=bool(cfg.get("render_tree_png", False)),
            tree_image_dpi=int(cfg.get("tree_image_dpi", 150)),
            open_tree_pdf=bool(cfg.get("open_tree_pdf", False) or cfg.get("show_tree_image", False)),
            export_tree=bool(cfg.get("export_tree", True)),
            resample_size=m if m > 0 else None,
        )
        logger.info("16 组对比表: %s", tune_paths.get("tune_results.csv", tune_paths.get("tune_results.json")))
    else:
        logger.info("阶段 4/4: VIPER 单组参数 (RUN_CONFIG cart_*) → viper_out/")
        viper_cfg = ViperConfig(
            n_round=n_round,
            max_depth=int(cfg.get("cart_max_depth", 5)),
            min_samples_leaf=int(cfg.get("cart_min_samples_leaf", 1)),
            min_samples_split=int(cfg.get("cart_min_samples_split", 2)),
            random_state=seed,
            resample_size=m if m > 0 else None,
            weight_noise_std=weight_noise,
            pick_best_by_full_acc=bool(cfg.get("viper_pick_best_round", True)),
            export_tree=bool(cfg.get("export_tree", True)),
            render_tree_pdf=bool(cfg.get("render_tree_pdf", True)),
            render_tree_png=bool(cfg.get("render_tree_png", False)),
            tree_image_dpi=int(cfg.get("tree_image_dpi", 150)),
            open_tree_pdf=bool(cfg.get("open_tree_pdf", False) or cfg.get("show_tree_image", False)),
            show_tree_image=bool(cfg.get("show_tree_image", False)),
            # 动态权重更新参数
            adaptive_resample=bool(cfg.get("viper_adaptive_resample", False)),
            adaptation_rate=float(cfg.get("viper_adaptation_rate", 0.1)),
            adapt_interval=int(cfg.get("viper_adapt_interval", 3)),
        )
        viper_result = run_viper_from_files(csv_path, weights_path, viper_out_dir, viper_cfg)

    sel_round = viper_result.selected_round
    sel_full = viper_result.selected_acc_full
    sel_resampled = viper_result.selected_acc_resampled

    # --- 规则集成（可选）---
    final_rules = viper_result.rules.copy()
    if bool(cfg.get("run_rule_ensemble", False)):
        logger.info("阶段 5/5: 规则集成（从 %d 轮规则中集成）", n_round)
        
        # 收集所有轮次的规则
        rules_per_round = []
        for round_result in viper_result.round_results:
            rules_per_round.append({
                'rules': round_result.rules,
                'acc_full': round_result.acc_full,
            })
        
        # 执行规则集成
        ensemble_rules = ensemble_rules_from_rounds(
            rules_per_round,
            max_rules=int(cfg.get("ensemble_max_rules", 100)),
            confidence_threshold=float(cfg.get("ensemble_confidence_threshold", 0.75)),
            min_support_rounds=int(cfg.get("ensemble_min_support_rounds", 2)),
            use_feature_pattern=bool(cfg.get("ensemble_feature_pattern", True)),
        )
        
        # 转换为 IF-THEN 格式
        final_rules = rules_to_if_then_strings(ensemble_rules)
        
        # 保存集成规则
        ensemble_rules_path = viper_out_dir / "ensemble_rules"
        save_ensemble_rules(ensemble_rules, ensemble_rules_path)
        
        logger.info("规则集成完成: 原始规则数=%d 集成后规则数=%d", 
                    sum(len(r['rules']) for r in rules_per_round), len(final_rules))

    rules_txt = viper_out_dir / "rules.txt"
    rules_json = viper_out_dir / "rules.json"
    summary_json = viper_out_dir / "viper_summary.json"
    tree_json = viper_out_dir / "tree.json"
    tree_nodes_csv = viper_out_dir / "tree_nodes.csv"
    tree_dot = viper_out_dir / "policy_tree_debug.dot"
    tree_pdf = viper_out_dir / "policy_tree.pdf"
    tree_png = viper_out_dir / "policy_tree.png"
    preview = final_rules[:10]

    # 保存集成后的规则到 rules.txt 和 rules.json（覆盖原始规则）
    if bool(cfg.get("run_rule_ensemble", False)) and final_rules:
        rules_txt.write_text("\n".join(final_rules) + "\n", encoding="utf-8")
        rules_json.write_text(
            json.dumps(final_rules, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # 更新 summary.json 中的 n_rules
        if summary_json.exists():
            with open(summary_json, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
            summary_data["n_rules"] = len(final_rules)
            with open(summary_json, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    result = PipelineResult(
        success=True,
        trajectory_csv=str(csv_path),
        output_dir=str(out_dir),
        n_samples=n,
        q_hat_pt=str(ckpt_path),
        l_hat_csv=str(l_hat_path),
        weights_csv=str(weights_path),
        rules_txt=str(rules_txt),
        rules_json=str(rules_json),
        viper_summary_json=str(summary_json),
        final_result_json=str(final_json_path),
        fqe_final_loss=float(fqe_result.final_loss),
        viper_last_acc_full=float(sel_full),
        viper_last_acc_resampled=float(sel_resampled),
        n_rules=len(final_rules),  # 使用集成后的规则数
        rules_preview=preview,
        tree_json=str(tree_json) if tree_json.is_file() else "",
        tree_nodes_csv=str(tree_nodes_csv) if tree_nodes_csv.is_file() else "",
        tree_dot=str(tree_dot) if tree_dot.is_file() else "",
        tree_pdf=str(tree_pdf) if tree_pdf.is_file() else "",
        tree_png=str(tree_png) if tree_png.is_file() else "",
    )

    payload = {
        "success": True,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "config": {k: v for k, v in cfg.items() if k != "verbose"},
        **asdict(result),
        "viper_selected_round": sel_round,
        "viper_tune_best": tune_best,
        "viper_tune_dir": str(viper_tune_dir) if tune_best else "",
        "rules": viper_result.rules,
    }
    final_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("已写入汇总结果: %s", final_json_path)

    if cfg.get("run_verify", True):
        from causal.decision_tree.verify_phase_link import verify_e2e

        logger.info("自动校验全流程衔接…")
        rc = verify_e2e(csv_path, out_dir, device=device, eps=float(cfg.get("weights_eps", 1e-6)))
        if rc != 0:
            raise RuntimeError("全流程校验未通过，请查看上方 FAIL 信息")

    return result


def _print_summary(r: PipelineResult) -> None:
    sep = "=" * 60
    print(sep)
    print("VIPER 决策树流水线 · 运行完成")
    print(sep)
    print(f"轨迹 CSV     : {r.trajectory_csv}")
    print(f"样本行数 N   : {r.n_samples}")
    print(f"输出目录     : {r.output_dir}")
    print(f"FQE loss     : {r.fqe_final_loss:.6f}")
    print(f"VIPER 准确率 : 全量={r.viper_last_acc_full:.4f}  重采样集={r.viper_last_acc_resampled:.4f}")
    print(f"规则条数     : {r.n_rules}")
    print(f"规则文件     : {r.rules_txt}")
    print(f"决策树 JSON  : {r.tree_json or '(未导出)'}")
    print(f"决策树节点表 : {r.tree_nodes_csv or '(未导出)'}")
    print(f"决策树 PDF   : {r.tree_pdf or '(未生成，需 pip install graphviz 且安装系统 Graphviz)'}")
    print(f"决策树 DOT   : {r.tree_dot or '(未导出)'}")
    print(f"决策树 PNG   : {r.tree_png or '(未生成，可加 --render-tree-png)'}")
    print(f"汇总 JSON    : {r.final_result_json}")
    print(sep)
    print("规则预览（前 10 条）:")
    for i, line in enumerate(r.rules_preview, 1):
        print(f"  {i}. {line}")
    if r.n_rules > len(r.rules_preview):
        print(f"  … 共 {r.n_rules} 条，完整内容见 rules.txt")
    print(sep)


def main() -> None:
    _setup_logging(bool(RUN_CONFIG.get("verbose", True)))
    try:
        result = run_full_pipeline(RUN_CONFIG)
        _print_summary(result)
    except Exception as exc:
        logger.exception("流水线失败")
        print(f"\n[失败] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
