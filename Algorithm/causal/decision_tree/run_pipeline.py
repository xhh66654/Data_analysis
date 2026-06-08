"""
VIPER 决策树离线全流程 —— 把大量轨迹提炼成可读的 IF-THEN 规则（主用途）。

本模块的定位是**规则展示 / 策略归纳**：
  · 读入轨迹 CSV
  · FQE → l_hat → weights：为每步打「决策重要性」
  · VIPER + CART：从（全表或指定一局）里抽出一条条决策规则 + 树图

默认 pipeline_mode="rules"：VIPER 用**全部行**建树，不 holdout 30% 做测试。
若需要论文式 train/val/test 指标，可设 pipeline_mode="eval"。

配置：修改下方 RUN_CONFIG（至少 trajectory_csv；只看某一局则设 only_episode）。

主要输出（{轨迹目录}/fqe_out/）：
  viper_out/rules.txt、policy_tree.pdf — 给人看的规则
  q_hat.pt、l_hat.csv、weights.csv     — 中间产物，供 step_explain 等复用

用法：
  1. 修改下方 RUN_CONFIG（至少填写 trajectory_csv）
  2. 在 Algorithm 目录执行：
       python causal/decision_tree/run_pipeline.py
     或：
       python -m causal.decision_tree.run_pipeline

入口：本文件 RUN_CONFIG + `python causal/decision_tree/run_pipeline.py`（或 `python -m causal.decision_tree.run_pipeline`）。
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

from causal.decision_tree.data_flow import DataFlowTracker, record_reward_norm, record_rule_ensemble
from causal.decision_tree.fqe import FQETrainConfig, load_q_hat, save_q_hat, train_q_hat
from causal.decision_tree.l_hat import compute_l_hat, l_hat_dataframe, save_l_hat_csv
from causal.decision_tree.rule_ensemble import (
    ensemble_rules_from_rounds,
    save_ensemble_rules,
    rules_to_if_then_strings,
)
from causal.decision_tree.trajectory_io import (
    ACTION_COL,
    EPISODE_COL,
    STATE_COLS,
    build_transitions,
    load_trajectory_csv,
    normalize_rewards,
    RewardNormConfig,
)
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
    "trajectory_csv": r"F:\cause_analysis\Algorithm\causal\trajectories\trajectory_LLdV3_S0_4.csv",
    # 全流程产物目录；留空 "" → 自动设为 {trajectory_csv 所在目录}/fqe_out/
    # 其中 viper_out/ 存放规则与决策树图
    "output_dir": "",
    # -------------------------------------------------------------------------
    # 阶段 1：FQE —— 训练 Q_hat 神经网络（影响最大，见 调参文档.md）
    # 输出：fqe_out/q_hat.pt；下游 l_hat / weights / VIPER 均依赖此 Q 估计
    # -------------------------------------------------------------------------
    # 训练轮数；loss 仍下降时可加大（大表可试 20～30）
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
    # 阶段 3：weights —— 抽样权重（VIPER 多轮共用）
    # 输出：fqe_out/weights.csv
    # -------------------------------------------------------------------------
    # weight_mode: uniform | advantage | margin（推荐 margin：按决策重要性 top1-top2 加权）
    "weight_mode": "margin",
    # 兼容旧配置；若设置则覆盖 weight_mode（True→margin，False→uniform）
    "viper_weighted_sampling": None,
    # 平滑项 ε；margin/advantage 模式生效
    "weights_eps": 1e-6,
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
    "viper_n_round": 20,
    # 在验证集上选最优轮（需 val_frac>0）
    "viper_pick_best_round": True,
    "cart_max_depth": 6,
    "cart_min_samples_leaf": 50,
    "cart_min_samples_split": 100,
    # 代价复杂度剪枝；>0 时自动剪枝（可试 1e-4～1e-3），0=仅用 max_depth
    "cart_ccp_alpha": 0.0,
    # 训练：不均衡时用 "balanced" 加权分裂；PDF/节点表展示真实类别分布（非加权计数）
    "cart_class_weight": "balanced",
    # 选轮指标：acc | macro_f1（不均衡数据推荐 macro_f1）
    "viper_selection_metric": "macro_f1",
    # oracle 重标注：y ← argmax_a Q(s,a)，消除探索噪声标签
    "oracle_relabel": True,
    # -------------------------------------------------------------------------
    # 规则提炼模式（主用途：大量数据 → 规则展示，不是泛化评估）
    # -------------------------------------------------------------------------
    # "rules"：全表参与 VIPER 建树（默认，符合「整库轨迹→规则」）
    # "eval" ：按 episode 划分 train/val/test，用于看泛化指标（旧实验用法）
    "pipeline_mode": "rules",
    # 只提炼某一局：填 episode 编号（如 3）；None=用下面 VIPER 所需的全部行（rules 模式下=全表）
    "only_episode": None,
    # 仅 pipeline_mode="eval" 时生效：
    "val_frac": 0.15,
    "test_frac": 0.15,
    "refit_final_tree_on_full_data": False,
    # 每轮 bootstrap 样本数；0=与 train 池同规模有放回抽样（不缩小 m，但可重复同一条）
    # 若设为 120000 等，则每轮 CART.fit 仅用该数量的有放回样本（显式裁到 12 万）
    "resample_size": 0,
    "weight_noise": 0.02,
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
    "ensemble_max_rules": 200,
    "ensemble_confidence_threshold": 0.0,
    "ensemble_min_support_rounds": 0,
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


def _resolve_pipeline_mode(cfg: dict) -> tuple[str, float, float, bool]:
    """返回 (mode, val_frac, test_frac, refit_on_full_data)。"""
    mode = str(cfg.get("pipeline_mode", "rules")).strip().lower()
    if mode == "rules":
        return mode, 0.0, 0.0, False
    if mode == "eval":
        return (
            mode,
            float(cfg.get("val_frac", 0.15)),
            float(cfg.get("test_frac", 0.15)),
            bool(cfg.get("refit_final_tree_on_full_data", True)),
        )
    raise ValueError(f"pipeline_mode 须为 rules 或 eval，得到 {mode!r}")


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

    mode, val_frac, test_frac, refit_on_full = _resolve_pipeline_mode(cfg)
    only_ep = cfg.get("only_episode")
    only_ep_int: int | None = int(only_ep) if only_ep is not None else None

    logger.info("读取轨迹: %s", csv_path)
    logger.info(
        "流水线模式: %s（%s） only_episode=%s",
        mode,
        "全表→规则展示，不 holdout" if mode == "rules" else "train/val/test 评估",
        only_ep_int if only_ep_int is not None else "全部",
    )
    df = load_trajectory_csv(str(csv_path))
    n = len(df)

    flow = DataFlowTracker()
    flow.set_origin(n, source=str(csv_path))

    # --- [DATA-CROP-01] 奖励归一化：裁剪 reward 数值，不删行 ---
    if bool(cfg.get("enable_reward_norm", False)):
        reward_norm_cfg = RewardNormConfig(
            clip_range=(
                float(cfg.get("reward_clip_min", -10.0)),
                float(cfg.get("reward_clip_max", 10.0)),
            ),
            standardize=True,
            per_episode=bool(cfg.get("reward_norm_per_episode", False)),
        )
        df, rstats = normalize_rewards(df, reward_norm_cfg)
        record_reward_norm(
            flow,
            n,
            clipped_count=int(rstats.get("clipped_count", 0)),
            clip_range=tuple(rstats.get("clip_range", (-10.0, 10.0))),
        )
    else:
        flow.record(
            "01",
            "跳过奖励归一化",
            n_in=n,
            n_out=n,
            reduces_rows=False,
            module="run_pipeline",
            note="enable_reward_norm=False",
        )

    # --- [DATA-CROP-02~05] FQE / l_hat / weights：全程保持全表 n 行 ---
    flow.record(
        "02",
        "FQE 训练 Q_hat",
        n_in=n,
        n_out=n,
        reduces_rows=False,
        module="fqe.train_q_hat",
        note="全表转移样本；不删行",
    )

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
    lh_df = l_hat_dataframe(df, lh)
    save_l_hat_csv(l_hat_path, lh_df)
    flow.record(
        "03",
        "计算 l_hat / a_star",
        n_in=n,
        n_out=n,
        reduces_rows=False,
        module="l_hat",
        note="逐行推理；行数与 CSV 一致",
    )

    if bool(cfg.get("oracle_relabel", True)) and "a_star" in lh_df.columns:
        n_star = int(lh_df["a_star"].nunique())
        if n_star < 2:
            dominant = int(lh_df["a_star"].mode().iloc[0])
            logger.warning(
                "oracle 标签仅 %d 类（全部为 action=%d）。"
                "小样本下 Q 网络易塌缩为常数策略 → 决策树仅 1 叶、指标虚高 1.0、规则极少。"
                "建议：增大轨迹（更多 episode）、或设 oracle_relabel=False 用行为动作 action 作标签。",
                n_star,
                dominant,
            )

    # --- 阶段 3: weights ---
    weight_mode = str(cfg.get("weight_mode", "margin"))
    legacy_ws = cfg.get("viper_weighted_sampling")
    ws_kw: dict = {}
    if legacy_ws is not None:
        ws_kw["weighted_sampling"] = bool(legacy_ws)
    logger.info("阶段 3/4: 计算 weights（mode=%s）", weight_mode)
    run_weights_from_l_hat_csv(
        l_hat_path,
        output_path=weights_path,
        eps=float(cfg.get("weights_eps", 1e-6)),
        weight_mode=weight_mode,
        **ws_kw,
    )
    flow.record(
        "04",
        "计算 VIPER 抽样权重",
        n_in=n,
        n_out=n,
        reduces_rows=False,
        module="weights",
        note=f"weight_mode={weight_mode}；weights.csv 行数=全表",
    )
    n_viper = n
    if only_ep_int is not None:
        n_viper = int((pd.to_numeric(df[EPISODE_COL], errors="coerce") == only_ep_int).sum())
        if n_viper == 0:
            raise ValueError(f"only_episode={only_ep_int} 在 CSV 中无数据行")
        flow.record(
            "05a",
            f"仅第 {only_ep_int} 局用于规则/树",
            n_in=n,
            n_out=n_viper,
            reduces_rows=True,
            module="run_pipeline",
            note="FQE/l_hat/weights 仍用全表；VIPER 只提炼该局 IF-THEN 规则",
        )
    flow.record(
        "05",
        "VIPER 规则提炼输入",
        n_in=n_viper if only_ep_int is not None else n,
        n_out=n_viper,
        reduces_rows=only_ep_int is not None,
        module="viper_cart.run_viper_from_files",
        note=(
            "rules 模式: 不划分 val/test，全表建树"
            if mode == "rules"
            else f"eval 模式: val_frac={val_frac} test_frac={test_frac}"
        ),
    )
    sampling_label = weight_mode

    # --- 阶段 4～6: VIPER + CART + 规则 ---
    m = int(cfg.get("resample_size", 0))
    n_round = int(cfg.get("viper_n_round", 8))
    weight_noise = float(cfg.get("weight_noise", 0.02))
    logger.info("阶段 4/4: VIPER (RUN_CONFIG cart_*) → viper_out/")
    cw = cfg.get("cart_class_weight")
    class_weight = None if cw in (None, "", "none", False) else cw
    viper_cfg = ViperConfig(
        n_round=n_round,
        max_depth=int(cfg.get("cart_max_depth", 5)),
        min_samples_leaf=int(cfg.get("cart_min_samples_leaf", 1)),
        min_samples_split=int(cfg.get("cart_min_samples_split", 2)),
        random_state=seed,
        resample_size=m if m > 0 else None,
        weight_noise_std=weight_noise,
        weighted_sampling=weight_mode != "uniform",
        pick_best_by_full_acc=bool(cfg.get("viper_pick_best_round", True)),
        class_weight=class_weight,
        ccp_alpha=float(cfg.get("cart_ccp_alpha", 0.0)),
        selection_metric=str(cfg.get("viper_selection_metric", "macro_f1")),
        export_tree=bool(cfg.get("export_tree", True)),
        render_tree_pdf=bool(cfg.get("render_tree_pdf", True)),
        render_tree_png=bool(cfg.get("render_tree_png", False)),
        tree_image_dpi=int(cfg.get("tree_image_dpi", 150)),
        open_tree_pdf=bool(cfg.get("open_tree_pdf", False) or cfg.get("show_tree_image", False)),
        show_tree_image=bool(cfg.get("show_tree_image", False)),
    )
    viper_result = run_viper_from_files(
        csv_path,
        weights_path,
        viper_out_dir,
        viper_cfg,
        val_frac=val_frac,
        test_frac=test_frac,
        oracle_relabel=bool(cfg.get("oracle_relabel", True)),
        l_hat_path=l_hat_path,
        data_flow=flow,
        refit_on_full_data=refit_on_full,
        only_episode=only_ep_int,
        pipeline_mode=mode,
    )

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
        ensemble_if_then = rules_to_if_then_strings(ensemble_rules)
        if ensemble_if_then:
            final_rules = ensemble_if_then
        else:
            logger.warning(
                "规则集成为空，保留 VIPER 选用树的 %d 条规则（勿将 n_rules 记为 0）",
                len(viper_result.rules),
            )
            final_rules = viper_result.rules.copy()

        # 保存集成规则
        ensemble_rules_path = viper_out_dir / "ensemble_rules"
        save_ensemble_rules(ensemble_rules, ensemble_rules_path)

        n_rules_raw = sum(len(r["rules"]) for r in rules_per_round)
        record_rule_ensemble(
            flow,
            n_rules_raw,
            len(final_rules),
            max_rules=int(cfg.get("ensemble_max_rules", 100)),
        )
        logger.info(
            "规则集成完成: 原始规则数=%d 集成后规则数=%d",
            n_rules_raw,
            len(final_rules),
        )

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
    
    flow.log_table()
    data_flow_path = flow.save(out_dir / "data_flow_report.json")

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
        n_rules=len(final_rules) if final_rules else len(viper_result.rules),
        rules_preview=preview,
        tree_json=str(tree_json) if tree_json.is_file() else "",
        tree_nodes_csv=str(tree_nodes_csv) if tree_nodes_csv.is_file() else "",
        tree_dot=str(tree_dot) if tree_dot.is_file() else "",
        tree_pdf=str(tree_pdf) if tree_pdf.is_file() else "",
        tree_png=str(tree_png) if tree_png.is_file() else "",
    )

    test_metrics = (getattr(viper_result, "metrics", None) or {}).get("test", {})
    payload = {
        "success": True,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "config": {k: v for k, v in cfg.items() if k != "verbose"},
        "data_flow_report": str(data_flow_path),
        "data_flow_summary_zh": flow.build_report()["summary_zh"],
        **asdict(result),
        "viper_selected_round": sel_round,
        "eval_val_metric": float(sel_full),
        "eval_selection_metric": str(cfg.get("viper_selection_metric", "macro_f1")),
        "test_accuracy": test_metrics.get("accuracy"),
        "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
        "test_macro_f1": test_metrics.get("macro_f1"),
        "metrics_json": str(viper_out_dir / "metrics.json"),
        "rules": viper_result.rules,
    }
    final_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("已写入汇总结果: %s", final_json_path)
    return result


def _print_summary(r: PipelineResult) -> None:
    sep = "=" * 60
    print(sep)
    print("VIPER 决策树流水线 · 运行完成")
    print(sep)
    print(f"轨迹 CSV     : {r.trajectory_csv}")
    print(f"样本行数 N   : {r.n_samples}  （CSV 全量，FQE/l_hat/weights 均用此规模）")
    try:
        dfr = json.loads(
            (Path(r.output_dir) / "data_flow_report.json").read_text(encoding="utf-8")
        )
        print(f"数据流转     : {dfr.get('summary_zh', '')}")
        for st in dfr.get("stages") or []:
            if st.get("reduces_rows"):
                print(
                    f"  [裁剪] {st['id']} {st['name_zh']}: "
                    f"{st['n_in']} → {st['n_out']} ({100*st['crop_ratio_from_origin']:.1f}% 全量)"
                )
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    print(f"输出目录     : {r.output_dir}")
    print(f"FQE loss     : {r.fqe_final_loss:.6f}")
    print(
        f"VIPER 指标   : 验证集选轮={r.viper_last_acc_full:.4f}  "
        f"重采样集={r.viper_last_acc_resampled:.4f}"
    )
    metrics_path = Path(r.output_dir) / "viper_out" / "metrics.json"
    if metrics_path.is_file():
        try:
            te = json.loads(metrics_path.read_text(encoding="utf-8")).get("test", {})
            if te.get("n", 0) > 0:
                print(
                    f"测试集       : acc={te['accuracy']:.4f}  "
                    f"balanced_acc={te['balanced_accuracy']:.4f}  "
                    f"macro_f1={te['macro_f1']:.4f}  (n={te['n']})"
                )
        except (json.JSONDecodeError, OSError, KeyError):
            pass
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
