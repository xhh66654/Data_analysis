#!/usr/bin/env python3
"""
run_explain — 单机单步决策解释主入口。

用法（推荐）::

  1. 修改下方 RUN_CONFIG（轨迹 CSV、要解释的局/步、FQE 训练参数）
  2. 在 Algorithm 目录执行：
       python causal/step_explain/run_explain.py

  默认会**根据 trajectory_csv 现场训练 FQE**，再解释指定一步；
  不会读取 decision_tree 已有的 fqe_out/q_hat.pt（除非显式开启 use_pretrained_q_hat）。

Python API::

    from causal.step_explain import run_from_config, RUN_CONFIG
    result = run_from_config(RUN_CONFIG)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# 保证从任意工作目录运行都能 import causal.*
_ALGORITHM_ROOT = Path(__file__).resolve().parents[2]
if str(_ALGORITHM_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALGORITHM_ROOT))

from causal.decision_tree.fqe import load_q_hat
from causal.step_explain.action_explain import DEFAULT_ACTION_LABELS, compare_actions
from causal.step_explain.fqe_train import train_fqe_for_explain
from causal.step_explain.dim_reduce import load_block_map, compute_block_importance
from causal.step_explain.effect_validation import format_validation_console, validate_explanation
from causal.step_explain.model_quality_report import (
    build_model_training_report,
    format_model_training_console,
)
from causal.step_explain.narrative import (
    build_explain_json,
    build_narrative,
    build_narrative_technical,
)
from causal.step_explain.trajectory_loader import load_csv, locate_row

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_DEFAULT_BLOCK_MAP = _HERE / "state_block_map.yaml"

# =============================================================================
# 在此填写你的数据与参数（运行前只需改这一块，与 decision_tree RUN_CONFIG 相同用法）
# =============================================================================
RUN_CONFIG: Dict[str, Any] = {
    # -------------------------------------------------------------------------
    # 输入路径
    # -------------------------------------------------------------------------
    # 【必填】离线轨迹 CSV（与 decision_tree 相同格式，一步一行）
    # 必需列：episode, action, reward, s_0…s_7；建议含 global_step, dw, truncated, s_next_*
    "trajectory_csv": str(_ALGORITHM_ROOT / "causal" / "trajectories" / "trajectory_LLdV3_S0_2.csv"),
    # -------------------------------------------------------------------------
    # FQE：默认现场训练（不加载已有 q_hat.pt）
    # -------------------------------------------------------------------------
    # False=每次运行都用 trajectory_csv 重新训练（推荐）
    # True=跳过训练，直接加载 q_hat_path（仅调试或复现时用）
    "use_pretrained_q_hat": False,
    # 仅 use_pretrained_q_hat=True 时生效
    "q_hat_path": "",
    # 训练轮数；loss 仍下降时可加大
    "fqe_epochs": 5,
    "fqe_device": "cuda",
    "fqe_target": "sarsa",
    "fqe_gamma": 0.99,
    "fqe_lr": 1e-3,
    "fqe_batch_size": 1024,
    "fqe_hidden": 256,
    "fqe_use_target_network": True,
    "fqe_target_tau": 0.005,
    "enable_reward_norm": True,
    "reward_clip_min": -10.0,
    "reward_clip_max": 10.0,
    "reward_norm_per_episode": False,
    "seed": 45,
    # -------------------------------------------------------------------------
    # 要解释的单步（三选一；row_index 优先于 episode+global_step）
    # -------------------------------------------------------------------------
    # 局号（与 CSV 列 episode 一致）
    "episode": 3,
    # 全局步号（与 CSV 列 global_step 一致）；无该列时按局内第 N 行
    "global_step": 294,
    # 直接指定 DataFrame 行号（>0 时覆盖 episode / global_step）；None=不用
    "row_index": None,
    # -------------------------------------------------------------------------
    # 输出
    # -------------------------------------------------------------------------
    # 解释结果目录；留空 "" → causal/step_explain/output/
    "output_dir": "",
    # 状态语义分块 YAML；留空 "" → 本目录 state_block_map.yaml
    "block_map_path": "",
    # -------------------------------------------------------------------------
    # 归因与叙述
    # -------------------------------------------------------------------------
    # 反事实基线：zero=块内置 0 | mean=块内替换为训练集该维均值
    "baseline": "zero",
    # 写入中文段落的最重要状态块数量
    "top_k_blocks": 2,
    # 叙述中列出的备选动作数量（不含当前动作）
    "top_k_alternatives": 2,
    # 解释阶段推理设备（可与 fqe_device 不同；无 GPU 时改为 cpu）
    "device": "cuda",
    # 叙述风格："plain"=通俗（默认，不出现 Q 值）| "technical"=含模型数值
    "narrative_style": "plain",
    # 是否运行效果验证并写出 validation_report.json
    "run_validation": True,
    # 验证阈值：归因 |Δ| 低于此认为「因素过弱」；动作 Q 极差低于此认为「区分度不足」
    "validation_min_block_delta": 0.05,
    "validation_min_action_spread": 0.1,
    # 是否打印 INFO 日志
    "verbose": True,
}


@dataclass
class ExplainQuery:
    """指定要解释的单步。三种方式任选其一。"""
    episode: Optional[int] = None
    global_step: Optional[int] = None
    row_index: Optional[int] = None


def _setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _resolve_output_dir(cfg: Dict[str, Any]) -> Path:
    raw = (cfg.get("output_dir") or "").strip()
    if raw:
        return Path(raw).resolve()
    return (_HERE / "output").resolve()


def _resolve_block_map(cfg: Dict[str, Any]) -> Path:
    raw = (cfg.get("block_map_path") or "").strip()
    if raw:
        return Path(raw).resolve()
    return _DEFAULT_BLOCK_MAP


def _prepare_q_hat(cfg: Dict[str, Any], traj: Path, out_dir: Path) -> Path:
    """现场训练 FQE，或（仅当配置允许时）加载已有 q_hat.pt。"""
    if bool(cfg.get("use_pretrained_q_hat")):
        raw = (cfg.get("q_hat_path") or "").strip()
        if not raw:
            raise ValueError("use_pretrained_q_hat=True 时必须设置 q_hat_path")
        q_path = Path(raw).resolve()
        if not q_path.is_file():
            raise FileNotFoundError(f"未找到预训练 Q 网络: {q_path}")
        logger.info("跳过 FQE 训练，加载已有模型: %s", q_path)
        return q_path

    q_path, _ = train_fqe_for_explain(traj, out_dir, cfg)
    return q_path


def _query_from_config(cfg: Dict[str, Any]) -> ExplainQuery:
    row = cfg.get("row_index")
    if row is not None and int(row) >= 0:
        return ExplainQuery(row_index=int(row))
    ep = cfg.get("episode")
    if ep is None:
        raise ValueError("RUN_CONFIG 须指定 episode，或设置 row_index")
    gs = cfg.get("global_step")
    return ExplainQuery(
        episode=int(ep),
        global_step=int(gs) if gs is not None else None,
    )


def run_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """按 RUN_CONFIG 字典执行单步解释。"""
    traj = Path((cfg.get("trajectory_csv") or "").strip()).resolve()
    if not traj.is_file():
        raise FileNotFoundError(f"轨迹 CSV 不存在: {traj}")

    out_dir = _resolve_output_dir(cfg)
    q_hat = _prepare_q_hat(cfg, traj, out_dir)
    block_map = _resolve_block_map(cfg)

    return run_explain(
        csv_path=traj,
        q_hat_path=q_hat,
        query=_query_from_config(cfg),
        out_dir=out_dir,
        block_map_path=block_map,
        baseline=str(cfg.get("baseline") or "zero"),
        top_k_blocks=int(cfg.get("top_k_blocks") or 2),
        top_k_alternatives=int(cfg.get("top_k_alternatives") or 2),
        device=str(cfg.get("device") or "cpu"),
        narrative_style=str(cfg.get("narrative_style") or "plain"),
        run_validation=bool(cfg.get("run_validation", True)),
        validation_min_block_delta=float(cfg.get("validation_min_block_delta") or 0.05),
        validation_min_action_spread=float(cfg.get("validation_min_action_spread") or 0.1),
    )


def run_explain(
    csv_path: str | Path,
    q_hat_path: str | Path,
    query: ExplainQuery,
    out_dir: str | Path = "step_explain_output",
    block_map_path: Optional[str | Path] = None,
    baseline: str = "zero",
    top_k_blocks: int = 2,
    top_k_alternatives: int = 2,
    device: str = "cpu",
    narrative_style: str = "plain",
    run_validation: bool = True,
    validation_min_block_delta: float = 0.05,
    validation_min_action_spread: float = 0.1,
) -> Dict[str, Any]:
    """
    主流程：加载数据 → 定位步 → 归因 → 生成解释。

    返回 dict，含 narrative_zh（中文段落）及完整 explain.json 字段。
    同时将 explain.json 写出到 out_dir/explain.json。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(csv_path)
    step = locate_row(
        df,
        episode=query.episode,
        global_step=query.global_step,
        row_index=query.row_index,
    )
    logger.info(
        "定位步：episode=%d global_step=%s row=%d action=%d reward=%.4f",
        step.episode, step.global_step, step.row_index, step.action, step.reward,
    )

    q_net, meta = load_q_hat(q_hat_path, device=device)
    model_training_report = build_model_training_report(
        q_hat_path,
        meta,
        explain_trajectory_csv=csv_path,
    )
    logger.info("Q 网络已加载；FQE final_loss=%s", meta.get("final_loss"))

    bmap_path = Path(block_map_path) if block_map_path else _DEFAULT_BLOCK_MAP
    blocks = load_block_map(bmap_path)

    state_means: Optional[np.ndarray] = None
    if baseline == "mean":
        state_cols = [f"s_{i}" for i in range(step.state.shape[0])]
        avail = [c for c in state_cols if c in df.columns]
        if len(avail) == len(state_cols):
            state_means = df[avail].mean().values.astype(np.float32)
            logger.info("已计算训练集状态均值（用于 mean baseline）")
        else:
            logger.warning("无法计算状态均值，回退到 zero baseline")
            baseline = "zero"

    action_cmp = compare_actions(
        q_net=q_net,
        state=step.state,
        chosen_action=step.action,
        action_labels=DEFAULT_ACTION_LABELS,
        top_k_alternatives=top_k_alternatives,
        device=device,
    )

    block_imps = compute_block_importance(
        q_net=q_net,
        state=step.state,
        action=step.action,
        blocks=blocks,
        baseline=baseline,
        state_means=state_means,
        device=device,
    )

    narrative_plain = build_narrative(
        episode=step.episode,
        step=step.global_step,
        reward=step.reward,
        action_cmp=action_cmp,
        block_importances=block_imps,
        top_k_blocks=top_k_blocks,
        style="plain",
    )
    narrative_technical = build_narrative_technical(
        episode=step.episode,
        step=step.global_step,
        reward=step.reward,
        action_cmp=action_cmp,
        block_importances=block_imps,
        top_k_blocks=top_k_blocks,
    )
    narrative_main = (
        narrative_plain if narrative_style == "plain" else narrative_technical
    )

    validation_report: Optional[Dict[str, Any]] = None
    if run_validation:
        validation_report = validate_explanation(
            action_cmp=action_cmp,
            block_importances=block_imps,
            reward=step.reward,
            q_meta=meta,
            min_block_delta=validation_min_block_delta,
            min_action_spread=validation_min_action_spread,
        )

    result = build_explain_json(
        episode=step.episode,
        step=step.global_step,
        row_index=step.row_index,
        reward=step.reward,
        action_cmp=action_cmp,
        block_importances=block_imps,
        top_k_blocks=top_k_blocks,
        narrative_zh=narrative_main,
        narrative_technical=narrative_technical,
        validation_report=validation_report,
        model_training_report=model_training_report,
    )
    out_path = out_dir / "explain.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("explain.json 已写出: %s", out_path.resolve())

    if validation_report is not None:
        val_path = out_dir / "validation_report.json"
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(validation_report, f, ensure_ascii=False, indent=2)
        logger.info("validation_report.json 已写出: %s", val_path.resolve())

    mt_path = out_dir / "model_training_report.json"
    with open(mt_path, "w", encoding="utf-8") as f:
        json.dump(model_training_report, f, ensure_ascii=False, indent=2)
    logger.info("model_training_report.json 已写出: %s", mt_path.resolve())

    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """命令行参数；默认值来自 RUN_CONFIG，用于覆盖配置中的单项。"""
    c = RUN_CONFIG
    p = argparse.ArgumentParser(
        description="单步决策解释：指定轨迹中的一步，输出中文归因段落",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="未传任何参数时，完全使用本文件顶部 RUN_CONFIG。",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=str(c.get("trajectory_csv") or ""),
        help="轨迹 CSV 路径",
    )
    p.add_argument(
        "--use-pretrained",
        action="store_true",
        default=bool(c.get("use_pretrained_q_hat")),
        help="不训练 FQE，改用 --q-hat 指定已有模型",
    )
    p.add_argument(
        "--q-hat",
        type=str,
        dest="q_hat",
        default=str(c.get("q_hat_path") or ""),
        help="仅 --use-pretrained 时：已有 q_hat.pt 路径",
    )
    p.add_argument(
        "--fqe-epochs",
        type=int,
        default=int(c.get("fqe_epochs") or 5),
        dest="fqe_epochs",
        help="现场训练 FQE 的 epoch 数",
    )
    p.add_argument("--episode", type=int, default=c.get("episode"), help="局号")
    p.add_argument("--step", type=int, default=c.get("global_step"), help="global_step")
    p.add_argument(
        "--row",
        type=int,
        default=c.get("row_index"),
        help="DataFrame 行号（指定后覆盖 episode/step）",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(c.get("output_dir") or ""),
        help="输出目录；留空则用 causal/step_explain/output/",
    )
    p.add_argument(
        "--block-map",
        type=str,
        default=str(c.get("block_map_path") or ""),
        dest="block_map",
        help="state_block_map.yaml 路径",
    )
    p.add_argument(
        "--baseline",
        choices=("zero", "mean"),
        default=str(c.get("baseline") or "zero"),
        help="反事实基线",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=int(c.get("top_k_blocks") or 2),
        dest="top_k",
        help="写入叙述的状态块数",
    )
    p.add_argument(
        "--top-k-alt",
        type=int,
        default=int(c.get("top_k_alternatives") or 2),
        dest="top_k_alt",
        help="叙述中备选动作数量",
    )
    p.add_argument(
        "--device",
        type=str,
        default=str(c.get("device") or "cpu"),
        help="cpu | cuda",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="减少日志输出",
    )
    return p.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(RUN_CONFIG)
    cfg["trajectory_csv"] = args.csv
    cfg["use_pretrained_q_hat"] = args.use_pretrained
    cfg["q_hat_path"] = args.q_hat
    cfg["fqe_epochs"] = args.fqe_epochs
    cfg["episode"] = args.episode
    cfg["global_step"] = args.step
    cfg["row_index"] = args.row
    cfg["output_dir"] = args.out
    cfg["block_map_path"] = args.block_map
    cfg["baseline"] = args.baseline
    cfg["top_k_blocks"] = args.top_k
    cfg["top_k_alternatives"] = args.top_k_alt
    cfg["device"] = args.device
    if args.quiet:
        cfg["verbose"] = False
    return cfg


def _print_summary(result: Dict[str, Any], out_dir: Path) -> None:
    sep = "=" * 60
    print()
    print(sep)
    print("单步决策解释 · 运行完成")
    print(sep)

    mt = result.get("model_training")
    if mt:
        print(format_model_training_console(mt))
        print(sep)

    q = result.get("query") or {}
    print(f"局 / 步      : episode={q.get('episode')}  global_step={q.get('global_step')}  row={q.get('row_index')}")
    ch = result.get("chosen_action") or {}
    plain = ch.get("plain_label") or ch.get("label")
    print(f"执行动作     : {plain} (id={ch.get('id')})  模型排名={ch.get('rank')}")
    print(f"是否模型最优 : {result.get('is_optimal')}")
    print(f"即时奖励     : {result.get('reward')}")
    blocks = result.get("block_importances") or []
    if blocks:
        print("关键因素     :", ", ".join(
            f"{b['block_name']}(影响强度={b['abs_delta']:.3f})" for b in blocks
        ))

    val = result.get("validation")
    if val:
        print("【本步解释效果验证】（针对当前 episode/step，不是全局训练指标）")
        print(format_validation_console(val))

    print(sep)
    print(f"q_hat.pt（本次训练）       : {out_dir / 'q_hat.pt'}")
    print(f"explain.json               : {out_dir / 'explain.json'}")
    print(f"model_training_report.json : {out_dir / 'model_training_report.json'}")
    print(f"validation_report.json     : {out_dir / 'validation_report.json'}")
    print(sep)
    print("【决策解释（通俗版）】")
    print(result.get("narrative_zh", ""))
    print(sep)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if argv:
        args = parse_args(argv)
        cfg = _config_from_args(args)
    else:
        cfg = dict(RUN_CONFIG)

    _setup_logging(bool(cfg.get("verbose", True)))

    try:
        out_dir = _resolve_output_dir(cfg)
        result = run_from_config(cfg)
        _print_summary(result, out_dir)
        return 0
    except Exception as exc:
        logger.exception("单步解释失败")
        print(f"\n[失败] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
