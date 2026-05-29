"""
流水线各阶段产物衔接校验与端到端一致性检查。

用于排查「FQE 输出能否正确喂给 l_hat」「l_hat 与 weights 公式是否一致」
「VIPER 规则文件是否齐全」等问题。支持单阶段校验与 --e2e 全流程复算对比。

主要函数：
  verify_toy_weights()   — 用文档小例子验证 l_hat→weights 公式
  verify_weights_csv()   — 检查 weights.csv 中 w_raw、weights 是否与公式一致
  verify()               — 对比 checkpoint 重算 l_hat 与磁盘 l_hat.csv
  verify_e2e()           — 检查 output_dir 下全部产物并复算关键数值
  verify_viper_outputs() — 检查 viper_out/ 下 rules、tree.json、summary 等

用法：
  python -m causal.decision_tree.verify_phase_link --e2e --csv ... --out-dir .../fqe_out
  python -m causal.decision_tree.verify_phase_link --csv ... --checkpoint ...

run_pipeline.py 在 run_verify=True 时会自动调用 verify_e2e。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .fqe import load_q_hat
from .l_hat import compute_l_hat, l_hat_dataframe
from .trajectory_io import ACTION_COL, EPISODE_COL, STATE_COLS, build_transitions, load_trajectory_csv
from .weights import DEFAULT_EPS, compute_uniform_weights, compute_weights, compute_mixed_weights, load_l_hat_csv


def verify_toy_weights(eps: float = DEFAULT_EPS) -> list[str]:
    """文档小例子：q_all=[1,2,1.5,0.5] → V=2；a=1→l_hat=0；a=3→l_hat=1.5。"""
    errors: list[str] = []
    v, q1, q3 = 2.0, 2.0, 0.5
    l_opt = v - q1
    l_sub = v - q3
    if not np.isclose(l_opt, 0.0):
        errors.append(f"最优动作 l_hat 应为 0，得到 {l_opt}")
    if not np.isclose(l_sub, 1.5):
        errors.append(f"次优动作 l_hat 应为 1.5，得到 {l_sub}")

    w_opt = compute_weights(np.array([l_opt]), eps=eps)
    if not np.isclose(w_opt.w_raw[0], eps):
        errors.append(f"a=1 时 w_raw 应为 eps，得到 {w_opt.w_raw[0]}")
    w_sub = compute_weights(np.array([l_sub]), eps=eps)
    if not np.isclose(w_sub.w_raw[0], 1.5 + eps):
        errors.append(f"a=3 时 w_raw 应为 1.5+eps，得到 {w_sub.w_raw[0]}")

    w_mix = compute_weights(np.array([l_opt, l_sub, l_sub]), eps=eps)
    raw_expect = np.array([eps, 1.5 + eps, 1.5 + eps])
    if not np.allclose(w_mix.w_raw, raw_expect):
        errors.append("混合 w_raw 与公式不一致")
    if not np.isclose(w_mix.weights.sum(), 1.0):
        errors.append("weights 之和不为 1")
    if (w_mix.weights < 0).any():
        errors.append("weights 含负值")
    return errors


def verify_weights_csv(
    weights_path: Path,
    eps: float = DEFAULT_EPS,
    *,
    weighted_sampling: bool = True,
    mixed_alpha: float = 0.7,
) -> list[str]:
    errors: list[str] = []
    df = pd.read_csv(weights_path, encoding="utf-8-sig")
    for col in ("l_hat", "w_raw", "weights"):
        if col not in df.columns:
            errors.append(f"weights.csv 缺少列 {col}")
            return errors
    l_hat = df["l_hat"].values.astype(np.float64)
    w_raw = df["w_raw"].values.astype(np.float64)
    weights = df["weights"].values.astype(np.float64)
    if weighted_sampling:
        expect = compute_mixed_weights(l_hat, alpha=mixed_alpha, eps=eps)
    else:
        expect = compute_uniform_weights(l_hat)
    if not np.allclose(w_raw, expect.w_raw, rtol=1e-5, atol=1e-8):
        errors.append("w_raw 与 mixed_weights 不一致")
    if not np.allclose(weights, expect.weights, rtol=1e-5, atol=1e-8):
        errors.append("weights 与 mixed_weights 不一致")
    if not np.isclose(weights.sum(), 1.0, rtol=1e-6, atol=1e-8):
        errors.append(f"weights 之和={weights.sum()} != 1")
    if (weights < 0).any():
        errors.append("weights 含负值")
    l_hat_path = weights_path.parent / "l_hat.csv"
    if l_hat_path.is_file() and len(df) != len(load_l_hat_csv(l_hat_path)):
        errors.append("weights.csv 行数与 l_hat.csv 不一致")
    return errors


def verify(csv_path: Path, ckpt_path: Path, device: str = "cpu") -> int:
    errors: list[str] = []
    warnings: list[str] = []

    df = load_trajectory_csv(str(csv_path))
    n = len(df)
    trans = build_transitions(df)
    states_df = df[STATE_COLS].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    actions = pd.to_numeric(df[ACTION_COL], errors="coerce").values.astype(np.int64)

    # 1) 训练用状态与 l_hat 用状态一致
    if not np.allclose(trans.s, states_df, rtol=0, atol=0):
        errors.append("build_transitions 的 s 与 l_hat 读取的 STATE_COLS 不一致")

    q_net, meta = load_q_hat(ckpt_path, device=device)
    state_dim = int(q_net.net[0].in_features)
    n_actions = int(q_net.net[-1].out_features)

    if state_dim != trans.s.shape[1]:
        errors.append(f"checkpoint state_dim={state_dim} != 数据 {trans.s.shape[1]}")
    if meta.get("n_actions") and int(meta["n_actions"]) != n_actions:
        errors.append(f"meta n_actions={meta['n_actions']} != 网络输出 {n_actions}")
    amax = int(actions.max())
    if amax >= n_actions:
        errors.append(f"动作最大值 {amax} >= n_actions={n_actions}")

    meta_csv = meta.get("csv")
    if meta_csv and Path(meta_csv).resolve() != csv_path.resolve():
        warnings.append(f"checkpoint 训练 CSV 与当前 CSV 不同:\n  meta: {meta_csv}\n  now:  {csv_path}")

    lh = compute_l_hat(q_net, states_df, actions, device=device, batch_size=512)
    if lh.n != n:
        errors.append(f"l_hat 行数 {lh.n} != CSV 行数 {n}")

    out = l_hat_dataframe(df, lh)

    # 2) Q_sa == Q_hat_a{action}
    for i in range(min(n, 5000)):
        a = int(actions[i])
        q_col = f"Q_hat_a{a}"
        if abs(out.loc[i, "Q_sa"] - out.loc[i, q_col]) > 1e-4:
            errors.append(f"行 {i}: Q_sa 与 {q_col} 不一致")
            break

    # 3) l_hat = V_hat - Q_sa, V_hat = max Q
    q_cols = [c for c in out.columns if c.startswith("Q_hat_a")]
    q_mat = out[q_cols].values
    v_from_max = q_mat.max(axis=1)
    if not np.allclose(out["V_hat"].values, v_from_max, rtol=1e-5, atol=1e-4):
        errors.append("V_hat 不等于各 Q_hat_a* 的最大值")
    l_recalc = out["V_hat"].values - out["Q_sa"].values
    if not np.allclose(out["l_hat"].values, l_recalc, rtol=1e-5, atol=1e-4):
        errors.append("l_hat != V_hat - Q_sa")

    # 4) l_hat >= 0（数值容差）
    if (out["l_hat"].values < -1e-4).any():
        errors.append("存在 l_hat < 0（违反 max-Q 结构）")

    # 5) episode/action 与源 CSV 对齐
    if not np.array_equal(out[EPISODE_COL].values, df[EPISODE_COL].values):
        errors.append("l_hat.csv 的 episode 与源 CSV 不对齐")
    if not np.array_equal(out[ACTION_COL].values, df[ACTION_COL].values):
        errors.append("l_hat.csv 的 action 与源 CSV 不对齐")

    print(f"CSV 行数: {n}")
    print(f"checkpoint: {ckpt_path}")
    print(f"state_dim={state_dim} n_actions={n_actions}")
    if meta:
        print(f"meta: csv={meta.get('csv')} target={meta.get('target')} final_loss={meta.get('final_loss')}")
    print(
        f"l_hat: mean={out['l_hat'].mean():.6f} std={out['l_hat'].std():.6f} "
        f"min={out['l_hat'].min():.6f} max={out['l_hat'].max():.6f}"
    )
    for w in warnings:
        print(f"WARN: {w}")
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    print("OK: 阶段1→阶段2 衔接检查通过")
    return 0


def verify_viper_outputs(
    csv_path: Path,
    weights_path: Path,
    viper_dir: Path,
) -> list[str]:
    errors: list[str] = []
    df = load_trajectory_csv(str(csv_path))
    n = len(df)

    rules_txt = viper_dir / "rules.txt"
    rules_json = viper_dir / "rules.json"
    summary_json = viper_dir / "viper_summary.json"
    tree_json = viper_dir / "tree.json"
    tree_nodes_csv = viper_dir / "tree_nodes.csv"
    tree_dot = viper_dir / "policy_tree_debug.dot"
    if not tree_dot.is_file():
        tree_dot = viper_dir / "policy_tree.dot"
    for p in (rules_txt, rules_json, summary_json, tree_json, tree_nodes_csv, tree_dot):
        if not p.is_file():
            errors.append(f"缺少 VIPER 输出: {p}")

    if errors:
        return errors

    w_df = pd.read_csv(weights_path, encoding="utf-8-sig")
    if len(w_df) != n:
        errors.append(f"weights 行数 {len(w_df)} != 轨迹 CSV {n}")

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    rounds = summary.get("rounds") or []
    if not rounds:
        errors.append("viper_summary.json 无 rounds 记录")

    rules_lines = [ln.strip() for ln in rules_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rules_from_json = json.loads(rules_json.read_text(encoding="utf-8"))
    if len(rules_lines) != len(rules_from_json):
        errors.append("rules.txt 与 rules.json 条数不一致")
    if summary.get("n_rules") != len(rules_lines):
        errors.append("summary.n_rules 与 rules 条数不一致")
    for i, line in enumerate(rules_lines[:5]):
        if not line.startswith("IF ") or " THEN " not in line:
            errors.append(f"规则格式异常 (行 {i+1}): {line[:80]}")
            break

    if rounds:
        last = rounds[-1]
        for key in ("train_accuracy_resampled", "full_data_accuracy", "n_resampled"):
            if key not in last:
                errors.append(f"summary 末轮缺少字段 {key}")
        if last.get("n_resampled") != n:
            errors.append(f"重采样规模 {last.get('n_resampled')} != N={n}")

    tj = json.loads(tree_json.read_text(encoding="utf-8"))
    root = tj.get("root") or {}
    if root.get("type") not in ("split", "leaf"):
        errors.append("tree.json 缺少有效 root 节点")
    if int(tj.get("n_leaves", 0)) < 1:
        errors.append("tree.json n_leaves 无效")

    sel_r = summary.get("selected_round")
    sel_acc = summary.get("selected_acc_full")
    print(
        f"VIPER: n_rules={len(rules_lines)} rounds={len(rounds)} "
        f"n_nodes={tj.get('n_nodes')} n_leaves={tj.get('n_leaves')} "
        f"选用轮={sel_r} acc_full={sel_acc}"
    )
    return errors


def verify_e2e(
    csv_path: Path,
    out_dir: Path,
    *,
    device: str = "cpu",
    eps: float = DEFAULT_EPS,
    weighted_sampling: bool = True,
) -> int:
    """全流程产物检查：q_hat.pt → l_hat → weights → viper_out。"""
    ckpt = out_dir / "q_hat.pt"
    l_hat_csv = out_dir / "l_hat.csv"
    weights_csv = out_dir / "weights.csv"
    viper_dir = out_dir / "viper_out"

    all_errors: list[str] = []
    required = [
        ("q_hat.pt", ckpt),
        ("l_hat.csv", l_hat_csv),
        ("weights.csv", weights_csv),
        ("viper_out/", viper_dir),
    ]
    print("=== 全流程验证 ===")
    print(f"CSV: {csv_path}")
    print(f"输出目录: {out_dir}")
    for label, p in required:
        ok = p.is_file() or (p.is_dir() and any(p.iterdir()))
        status = "存在" if ok else "缺失"
        print(f"  [{status}] {label}")
        if not ok:
            all_errors.append(f"缺少产物: {p}")

    if all_errors:
        for e in all_errors:
            print(f"FAIL: {e}")
        return 1

    final_json = out_dir / "final_result.json"
    mixed_alpha = 0.7
    if final_json.is_file():
        try:
            saved_cfg = json.loads(final_json.read_text(encoding="utf-8")).get("config", {})
            if "viper_weighted_sampling" in saved_cfg:
                weighted_sampling = bool(saved_cfg["viper_weighted_sampling"])
            mixed_alpha = float(saved_cfg.get("mixed_alpha", 0.7))
        except (json.JSONDecodeError, OSError):
            pass

    rc = verify(csv_path, ckpt, device=device)
    if rc != 0:
        return rc

    w_err = verify_weights_csv(weights_csv, eps=eps, weighted_sampling=weighted_sampling, mixed_alpha=mixed_alpha)
    for e in w_err:
        print(f"FAIL weights: {e}")
    if w_err:
        return 1
    print(f"OK: weights.csv ({weights_csv})")

    v_err = verify_viper_outputs(csv_path, weights_csv, viper_dir)
    for e in v_err:
        print(f"FAIL viper: {e}")
    if v_err:
        return 1
    print("OK: VIPER 输出 (rules + summary + tree)")

    print("\n=== 全流程验证通过 ===")
    return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--weights-csv", type=str, default="")
    p.add_argument("--eps", type=float, default=DEFAULT_EPS)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--toy-only", action="store_true", help="只跑文档数值小例子")
    p.add_argument(
        "--e2e",
        action="store_true",
        help="检查 out-dir 下全流程产物（需先 python -m causal.decision_tree --phase all）",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="fqe_out 目录；--e2e 时默认 {csv 父目录}/fqe_out",
    )
    args = p.parse_args()

    errors: list[str] = []
    errors.extend(verify_toy_weights(eps=args.eps))
    if errors:
        for e in errors:
            print(f"FAIL toy: {e}")
        sys.exit(1)
    print("OK: 文档小例子 (l_hat → w_raw → weights)")

    if args.toy_only:
        sys.exit(0)

    if args.e2e:
        if not args.csv:
            p.error("--e2e 需要 --csv")
        csv_p = Path(args.csv)
        out_p = Path(args.out_dir) if args.out_dir.strip() else csv_p.parent / "fqe_out"
        sys.exit(verify_e2e(csv_p, out_p, device=args.device, eps=args.eps))

    if args.weights_csv:
        w_err = verify_weights_csv(Path(args.weights_csv), eps=args.eps)
        for e in w_err:
            print(f"FAIL weights: {e}")
        if w_err:
            sys.exit(1)
        print(f"OK: weights.csv 检查通过 ({args.weights_csv})")

    if args.csv and args.checkpoint:
        sys.exit(verify(Path(args.csv), Path(args.checkpoint), args.device))

    if not args.weights_csv:
        p.error("请提供 --csv 与 --checkpoint，或 --weights-csv，或 --toy-only")
    sys.exit(0)


if __name__ == "__main__":
    main()
