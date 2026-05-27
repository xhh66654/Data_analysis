"""
校验 VIPER 导出的 tree.json：从根到叶的路径约束是否逻辑自洽。

递归遍历 tree.json 的 root 节点，沿左（<= threshold）/ 右（> threshold）分支
累积每个特征 s_i 的 (lo, hi] 区间约束，检查：
  · 是否存在 lo > hi 的空区间（规则逻辑矛盾）
  · 父子节点样本数 n_samples 是否一致
  · 特定特征（如 s_1）是否出现不可能的双重约束
  · 叶节点 class_counts 是否全为零

返回码：0=通过，1=发现矛盾或错误。

用法：
  python -m causal.decision_tree.validate_tree_json [path/to/tree.json]
  默认路径：causal/trajectories/fqe_out/viper_out/tree.json

建议在导出 rules.txt 后人工抽查或 CI 中调用，确保 IF-THEN 规则与树结构一致。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def validate_tree_json(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    root = data["root"]
    eps = 1e-9

    errors: list[str] = []
    warnings: list[str] = []
    leaves: list[dict] = []

    def walk(node: dict, constraints: dict[str, tuple], path: list[str]) -> None:
        if node["type"] == "leaf":
            leaves.append(
                {
                    "path": list(path),
                    "prediction": node["prediction"],
                    "n_samples": node["n_samples"],
                    "constraints": dict(constraints),
                }
            )
            for feat, (lo, hi) in constraints.items():
                if lo is not None and hi is not None and lo > hi + eps:
                    errors.append(
                        f"空区间 {feat}: ({lo}, {hi}] 路径={' -> '.join(path)} -> {node['prediction']}"
                    )
            return

        feat = node["feature"]
        thr = float(node["threshold"])
        tag = f"{feat}@{node['node_id']}(<={thr:.4g}|>{thr:.4g}) n={node['n_samples']}"
        path.append(tag)

        lo_p, hi_p = constraints.get(feat, (None, None))
        c_left = dict(constraints)
        c_left[feat] = (lo_p, thr if hi_p is None else min(hi_p, thr))
        walk(node["left"], c_left, path)

        lo_p, hi_p = constraints.get(feat, (None, None))
        c_right = dict(constraints)
        c_right[feat] = (max(lo_p, thr) if lo_p is not None else thr, hi_p)
        walk(node["right"], c_right, path)

        path.pop()

        if node["left"]["n_samples"] + node["right"]["n_samples"] != node["n_samples"]:
            warnings.append(
                f"节点 {node['node_id']} 子样本和 {node['left']['n_samples']}+{node['right']['n_samples']}"
                f" != {node['n_samples']}"
            )

    walk(root, {}, [])

    # 用户关心的 s_1 矛盾：祖先 <=1.234 且 要求 >1.379
    t234 = 1.2335858941078186
    t379 = 1.3791329264640808
    bad_s1 = []
    ok_band = []
    for leaf in leaves:
        c = leaf["constraints"].get("s_1")
        if not c:
            continue
        lo, hi = c
        if hi is not None and hi <= t234 + eps and lo is not None and lo > t379 - eps:
            bad_s1.append(leaf)
        if lo is not None and lo >= t234 - eps and hi is not None and hi <= t379 + eps:
            ok_band.append(leaf)

    zero_cc = 0

    def count_zero(node: dict) -> None:
        nonlocal zero_cc
        if node["type"] == "leaf":
            cc = node.get("class_counts") or {}
            if cc and all(int(v) == 0 for v in cc.values()):
                zero_cc += 1
            return
        count_zero(node["left"])
        count_zero(node["right"])

    count_zero(root)

    print(f"文件: {path}")
    print(f"meta: n_nodes={data.get('n_nodes')} n_leaves={data.get('n_leaves')} max_depth={data.get('max_depth')}")
    print(f"叶节点数: {len(leaves)}")
    print(f"逻辑矛盾 (特征区间为空): {len(errors)}")
    for e in errors[:8]:
        print(f"  [ERROR] {e}")
    print(f"样本数和不等警告: {len(warnings)}")
    if warnings[:3]:
        for w in warnings[:3]:
            print(f"  [WARN] {w}")
    print(f"s_1<={t234:.4f} 且 s_1>{t379:.4f} 的叶路径: {len(bad_s1)}")
    print(f"s_1 在 ({t234:.4f}, {t379:.4f}] 的叶路径: {len(ok_band)} (与 rules 中 >1.234 且 <=1.379 一致)")
    print(f"class_counts 全为 0 的叶: {zero_cc}/{len(leaves)}")

    if not errors and not bad_s1:
        print("\n结论: tree.json 路径约束自洽，无「s_1<=1.234 下再 s_1>1.379」类矛盾。")
        return 0
    return 1


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "trajectories/fqe_out/viper_out/tree.json"
    sys.exit(validate_tree_json(p))
