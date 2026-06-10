"""
决策树与规则集一致性校验。

用于回答：规则是否真的来自「根 → 叶」路径？合并后是否仍与树一致？
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import TREE_LEAF

from src.module_a_rules.extract_rules import Rule, RuleCondition, extract_rules_from_tree
from src.module_a_rules.merge_rules import _matches, merge_rules


@dataclass
class TreeRulesVerification:
    """
    决策树与规则集一致性校验的结果摘要。

    记录叶节点数、原始/合并规则数、路径一一对应关系、
    与 ``tree.predict`` 的一致性以及训练集覆盖率等指标。
    """

    n_leaves: int
    n_raw_rules: int
    n_merged_rules: int
    raw_paths_one_to_one: bool
    raw_matches_tree_predict: bool
    merged_matches_tree_predict: bool
    tree_accuracy_on_train: float
    raw_rules_coverage: float
    merged_rules_coverage: float
    details: Dict[str, Any]


def _leaf_paths_from_tree(tree: DecisionTreeClassifier) -> List[Dict[str, Any]]:
    """
    从根 DFS 到每个叶节点，得到与 ``extract_rules`` 同构的路径。

    参数:
        tree: 已训练的决策树分类器。

    返回:
        每条路径对应一个字典，含条件元组、动作、支持度与节点 ID。
    """
    t = tree.tree_
    paths: List[Dict[str, Any]] = []

    def _dfs(node_id: int, path: List[RuleCondition]) -> None:
        """深度优先遍历到叶节点，收集路径条件与叶节点统计。"""
        left = int(t.children_left[node_id])
        if left == TREE_LEAF:
            values = t.value[node_id][0]
            best_idx = int(values.argmax())
            paths.append(
                {
                    "conditions": tuple(
                        (c.feature_idx, c.op, float(c.threshold)) for c in path
                    ),
                    "action": tree.classes_[best_idx],
                    "support": int(values.sum()),
                    "node_id": node_id,
                }
            )
            return
        feat_idx = int(t.feature[node_id])
        threshold = float(t.threshold[node_id])
        _dfs(left, path + [RuleCondition(feat_idx, "<=", threshold)])
        _dfs(
            int(t.children_right[node_id]),
            path + [RuleCondition(feat_idx, ">", threshold)],
        )

    _dfs(0, [])
    return paths


def _rule_signature(rule: Rule) -> Tuple:
    """
    生成规则签名，用于与树路径比较。

    参数:
        rule: 规则对象。

    返回:
        ``(conditions_tuple, action, support)`` 元组。
    """
    return (
        tuple((c.feature_idx, c.op, float(c.threshold)) for c in rule.conditions),
        rule.action,
        int(rule.support),
    )


def _path_signature(path: Dict[str, Any]) -> Tuple:
    """
    生成树路径签名，用于与规则比较。

    参数:
        path: 由 ``_leaf_paths_from_tree`` 产生的路径字典。

    返回:
        ``(conditions, action, support)`` 元组。
    """
    return (path["conditions"], path["action"], int(path["support"]))


def _predict_most_specific(rules: List[Rule], x: np.ndarray) -> Optional[object]:
    """
    用条件最多的命中规则预测单个样本的动作。

    参数:
        rules: 规则列表。
        x: 归一化后的特征向量。

    返回:
        预测动作；无命中时返回 ``None``。
    """
    hits = [r for r in rules if _matches(r, x)]
    if not hits:
        return None
    return max(hits, key=lambda r: len(r.conditions)).action


def verify_tree_and_rules(
    tree: DecisionTreeClassifier,
    X_pre: np.ndarray,
    y: np.ndarray,
    *,
    merged_rules: Optional[List[Rule]] = None,
) -> TreeRulesVerification:
    """
    校验决策树与规则集之间的一致性关系。

    检查项：
    1. 原始规则条数 = 叶节点数，且每条规则对应唯一根→叶路径；
    2. 原始规则在训练集上的预测与 ``tree.predict`` 完全一致；
    3. 合并规则与 ``tree.predict`` 是否一致（合并会放宽条件，可能不一致）。

    参数:
        tree: 已训练的决策树分类器。
        X_pre: 归一化后的训练特征矩阵。
        y: 真实动作标签数组。
        merged_rules: 可选的已合并规则列表；为 ``None`` 时自动调用 ``merge_rules``。

    返回:
        ``TreeRulesVerification`` 校验结果对象。
    """
    raw_rules = extract_rules_from_tree(tree)
    merged = merged_rules if merged_rules is not None else merge_rules(raw_rules)
    paths = _leaf_paths_from_tree(tree)

    raw_sigs = {_rule_signature(r) for r in raw_rules}
    path_sigs = {_path_signature(p) for p in paths}
    one_to_one = raw_sigs == path_sigs

    y_tree = tree.predict(X_pre)
    tree_acc = float(np.mean(y_tree == y))

    raw_preds = np.array([_predict_most_specific(raw_rules, xi) for xi in X_pre])
    merged_preds = np.array([_predict_most_specific(merged, xi) for xi in X_pre])

    raw_match = bool(np.all(raw_preds == y_tree))
    merged_match = bool(np.all(merged_preds == y_tree))

    def _coverage(rules: List[Rule]) -> float:
        """计算规则集在训练集上的标签覆盖率。"""
        ok = 0
        for xi, yi in zip(X_pre, y):
            pred = _predict_most_specific(rules, xi)
            if pred is not None and pred == yi:
                ok += 1
        return ok / len(y) if len(y) else 0.0

    return TreeRulesVerification(
        n_leaves=len(paths),
        n_raw_rules=len(raw_rules),
        n_merged_rules=len(merged),
        raw_paths_one_to_one=one_to_one,
        raw_matches_tree_predict=raw_match,
        merged_matches_tree_predict=merged_match,
        tree_accuracy_on_train=tree_acc,
        raw_rules_coverage=_coverage(raw_rules),
        merged_rules_coverage=_coverage(merged),
        details={
            "only_in_raw_rules": len(raw_sigs - path_sigs),
            "only_in_tree_paths": len(path_sigs - raw_sigs),
            "raw_mismatch_samples": int(np.sum(raw_preds != y_tree)),
            "merged_mismatch_samples": int(np.sum(merged_preds != y_tree)),
        },
    )
