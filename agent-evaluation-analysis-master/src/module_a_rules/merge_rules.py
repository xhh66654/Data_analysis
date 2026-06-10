"""
规则合并：剔除冗余规则，提升规则集可读性和泛化性。

================================================================================
说明（小白友好版）：
================================================================================

决策树提取出的规则往往数量很多（叶节点有多少就有多少条），
而且同一个动作可能有许多条件非常相似的规则，读起来冗长。

合并策略（从简到难）：

策略1：子集合并
    "如果规则A的条件 是 规则B的条件的子集，
     且两者动作相同，则规则B是多余的——规则A已经包含了规则B的所有情形"
    → 删掉规则B，保留更宽松（条件更少）的规则A

    例如：
        规则A：IF 敌机距离 <= 较近  THEN 发射导弹
        规则B：IF 敌机距离 <= 较近 AND 自身血量 > 良好  THEN 发射导弹
        → 规则B多余，删掉

策略2：区间合并（同一特征、同向、相邻阈值、同动作）
    "两条规则除了某个特征的阈值不同，其余条件完全一样，
     且两条规则的阈值可以合并成一个更大的区间"
    → 合并为一条阈值更宽松的规则

    例如：
        规则A：IF 敌机距离 <= 50km  THEN 发射导弹
        规则B：IF 敌机距离 <= 80km  THEN 发射导弹
        → 合并为：IF 敌机距离 <= 80km  THEN 发射导弹

rules_coverage：
    "评估规则集在数据上的准确率"
    用提取出的规则集逐条匹配测试样本，
    统计被规则正确覆盖的比例（类似分类准确率）。
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from src.module_a_rules.extract_rules import Rule, RuleCondition


# ==============================================================================
# 策略1：子集合并（条件更少的规则覆盖条件更多的规则）
# ==============================================================================

def _conditions_to_frozenset(conditions: List[RuleCondition]) -> FrozenSet[Tuple]:
    """
    把条件列表转成 frozenset，方便做子集比较。

    每个条件表示为 (feature_idx, op, threshold) 三元组。

    参数:
        conditions: 规则条件列表。

    返回:
        条件的不可变集合。
    """
    return frozenset((c.feature_idx, c.op, c.threshold) for c in conditions)


def _subset_merge(rules: List[Rule]) -> List[Rule]:
    """
    策略1：删除那些"条件是其他同动作规则超集"的冗余规则。

    如果规则A的条件集合 ⊆ 规则B的条件集合，且 action(A) == action(B)，
    则规则B多余（A已经更宽松地覆盖了B的情形），删除B。

    复杂度：O(N²)，规则数量通常不超过几百条，可接受。

    参数:
        rules: 原始规则列表。

    返回:
        删除冗余规则后的列表。
    """
    # 按动作分组
    by_action: Dict[object, List[Rule]] = {}
    for rule in rules:
        by_action.setdefault(rule.action, []).append(rule)

    kept: List[Rule] = []
    for action, group in by_action.items():
        # 对每组规则，找出哪些是冗余的（被其他规则"包含"）
        cond_sets = [_conditions_to_frozenset(r.conditions) for r in group]
        redundant = set()

        for i in range(len(group)):
            if i in redundant:
                continue
            for j in range(len(group)):
                if i == j or j in redundant:
                    continue
                # 如果 i 的条件集 ⊆ j 的条件集，则 j 是冗余的
                if cond_sets[i] < cond_sets[j]:  # 严格子集
                    redundant.add(j)

        for i, rule in enumerate(group):
            if i not in redundant:
                kept.append(rule)

    return kept


# ==============================================================================
# 策略2：区间合并（同特征、同动作、阈值可扩展）
# ==============================================================================

def _interval_merge(rules: List[Rule]) -> List[Rule]:
    """
    策略2：对同动作、除一个特征阈值外其余条件完全相同的规则进行区间合并。

    合并逻辑：
        规则A：IF feat_k <= 50  AND [其他条件]  THEN 动作X
        规则B：IF feat_k <= 80  AND [其他条件]  THEN 动作X
        → 合并为：IF feat_k <= 80  AND [其他条件]  THEN 动作X
          （保留阈值更大的那条，覆盖范围更宽）

        规则A：IF feat_k > 50  AND [其他条件]  THEN 动作X
        规则B：IF feat_k > 30  AND [其他条件]  THEN 动作X
        → 合并为：IF feat_k > 30  AND [其他条件]  THEN 动作X
          （保留阈值更小的那条，覆盖范围更宽）

    参数:
        rules: 子集合并后的规则列表。

    返回:
        区间合并后的规则列表。
    """
    # 用一个简单的贪心思路：
    # 把规则组织成 "签名 → 规则列表"，
    # 签名 = (动作, 除某一特定条件外的所有其他条件的 frozenset + 特征idx + op)
    # 相同签名的规则说明"只有阈值不同"，可以合并

    # 为了避免“先标记后比较”导致的误删，改为按签名分组后再做阈值归并。
    # 签名：动作 + 条件数量 + (变化条件索引, feature_idx, op) + 其余固定条件集合
    groups: Dict[Tuple, List[Rule]] = {}
    passthrough: List[Rule] = []

    for rule in rules:
        if not rule.conditions:
            passthrough.append(rule)
            continue
        grouped = False
        for idx, cond in enumerate(rule.conditions):
            if cond.op not in ("<=", ">"):
                continue
            fixed = tuple(
                sorted(
                    (c.feature_idx, c.op, c.threshold)
                    for j, c in enumerate(rule.conditions)
                    if j != idx
                )
            )
            key = (rule.action, len(rule.conditions), idx, cond.feature_idx, cond.op, fixed)
            groups.setdefault(key, []).append(rule)
            grouped = True
        if not grouped:
            passthrough.append(rule)

    merged: List[Rule] = list(passthrough)
    for key, group in groups.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        idx = int(key[2])
        op = str(key[4])
        if op == "<=":
            chosen = max(group, key=lambda r: float(r.conditions[idx].threshold))
        else:  # op == ">"
            chosen = min(group, key=lambda r: float(r.conditions[idx].threshold))
        merged.append(chosen)

    # 去重（同一条规则可能因多个可变索引重复进入 merged）
    uniq: Dict[Tuple, Rule] = {}
    for r in merged:
        sig = (
            r.action,
            tuple((c.feature_idx, c.op, float(c.threshold)) for c in r.conditions),
        )
        if sig not in uniq or float(r.support) > float(uniq[sig].support):
            uniq[sig] = r
    return list(uniq.values())


# ==============================================================================
# 主函数：合并规则
# ==============================================================================

def merge_rules(rules: List[Rule]) -> List[Rule]:
    """
    对规则集进行两轮合并，减少冗余，提升可读性。

    合并顺序：
        第1轮：子集合并（删除被更宽松规则覆盖的冗余规则）
        第2轮：区间合并（合并只有阈值不同的同动作规则）

    Parameters
    ----------
    rules : 原始规则列表（来自 extract_rules_from_tree）

    Returns
    -------
    合并后的规则列表（数量 ≤ 原始数量），按支持度降序排列

    示例：
        raw_rules = extract_rules_from_tree(tree, preprocessor)
        print(f"合并前: {len(raw_rules)} 条")
        merged = merge_rules(raw_rules)
        print(f"合并后: {len(merged)} 条")
    """
    if not rules:
        return []

    # 第1轮：子集合并
    after_subset = _subset_merge(rules)

    # 第2轮：区间合并
    after_interval = _interval_merge(after_subset)

    # 按支持度降序排列（覆盖样本多的规则排前面）
    after_interval.sort(key=lambda r: r.support, reverse=True)

    return after_interval


# ==============================================================================
# 评估：规则集在数据上的准确率
# ==============================================================================

def rules_coverage(
    rules: List[Rule],
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    评估规则集在数据集 (X, y) 上的准确率（规则覆盖率）。

    匹配逻辑（第一条命中优先）：
        对每个样本，按顺序遍历规则列表，
        找到第一条所有条件都满足的规则，
        用该规则的 action 作为预测值。
        如果没有任何规则命中，该样本算错误。

    Parameters
    ----------
    rules : 规则列表（已按支持度降序排列）
    X     : (N, n_features) 观测特征矩阵（归一化后）
    y     : (N,) 真实动作标签

    Returns
    -------
    accuracy : 0.0 ~ 1.0，被规则正确覆盖的样本比例

    示例：
        acc = rules_coverage(merged_rules, X_test, y_test)
        print(f"规则集覆盖率: {acc:.2%}")  # 例如 "规则集覆盖率: 87.50%"
    """
    if len(rules) == 0 or len(X) == 0:
        return 0.0

    correct = 0
    for xi, yi in zip(X, y):
        pred = _predict_one(rules, xi)
        if pred is not None and pred == yi:
            correct += 1

    return correct / len(X)


def _predict_one(rules: List[Rule], x: np.ndarray) -> Optional[object]:
    """
    用规则集对单个样本进行预测（第一条命中优先）。

    Parameters
    ----------
    rules : 规则列表
    x     : 单个样本的观测特征向量

    Returns
    -------
    命中规则的动作；如果没有规则命中，返回 None
    """
    for rule in rules:
        if _matches(rule, x):
            return rule.action
    return None


def _matches(rule: Rule, x: np.ndarray) -> bool:
    """
    判断样本 x 是否满足规则 rule 的所有条件。

    所有条件都满足（AND 关系）才算命中。

    参数:
        rule: 待匹配的规则。
        x: 归一化后的特征向量。

    返回:
        全部条件满足时返回 ``True``，否则返回 ``False``。
    """
    for cond in rule.conditions:
        val = x[cond.feature_idx]
        if cond.op == "<=" and not (val <= cond.threshold):
            return False
        if cond.op == ">" and not (val > cond.threshold):
            return False
    return True
