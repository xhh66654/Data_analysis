"""
从规则集构建「规则树」结构，并解释规则合并过程。

说明：
- 决策树提取的 raw 规则天然是一棵二叉树（每条根→叶路径一条规则）。
- merge_rules 后的规则集是「规则列表」，不一定还能还原成唯一二叉决策树。
- 本模块提供：
  1) merge 分步说明（子集合并 / 区间合并）
  2) 将规则集组装为前缀树（prefix tree），用于树形展示
  3) 检测合并后是否仍可当作决策树使用（无重叠冲突）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.module_a_rules.extract_rules import Rule, RuleCondition
from src.module_a_rules.merge_rules import _interval_merge, _matches, _subset_merge


@dataclass
class RuleTreeNode:
    """
    规则树节点，兼容前缀树与二叉决策树的统一表示。

    分裂节点记录特征、运算符与阈值；叶节点记录动作、支持度与置信度。
    子节点以边标签为键组织，可检测同层条件重叠冲突。
    """

    node_id: str
    node_type: str  # "split" | "leaf" | "empty"
    # split
    feature_idx: Optional[int] = None
    feature_name: Optional[str] = None
    op: Optional[str] = None
    threshold: Optional[float] = None
    threshold_text: Optional[str] = None
    # leaf
    action: Optional[object] = None
    action_text: Optional[str] = None
    support: int = 0
    confidence: float = 0.0
    rule_indices: List[int] = field(default_factory=list)
    # children: edge_label -> child
    children: Dict[str, "RuleTreeNode"] = field(default_factory=dict)
    # 同一层是否存在条件重叠（无法构成决策树分裂）
    has_overlap_conflict: bool = False


@dataclass
class MergeStepReport:
    """
    单轮规则合并步骤的报告摘要。

    记录合并前后规则数量及被删除规则的详情。
    """

    name: str
    before_count: int
    after_count: int
    removed_rules: List[Dict[str, Any]]


@dataclass
class RuleTreeBuildResult:
    """
    规则树构建结果。

    包含树根、树类型（决策树等价或带冲突的前缀树）、
    是否与决策树兼容及合并步骤报告。
    """

    tree: RuleTreeNode
    tree_kind: str  # "decision_tree_equivalent" | "prefix_tree_with_conflicts"
    is_decision_tree_compatible: bool
    overlap_conflicts: List[str]
    merge_steps: List[MergeStepReport]


def explain_rule_merge_steps(raw_rules: List[Rule]) -> List[MergeStepReport]:
    """
    逐步记录子集合并与区间合并删除了哪些规则。

    参数:
        raw_rules: 合并前的原始规则列表。

    返回:
        包含 ``subset_merge`` 与 ``interval_merge`` 两步报告的列表。
    """
    after_subset = _subset_merge(raw_rules)
    subset_removed = _diff_removed(raw_rules, after_subset)

    after_interval = _interval_merge(after_subset)
    interval_removed = _diff_removed(after_subset, after_interval)

    return [
        MergeStepReport(
            name="subset_merge",
            before_count=len(raw_rules),
            after_count=len(after_subset),
            removed_rules=subset_removed,
        ),
        MergeStepReport(
            name="interval_merge",
            before_count=len(after_subset),
            after_count=len(after_interval),
            removed_rules=interval_removed,
        ),
    ]


def _diff_removed(before: List[Rule], after: List[Rule]) -> List[Dict[str, Any]]:
    """
    比较合并前后规则集，提取被删除规则的详情。

    参数:
        before: 合并前的规则列表。
        after: 合并后的规则列表。

    返回:
        被删除规则的字典列表，含索引、动作、条件等字段。
    """
    after_sigs = {_rule_sig(r) for r in after}
    removed = []
    for i, r in enumerate(before):
        sig = _rule_sig(r)
        if sig not in after_sigs:
            removed.append(
                {
                    "rule_index": i,
                    "action": str(r.action),
                    "support": int(r.support),
                    "confidence": round(float(r.confidence), 4),
                    "conditions": [
                        {
                            "feature_idx": c.feature_idx,
                            "op": c.op,
                            "threshold": float(c.threshold),
                        }
                        for c in r.conditions
                    ],
                }
            )
    return removed


def _rule_sig(rule: Rule) -> Tuple:
    """
    生成规则的唯一签名元组，用于去重与差异比较。

    参数:
        rule: 规则对象。

    返回:
        ``(action, conditions_tuple)`` 形式的不可变签名。
    """
    return (
        rule.action,
        tuple((c.feature_idx, c.op, float(c.threshold)) for c in rule.conditions),
    )


def build_rule_tree(
    rules: List[Rule],
    feature_names: List[str],
    preprocessor: Optional[Any] = None,
) -> RuleTreeBuildResult:
    """
    将规则列表组装为树形结构（按条件前缀共享路径）。

    若同一节点下出现「同特征、同方向、多个阈值均可满足」的分支，则标记冲突；
    此时只能作为 prefix tree 展示，不能视为唯一决策树。

    参数:
        rules: 规则列表。
        feature_names: 特征名列表。
        preprocessor: 可选预处理器，用于格式化阈值文本。

    返回:
        ``RuleTreeBuildResult``，含树根、树类型与冲突信息。
    """
    merge_steps = explain_rule_merge_steps(rules) if len(rules) > 0 else []
    root = RuleTreeNode(node_id="root", node_type="empty")
    conflicts: List[str] = []

    for idx, rule in enumerate(rules):
        _insert_rule_path(
            root,
            rule,
            rule_index=idx,
            feature_names=feature_names,
            preprocessor=preprocessor,
            conflicts=conflicts,
        )

    _finalize_empty_root(root)
    compatible = len(conflicts) == 0 and not root.has_overlap_conflict
    tree_kind = "decision_tree_equivalent" if compatible else "prefix_tree_with_conflicts"

    return RuleTreeBuildResult(
        tree=root,
        tree_kind=tree_kind,
        is_decision_tree_compatible=compatible,
        overlap_conflicts=conflicts,
        merge_steps=merge_steps,
    )


def _insert_rule_path(
    node: RuleTreeNode,
    rule: Rule,
    rule_index: int,
    feature_names: List[str],
    preprocessor: Optional[Any],
    conflicts: List[str],
    depth: int = 0,
) -> None:
    """
    将单条规则按条件前缀递归插入规则树。

    参数:
        node: 当前树节点。
        rule: 待插入的规则（条件列表会逐层剥离）。
        rule_index: 规则在原始列表中的索引。
        feature_names: 特征名列表。
        preprocessor: 可选预处理器。
        conflicts: 冲突消息列表（就地追加）。
        depth: 当前递归深度。
    """
    if not rule.conditions:
        _attach_leaf(node, rule, rule_index)
        return

    cond = rule.conditions[0]
    edge = _edge_label(cond)
    child = node.children.get(edge)
    if child is None:
        child = RuleTreeNode(
            node_id=f"{node.node_id}/{edge}",
            node_type="split",
            feature_idx=cond.feature_idx,
            feature_name=_feat_name(cond.feature_idx, feature_names),
            op=cond.op,
            threshold=float(cond.threshold),
            threshold_text=_format_threshold(cond, feature_names, preprocessor),
        )
        node.children[edge] = child
        node.node_type = "split" if node.node_type == "empty" else node.node_type

    _check_sibling_overlap(node, cond, conflicts)
    _insert_rule_path(
        child,
        Rule(
            conditions=rule.conditions[1:],
            action=rule.action,
            support=rule.support,
            confidence=rule.confidence,
        ),
        rule_index=rule_index,
        feature_names=feature_names,
        preprocessor=preprocessor,
        conflicts=conflicts,
        depth=depth + 1,
    )


def _check_sibling_overlap(
    parent: RuleTreeNode,
    cond: RuleCondition,
    conflicts: List[str],
) -> None:
    """
    检测同层兄弟节点是否存在条件重叠冲突。

    同层若存在「同特征、同比较符、不同阈值」的分支，样本可能同时命中多条路径，
    此时规则集不能视为唯一决策树（需要规则优先级，而不是树遍历）。

    参数:
        parent: 父分裂节点。
        cond: 当前待插入的条件。
        conflicts: 冲突消息列表（就地追加）。
    """
    for edge, ch in parent.children.items():
        if edge == "__leaf__" or ch.feature_idx is None:
            continue
        if ch.feature_idx == cond.feature_idx and ch.op == cond.op:
            if float(ch.threshold) == float(cond.threshold):
                continue
            parent.has_overlap_conflict = True
            msg = (
                f"节点 {parent.node_id} 存在重叠分支："
                f"feature={cond.feature_idx} {cond.op} "
                f"阈值 {ch.threshold} 与 {cond.threshold}"
            )
            if msg not in conflicts:
                conflicts.append(msg)


def _attach_leaf(node: RuleTreeNode, rule: Rule, rule_index: int) -> None:
    """
    在节点下挂载或更新叶节点。

    若已存在叶节点，合并 ``rule_indices`` 并保留支持度更高的动作。

    参数:
        node: 父节点。
        rule: 对应规则。
        rule_index: 规则索引。
    """
    leaf = node.children.get("__leaf__")
    if leaf is None:
        leaf = RuleTreeNode(
            node_id=f"{node.node_id}/leaf",
            node_type="leaf",
            action=rule.action,
            action_text=_format_action_short(rule.action),
            support=int(rule.support),
            confidence=float(rule.confidence),
            rule_indices=[rule_index],
        )
        node.children["__leaf__"] = leaf
        return
    leaf.rule_indices.append(rule_index)
    if int(rule.support) > int(leaf.support):
        leaf.action = rule.action
        leaf.action_text = _format_action_short(rule.action)
        leaf.support = int(rule.support)
        leaf.confidence = float(rule.confidence)


def _finalize_empty_root(root: RuleTreeNode) -> None:
    """
    将无子节点的空根节点标记为「无规则」叶节点。

    参数:
        root: 规则树根节点。
    """
    if root.node_type == "empty" and not root.children:
        root.node_type = "leaf"
        root.action_text = "（无规则）"


def _edge_label(cond: RuleCondition) -> str:
    """
    为规则条件生成前缀树边的唯一标签字符串。

    参数:
        cond: 规则条件。

    返回:
        含特征索引、运算符与阈值的边标签。
    """
    return f"f{cond.feature_idx}_{cond.op}_{cond.threshold:.6f}"


def _feat_name(feat_idx: int, feature_names: List[str]) -> str:
    """
    根据列索引返回特征名。

    参数:
        feat_idx: 特征列索引。
        feature_names: 特征名列表。

    返回:
        特征名；越界时返回 ``feature_{idx}``。
    """
    if 0 <= feat_idx < len(feature_names):
        return str(feature_names[feat_idx])
    return f"feature_{feat_idx}"


def _format_threshold(
    cond: RuleCondition,
    feature_names: List[str],
    preprocessor: Optional[Any],
) -> str:
    """
    将规则条件格式化为人类可读的阈值文本。

    参数:
        cond: 规则条件。
        feature_names: 特征名列表。
        preprocessor: 可选预处理器，用于反归一化与语义标签。

    返回:
        阈值描述字符串。
    """
    if preprocessor is None:
        return f"{_feat_name(cond.feature_idx, feature_names)} {cond.op} {cond.threshold:.4f}"
    try:
        raw = float(preprocessor.denormalize_threshold(cond.feature_idx, cond.threshold))
        feat = str(preprocessor.get_feature_name(cond.feature_idx))
        sem = str(preprocessor.discretize_label(feat, raw))
        return f"{feat} {cond.op} {raw:.3f}（{sem}）"
    except Exception:
        return f"{_feat_name(cond.feature_idx, feature_names)} {cond.op} {cond.threshold:.4f}"


def _format_action_short(action: object) -> str:
    """
    将动作格式化为适合树节点展示的短文本。

    参数:
        action: 动作值（字符串或可转字符串对象）。

    返回:
        不超过 60 字符的动作描述。
    """
    s = str(action)
    if len(s) <= 60:
        return s
    return s[:59] + "…"


def predict_by_rule_tree(
    root: RuleTreeNode,
    x: np.ndarray,
    *,
    prefer_most_specific: bool = True,
) -> Optional[object]:
    """
    按规则树遍历样本并预测动作。

    决策树兼容结构：每层的 yes/no 最多命中一个分支。
    若存在冲突（prefix tree），回退为「收集所有命中叶节点 + 取最长路径」。

    参数:
        root: 规则树根节点。
        x: 归一化后的特征向量。
        prefer_most_specific: 冲突时是否优先取路径最深的叶节点。

    返回:
        预测动作；无命中时返回 ``None``。
    """
    if not root.has_overlap_conflict:
        leaf = _walk_decision_style(root, x)
        if leaf is not None:
            return leaf.action

    hits: List[Tuple[int, RuleTreeNode]] = []
    _collect_matching_leaves(root, x, 0, hits)
    if not hits:
        return None
    if prefer_most_specific:
        _, leaf = max(hits, key=lambda t: t[0])
        return leaf.action
    return hits[0][1].action


def _walk_decision_style(node: RuleTreeNode, x: np.ndarray) -> Optional[RuleTreeNode]:
    """
    按二叉决策树风格自顶向下遍历，返回命中的叶节点。

    参数:
        node: 当前节点。
        x: 特征向量。

    返回:
        命中的叶节点；无法遍历时返回 ``None``。
    """
    if node.node_type == "leaf":
        return node
    if node.node_type == "split" and node.feature_idx is not None:
        val = x[node.feature_idx]
        if "yes" in node.children and "no" in node.children:
            if val <= float(node.threshold):
                return _walk_decision_style(node.children["yes"], x)
            return _walk_decision_style(node.children["no"], x)
        for edge, child in node.children.items():
            if edge == "__leaf__":
                continue
            if child.op == "<=" and val <= float(child.threshold):
                return _walk_decision_style(child, x)
            if child.op == ">" and val > float(child.threshold):
                return _walk_decision_style(child, x)
    if "__leaf__" in node.children:
        return node.children["__leaf__"]
    return None


def _collect_matching_leaves(
    node: RuleTreeNode,
    x: np.ndarray,
    depth: int,
    out: List[Tuple[int, RuleTreeNode]],
) -> None:
    """
    在前缀树模式下收集所有条件满足的叶节点。

    参数:
        node: 当前节点。
        x: 特征向量。
        depth: 当前深度。
        out: 输出列表，元素为 ``(depth, leaf_node)`` 元组（就地追加）。
    """
    if node.node_type == "leaf" and node.action is not None:
        out.append((depth, node))
        return
    matched = False
    for edge, child in node.children.items():
        if edge == "__leaf__":
            out.append((depth, child))
            continue
        if child.node_type != "split" or child.feature_idx is None:
            continue
        val = x[child.feature_idx]
        ok = (child.op == "<=" and val <= float(child.threshold)) or (
            child.op == ">" and val > float(child.threshold)
        )
        if ok:
            matched = True
            _collect_matching_leaves(child, x, depth + 1, out)
    if not matched and "__leaf__" in node.children:
        out.append((depth, node.children["__leaf__"]))


def rule_tree_to_dict(node: RuleTreeNode) -> Dict[str, Any]:
    """
    将规则树节点递归序列化为 JSON 友好结构。

    参数:
        node: 规则树节点。

    返回:
        可供前端渲染树形图的字典。
    """
    base: Dict[str, Any] = {
        "node_id": node.node_id,
        "node_type": node.node_type,
        "has_overlap_conflict": node.has_overlap_conflict,
    }
    if node.node_type == "split":
        base.update(
            {
                "feature_idx": node.feature_idx,
                "feature_name": node.feature_name,
                "op": node.op,
                "threshold": node.threshold,
                "threshold_text": node.threshold_text,
                "children": {
                    edge: rule_tree_to_dict(ch)
                    for edge, ch in node.children.items()
                    if edge != "__leaf__"
                },
            }
        )
        if "__leaf__" in node.children:
            base["default_leaf"] = rule_tree_to_dict(node.children["__leaf__"])
    elif node.node_type == "leaf":
        base.update(
            {
                "action": str(node.action) if node.action is not None else None,
                "action_text": node.action_text,
                "support": node.support,
                "confidence": round(float(node.confidence), 4),
                "rule_indices": list(node.rule_indices),
            }
        )
    else:
        base["children"] = {
            edge: rule_tree_to_dict(ch) for edge, ch in node.children.items()
        }
    return base


def build_rule_tree_from_sklearn_tree(
    tree: Any,
    feature_names: List[str],
    preprocessor: Optional[Any] = None,
    max_depth: Optional[int] = None,
) -> RuleTreeBuildResult:
    """
    直接从 sklearn 决策树构建规则树（保证是二叉决策树结构）。

    用于希望「合并后仍看树」时，优先展示模型原生树，而不是 merge 后的扁平规则。

    参数:
        tree: 已训练的 ``DecisionTreeClassifier``。
        feature_names: 特征名列表。
        preprocessor: 可选预处理器。
        max_depth: 展示用的最大深度；``None`` 表示不截断。

    返回:
        ``RuleTreeBuildResult``，``tree_kind`` 恒为 ``decision_tree_equivalent``。
    """
    from sklearn.tree import _tree

    t = tree.tree_
    conflicts: List[str] = []

    def _build(node_id: int, depth: int) -> RuleTreeNode:
        """递归将 sklearn 树节点转为 ``RuleTreeNode``。"""
        left = int(t.children_left[node_id])
        is_leaf = left == _tree.TREE_LEAF
        if is_leaf or (max_depth is not None and depth >= max_depth):
            vals = t.value[node_id][0]
            bi = int(vals.argmax())
            support = int(vals.sum())
            conf = float(vals[bi] / support) if support > 0 else 0.0
            action = tree.classes_[bi]
            return RuleTreeNode(
                node_id=f"n{node_id}",
                node_type="leaf",
                action=action,
                action_text=_format_action_short(action),
                support=support,
                confidence=conf,
                rule_indices=[node_id],
            )

        feat_idx = int(t.feature[node_id])
        th = float(t.threshold[node_id])
        cond_le = RuleCondition(feat_idx, "<=", th)
        cond_gt = RuleCondition(feat_idx, ">", th)
        split_node = RuleTreeNode(
            node_id=f"n{node_id}",
            node_type="split",
            feature_idx=feat_idx,
            feature_name=_feat_name(feat_idx, feature_names),
            op="<=",
            threshold=th,
            threshold_text=_format_threshold(cond_le, feature_names, preprocessor),
        )
        split_node.children["yes"] = _build(left, depth + 1)
        split_node.children["no"] = _build(int(t.children_right[node_id]), depth + 1)
        return split_node

    root = _build(0, 0)
    return RuleTreeBuildResult(
        tree=root,
        tree_kind="decision_tree_equivalent",
        is_decision_tree_compatible=True,
        overlap_conflicts=conflicts,
        merge_steps=[],
    )


def merge_steps_to_dict(steps: List[MergeStepReport]) -> List[Dict[str, Any]]:
    """
    将合并步骤报告列表转为 JSON 友好字典列表。

    参数:
        steps: ``MergeStepReport`` 列表。

    返回:
        含合并前后数量与被删规则详情的字典列表。
    """
    return [
        {
            "name": s.name,
            "before_count": s.before_count,
            "after_count": s.after_count,
            "removed_count": len(s.removed_rules),
            "removed_rules": s.removed_rules,
        }
        for s in steps
    ]


def export_rule_tree_pdf(
    build_result: RuleTreeBuildResult,
    out_path: str,
    title: str = "",
) -> str:
    """
    将规则树导出为 PDF（通过 Graphviz 渲染）。

    参数:
        build_result: 规则树构建结果。
        out_path: 输出路径（可带或不带 ``.pdf`` 后缀）。
        title: PDF 标题。

    返回:
        实际生成的 PDF 文件路径字符串。
    """
    from src.viz.tree_plot import _choose_writable_output_path, _dot_escape, _render_dot_file

    body_lines: List[str] = []
    _emit_rule_tree_dot(build_result.tree, body_lines, counter=[0])
    n_leaves = max(1, _count_leaves(build_result.tree))
    nodesep = 0.9 if n_leaves <= 8 else 1.2
    ranksep = 1.2 if n_leaves <= 8 else 1.6
    dot = f'''digraph RuleTree {{
    charset="UTF-8";
    labelloc="t";
    label="{_dot_escape(title)}";
    fontname="Microsoft YaHei";
    graph [rankdir=TB, nodesep={nodesep}, ranksep={ranksep}];
    node [shape=box, style="rounded,filled", fontname="Microsoft YaHei", fontsize=10];
    edge [fontname="Microsoft YaHei", fontsize=9];
{chr(10).join(body_lines)}
}}
'''
    out_base = Path(out_path)
    if out_base.suffix.lower() == ".pdf":
        out_base = out_base.with_suffix("")
    out_file = _render_dot_file(dot, out_base, "pdf")
    return out_file


def _count_leaves(node: RuleTreeNode) -> int:
    """
    递归统计规则树的叶节点数量。

    参数:
        node: 规则树节点。

    返回:
        以该节点为根的子树中叶节点总数。
    """
    if node.node_type == "leaf":
        return 1
    n = 0
    for ch in node.children.values():
        n += _count_leaves(ch)
    return n


def _emit_rule_tree_dot(
    node: RuleTreeNode,
    lines: List[str],
    counter: List[int],
    parent_id: Optional[str] = None,
    edge_label: Optional[str] = None,
) -> str:
    """
    递归生成 Graphviz DOT 格式的节点与边定义行。

    参数:
        node: 当前规则树节点。
        lines: DOT 行列表（就地追加）。
        counter: 单元素列表，用作节点 ID 递增计数器。
        parent_id: 父节点 DOT ID。
        edge_label: 父到当前节点的边标签。

    返回:
        当前节点的 DOT 节点 ID。
    """
    nid = f"rt{counter[0]}"
    counter[0] += 1

    if node.node_type == "split":
        label = node.threshold_text or f"{node.feature_name} {node.op} {node.threshold}"
        if node.has_overlap_conflict:
            label += "\\n（分支可能重叠）"
        lines.append(f'    {nid} [label="{_dot_escape_simple(label)}"];')
    else:
        label = (
            f"主导动作 = {node.action_text or node.action}\\n"
            f"samples = {node.support}\\n"
            f"置信度 = {node.confidence:.1%}"
        )
        lines.append(f'    {nid} [label="{_dot_escape_simple(label)}"];')

    if parent_id is not None and edge_label is not None:
        lines.append(f'    {parent_id} -> {nid} [label="{_dot_escape_simple(edge_label)}"];')

    if node.node_type == "split":
        yes = node.children.get("yes")
        no = node.children.get("no")
        if yes is not None and no is not None:
            _emit_rule_tree_dot(yes, lines, counter, nid, "是")
            _emit_rule_tree_dot(no, lines, counter, nid, "否")
        else:
            for edge, ch in node.children.items():
                if edge == "__leaf__":
                    _emit_rule_tree_dot(ch, lines, counter, nid, "默认")
                    continue
                lbl = "是" if "<=" in edge else "否"
                _emit_rule_tree_dot(ch, lines, counter, nid, lbl)
    return nid


def _dot_escape_simple(text: str) -> str:
    """
    对 DOT 标签文本做简单转义。

    参数:
        text: 原始标签文本。

    返回:
        转义反斜杠与双引号后的安全字符串。
    """
    return text.replace("\\", "\\\\").replace('"', '\\"')
