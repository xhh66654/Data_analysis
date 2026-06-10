"""
规则提取：从训练好的决策树中提取 IF-THEN 规则集。

================================================================================
说明（小白友好版）：
================================================================================

图中步骤5：
    "提取规则，形成规则集"
    "规则抽取就是先训练决策树，然后根据决策树，从根到叶子，拿到规则集"

决策树的每一条从根节点到叶节点的路径就是一条规则：

    根节点
    ├── 敌机距离 <= 50km
    │   ├── 自身血量 > 0.5  →  叶子：动作=发射导弹（支持度=3，置信度=100%）
    │   └── 自身血量 <= 0.5 →  叶子：动作=机动规避（支持度=2，置信度=100%）
    └── 敌机距离 > 50km
        └── ...              →  叶子：动作=开启雷达（支持度=2，置信度=100%）

对应提取出的规则就是：
    规则1：如果 敌机距离 <= 较近 且 自身血量 > 良好：→ 发射导弹（置信度: 1.00）
    规则2：如果 敌机距离 <= 较近 且 自身血量 <= 良好：→ 机动规避（置信度: 1.00）
    规则3：如果 敌机距离 > 较近：→ 开启雷达（置信度: 1.00）

算法：DFS（深度优先搜索）从根遍历到每个叶节点，
      沿途收集条件（左子树=满足条件，右子树=不满足条件），
      到达叶子时把这条路径上的所有条件组合成一条规则。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import TREE_UNDEFINED, TREE_LEAF

from src.module_a_rules.preprocess import Preprocessor


# ==============================================================================
# 数据结构
# ==============================================================================

@dataclass
class RuleCondition:
    """
    一条规则中的单个条件：某特征 满足/不满足 某阈值。

    例如："敌机距离 <= 50.0"  或  "自身血量 > 0.5"

    Attributes
    ----------
    feature_idx : 特征在观测向量中的列索引
    op          : 比较运算符，"<=" 表示左子树，">" 表示右子树
    threshold   : 决策树节点的分裂阈值（归一化后的值）
    """
    feature_idx: int
    op: str          # "<=" 或 ">"
    threshold: float

    def to_text(self, pre: Optional[Preprocessor] = None) -> str:
        """
        把这个条件翻译成人类可读的文字。

        如果提供了预处理器（pre），会把阈值还原到原始单位，
        并用语义标签表示（例如"较近"而不是"50.0"）。

        参数:
            pre: 可选预处理器，用于反归一化与语义标签。

        返回:
            可读的条件描述字符串。

        示例：
            无预处理器：敌机距离 <= 0.342
            有预处理器：敌机距离 <= 较近（50.0km）
        """
        if pre is not None:
            feat_name = pre.get_feature_name(self.feature_idx)
            # 把归一化后的阈值还原到原始单位
            raw_thresh = pre.denormalize_threshold(self.feature_idx, self.threshold)
            # 获取语义标签
            label = pre.discretize_label(feat_name, raw_thresh)
            return f"{feat_name} {self.op} {label}（{raw_thresh:.2f}）"
        else:
            return f"特征{self.feature_idx} {self.op} {self.threshold:.4f}"


@dataclass
class Rule:
    """
    一条完整的 IF-THEN 规则。

    表示：如果所有条件都满足，则执行对应动作。

    Attributes
    ----------
    conditions  : 条件列表，每个条件是一个 RuleCondition
    action      : 动作（决策树叶节点中样本数最多的类别）
    support     : 支持度（该叶节点覆盖的训练样本数）
    confidence  : 置信度（该动作在叶节点中的纯度，0~1）
                  置信度=1.0 表示该叶节点所有样本都选这个动作
    """
    conditions: List[RuleCondition] = field(default_factory=list)
    action: object = None
    support: int = 0
    confidence: float = 0.0

    def to_text(
        self,
        pre: Optional[Preprocessor] = None,
        action_names: Optional[Dict[int, str]] = None,
    ) -> str:
        """
        把这条规则格式化为 "IF 条件1 AND 条件2 ... THEN 动作" 的可读文字。

        参数:
            pre: 预处理器，用于翻译特征阈值。
            action_names: 动作名称映射 ``{动作索引: 动作名}``；
                若 ``action`` 已是字符串则忽略。

        返回:
            可读的 IF-THEN 规则字符串。

        示例：
            IF 敌机距离 <= 较近（50.0km） AND 自身血量 > 良好（0.50）
            THEN 发射导弹
            （支持度: 3，置信度: 1.00）
        """
        # 翻译每个条件
        if self.conditions:
            cond_parts = [c.to_text(pre) for c in self.conditions]
            cond_str = "\n     AND ".join(cond_parts)
            prefix = f"IF {cond_str}"
        else:
            prefix = "IF 无条件（默认规则）"

        # 翻译动作
        if isinstance(self.action, str):
            action_str = self.action
        elif action_names and self.action in action_names:
            action_str = action_names[self.action]
        else:
            action_str = str(self.action)

        return (
            f"{prefix}\n"
            f"THEN {action_str}\n"
            f"（支持度: {self.support}，置信度: {self.confidence:.2f}）"
        )


# ==============================================================================
# 步骤5：从决策树 DFS 提取规则
# ==============================================================================

def extract_rules_from_tree(
    tree: DecisionTreeClassifier,
    preprocessor: Optional[Preprocessor] = None,
) -> List[Rule]:
    """
    深度优先遍历 sklearn 决策树，提取所有从根到叶的 IF-THEN 规则。

    图中步骤5：
        "提取规则，形成规则集"

    算法思路（DFS）：
        从根节点出发，递归地往下走：
        - 向左走（<=）：把 "特征 <= 阈值" 加入当前路径的条件列表
        - 向右走（>）：把 "特征 > 阈值" 加入当前路径的条件列表
        - 到达叶节点：把当前路径上的所有条件打包成一条 Rule

    Parameters
    ----------
    tree         : 已训练好的 sklearn DecisionTreeClassifier
    preprocessor : 预处理器（可选，用于在规则中显示原始单位和语义标签）

    Returns
    -------
    rules : Rule 列表，按支持度从大到小排序（覆盖样本多的规则排在前面）
    """
    rules: List[Rule] = []

    # 递归 DFS 函数
    def _dfs(node_id: int, path: List[RuleCondition]) -> None:
        """
        深度优先遍历决策树节点，到达叶节点时提取规则。

        参数:
            node_id: 当前树节点索引。
            path: 从根节点到当前节点经过的所有条件列表。
        """
        left_child  = tree.tree_.children_left[node_id]
        right_child = tree.tree_.children_right[node_id]

        # 判断是否为叶节点：sklearn 用 TREE_LEAF（值为 -1）标记叶节点的子节点
        # 注意：TREE_UNDEFINED = -2（用于特征索引），TREE_LEAF = -1（用于子节点）
        is_leaf = (left_child == TREE_LEAF)

        if is_leaf:
            # ---- 到达叶节点：提取这条规则 ----
            # tree_.value[node_id] 形如 [[n_class0, n_class1, ...]]
            node_value = tree.tree_.value[node_id][0]
            best_class_idx = int(np.argmax(node_value))          # 多数类的索引
            action = tree.classes_[best_class_idx]               # 对应的动作
            support = int(node_value.sum())                      # 总样本数
            confidence = float(node_value[best_class_idx] / support) if support > 0 else 0.0

            rules.append(Rule(
                conditions=list(path),   # 复制当前路径的条件列表
                action=action,
                support=support,
                confidence=confidence,
            ))
            return

        # ---- 非叶节点：取出分裂特征和阈值，继续往下递归 ----
        feat_idx  = int(tree.tree_.feature[node_id])
        threshold = float(tree.tree_.threshold[node_id])

        # 向左子树走：特征值 <= 阈值
        _dfs(
            left_child,
            path + [RuleCondition(feat_idx, "<=", threshold)],
        )
        # 向右子树走：特征值 > 阈值
        _dfs(
            right_child,
            path + [RuleCondition(feat_idx, ">", threshold)],
        )

    # 从根节点（索引=0）开始 DFS，初始路径为空
    _dfs(node_id=0, path=[])

    # 按支持度从大到小排序（覆盖更多样本的规则更可靠，排前面）
    rules.sort(key=lambda r: r.support, reverse=True)
    return rules


def rules_to_text(
    rules: List[Rule],
    preprocessor: Optional[Preprocessor] = None,
    action_names: Optional[Dict[int, str]] = None,
    top_k: Optional[int] = None,
) -> str:
    """
    把规则列表转成完整的可读文本，便于展示和调试。

    Parameters
    ----------
    rules        : 规则列表
    preprocessor : 预处理器（用于翻译特征阈值）
    action_names : 动作名称映射
    top_k        : 只展示前 k 条（支持度最高的），None=全部

    Returns
    -------
    多行文本字符串，每条规则之间用分割线隔开
    """
    display_rules = rules[:top_k] if top_k else rules
    lines = [f"共提取到 {len(rules)} 条规则，展示前 {len(display_rules)} 条：\n"]
    for i, rule in enumerate(display_rules, start=1):
        lines.append(f"【规则 {i}】")
        lines.append(rule.to_text(pre=preprocessor, action_names=action_names))
        lines.append("-" * 50)
    return "\n".join(lines)
