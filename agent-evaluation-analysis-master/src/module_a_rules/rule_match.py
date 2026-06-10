"""
规则匹配与本地持久化（按需遍历解释）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.module_a_rules.extract_rules import Rule, RuleCondition
from src.module_a_rules.merge_rules import _matches


def match_rules(
    rules: List[Rule],
    x: np.ndarray,
    *,
    preprocessor=None,
    x_raw: Optional[np.ndarray] = None,
    all_matching: bool = False,
) -> List[Rule]:
    """
    返回命中样本 x 的规则列表。

    参数:
        rules: 待匹配的规则列表。
        x: 归一化后的特征向量；若提供 ``x_raw`` 则忽略。
        preprocessor: 预处理器，配合 ``x_raw`` 做变换时使用。
        x_raw: 原始特征向量，需配合 ``preprocessor.transform``。
        all_matching: ``True`` 返回全部命中规则；``False`` 仅返回条件最多的一条。

    返回:
        命中的规则列表；无命中时返回空列表。
    """
    if x_raw is not None:
        if preprocessor is None:
            raise ValueError("match_rules 使用 x_raw 时必须提供 preprocessor")
        x = preprocessor.transform(np.asarray(x_raw, dtype=float).reshape(1, -1))[0]

    x = np.asarray(x, dtype=float).ravel()
    hits = [r for r in rules if _matches(r, x)]
    if not hits:
        return []
    if all_matching:
        return sorted(hits, key=lambda r: len(r.conditions), reverse=True)
    return [max(hits, key=lambda r: len(r.conditions))]


def predict_from_rules(
    rules: List[Rule],
    x: np.ndarray,
    *,
    preprocessor=None,
    x_raw: Optional[np.ndarray] = None,
) -> Optional[object]:
    """
    用最具体命中规则预测动作。

    参数:
        rules: 规则列表。
        x: 归一化后的特征向量。
        preprocessor: 预处理器，配合 ``x_raw`` 使用。
        x_raw: 原始特征向量。

    返回:
        命中规则的动作；无命中时返回 ``None``。
    """
    if x_raw is not None:
        if preprocessor is None:
            raise ValueError("predict_from_rules 使用 x_raw 时必须提供 preprocessor")
        x = preprocessor.transform(np.asarray(x_raw, dtype=float).reshape(1, -1))[0]
    hits = match_rules(rules, x, all_matching=True)
    return hits[0].action if hits else None


def rules_to_jsonable(rules: List[Rule], feature_names: List[str]) -> List[Dict[str, Any]]:
    """
    将规则列表转为可 JSON 序列化的字典列表。

    参数:
        rules: 规则对象列表。
        feature_names: 特征名列表，用于翻译 ``feature_idx``。

    返回:
        每条规则对应一个字典，含条件、动作、支持度与置信度。
    """
    out: List[Dict[str, Any]] = []
    for i, rule in enumerate(rules, start=1):
        conds = []
        for c in rule.conditions:
            fname = (
                feature_names[c.feature_idx]
                if 0 <= c.feature_idx < len(feature_names)
                else f"feature_{c.feature_idx}"
            )
            conds.append(
                {
                    "feature": fname,
                    "feature_idx": c.feature_idx,
                    "op": c.op,
                    "threshold_norm": float(c.threshold),
                }
            )
        out.append(
            {
                "rule_index": i,
                "conditions": conds,
                "action": str(rule.action),
                "support": int(rule.support),
                "confidence": float(rule.confidence),
            }
        )
    return out


def rules_from_jsonable(payload: List[Dict[str, Any]]) -> List[Rule]:
    """
    从 JSON 友好的字典列表还原规则对象。

    参数:
        payload: 由 ``rules_to_jsonable`` 或 JSON 文件产生的规则字典列表。

    返回:
        ``Rule`` 对象列表。
    """
    rules: List[Rule] = []
    for item in payload:
        conds = [
            RuleCondition(
                feature_idx=int(c["feature_idx"]),
                op=str(c["op"]),
                threshold=float(c["threshold_norm"]),
            )
            for c in item.get("conditions", [])
        ]
        rules.append(
            Rule(
                conditions=conds,
                action=item.get("action"),
                support=int(item.get("support", 0)),
                confidence=float(item.get("confidence", 0.0)),
            )
        )
    return rules


def save_rules_json(
    path: Union[str, Path],
    rules: List[Rule],
    *,
    feature_names: List[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    将规则集持久化为 JSON 文件。

    参数:
        path: 输出文件路径。
        rules: 规则对象列表。
        feature_names: 特征名列表。
        metadata: 可选元数据（任务 ID、准确率等）。

    返回:
        写入文件的 ``Path`` 对象。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "metadata": metadata or {},
        "feature_names": feature_names,
        "rules": rules_to_jsonable(rules, feature_names),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return path


def load_rules_json(path: Union[str, Path]) -> tuple[List[Rule], List[str], Dict[str, Any]]:
    """
    从 JSON 文件加载规则集。

    参数:
        path: 规则 JSON 文件路径。

    返回:
        三元组 ``(rules, feature_names, metadata)``。
    """
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    feature_names = list(doc.get("feature_names") or [])
    rules = rules_from_jsonable(doc.get("rules") or [])
    return rules, feature_names, dict(doc.get("metadata") or {})
