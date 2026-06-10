"""
模块 A：基于规则抽取的智能溯因（近似 VIPER + 决策树）。

流程：
    1. collect_data    从 InferenceRecord 提取 (观测, 动作, 奖励) 样本
    2. preprocess      量纲归一化 + 离散化 + 阈值反向还原
    3. viper           迭代训练带 return-to-go 权重的 CART 决策树
    4. extract_rules   DFS 把树转成 IF-THEN 规则 + 语义标签翻译
    5. merge_rules     合并冗余规则 + 覆盖率评估
"""
from .collect_data import (
    collect_from_record,
    collect_from_records,
    collect_multi_agent,
    compute_return_to_go,
)
# 注意：新版 collect_from_record 签名为 (record, agent_id, action_item)
# 返回 (X, y, rewards, feature_names)，比旧版多了 feature_names
from .preprocess import Preprocessor
from .agent_profile import (
    AgentPreprocessorProfile,
    fit_preprocessor_with_profile,
    load_profile,
    profile_id_for,
    schema_fingerprint,
)
from .viper import VIPERData, VIPERResult
from .rule_match import (
    load_rules_json,
    match_rules,
    predict_from_rules,
    save_rules_json,
)
from .pipeline import run_rule_extraction_for_label
from .extract_rules import extract_rules_from_tree, rules_to_text, Rule, RuleCondition
from .merge_rules import merge_rules, rules_coverage
from .verify_tree_rules import TreeRulesVerification, verify_tree_and_rules
from .rule_tree import (
    RuleTreeBuildResult,
    RuleTreeNode,
    build_rule_tree,
    build_rule_tree_from_sklearn_tree,
    explain_rule_merge_steps,
    export_rule_tree_pdf,
    rule_tree_to_dict,
)

__all__ = [
    # collect_data
    "collect_from_record",
    "collect_from_records",
    "collect_multi_agent",
    "compute_return_to_go",
    # preprocess
    "Preprocessor",
    "AgentPreprocessorProfile",
    "fit_preprocessor_with_profile",
    "load_profile",
    "profile_id_for",
    "schema_fingerprint",
    "run_rule_extraction_for_label",
    "match_rules",
    "predict_from_rules",
    "save_rules_json",
    "load_rules_json",
    # viper
    "VIPERData",
    "VIPERResult",
    # extract_rules
    "extract_rules_from_tree",
    "rules_to_text",
    "Rule",
    "RuleCondition",
    # merge_rules
    "merge_rules",
    "rules_coverage",
    # verify
    "verify_tree_and_rules",
    "TreeRulesVerification",
    "build_rule_tree",
    "build_rule_tree_from_sklearn_tree",
    "explain_rule_merge_steps",
    "export_rule_tree_pdf",
    "rule_tree_to_dict",
    "RuleTreeBuildResult",
    "RuleTreeNode",
]
