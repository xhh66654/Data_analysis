"""
规则集成模块：利用多轮 VIPER 结果提升准确率。

核心思想：
  1. 收集多轮 VIPER 生成的规则
  2. 通过投票机制融合多轮规则
  3. 只保留高置信度、高频出现的规则
  4. 限制最终规则数量，保持可读性

主要类型与函数：
  RuleEnsemble        — 规则集成器类
  RuleInfo            — 单条规则信息
  ensemble_rules_from_rounds() — 从多轮规则列表集成
  prune_rules()       — 规则剪枝（按置信度、覆盖度等）
  merge_similar_rules() — 合并相似规则
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RuleInfo:
    """单条规则的详细信息"""
    conditions: tuple  # 条件元组（已排序去重）
    action: int | str  # 动作编号或名称
    confidence: float = 1.0  # 置信度（0-1）
    support_count: int = 1  # 支持轮数
    coverage: float = 0.0  # 覆盖样本比例
    source_rounds: list[int] = field(default_factory=list)  # 来源轮次


class RuleEnsemble:
    """规则集成器：融合多轮 VIPER 规则"""
    
    def __init__(self, voting_method: str = 'weighted', confidence_threshold: float = 0.7):
        """
        参数：
            voting_method: 投票方式 'weighted'（加权）或 'majority'（多数）
            confidence_threshold: 置信度阈值，低于此值的规则被过滤
        """
        self.voting_method = voting_method
        self.confidence_threshold = confidence_threshold
        self.rules_by_condition = defaultdict(list)  # {条件键: [规则信息列表]}
    
    def add_rules(self, rules: list[dict], round_idx: int = 0, round_weight: float = 1.0):
        """
        添加一轮规则
        
        参数：
            rules: 规则列表，每条规则为 {'conditions': ..., 'action': ..., 'confidence': ...}
            round_idx: 轮次索引
            round_weight: 本轮权重（可基于准确率设置）
        """
        for rule in rules:
            # 将条件转换为可哈希的键（排序去重）
            cond_key = tuple(sorted(rule['conditions']))
            
            self.rules_by_condition[cond_key].append({
                'action': rule['action'],
                'weight': round_weight,
                'confidence': rule.get('confidence', 1.0),
                'round_idx': round_idx,
            })
    
    def ensemble(self, max_rules: int = 100, min_support: int = 2) -> list[RuleInfo]:
        """
        集成多轮规则
        
        参数：
            max_rules: 最终规则数上限
            min_support: 最少支持轮数
        
        返回：
            集成后的规则列表（按置信度×支持度排序）
        """
        final_rules = []
        
        for cond_key, action_list in self.rules_by_condition.items():
            # 筛选高置信度规则
            valid_actions = [
                a for a in action_list 
                if a['confidence'] >= self.confidence_threshold
            ]
            
            # 至少需要 min_support 轮支持
            if len(valid_actions) < min_support:
                continue
            
            # 投票决策
            if self.voting_method == 'weighted':
                # 加权投票：权重 × 置信度
                action_scores = defaultdict(float)
                for a in valid_actions:
                    action_scores[a['action']] += a['weight'] * a['confidence']
                final_action = max(action_scores, key=action_scores.get)
                confidence = max(a['confidence'] for a in valid_actions)
            
            else:  # majority voting
                # 多数投票
                actions = [a['action'] for a in valid_actions]
                final_action = Counter(actions).most_common(1)[0][0]
                confidence = sum(1 for a in valid_actions if a['action'] == final_action) / len(valid_actions)
            
            # 收集来源轮次
            source_rounds = sorted(set(a['round_idx'] for a in valid_actions))
            
            final_rules.append(RuleInfo(
                conditions=cond_key,
                action=final_action,
                confidence=confidence,
                support_count=len(valid_actions),
                source_rounds=source_rounds,
            ))
        
        # 按置信度×支持度排序，取前 max_rules
        final_rules.sort(
            key=lambda r: r.confidence * r.support_count,
            reverse=True
        )
        
        logger.info(
            "规则集成完成: 原始规则数=%d 集成后规则数=%d 置信度范围=[%.4f, %.4f]",
            sum(len(v) for v in self.rules_by_condition.values()),
            len(final_rules[:max_rules]),
            min(r.confidence for r in final_rules) if final_rules else 0,
            max(r.confidence for r in final_rules) if final_rules else 0,
        )
        
        return final_rules[:max_rules]

    def ensemble_by_feature_pattern(self, max_rules: int = 100, min_support: int = 2) -> list[RuleInfo]:
        """
        按特征模式集成规则：相同模式（特征+操作符）的规则合并，阈值取中位数

        参数：
            max_rules: 最终规则数上限
            min_support: 最少支持轮数

        返回：
            集成后的规则列表
        """
        # Step 1: 按 {模式键 + 动作} 分组
        pattern_groups: dict[tuple, list[dict]] = defaultdict(list)

        for cond_key, action_list in self.rules_by_condition.items():
            pattern_key = _conditions_to_pattern_key(cond_key)
            if not pattern_key:
                continue

            for a in action_list:
                if a['confidence'] < self.confidence_threshold:
                    continue
                pattern_groups[(pattern_key, a['action'])].append({
                    'cond_key': cond_key,
                    'pattern_key': pattern_key,
                    'action': a['action'],
                    'weight': a['weight'],
                    'confidence': a['confidence'],
                    'round_idx': a['round_idx'],
                })

        # Step 2: 对每个模式+动作组，投票选动作，取中位数阈值
        final_rules = []
        for (pattern_key, action), group in pattern_groups.items():
            if len(group) < min_support:
                continue

            # 收集每个特征的阈值
            feature_thresholds: dict[str, list[float]] = defaultdict(list)
            all_source_rounds = set()

            for entry in group:
                for cond_str in entry['cond_key']:
                    f, op, v = extract_condition_parts(cond_str)
                    if f:
                        feature_thresholds[f'{f}{op}'].append(v)
                all_source_rounds.add(entry['round_idx'])

            # 取中位数阈值
            median_conditions = []
            for feature_op, thresholds in sorted(feature_thresholds.items()):
                median_val = np.median(thresholds)
                median_conditions.append(f'{feature_op} {round(float(median_val), 4)}')

            # 计算置信度和支持数
            if self.voting_method == 'weighted':
                confidence = sum(e['weight'] * e['confidence'] for e in group) / sum(e['weight'] for e in group)
            else:
                confidence = max(e['confidence'] for e in group)

            final_rules.append(RuleInfo(
                conditions=tuple(sorted(median_conditions)),
                action=action,
                confidence=round(float(confidence), 4),
                support_count=len(group),
                source_rounds=sorted(all_source_rounds),
            ))

        # 按置信度×支持度排序
        final_rules.sort(key=lambda r: r.confidence * r.support_count, reverse=True)

        logger.info(
            "规则集成（模式去重）完成: 原始规则数=%d 集成后规则数=%d 置信度范围=[%.4f, %.4f]",
            sum(len(v) for v in self.rules_by_condition.values()),
            len(final_rules[:max_rules]),
            min(r.confidence for r in final_rules) if final_rules else 0,
            max(r.confidence for r in final_rules) if final_rules else 0,
        )

        return final_rules[:max_rules]


def ensemble_rules_from_rounds(
    rules_per_round: list[dict],
    max_rules: int = 100,
    confidence_threshold: float = 0.7,
    min_support_rounds: int = 2,
    use_feature_pattern: bool = True,
) -> list[RuleInfo]:
    """
    从多轮规则列表集成

    参数：
        rules_per_round: 每轮规则的列表，格式为 [{'rules': [...], 'acc_full': ...}, ...]
        max_rules: 最终规则数上限
        confidence_threshold: 置信度阈值
        min_support_rounds: 最少支持轮数
        use_feature_pattern: 是否按特征模式去重（取中位数阈值）

    返回：
        集成后的规则列表
    """
    ensemble = RuleEnsemble(
        voting_method='weighted',
        confidence_threshold=confidence_threshold,
    )

    for round_idx, round_data in enumerate(rules_per_round):
        weight = round_data.get('acc_full', 1.0)
        rules = round_data.get('rules', [])

        formatted_rules = []
        for rule in rules:
            if isinstance(rule, str):
                action = rule.split('THEN')[-1].strip() if 'THEN' in rule else 'unknown'
                formatted_rules.append({
                    'conditions': parse_rule_conditions(rule),
                    'action': action,
                    'confidence': weight,
                })
            elif isinstance(rule, dict):
                formatted_rules.append(rule)

        ensemble.add_rules(formatted_rules, round_idx=round_idx, round_weight=weight)

    if use_feature_pattern:
        result = ensemble.ensemble_by_feature_pattern(max_rules=max_rules, min_support=min_support_rounds)
    else:
        result = ensemble.ensemble(max_rules=max_rules, min_support=min_support_rounds)

    if len(result) == 0 and min_support_rounds > 1:
        logger.warning('规则集成结果为0（min_support=%d），回退到 min_support=1 去重', min_support_rounds)
        if use_feature_pattern:
            result = ensemble.ensemble_by_feature_pattern(max_rules=max_rules, min_support=1)
        else:
            result = ensemble.ensemble(max_rules=max_rules, min_support=1)

    if len(result) == 0:
        logger.warning('规则集成仍为0，可能 confidence_threshold=%.2f 高于所有轮次准确率', confidence_threshold)

    return result


def parse_rule_conditions(rule_str: str, round_digits: int = 4) -> tuple:
    """
    从规则字符串中提取条件部分，对阈值进行四舍五入以支持跨轮去重
    
    参数：
        rule_str: 规则字符串，如 "IF s_0 > 0.5 AND s_1 < 0.3 THEN action_1"
        round_digits: 阈值保留的小数位数，默认4位
    
    返回：
        条件元组，如 ('s_0 > 0.5', 's_1 < 0.3')
    """
    import re

    rule_str = rule_str.strip()

    if rule_str.startswith('IF '):
        rule_str = rule_str[3:]

    if ' THEN ' in rule_str:
        rule_str = rule_str.split(' THEN ')[0]

    conditions = []
    for c in rule_str.split(' AND '):
        c = c.strip()
        if not c:
            continue
        # 对数值阈值进行四舍五入（只匹配带小数点的数字）
        c = re.sub(
            r'(-?\d+\.\d+)',
            lambda m: f'{round(float(m.group(1)), round_digits)}',
            c,
        )
        conditions.append(c)

    return tuple(conditions)


def parse_rule_pattern(rule_str: str) -> tuple:
    """
    提取规则的特征模式签名（只保留特征名和操作符，忽略阈值）

    参数：
        rule_str: 规则字符串，如 "IF s_0 > 0.5 AND s_1 < 0.3 THEN action_1"

    返回：
        模式元组，如 ('s_0>', 's_1<')
    """
    import re

    rule_str = rule_str.strip()
    if rule_str.startswith('IF '):
        rule_str = rule_str[3:]
    if ' THEN ' in rule_str:
        rule_str = rule_str.split(' THEN ')[0]

    patterns = []
    for c in rule_str.split(' AND '):
        c = c.strip()
        if not c:
            continue
        m = re.match(r'(s_\d+)\s*(<=|<|>=|>)', c)
        if m:
            patterns.append(m.group(1) + m.group(2))

    return tuple(sorted(patterns))


def extract_condition_parts(cond_str: str) -> tuple[str | None, str | None, float | None]:
    """
    从条件字符串中提取 (特征名, 操作符, 阈值)

    参数：
        cond_str: 如 "s_0 <= 0.5"

    返回：
        (feature, op, value) 或 (None, None, None)
    """
    import re
    m = re.match(r'(s_\d+)\s*(<=|<|>=|>)\s*(-?\d+\.?\d*)', cond_str.strip())
    if m:
        return m.group(1), m.group(2), float(m.group(3))
    return None, None, None


def _conditions_to_pattern_key(conditions: tuple) -> tuple:
    """
    将条件元组转换为模式键（特征+操作符）

    参数：
        conditions: 条件元组，如 ('s_0 <= 0.5', 's_1 > 0.3')

    返回：
        模式键，如 ('s_0<=', 's_1>')
    """
    return tuple(
        f + o
        for f, o, _ in (extract_condition_parts(c) for c in conditions)
        if f is not None
    )


def prune_rules(
    rules: list[RuleInfo],
    max_rules: int = 100,
    min_confidence: float = 0.7,
    min_coverage: float = 0.0,
) -> list[RuleInfo]:
    """
    规则剪枝：按多种条件筛选
    
    参数：
        rules: 规则列表
        max_rules: 规则数上限
        min_confidence: 最小置信度
        min_coverage: 最小覆盖度
    
    返回：
        剪枝后的规则列表
    """
    # 按置信度筛选
    filtered = [r for r in rules if r.confidence >= min_confidence]
    
    # 按覆盖度筛选
    if min_coverage > 0:
        filtered = [r for r in filtered if r.coverage >= min_coverage]
    
    # 按置信度×支持度排序
    filtered.sort(key=lambda r: r.confidence * r.support_count, reverse=True)
    
    return filtered[:max_rules]


def merge_similar_rules(rules: list[RuleInfo], similarity_threshold: float = 0.8) -> list[RuleInfo]:
    """
    合并相似规则（简化实现）
    
    参数：
        rules: 规则列表
        similarity_threshold: 相似度阈值
    
    返回：
        合并后的规则列表
    """
    merged = []
    used = set()
    
    for i, rule1 in enumerate(rules):
        if i in used:
            continue
        
        # 找相似规则
        similar = [rule1]
        used.add(i)
        
        for j, rule2 in enumerate(rules):
            if j in used:
                continue
            
            # 简单相似度计算：条件集合交集比例
            conds1 = set(rule1.conditions)
            conds2 = set(rule2.conditions)
            if not conds1 or not conds2:
                continue
            
            intersection = len(conds1 & conds2)
            union = len(conds1 | conds2)
            similarity = intersection / union if union > 0 else 0
            
            if similarity >= similarity_threshold and rule1.action == rule2.action:
                similar.append(rule2)
                used.add(j)
        
        # 合并相似规则
        if len(similar) > 1:
            # 保留置信度最高的规则作为代表
            merged.append(max(similar, key=lambda r: r.confidence))
            logger.debug(f"合并了 {len(similar)} 条相似规则")
        else:
            merged.append(rule1)
    
    return merged


def rules_to_if_then_strings(rules: list[RuleInfo], class_mapping: dict | None = None) -> list[str]:
    """
    将规则转换为 IF-THEN 字符串格式
    
    参数：
        rules: 规则列表
        class_mapping: 动作编号到名称的映射
    
    返回：
        IF-THEN 规则字符串列表
    """
    result = []
    
    for rule in rules:
        # 构建条件部分
        cond_parts = []
        for cond in rule.conditions:
            if isinstance(cond, str):
                cond_parts.append(cond)
            elif isinstance(cond, tuple) and len(cond) == 3:
                # (特征名, 操作符, 阈值)
                feature, op, value = cond
                cond_parts.append(f"{feature} {op} {value}")
            else:
                cond_parts.append(str(cond))
        
        conditions = " AND ".join(cond_parts)
        
        # 获取动作名称
        action = rule.action
        if class_mapping and isinstance(action, (int, np.integer)):
            action = class_mapping.get(action, str(action))
        
        # 添加置信度信息
        rounds = rule.source_rounds or []
        delta = f" [置信度={rule.confidence:.2f}, 出现={rule.support_count}次/{len(rounds)}轮]"
        
        result.append(f"IF {conditions} THEN {action}{delta}")
    
    return result


def save_ensemble_rules(rules: list[RuleInfo], output_path: str | Path, class_mapping: dict | None = None):
    """
    保存集成规则到文件
    
    参数：
        rules: 规则列表
        output_path: 输出文件路径
        class_mapping: 动作编号到名称的映射
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为 IF-THEN 格式
    if_then_rules = rules_to_if_then_strings(rules, class_mapping)
    output_path.with_suffix('.txt').write_text(
        '\n'.join(if_then_rules) + '\n',
        encoding='utf-8'
    )
    
    # 保存为 JSON 格式（包含详细信息）
    rules_json = []
    for rule in rules:
        rules_json.append({
            'conditions': list(rule.conditions),
            'action': rule.action,
            'confidence': rule.confidence,
            'support_count': rule.support_count,
            'source_rounds': rule.source_rounds,
            'coverage': rule.coverage,
        })
    
    output_path.with_suffix('.json').write_text(
        json.dumps(rules_json, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    
    logger.info("已保存集成规则: %s (%d 条)", output_path.resolve(), len(rules))