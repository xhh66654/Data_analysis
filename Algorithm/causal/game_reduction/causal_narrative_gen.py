#!/usr/bin/env python3
"""
因果溯因自然语言解释生成模块。

阶段 1：机械式因果链生成（无需标注）
- 从反事实溯因 JSON 中提取关键因素
- 按动作向量变化量排序
- 生成定量的、结构化的因果链表述

后续阶段：见 NARRATIVE_GENERATION_ROADMAP.md
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CausalFactor:
    """单个因果因素的描述"""
    rank: int                              # 排序位置（1, 2, 3...）
    block_name: str                        # 父变量块名
    l2_impact: float                       # 动作向量变化量（L2距离）
    intensity_level: str                   # 影响强度等级（"无关", "轻微", "中等", "强烈"）
    strongest_behavior: str                # 受影响最大的行为标签
    behavior_baseline_prob: float          # 基线概率
    behavior_after_removal_prob: float     # 移除后概率
    behavior_delta_p: float                # 概率变化
    cosine_similarity: float               # 预测方向余弦相似度
    mse_change: float                      # MSE 相对变化


def infer_intensity_level(
    l2_change: float,
    thresholds: list[float] | None = None
) -> str:
    """
    根据 L2 变化量推断影响强度等级。
    
    Args:
        l2_change: 动作向量的 L2 距离（干预前后）
        thresholds: 强度阈值，默认 [0, 0.3, 0.6, 1.0]
        
    Returns:
        强度等级：'无关', '轻微', '中等', '强烈'
    """
    if thresholds is None:
        thresholds = [0.0, 0.3, 0.6, 1.0]
    
    levels = ["无关", "轻微", "中等", "强烈"]
    
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= l2_change < thresholds[i + 1]:
            return levels[i]
    
    # 如果超过最后一个阈值
    return levels[-1]


def mechanical_explanation(
    cf_row: dict[str, Any],
    max_factors: int = 3,
    intensity_thresholds: list[float] | None = None
) -> str:
    """
    生成机械式因果链解释（阶段 1）。
    
    从单因子干预中提取关键因素，按影响大小排序，
    生成定量的、结构化的中文表述。
    
    Args:
        cf_row: counterfactual_abduction.json 中的单行数据
        max_factors: 最多提取几个关键因素（默认 3）
        intensity_thresholds: 强度等级阈值（默认 [0, 0.3, 0.6, 1.0]）
        
    Returns:
        格式化的中文解释文本（多行）
    
    Example:
        >>> cf_row = {...}  # 从 JSON 读取
        >>> explanation = mechanical_explanation(cf_row, max_factors=3)
        >>> print(explanation)
        1. 因素 s_mean__pooled_neighbors（强烈影响）：
           移除后动作向量变化 0.619；press-forward 概率下降 8.22%
        2. 因素 a_mean__pooled_neighbors（强烈影响）：
           移除后动作向量变化 0.822；maneuver 概率下降 5.04%
    """
    interventions = cf_row.get('single_block_interventions', [])
    
    if not interventions:
        return "（无单因子干预数据）"
    
    # 按 L2 变化量降序排序
    sorted_by_impact = sorted(
        interventions,
        key=lambda x: x.get('l2_pred_change', 0),
        reverse=True
    )[:max_factors]
    
    lines = []
    for rank, inter in enumerate(sorted_by_impact, 1):
        block = inter.get('block', 'unknown')
        impact = inter.get('l2_pred_change', 0.0)
        intensity = infer_intensity_level(impact, intensity_thresholds)
        
        behav_drop = inter.get('largest_behavior_prob_drop', {})
        behavior = behav_drop.get('behavior', 'unknown')
        delta_p = abs(behav_drop.get('delta_p', 0.0))
        
        cosine_sim = inter.get('cosine_similarity', 0.0)
        
        lines.append(
            f"{rank}. 因素 {block}（{intensity}影响）：\n"
            f"   动作向量变化 {impact:.3f}；方向相似度 {cosine_sim:.3f}；"
            f"{behavior} 行为概率下降 {delta_p:.2%}"
        )
    
    return "\n".join(lines)


def causal_chain_ranking(
    cf_row: dict[str, Any],
    max_factors: int = 5
) -> list[dict[str, Any]]:
    """
    生成结构化的因果链排序表（便于下游处理或 JSON 输出）。
    
    Args:
        cf_row: counterfactual_abduction.json 中的单行数据
        max_factors: 最多提取几个因素
        
    Returns:
        因果因素列表，按影响大小排序
    
    Example:
        >>> ranking = causal_chain_ranking(cf_row)
        >>> for factor in ranking:
        ...     print(f"{factor['rank']}: {factor['block']} - {factor['intensity_level']}")
    """
    interventions = cf_row.get('single_block_interventions', [])
    
    sorted_by_impact = sorted(
        interventions,
        key=lambda x: x.get('l2_pred_change', 0),
        reverse=True
    )[:max_factors]
    
    ranking = []
    for rank, inter in enumerate(sorted_by_impact, 1):
        block = inter.get('block', 'unknown')
        l2_impact = inter.get('l2_pred_change', 0.0)
        intensity = infer_intensity_level(l2_impact)
        
        behav_drop = inter.get('largest_behavior_prob_drop', {})
        
        factor = {
            "rank": rank,
            "block": block,
            "l2_impact": l2_impact,
            "intensity_level": intensity,
            "strongest_behavior_change": {
                "behavior": behav_drop.get('behavior', 'unknown'),
                "baseline_prob": behav_drop.get('baseline_p', 0.0),
                "after_removal_prob": behav_drop.get('counterfactual_p', 0.0),
                "delta_p": behav_drop.get('delta_p', 0.0)
            },
            "cosine_similarity": inter.get('cosine_similarity', 0.0),
            "mse_baseline": inter.get('mse_baseline_vs_true_action', 0.0),
            "mse_counterfactual": inter.get('mse_counterfactual_vs_true_action', 0.0)
        }
        ranking.append(factor)
    
    return ranking


def enrich_counterfactual_with_narrative(
    cf_json_path: str | Path,
    output_path: str | Path | None = None,
    max_factors: int = 3
) -> dict[str, Any]:
    """
    读取反事实溯因 JSON，补充因果链排序字段，写回。
    
    Args:
        cf_json_path: 原始 counterfactual_abduction.json 路径
        output_path: 输出路径（默认原地覆盖）
        max_factors: 每行最多提取的因素数
        
    Returns:
        增强后的 JSON 对象
    """
    cf_json_path = Path(cf_json_path)
    cf_data = json.loads(cf_json_path.read_text(encoding='utf-8'))
    
    # 为每行补充因果链排序
    for row in cf_data.get('rows', []):
        row['causal_chain_ranking'] = causal_chain_ranking(row, max_factors)
        row['mechanical_explanation'] = mechanical_explanation(row, max_factors)
    
    # 写出
    if output_path is None:
        output_path = cf_json_path
    else:
        output_path = Path(output_path)
    
    output_path.write_text(
        json.dumps(cf_data, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    
    return cf_data


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print(f"  python {sys.argv[0]} <counterfactual_abduction.json> [output.json]")
        sys.exit(1)
    
    cf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"正在处理 {cf_path}...")
    result = enrich_counterfactual_with_narrative(cf_path, output_path)
    
    out = output_path or cf_path
    print(f"已写入 {out}")
    
    # 示例打印第一行
    if result['rows']:
        row0 = result['rows'][0]
        print("\n=== 第 0 行示例 ===")
        print(row0.get('mechanical_explanation', ''))
