#!/usr/bin/env python3
"""
模型验证示例脚本。

用于验证三个代理模型（策略模型、转移模型、奖励模型）的拟合程度和正确率。
"""
import json
import sys
from pathlib import Path
from typing import Dict

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.module_c_counterfactual.data_loader import (
    list_inference_task_ids,
    load_inference_records,
)
from src.module_c_counterfactual.policy_model import PolicySurrogate
from src.module_c_counterfactual.reward_model import RewardModel
from src.module_c_counterfactual.transition_model import TransitionModel


def validate_all_models(
    inference_task_id: str,
    agent_id: int,
    val_size: float = 0.2,
) -> Dict[str, Dict]:
    """
    验证所有三个代理模型。

    参数:
        inference_task_id: 推理任务 ID。
        agent_id: 智能体编号。
        val_size: 验证集比例。

    返回:
        包含各模型验证结果的字典。
    """
    print(f"[INFO] 加载推理记录: {inference_task_id}")
    records = load_inference_records(inference_task_id)
    print(f"[INFO] 加载到 {len(records)} 条仿真记录")

    results = {}

    # 1. 验证策略模型
    print("\n[INFO] 训练并验证策略模型...")
    policy = PolicySurrogate(mode="composed")
    policy.fit_records(records, agent_id)

    # 使用第一条记录进行验证
    if records:
        policy_result = policy.validate(records[0], agent_id, val_size=val_size)
        results["policy_model"] = policy_result.to_dict()
        print(f"  训练损失: {policy_result.train_loss:.4f}")
        print(f"  验证损失: {policy_result.val_loss:.4f}")
        print(f"  训练准确率: {policy_result.train_metrics.get('accuracy', 0):.4f}")
        print(f"  验证准确率: {policy_result.val_metrics.get('accuracy', 0):.4f}")
        print(f"  训练 F1: {policy_result.train_metrics.get('f1_weighted', 0):.4f}")
        print(f"  验证 F1: {policy_result.val_metrics.get('f1_weighted', 0):.4f}")

    # 2. 验证转移模型
    print("\n[INFO] 训练并验证转移模型...")
    transition = TransitionModel()
    transition.fit(records, agent_id)

    transition_result = transition.validate(records, agent_id, val_size=val_size)
    results["transition_model"] = transition_result.to_dict()
    print(f"  训练损失 (MSE): {transition_result.train_loss:.6f}")
    print(f"  验证损失 (MSE): {transition_result.val_loss:.6f}")
    print(f"  训练 R²: {transition_result.train_metrics.get('r2', 0):.4f}")
    print(f"  验证 R²: {transition_result.val_metrics.get('r2', 0):.4f}")
    print(f"  训练 MAE: {transition_result.train_metrics.get('mae', 0):.4f}")
    print(f"  验证 MAE: {transition_result.val_metrics.get('mae', 0):.4f}")

    # 3. 验证奖励模型
    print("\n[INFO] 训练并验证奖励模型...")
    reward = RewardModel()
    reward.fit(records, agent_id)

    reward_result = reward.validate(records, agent_id, val_size=val_size)
    results["reward_model"] = reward_result.to_dict()
    print(f"  训练损失 (MSE): {reward_result.train_loss:.6f}")
    print(f"  验证损失 (MSE): {reward_result.val_loss:.6f}")
    print(f"  训练 R²: {reward_result.train_metrics.get('r2', 0):.4f}")
    print(f"  验证 R²: {reward_result.val_metrics.get('r2', 0):.4f}")
    print(f"  训练 MAE: {reward_result.train_metrics.get('mae', 0):.4f}")
    print(f"  验证 MAE: {reward_result.val_metrics.get('mae', 0):.4f}")

    return results


def main():
    """主函数。"""
    # 获取可用的任务 ID
    task_ids = list_inference_task_ids()
    if not task_ids:
        print("[ERROR] 未找到推理任务，请先生成模拟数据")
        return

    # 使用第一个任务进行验证
    task_id = task_ids[0]
    agent_id = 1

    print(f"=" * 60)
    print(f"模型验证示例")
    print(f"任务 ID: {task_id}")
    print(f"智能体 ID: {agent_id}")
    print(f"=" * 60)

    # 执行验证
    results = validate_all_models(task_id, agent_id)

    # 保存结果到文件
    output_file = "model_validation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] 验证结果已保存到 {output_file}")


if __name__ == "__main__":
    main()