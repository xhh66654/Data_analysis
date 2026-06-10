"""
比较不同 K 下机械论 top 特征重叠率（K 采样稳定性评估）。

用法：
    py scripts/eval_k_sampling_stability.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.module_c_counterfactual.data_loader import load_inference_records
from src.service import counterfactual_service


def _top_feature_names(result: dict, k: int = 5) -> list[str]:
    """从反事实结果中提取前 k 个关键特征名。

    参数:
        result: ``counterfactual_service`` 返回字典。
        k: 最多取用的特征数量。

    返回:
        特征名称字符串列表。
    """
    names = []
    for f in result.get("key_features") or []:
        feat = f.get("feature")
        if feat:
            names.append(str(feat))
        if len(names) >= k:
            break
    return names


def main() -> None:
    """比较 K=50 与 K=200 下机械论 top 特征 Jaccard 重叠率。"""
    task_id = os.environ.get("EVAL_TASK_ID", "INF_A_001")
    agent_id = int(os.environ.get("EVAL_AGENT_ID", "1"))
    records = load_inference_records(task_id)
    record = records[0]
    dec = None
    t_query = 0
    for t in range(record.total_steps):
        d = record.get_decision_at(t, agent_id)
        if d and d.content:
            dec = dict(d.content)
            t_query = t
            break
    if not dec or t_query >= record.total_steps - 1:
        print("无可用决策步")
        return

    base = counterfactual_service(
        agent_id=agent_id,
        inference_task_id=task_id,
        sim_id=record.sim_id,
        decision_content=dec,
        query_step=t_query,
        cf_level="one_step",
        use_k_sampling=True,
        k_samples=50,
        k_seed=1,
    )
    large = counterfactual_service(
        agent_id=agent_id,
        inference_task_id=task_id,
        sim_id=record.sim_id,
        decision_content=dec,
        query_step=t_query,
        cf_level="one_step",
        use_k_sampling=True,
        k_samples=200,
        k_seed=1,
    )
    a = set(_top_feature_names(base))
    b = set(_top_feature_names(large))
    overlap = len(a & b) / max(len(a | b), 1)
    report = {
        "task_id": task_id,
        "sim_id": record.sim_id,
        "t_query": t_query,
        "top5_k50": list(a),
        "top5_k200": list(b),
        "jaccard_top5": round(overlap, 3),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
