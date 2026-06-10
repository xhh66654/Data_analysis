#!/usr/bin/env python3
"""
对比 Module C 反事实解释在「全 task Preprocessor profile」下的特征分档稳定性。

用法（PowerShell）：
    py scripts/eval_agent_scale_stability.py
    py scripts/eval_agent_scale_stability.py --task-a INF_A_001 --task-b INF_A_002 --agent-id 1
    py scripts/eval_agent_scale_stability.py --json output/scale_stability.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.module_a_rules.agent_profile import load_profile, profile_id_for, schema_fingerprint
from src.module_c_counterfactual.data_loader import load_inference_records
from src.module_c_counterfactual.surrogate_cache import clear_surrogate_bundle_cache
from src.service import counterfactual_service


def _pick_decision(records, agent_id: int):
    """从记录集中选取第一个含有效决策的步。

    参数:
        records: 推理记录列表。
        agent_id: 智能体 ID。

    返回:
        ``(sim_id, step, decision_content)`` 三元组。

    抛出:
        RuntimeError: 找不到任何有效决策时。
    """
    record = records[0]
    for t in range(record.total_steps):
        dec = record.get_decision_at(t, agent_id)
        if dec is not None and dec.content:
            return record.sim_id, t, dict(dec.content)
    raise RuntimeError(f"agent_id={agent_id} 无决策")


def run_eval(
    task_a: str,
    task_b: str,
    agent_id: int,
    *,
    output_json: str | None = None,
) -> dict:
    """对比两个任务下同一智能体特征分档标签的一致性。

    参数:
        task_a: 第一个推理任务 ID。
        task_b: 第二个推理任务 ID。
        agent_id: 智能体 ID。
        output_json: 可选报告输出路径。

    返回:
        含 ``label_consistency_rate`` 等指标的字典。
    """
    clear_surrogate_bundle_cache()
    out_dir = os.environ.get("ANALYSIS_OUTPUT_DIR", "./output")
    os.makedirs(out_dir, exist_ok=True)

    records_a = load_inference_records(task_a)
    records_b = load_inference_records(task_b)
    sim_a, step_a, dc_a = _pick_decision(records_a, agent_id)
    sim_b, step_b, dc_b = _pick_decision(records_b, agent_id)

    r_a = counterfactual_service(
        agent_id=agent_id,
        inference_task_id=task_a,
        sim_id=sim_a,
        decision_content=dc_a,
        query_step=step_a,
        cf_level="local",
        explain_with_llm=False,
    )
    fp = schema_fingerprint(records_a[0], agent_id)
    pid = profile_id_for(agent_id, fp)
    prof_after_a = load_profile(pid)

    clear_surrogate_bundle_cache()
    r_b = counterfactual_service(
        agent_id=agent_id,
        inference_task_id=task_b,
        sim_id=sim_b,
        decision_content=dc_b,
        query_step=step_b,
        cf_level="local",
        explain_with_llm=False,
    )
    prof_after_b = load_profile(pid)

    labels_a = {f["feature"]: f["label"] for f in r_a.get("key_features", [])}
    labels_b = {f["feature"]: f["label"] for f in r_b.get("key_features", [])}
    shared = sorted(set(labels_a) & set(labels_b))
    same_label = sum(1 for k in shared if labels_a[k] == labels_b[k])

    report = {
        "task_a": task_a,
        "task_b": task_b,
        "agent_id": agent_id,
        "profile_id": pid,
        "agent_profile_version_after_a": prof_after_a.version if prof_after_a else None,
        "agent_profile_version_after_b": prof_after_b.version if prof_after_b else None,
        "shared_features": len(shared),
        "consistent_labels": same_label,
        "label_consistency_rate": (same_label / len(shared)) if shared else None,
        "sample_labels_a": {k: labels_a[k] for k in shared[:5]},
        "sample_labels_b": {k: labels_b[k] for k in shared[:5]},
    }

    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Wrote {path}")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    """解析命令行并运行跨任务 Preprocessor 标尺稳定性评估。"""
    parser = argparse.ArgumentParser(description="评估 per-agent Preprocessor 标尺跨 task 稳定性")
    parser.add_argument("--task-a", default="INF_A_001")
    parser.add_argument("--task-b", default="INF_A_002")
    parser.add_argument("--agent-id", type=int, default=1)
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args()
    run_eval(args.task_a, args.task_b, args.agent_id, output_json=args.json_out)


if __name__ == "__main__":
    main()
