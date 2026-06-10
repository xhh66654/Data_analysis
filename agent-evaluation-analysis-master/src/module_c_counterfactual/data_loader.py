"""
数据加载层：从模拟数据库表（或真实 Doris 数据库）加载推理数据。

================================================================================
数据库表结构设计：
================================================================================

模拟两张表（对应 Doris 数据库表结构）：

表1：inference_task（推理数据基础信息表，每行 = 一条推理数据）
    字段：
        task_id           : 推理任务 id（多个推理数据可共享同一 task_id）
        sim_id            : 推理数据 id / 仿真 id（唯一标识一局仿真）
        agents_json       : 智能体信息列表，JSON 格式
        observation_space : 观测项名称列表，JSON 格式
        action_items_json : 动作项定义列表，JSON 格式
        total_steps       : 总决策步数

表2：inference_step（每步决策与观测流水表）
    字段：
        task_id         : 关联推理任务 id（外键）
        sim_id          : 关联推理数据 id（外键，与 task_id 联合定位一局仿真）
        step            : 时间步编号（0-based）
        agent_id        : 智能体 id
        decision_json   : 该步该智能体的决策内容，JSON 格式
        obs_json        : 该步该智能体的观测内容，JSON 格式（嵌套字典）
        reward          : 该步全局奖励值

================================================================================
MOCK_MODE 说明：
================================================================================

True  → 从本地 JSON 文件（模拟数据库表）加载
False → 连接真实 Doris 数据库（PyMySQL）

本地 JSON 文件路径：
    data/mock_records/inference_task.json
    data/mock_records/inference_step.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.module_c_counterfactual.inference_record import (
    ActionItem,
    AgentDecision,
    AgentMeta,
    AgentObservation,
    InferenceRecord,
    StepDecision,
    StepObservation,
)

# ==============================================================================
# 配置项
# ==============================================================================

MOCK_MODE = True

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "mock_records"
_JSON_TABLE_CACHE: Dict[str, Tuple[float, List[Dict]]] = {}

DORIS_CONFIG = {
    "host": "127.0.0.1",
    "port": 9030,
    "user": "root",
    "password": "",
    "database": "simulation_db",
    "charset": "utf8mb4",
}


# ==============================================================================
# 对外接口
# ==============================================================================

def load_inference_records(inference_task_id: str) -> List[InferenceRecord]:
    """
    根据推理任务 id 加载该任务下的全部推理数据记录。

    Parameters
    ----------
    inference_task_id : 推理任务 id（前端用户选择的任务标识符）

    Returns
    -------
    InferenceRecord 列表；无匹配时返回空列表
    """
    if MOCK_MODE:
        return _load_records_from_json(inference_task_id)
    return _load_records_from_doris(inference_task_id)


def load_inference_record(task_id: str, sim_id: Optional[str] = None) -> Optional[InferenceRecord]:
    """
    加载指定推理任务下的推理数据（兼容旧接口）。

    规则抽取请优先使用 ``load_inference_records`` 加载全部推理数据。

    参数:
        task_id: 推理任务 id。
        sim_id: 可选仿真局 id；为 None 时返回第一条记录。

    返回:
        匹配的 InferenceRecord；无数据时 None。
    """
    records = load_inference_records(task_id)
    if not records:
        return None
    if sim_id is None:
        return records[0]
    for rec in records:
        if rec.sim_id == sim_id:
            return rec
    return None


def list_inference_task_ids() -> List[str]:
    """
    列出当前可用的推理任务 id（去重）。

    Returns
    -------
    inference_task_id 字符串列表
    """
    if MOCK_MODE:
        task_file = _DATA_DIR / "inference_task.json"
        if not task_file.exists():
            return []
        rows = json.loads(task_file.read_text(encoding="utf-8"))
        seen: set[str] = set()
        result: List[str] = []
        for row in rows:
            tid = row["task_id"]
            if tid not in seen:
                seen.add(tid)
                result.append(tid)
        return result
    raise NotImplementedError("正式模式尚未实现，请将 MOCK_MODE 设为 True")


def list_available_tasks() -> List[str]:
    """兼容别名，等同于 ``list_inference_task_ids``。"""
    return list_inference_task_ids()


# ==============================================================================
# 本地 JSON 文件加载（模拟数据库表）
# ==============================================================================

def _load_records_from_json(inference_task_id: str) -> List[InferenceRecord]:
    """
    从本地 JSON 模拟表加载指定任务下的全部推理记录。

    参数:
        inference_task_id: 推理任务 id。

    返回:
        InferenceRecord 列表。

    抛出:
        FileNotFoundError: 模拟数据文件不存在时。
    """
    task_file = _DATA_DIR / "inference_task.json"
    step_file = _DATA_DIR / "inference_step.json"

    if not task_file.exists() or not step_file.exists():
        raise FileNotFoundError(
            f"模拟数据文件不存在，请先运行 scripts/generate_mock_data.py 生成数据。\n"
            f"期望路径：{task_file}"
        )

    all_tasks: List[Dict] = _load_json_table_cached(task_file)
    task_rows = [r for r in all_tasks if r["task_id"] == inference_task_id]
    if not task_rows:
        return []

    all_steps: List[Dict] = _load_json_table_cached(step_file)

    records: List[InferenceRecord] = []
    for task_row in task_rows:
        sim_id = task_row["sim_id"]
        step_rows = [
            row for row in all_steps
            if row["task_id"] == inference_task_id and row.get("sim_id", sim_id) == sim_id
        ]
        record = _build_record_from_row(task_row, step_rows)
        if record is not None:
            records.append(record)
    return records


def _build_record_from_row(task_row: Dict, step_rows: List[Dict]) -> Optional[InferenceRecord]:
    """将一行 inference_task 与对应 step 行组装为 InferenceRecord。"""
    task_id = task_row["task_id"]
    sim_id = task_row["sim_id"]

    agents: List[AgentMeta] = [
        AgentMeta(
            agent_id=a["agent_id"],
            agent_name=a["agent_name"],
            equipment_type=a["equipment_type"],
            equipment_units=list(a.get("equipment_units") or ["__default__"]),
        )
        for a in task_row["agents_json"]
    ]

    observation_space: List[str] = task_row["observation_space"]

    action_items: List[ActionItem] = [
        ActionItem(
            name=ai["name"],
            possible_values=ai.get("possible_values", []),
            is_continuous=ai.get("is_continuous", False),
        )
        for ai in task_row["action_items_json"]
    ]

    total_steps: int = task_row["total_steps"]

    step_data: Dict[int, Dict[int, Dict]] = defaultdict(dict)
    rewards_by_step: Dict[int, float] = {}

    for row in step_rows:
        t = row["step"]
        aid = row["agent_id"]
        step_data[t][aid] = row
        rewards_by_step[t] = float(row["reward"])

    decisions: List[StepDecision] = []
    observations: List[StepObservation] = []
    rewards: List[float] = []

    for t in range(total_steps):
        agent_rows = step_data.get(t, {})
        step_decs = []
        step_obs = []
        for agent in agents:
            aid = agent.agent_id
            row = agent_rows.get(aid)
            if row is None:
                continue
            step_decs.append(AgentDecision(
                agent_id=aid,
                content=row["decision_json"],
            ))
            step_obs.append(AgentObservation(
                agent_id=aid,
                obs_values=row["obs_json"],
            ))

        decisions.append(StepDecision(step=t, decisions=step_decs))
        observations.append(StepObservation(step=t, observations=step_obs))
        rewards.append(rewards_by_step.get(t, 0.0))

    return InferenceRecord(
        task_id=task_id,
        sim_id=sim_id,
        agents=agents,
        observation_space=observation_space,
        action_items=action_items,
        decisions=decisions,
        observations=observations,
        total_steps=total_steps,
        rewards=rewards,
    )


def _load_json_table_cached(path: Path) -> List[Dict]:
    """
    读取本地 JSON 表并按 mtime 做进程内缓存，减少重复 I/O 和反序列化开销。
    """
    mtime = path.stat().st_mtime
    key = str(path)
    hit = _JSON_TABLE_CACHE.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    rows: List[Dict] = json.loads(path.read_text(encoding="utf-8"))
    _JSON_TABLE_CACHE[key] = (mtime, rows)
    return rows


def _load_from_json(task_id: str) -> Optional[InferenceRecord]:
    """
    兼容旧内部调用，返回任务下第一条推理数据。

    参数:
        task_id: 推理任务 id。

    返回:
        第一条 InferenceRecord；无数据时 None。
    """
    records = _load_records_from_json(task_id)
    return records[0] if records else None


# ==============================================================================
# 正式 Doris 数据库查询（MOCK_MODE=False 时使用）
# ==============================================================================

def _load_records_from_doris(inference_task_id: str) -> List[InferenceRecord]:
    """
    从 Doris 数据库中查询推理任务下的全部推理数据。

    数据库表结构：
        inference_task (task_id, sim_id, agents_json, observation_space, action_items_json, total_steps)
        inference_step (task_id, sim_id, step, agent_id, decision_json, obs_json, reward)
    """
    # TODO（正式接入数据库时实现）：
    #
    #   import pymysql
    #   conn = pymysql.connect(**DORIS_CONFIG)
    #   cursor = conn.cursor(pymysql.cursors.DictCursor)
    #
    #   cursor.execute(
    #       "SELECT * FROM inference_task WHERE task_id=%s",
    #       (inference_task_id,)
    #   )
    #   task_rows = cursor.fetchall()
    #   if not task_rows:
    #       return []
    #
    #   cursor.execute(
    #       "SELECT * FROM inference_step WHERE task_id=%s ORDER BY sim_id, step, agent_id",
    #       (inference_task_id,)
    #   )
    #   all_step_rows = cursor.fetchall()
    #   cursor.close(); conn.close()
    #
    #   records = []
    #   for task_row in task_rows:
    #       sim_id = task_row["sim_id"]
    #       step_rows = [r for r in all_step_rows if r["sim_id"] == sim_id]
    #       records.append(_build_record_from_row(task_row, step_rows))
    #   return records
    raise NotImplementedError("正式 Doris 查询尚未实现，请将 MOCK_MODE 设为 True 使用本地数据")


def _load_from_doris(task_id: str) -> Optional[InferenceRecord]:
    """
    兼容旧接口，返回任务下第一条推理数据。

    参数:
        task_id: 推理任务 id。

    返回:
        第一条 InferenceRecord；无数据时 None。
    """
    records = _load_records_from_doris(task_id)
    return records[0] if records else None
