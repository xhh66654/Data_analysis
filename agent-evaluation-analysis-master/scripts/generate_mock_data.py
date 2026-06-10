"""
模拟数据生成脚本（多装备类型版，支持大规模）。

生成 Doris 模拟表：
    inference_task.json  — 每行 = 一局仿真（sim_id）
    inference_step.json  — 每行 = 一步 × 一个智能体

业务语义：
    - 一个推理任务（inference_task_id）包含多局仿真（sim_id）
    - 一局可有数百～千余步决策
    - 支持「多智能体共享任务」与「每智能体独立多任务」两种组织方式

预设规模（--preset）：
    dev    — 开发/单测（8~16 步/局，3 局/任务，秒级）
    full   — **推荐全量**（600~1200 步/局，5~8 局/任务，A+B+C，约 20 万 step 行）
    medium — 中等（50~100 步/局）
    large  — 偏大（模块 A 深度验证）
    huge   — 超大（慎用）

业务默认尺度（full）：
    一局决策 600~1200 条；一次推理任务 5~8 局仿真。

示例：
    py scripts/generate_mock_data.py --preset full
    py scripts/generate_mock_data.py --preset dev
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "mock_records"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_STEPS = 8
MAX_STEPS = 16

_sim_counter = {"A": 0, "B": 0, "C": 0}

PRESETS: Dict[str, Dict[str, Any]] = {
    "dev": {
        "min_steps": 8,
        "max_steps": 16,
        "min_sims_a": 3,
        "max_sims_a": 3,
        "min_sims_b": 2,
        "max_sims_b": 2,
        "min_sims_c": 2,
        "max_sims_c": 2,
        "n_a": 5,
        "n_b": 5,
        "n_c": 5,
        "agent_tasks_a": 0,
        "agent_tasks_b": 0,
        "agent_tasks_c": 0,
        "multi_min_sims": 2,
        "multi_max_sims": 2,
        "with_boundary": False,
        "compact": False,
    },
    "full": {
        "min_steps": 600,
        "max_steps": 1200,
        "min_sims_a": 5,
        "max_sims_a": 8,
        "min_sims_b": 5,
        "max_sims_b": 8,
        "min_sims_c": 5,
        "max_sims_c": 8,
        "n_a": 5,
        "n_b": 3,
        "n_c": 3,
        "agent_tasks_a": 2,
        "agent_tasks_b": 2,
        "agent_tasks_c": 2,
        "multi_min_sims": 5,
        "multi_max_sims": 8,
        "with_boundary": True,
        "compact": True,
    },
    "medium": {
        "min_steps": 50,
        "max_steps": 100,
        "min_sims_a": 10,
        "max_sims_a": 10,
        "min_sims_b": 8,
        "max_sims_b": 8,
        "min_sims_c": 8,
        "max_sims_c": 8,
        "n_a": 10,
        "n_b": 8,
        "n_c": 8,
        "agent_tasks_a": 3,
        "agent_tasks_b": 2,
        "agent_tasks_c": 2,
        "multi_min_sims": 5,
        "multi_max_sims": 5,
        "with_boundary": True,
        "compact": True,
    },
    "large": {
        "min_steps": 600,
        "max_steps": 1200,
        "min_sims_a": 8,
        "max_sims_a": 12,
        "min_sims_b": 8,
        "max_sims_b": 12,
        "min_sims_c": 8,
        "max_sims_c": 12,
        "n_a": 3,
        "n_b": 3,
        "n_c": 3,
        "agent_tasks_a": 3,
        "agent_tasks_b": 2,
        "agent_tasks_c": 2,
        "multi_min_sims": 8,
        "multi_max_sims": 12,
        "with_boundary": False,
        "compact": True,
    },
    "huge": {
        "min_steps": 1000,
        "max_steps": 1500,
        "min_sims_a": 20,
        "max_sims_a": 30,
        "min_sims_b": 15,
        "max_sims_b": 25,
        "min_sims_c": 15,
        "max_sims_c": 25,
        "n_a": 5,
        "n_b": 5,
        "n_c": 5,
        "agent_tasks_a": 10,
        "agent_tasks_b": 5,
        "agent_tasks_c": 5,
        "multi_min_sims": 15,
        "multi_max_sims": 25,
        "with_boundary": False,
        "compact": True,
    },
}


# ==============================================================================
# 场景A：歼-20 空战（3 个智能体）
# ==============================================================================

SCENARIO_A_AGENTS = [
    {"agent_id": 1, "agent_name": "Alpha编队", "equipment_type": "歼-20"},
    {"agent_id": 2, "agent_name": "Bravo编队", "equipment_type": "歼-20"},
    {"agent_id": 3, "agent_name": "Charlie编队", "equipment_type": "歼-20"},
]
SCENARIO_A_OBS_SPACE = ["自身状态", "敌机距离", "敌机状态"]
SCENARIO_A_ACTION_ITEMS = [
    {"name": "雷达开关控制", "possible_values": ["开", "关"], "is_continuous": False},
    {"name": "雷达方向控制", "possible_values": ["左扫", "右扫", "正前方"], "is_continuous": False},
    {"name": "武器控制", "possible_values": ["发射导弹", "不发射"], "is_continuous": False},
    {"name": "机动控制", "possible_values": ["规避", "追击", "保持"], "is_continuous": False},
]


def _clamp(v: float, lo: float, hi: float) -> float:
    """将数值限制在闭区间 [lo, hi] 内。

    参数:
        v: 待裁剪的数值。
        lo: 下界。
        hi: 上界。

    返回:
        裁剪后的浮点数。
    """
    return max(lo, min(hi, v))


def _next_sim_id(prefix: str) -> str:
    """生成递增的仿真局 ID。

    参数:
        prefix: 场景前缀（如 ``"A"``、``"B"``、``"C"``）。

    返回:
        形如 ``SIM_{prefix}_{序号:06d}`` 的字符串。
    """
    _sim_counter[prefix] += 1
    return f"SIM_{prefix}_{_sim_counter[prefix]:06d}"


def _sample_steps(explicit: Optional[int] = None) -> int:
    """采样单局决策步数。

    参数:
        explicit: 若指定则直接返回该值；否则在 ``MIN_STEPS``～``MAX_STEPS`` 间随机。

    返回:
        本局总步数。
    """
    if explicit is not None:
        return explicit
    return random.randint(MIN_STEPS, MAX_STEPS)


def _sample_sims(min_sims: int, max_sims: int, explicit: Optional[int] = None) -> int:
    """采样单个推理任务包含的仿真局数。

    参数:
        min_sims: 最少局数。
        max_sims: 最多局数。
        explicit: 若指定则直接返回该值。

    返回:
        本任务的仿真局数。
    """
    if explicit is not None:
        return explicit
    if min_sims >= max_sims:
        return min_sims
    return random.randint(min_sims, max_sims)


def _append_sim_steps(
    task_rows: List[Dict],
    step_sink: Callable[[List[Dict]], None],
    inference_task_id: str,
    sim_id: str,
    agents: List[Dict],
    obs_space: List[str],
    action_items: List[Dict],
    simulate_fn,
    *,
    n_steps: Optional[int] = None,
) -> int:
    """生成一局仿真并流式写出 step 行。

    参数:
        task_rows: 待追加的 inference_task 行列表（原地修改）。
        step_sink: 接收 step 批次并写盘的回调。
        inference_task_id: 推理任务 ID。
        sim_id: 本局仿真 ID。
        agents: 参与本局的智能体元数据列表。
        obs_space: 观测空间名称列表。
        action_items: 动作项定义列表。
        simulate_fn: ``(n_steps, agent_id) -> (obs, decisions, rewards)`` 仿真函数。
        n_steps: 固定步数；为 ``None`` 时随机采样。

    返回:
        本局实际生成的步数。
    """
    steps = _sample_steps(n_steps)

    task_rows.append({
        "task_id": inference_task_id,
        "sim_id": sim_id,
        "agents_json": agents,
        "observation_space": obs_space,
        "action_items_json": action_items,
        "total_steps": steps,
    })

    all_obs, all_dec, all_rews = {}, {}, {}
    for agent in agents:
        aid = agent["agent_id"]
        all_obs[aid], all_dec[aid], all_rews[aid] = simulate_fn(steps, aid)

    batch: List[Dict] = []
    for t in range(steps):
        if len(agents) > 1:
            global_reward = round(
                sum(all_rews[a["agent_id"]][t] for a in agents) / len(agents), 4
            )
        else:
            global_reward = round(all_rews[agents[0]["agent_id"]][t], 4)

        for agent in agents:
            aid = agent["agent_id"]
            batch.append({
                "task_id": inference_task_id,
                "sim_id": sim_id,
                "step": t,
                "agent_id": aid,
                "decision_json": all_dec[aid][t],
                "obs_json": all_obs[aid][t],
                "reward": global_reward,
            })
        # 分批刷盘，避免单局步数上千时占用过多内存
        if len(batch) >= 500:
            step_sink(batch)
            batch.clear()

    if batch:
        step_sink(batch)
    return steps


def _simulate_fighter(n_steps: int, agent_id: int):
    """模拟歼-20 空战单智能体轨迹。

    参数:
        n_steps: 仿真步数。
        agent_id: 智能体 ID（影响初始距离等状态）。

    返回:
        ``(obs_list, decision_list, rewards)`` 三元组，每步一条记录。
    """
    hp = 1.0
    speed = 1.2 + random.uniform(-0.1, 0.1)
    altitude = 8.0 + random.uniform(-1.0, 1.0)
    dist = 90.0 - (agent_id - 1) * 5 + random.uniform(-10, 10)
    alt_diff = random.uniform(-2.0, 2.0)
    enemy_alive = 1
    locked = 0
    missiles = 6

    obs_list, decision_list, rewards = [], [], []

    for _ in range(n_steps):
        threat = _clamp(1.0 - dist / 100.0 + random.uniform(-0.1, 0.1), 0.0, 1.0)
        obs = {
            "自身状态": {
                "血量": round(hp, 3),
                "速度_马赫": round(speed, 3),
                "高度_km": round(altitude, 3),
            },
            "敌机距离": {
                "水平距离_km": round(dist, 3),
                "高度差_km": round(alt_diff, 3),
            },
            "敌机状态": {
                "存活": enemy_alive,
                "锁定中": locked,
                "威胁等级": round(threat, 3),
            },
        }
        obs_list.append(obs)

        if not enemy_alive:
            radar_sw, radar_dir, weapon, maneuver = "关", random.choice(["左扫", "右扫", "正前方"]), "不发射", "保持"
            reward = 0.1
        elif threat > 0.65 and random.random() > 0.2:
            radar_sw, radar_dir, weapon, maneuver = "开", "正前方", "不发射", "规避"
            reward = -0.1 + random.uniform(-0.05, 0.05)
        elif dist > 70:
            radar_sw, radar_dir, weapon, maneuver = "开", random.choice(["左扫", "右扫", "正前方"]), "不发射", "追击"
            reward = 0.05
        elif dist > 40:
            radar_sw, radar_dir, weapon, maneuver = "开", "正前方", "不发射", "追击"
            reward = 0.1
        elif missiles > 0 and random.random() > 0.3:
            radar_sw, radar_dir, weapon, maneuver = "开", "正前方", "发射导弹", "追击"
            missiles -= 1
            reward = 0.4 + random.uniform(-0.1, 0.2)
        else:
            radar_sw, radar_dir, weapon, maneuver = "开", "正前方", "不发射", "保持"
            reward = 0.05

        decision_list.append({
            "雷达开关控制": radar_sw,
            "雷达方向控制": radar_dir,
            "武器控制": weapon,
            "机动控制": maneuver,
        })
        rewards.append(round(reward, 4))

        if maneuver == "追击":
            dist = _clamp(dist - random.uniform(5, 15), 5, 120)
        elif maneuver == "规避":
            dist = _clamp(dist + random.uniform(5, 15), 5, 120)
        else:
            dist = _clamp(dist + random.uniform(-5, 5), 5, 120)
        alt_diff = _clamp(alt_diff + random.uniform(-1, 1), -5, 5)
        speed = _clamp(speed + random.uniform(-0.2, 0.2), 0.8, 2.0)
        altitude = _clamp(altitude + random.uniform(-0.5, 0.5), 3.0, 15.0)
        if threat > 0.7 and maneuver != "规避" and random.random() > 0.5:
            hp = _clamp(hp - random.uniform(0.05, 0.15), 0.0, 1.0)
        if weapon == "发射导弹" and random.random() > 0.5 and dist < 50:
            enemy_alive = 0
        if not enemy_alive:
            locked = 0
        elif dist < 60:
            locked = 1

    return obs_list, decision_list, rewards


def generate_scenario_a(
    n_inference_tasks: int,
    min_sims: int,
    max_sims: int,
    task_rows: List[Dict],
    step_sink: Callable[[List[Dict]], None],
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> None:
    """生成场景 A 多智能体共享推理任务（``INF_A_001``～``INF_A_NNN``）。

    参数:
        n_inference_tasks: 共享任务数量。
        min_sims: 每任务最少仿真局数。
        max_sims: 每任务最多仿真局数。
        task_rows: inference_task 行累积列表。
        step_sink: step 批次写盘回调。
        progress: 可选进度打印回调。
    """
    for i in range(1, n_inference_tasks + 1):
        inference_task_id = f"INF_A_{i:03d}"
        n_sims = _sample_sims(min_sims, max_sims)
        if progress:
            progress(f"  共享任务 {inference_task_id}（{n_sims} 局）")
        for _ in range(n_sims):
            sim_id = _next_sim_id("A")
            _append_sim_steps(
                task_rows, step_sink,
                inference_task_id, sim_id,
                SCENARIO_A_AGENTS, SCENARIO_A_OBS_SPACE, SCENARIO_A_ACTION_ITEMS,
                _simulate_fighter,
            )


def generate_per_agent_tasks(
    prefix: str,
    scenario_tag: str,
    agents: List[Dict],
    obs_space: List[str],
    action_items: List[Dict],
    simulate_fn,
    tasks_per_agent: int,
    min_sims: int,
    max_sims: int,
    task_rows: List[Dict],
    step_sink: Callable[[List[Dict]], None],
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> None:
    """为每个智能体生成独立多任务（``INF_{prefix}_AG{agent_id}_{idx:03d}``）。

    参数:
        prefix: 任务 ID 前缀（如 ``"A"``）。
        scenario_tag: 仿真 ID 计数器前缀。
        agents: 智能体元数据列表。
        obs_space: 观测空间名称列表。
        action_items: 动作项定义列表。
        simulate_fn: 单智能体仿真函数。
        tasks_per_agent: 每智能体独立任务数。
        min_sims: 每任务最少局数。
        max_sims: 每任务最多局数。
        task_rows: inference_task 行累积列表。
        step_sink: step 批次写盘回调。
        progress: 可选进度打印回调。
    """
    for agent in agents:
        aid = agent["agent_id"]
        solo = [agent]
        for idx in range(1, tasks_per_agent + 1):
            inference_task_id = f"INF_{prefix}_AG{aid}_{idx:03d}"
            n_sims = _sample_sims(min_sims, max_sims)
            if progress:
                progress(f"  智能体 {aid} 任务 {inference_task_id}（{n_sims} 局）")
            for _ in range(n_sims):
                sim_id = _next_sim_id(scenario_tag)
                _append_sim_steps(
                    task_rows, step_sink,
                    inference_task_id, sim_id,
                    solo, obs_space, action_items,
                    simulate_fn,
                )


# ==============================================================================
# 场景B：地面雷达站
# ==============================================================================

SCENARIO_B_AGENTS = [
    {"agent_id": 1, "agent_name": "雷达站_A", "equipment_type": "主动雷达"},
    {"agent_id": 2, "agent_name": "雷达站_B", "equipment_type": "被动雷达"},
]
SCENARIO_B_OBS_SPACE = ["雷达状态", "目标信号"]
SCENARIO_B_ACTION_ITEMS = [
    {"name": "功率调节", "possible_values": ["低", "中", "高"], "is_continuous": False},
    {"name": "扫描模式", "possible_values": ["全向", "定向", "跟踪"], "is_continuous": False},
    {"name": "发射干扰", "possible_values": ["是", "否"], "is_continuous": False},
]


def _simulate_radar(n_steps: int, agent_id: int):
    """模拟地面雷达站单智能体轨迹。

    参数:
        n_steps: 仿真步数。
        agent_id: 智能体 ID（当前未区分行为，保留接口一致性）。

    返回:
        ``(obs_list, decision_list, rewards)`` 三元组。
    """
    power_kw = 50.0 + random.uniform(-10, 10)
    azimuth = random.uniform(0, 360)
    elev = random.uniform(0, 45)
    signal_strength = random.uniform(-80, -40)
    freq_ghz = 9.0 + random.uniform(-1, 1)
    target_dist = 200 + random.uniform(-50, 50)

    obs_list, decision_list, rewards = [], [], []

    for _ in range(n_steps):
        obs = {
            "雷达状态": {
                "功率_kw": round(power_kw, 2),
                "方位角_deg": round(azimuth % 360, 2),
                "仰角_deg": round(elev, 2),
            },
            "目标信号": {
                "强度_dbm": round(signal_strength, 2),
                "频率_ghz": round(freq_ghz, 3),
                "距离_km": round(target_dist, 2),
            },
        }
        obs_list.append(obs)

        if signal_strength > -55:
            scan, power_lvl, jamming, reward = "跟踪", "高" if target_dist < 150 else "中", "否", 0.3
        elif signal_strength > -70:
            scan, power_lvl, jamming, reward = "定向", "中", random.choice(["是", "否"]), 0.1
        else:
            scan, power_lvl, jamming, reward = "全向", "低", "否", 0.0

        decision_list.append({"功率调节": power_lvl, "扫描模式": scan, "发射干扰": jamming})
        rewards.append(round(reward, 4))

        power_kw = _clamp(power_kw + {"低": -5, "中": 0, "高": 5}[power_lvl] + random.uniform(-2, 2), 10, 100)
        azimuth = (azimuth + random.uniform(-15, 15)) % 360
        elev = _clamp(elev + random.uniform(-3, 3), 0, 90)
        signal_strength = _clamp(signal_strength + random.uniform(-5, 5), -100, -20)
        freq_ghz = _clamp(freq_ghz + random.uniform(-0.1, 0.1), 7.0, 12.0)
        target_dist = _clamp(target_dist + random.uniform(-20, 20), 20, 500)

    return obs_list, decision_list, rewards


def generate_scenario_b(
    n_inference_tasks: int,
    min_sims: int,
    max_sims: int,
    task_rows: List[Dict],
    step_sink: Callable[[List[Dict]], None],
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> None:
    """生成场景 B 雷达站共享推理任务（``INF_B_001``～``INF_B_NNN``）。

    参数:
        n_inference_tasks: 共享任务数量。
        min_sims: 每任务最少仿真局数。
        max_sims: 每任务最多仿真局数。
        task_rows: inference_task 行累积列表。
        step_sink: step 批次写盘回调。
        progress: 可选进度打印回调。
    """
    for i in range(1, n_inference_tasks + 1):
        inference_task_id = f"INF_B_{i:03d}"
        n_sims = _sample_sims(min_sims, max_sims)
        if progress:
            progress(f"  共享任务 {inference_task_id}（{n_sims} 局）")
        for _ in range(n_sims):
            sim_id = _next_sim_id("B")
            _append_sim_steps(
                task_rows, step_sink,
                inference_task_id, sim_id,
                SCENARIO_B_AGENTS, SCENARIO_B_OBS_SPACE, SCENARIO_B_ACTION_ITEMS,
                _simulate_radar,
            )


# ==============================================================================
# 场景C：侦察机
# ==============================================================================

SCENARIO_C_AGENTS = [
    {"agent_id": 1, "agent_name": "侦察机_X", "equipment_type": "侦察机"},
]
SCENARIO_C_OBS_SPACE = ["位置", "传感器状态", "目标区域"]
SCENARIO_C_ACTION_ITEMS = [
    {"name": "飞行模式", "possible_values": ["盘旋", "直飞", "爬升", "下降"], "is_continuous": False},
    {"name": "传感器模式", "possible_values": ["拍照", "视频", "休眠"], "is_continuous": False},
]


def _simulate_recon(n_steps: int, agent_id: int):
    """模拟侦察机单智能体轨迹。

    参数:
        n_steps: 仿真步数。
        agent_id: 智能体 ID（保留接口一致性）。

    返回:
        ``(obs_list, decision_list, rewards)`` 三元组。
    """
    lon = 116.0 + random.uniform(-0.5, 0.5)
    lat = 39.0 + random.uniform(-0.5, 0.5)
    alt = 10.0 + random.uniform(-2, 2)
    battery = 1.0
    temp = 25.0 + random.uniform(-5, 5)
    resolution = 0.5 + random.uniform(-0.1, 0.1)
    n_targets = random.randint(0, 8)
    threat = round(random.uniform(0.0, 0.8), 2)
    coverage = round(random.uniform(0.1, 0.5), 2)

    obs_list, decision_list, rewards = [], [], []

    for _ in range(n_steps):
        obs = {
            "位置": {"经度": round(lon, 5), "纬度": round(lat, 5), "高度_km": round(alt, 3)},
            "传感器状态": {
                "电量": round(battery, 3),
                "温度_摄氏": round(temp, 1),
                "分辨率_m": round(resolution, 3),
            },
            "目标区域": {
                "目标数": n_targets,
                "威胁等级": threat,
                "覆盖率": round(coverage, 3),
            },
        }
        obs_list.append(obs)

        if battery < 0.2:
            fly_mode, sensor, reward = "下降", "休眠", -0.2
        elif threat > 0.6:
            fly_mode, sensor, reward = "盘旋", "拍照", 0.1
        elif coverage < 0.3 and n_targets > 2:
            fly_mode, sensor, reward = "直飞", "视频", 0.3
        elif n_targets > 4:
            fly_mode, sensor, reward = "盘旋", "视频", 0.25
        else:
            fly_mode = "直飞"
            sensor = "拍照" if random.random() > 0.3 else "休眠"
            reward = 0.05

        decision_list.append({"飞行模式": fly_mode, "传感器模式": sensor})
        rewards.append(round(reward, 4))

        lon += random.uniform(-0.01, 0.01)
        lat += random.uniform(-0.01, 0.01)
        alt = _clamp(
            alt + {"爬升": 0.5, "下降": -0.5, "直飞": 0.0, "盘旋": 0.0}[fly_mode]
            + random.uniform(-0.1, 0.1),
            1.0, 20.0,
        )
        battery = _clamp(battery - (0.02 if sensor == "休眠" else 0.05) + random.uniform(-0.01, 0.01), 0.0, 1.0)
        temp = _clamp(temp + random.uniform(-2, 3), 10.0, 80.0)
        resolution = _clamp(resolution + random.uniform(-0.05, 0.05), 0.1, 2.0)
        n_targets = max(0, n_targets + random.randint(-1, 2))
        threat = round(_clamp(threat + random.uniform(-0.1, 0.1), 0.0, 1.0), 2)
        coverage = round(
            _clamp(coverage + (0.05 if sensor != "休眠" else 0.0) + random.uniform(-0.02, 0.02), 0.0, 1.0),
            3,
        )

    return obs_list, decision_list, rewards


def generate_scenario_c(
    n_inference_tasks: int,
    min_sims: int,
    max_sims: int,
    task_rows: List[Dict],
    step_sink: Callable[[List[Dict]], None],
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> None:
    """生成场景 C 侦察机共享推理任务（``INF_C_001``～``INF_C_NNN``）。

    参数:
        n_inference_tasks: 共享任务数量。
        min_sims: 每任务最少仿真局数。
        max_sims: 每任务最多仿真局数。
        task_rows: inference_task 行累积列表。
        step_sink: step 批次写盘回调。
        progress: 可选进度打印回调。
    """
    for i in range(1, n_inference_tasks + 1):
        inference_task_id = f"INF_C_{i:03d}"
        n_sims = _sample_sims(min_sims, max_sims)
        if progress:
            progress(f"  共享任务 {inference_task_id}（{n_sims} 局）")
        for _ in range(n_sims):
            sim_id = _next_sim_id("C")
            _append_sim_steps(
                task_rows, step_sink,
                inference_task_id, sim_id,
                SCENARIO_C_AGENTS, SCENARIO_C_OBS_SPACE, SCENARIO_C_ACTION_ITEMS,
                _simulate_recon,
            )


# ==============================================================================
# 多装备个体
# ==============================================================================

SCENARIO_A_MULTI_AGENTS = [
    {
        "agent_id": 1,
        "agent_name": "双机编队",
        "equipment_type": "歼-20编队",
        "equipment_units": ["alpha_1", "alpha_2"],
    },
]


def _simulate_fighter_multi_unit(n_steps: int, agent_id: int):
    """模拟多装备个体（双机编队）空战轨迹。

    参数:
        n_steps: 仿真步数。
        agent_id: 智能体 ID。

    返回:
        ``(obs_list, decision_list, rewards)``；观测与决策按装备个体嵌套字典组织。
    """
    units = ["alpha_1", "alpha_2"]
    obs_list: List[Dict] = []
    decision_list: List[Dict] = []
    rewards: List[float] = []

    state = {
        u: {
            "hp": 1.0,
            "speed": 1.2 + 0.1 * i,
            "altitude": 8.0,
            "dist": 90.0 - 10 * i,
            "alt_diff": 0.5 * i,
            "enemy_alive": 1,
            "locked": 0,
            "missiles": 4,
        }
        for i, u in enumerate(units)
    }

    for _ in range(n_steps):
        step_obs: Dict[str, Dict] = {}
        step_dec: Dict[str, Dict] = {}
        step_reward = 0.0

        for u in units:
            st = state[u]
            threat = _clamp(1.0 - st["dist"] / 100.0 + random.uniform(-0.1, 0.1), 0.0, 1.0)
            step_obs[u] = {
                "自身状态": {
                    "血量": round(st["hp"], 3),
                    "速度_马赫": round(st["speed"], 3),
                    "高度_km": round(st["altitude"], 3),
                },
                "敌机距离": {
                    "水平距离_km": round(st["dist"], 3),
                    "高度差_km": round(st["alt_diff"], 3),
                },
                "敌机状态": {
                    "存活": st["enemy_alive"],
                    "锁定中": st["locked"],
                    "威胁等级": round(threat, 3),
                },
            }

            if not st["enemy_alive"]:
                radar_sw, radar_dir, weapon, maneuver = "关", "正前方", "不发射", "保持"
                r = 0.1
            elif threat > 0.65:
                radar_sw, radar_dir, weapon, maneuver = "开", "正前方", "不发射", "规避"
                r = -0.05
            elif st["dist"] > 50:
                radar_sw, radar_dir, weapon, maneuver = "开", "左扫", "不发射", "追击"
                r = 0.08
            elif st["missiles"] > 0:
                radar_sw, radar_dir, weapon, maneuver = "开", "正前方", "发射导弹", "追击"
                st["missiles"] -= 1
                r = 0.35
            else:
                radar_sw, radar_dir, weapon, maneuver = "开", "正前方", "不发射", "保持"
                r = 0.05

            step_dec[u] = {
                "雷达开关控制": radar_sw,
                "雷达方向控制": radar_dir,
                "武器控制": weapon,
                "机动控制": maneuver,
            }
            step_reward += r

            if maneuver == "追击":
                st["dist"] = _clamp(st["dist"] - random.uniform(3, 10), 5, 120)
            elif maneuver == "规避":
                st["dist"] = _clamp(st["dist"] + random.uniform(3, 10), 5, 120)
            if weapon == "发射导弹" and random.random() > 0.6:
                st["enemy_alive"] = 0
            st["locked"] = 1 if st["dist"] < 60 and st["enemy_alive"] else 0

        obs_list.append(step_obs)
        decision_list.append(step_dec)
        rewards.append(round(step_reward / len(units), 4))

    return obs_list, decision_list, rewards


def generate_scenario_a_multi(
    task_rows: List[Dict],
    step_sink: Callable[[List[Dict]], None],
    *,
    min_sims: int = 2,
    max_sims: int = 2,
) -> None:
    """生成多装备个体任务 ``INF_A_MULTI``。

    参数:
        task_rows: inference_task 行累积列表。
        step_sink: step 批次写盘回调。
        min_sims: 最少仿真局数。
        max_sims: 最多仿真局数。
    """
    inference_task_id = "INF_A_MULTI"
    n_sims = _sample_sims(min_sims, max_sims)
    for _ in range(n_sims):
        sim_id = _next_sim_id("A")
        _append_sim_steps(
            task_rows, step_sink,
            inference_task_id, sim_id,
            SCENARIO_A_MULTI_AGENTS, SCENARIO_A_OBS_SPACE, SCENARIO_A_ACTION_ITEMS,
            _simulate_fighter_multi_unit,
        )


def generate_single_sim_boundary(
    task_rows: List[Dict],
    step_sink: Callable[[List[Dict]], None],
) -> None:
    """生成边界测试任务 ``INF_A_SINGLE``（固定 12 步短局）。

    参数:
        task_rows: inference_task 行累积列表。
        step_sink: step 批次写盘回调。
    """
    sim_id = _next_sim_id("A")
    _append_sim_steps(
        task_rows, step_sink,
        "INF_A_SINGLE", sim_id,
        SCENARIO_A_AGENTS[:2], SCENARIO_A_OBS_SPACE, SCENARIO_A_ACTION_ITEMS,
        _simulate_fighter,
        n_steps=12,
    )


# ==============================================================================
# JSON 流式写出
# ==============================================================================

class _StepJsonWriter:
    """流式写入 step JSON 数组，避免超大数据全进内存。"""

    def __init__(self, path: Path, *, compact: bool) -> None:
        """初始化 JSON 数组写盘器。

        参数:
            path: 输出文件路径。
            compact: 为 ``True`` 时使用紧凑 JSON（无缩进）。
        """
        self.path = path
        self.compact = compact
        self._n = 0
        self._fh = path.open("w", encoding="utf-8")
        self._fh.write("[\n")

    def write_batch(self, rows: List[Dict]) -> None:
        """追加一批 step 行到 JSON 数组。

        参数:
            rows: step 记录字典列表。
        """
        for row in rows:
            if self._n > 0:
                self._fh.write(",\n")
            if self.compact:
                self._fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            else:
                self._fh.write(json.dumps(row, ensure_ascii=False, indent=2))
            self._n += 1

    def close(self) -> int:
        """关闭文件并补全 JSON 数组尾部。

        返回:
            已写入的 step 行总数。
        """
        self._fh.write("\n]\n")
        self._fh.close()
        return self._n


def _write_task_json(path: Path, rows: List[Dict], *, compact: bool) -> None:
    """将 inference_task 行列表写入 JSON 文件。

    参数:
        path: 输出文件路径。
        rows: task 记录字典列表。
        compact: 是否使用紧凑 JSON 格式。
    """
    kwargs: Dict[str, Any] = {"ensure_ascii": False}
    if compact:
        kwargs["separators"] = (",", ":")
    else:
        kwargs["indent"] = 2
    path.write_text(json.dumps(rows, **kwargs), encoding="utf-8")


def _avg_sims(min_s: int, max_s: int) -> float:
    """计算仿真局数区间的算术均值。

    参数:
        min_s: 最少局数。
        max_s: 最多局数。

    返回:
        平均局数估计值。
    """
    return (min_s + max_s) / 2.0


def _estimate_scale(args) -> Dict[str, Any]:
    """粗算生成规模，供用户确认。

    参数:
        args: 解析后的命令行命名空间（含各场景步数、局数、任务数等）。

    返回:
        含 ``avg_steps``、``total_sims``、``est_step_rows`` 等键的估计字典。
    """
    agents_a = len(SCENARIO_A_AGENTS)
    agents_b = len(SCENARIO_B_AGENTS)
    agents_c = len(SCENARIO_C_AGENTS)
    avg_steps = (args.min_steps + args.max_steps) / 2.0
    avg_sa = _avg_sims(args.min_sims_a, args.max_sims_a)
    avg_sb = _avg_sims(args.min_sims_b, args.max_sims_b)
    avg_sc = _avg_sims(args.min_sims_c, args.max_sims_c)
    avg_multi = _avg_sims(args.multi_min_sims, args.multi_max_sims)

    shared_sims = args.n_a * avg_sa + args.n_b * avg_sb + args.n_c * avg_sc
    agent_sims = (
        agents_a * args.agent_tasks_a * avg_sa
        + agents_b * args.agent_tasks_b * avg_sb
        + agents_c * args.agent_tasks_c * avg_sc
    )
    multi_sims = avg_multi
    boundary = 1 if args.with_boundary else 0
    total_sims = int(shared_sims + agent_sims + multi_sims + boundary)

    step_rows_est = int(
        avg_steps * (
            args.n_a * avg_sa * agents_a
            + args.n_b * avg_sb * agents_b
            + args.n_c * avg_sc * agents_c
            + agents_a * args.agent_tasks_a * avg_sa
            + agents_b * args.agent_tasks_b * avg_sb
            + agents_c * args.agent_tasks_c * avg_sc
            + multi_sims
            + boundary * 2
        )
    )
    return {
        "avg_steps": int(avg_steps),
        "avg_sims": int((avg_sa + avg_sb + avg_sc) / 3),
        "total_sims": total_sims,
        "est_step_rows": step_rows_est,
        "est_task_rows": total_sims,
    }


def run_generate(args) -> None:
    """按配置生成全部场景的 mock 数据并写盘。

    参数:
        args: ``_resolve_args`` 返回的命名空间，含 preset、步数、局数等参数。
    """
    global MIN_STEPS, MAX_STEPS
    MIN_STEPS = args.min_steps
    MAX_STEPS = args.max_steps

    _sim_counter["A"] = 0
    _sim_counter["B"] = 0
    _sim_counter["C"] = 0

    est = _estimate_scale(args)
    print("开始生成多装备类型模拟数据...")
    print(f"  预设/规模 : preset={args.preset or 'custom'}")
    print(f"  每局步数  : {args.min_steps} ~ {args.max_steps}（均值约 {est['avg_steps']}）")
    print(
        f"  每任务局数: A {args.min_sims_a}~{args.max_sims_a}  "
        f"B {args.min_sims_b}~{args.max_sims_b}  C {args.min_sims_c}~{args.max_sims_c}"
    )
    print(f"  预计仿真局: ~{est['total_sims']} 局")
    print(f"  预计 step 行: ~{est['est_step_rows']:,}")
    if est["est_step_rows"] > 500_000:
        print("  （大规模生成，可能需要数分钟，请耐心等待）")

    task_rows: List[Dict] = []
    step_writer = _StepJsonWriter(OUT_DIR / "inference_step.json", compact=args.compact)
    step_sink = step_writer.write_batch
    t0 = time.time()

    def progress(msg: str) -> None:
        """将进度消息打印到标准输出。

        参数:
            msg: 进度描述文本。
        """
        print(msg)

    if not args.only_b and not args.only_c:
        print(f"场景A（歼-20，{len(SCENARIO_A_AGENTS)} 智能体）")
        print(
            f"  共享任务 INF_A_001 ~ INF_A_{args.n_a:03d}，"
            f"每任务 {args.min_sims_a}~{args.max_sims_a} 局"
        )
        generate_scenario_a(
            args.n_a, args.min_sims_a, args.max_sims_a, task_rows, step_sink, progress=progress
        )
        if args.agent_tasks_a > 0:
            print(f"  每智能体独立任务 × {args.agent_tasks_a}")
            generate_per_agent_tasks(
                "A", "A",
                SCENARIO_A_AGENTS, SCENARIO_A_OBS_SPACE, SCENARIO_A_ACTION_ITEMS,
                _simulate_fighter,
                args.agent_tasks_a, args.min_sims_a, args.max_sims_a,
                task_rows, step_sink, progress=progress,
            )
        print(f"  多装备个体 INF_A_MULTI，{args.multi_min_sims}~{args.multi_max_sims} 局")
        generate_scenario_a_multi(
            task_rows, step_sink, min_sims=args.multi_min_sims, max_sims=args.multi_max_sims
        )
        if args.with_boundary:
            print("  边界测试 INF_A_SINGLE（1 局）")
            generate_single_sim_boundary(task_rows, step_sink)

    if not args.only_a and not args.only_c:
        print(
            f"场景B（雷达站）：INF_B_001 ~ INF_B_{args.n_b:03d}，"
            f"每任务 {args.min_sims_b}~{args.max_sims_b} 局"
        )
        generate_scenario_b(
            args.n_b, args.min_sims_b, args.max_sims_b, task_rows, step_sink, progress=progress
        )
        if args.agent_tasks_b > 0:
            generate_per_agent_tasks(
                "B", "B",
                SCENARIO_B_AGENTS, SCENARIO_B_OBS_SPACE, SCENARIO_B_ACTION_ITEMS,
                _simulate_radar,
                args.agent_tasks_b, args.min_sims_b, args.max_sims_b,
                task_rows, step_sink, progress=progress,
            )

    if not args.only_a and not args.only_b:
        print(
            f"场景C（侦察机）：INF_C_001 ~ INF_C_{args.n_c:03d}，"
            f"每任务 {args.min_sims_c}~{args.max_sims_c} 局"
        )
        generate_scenario_c(
            args.n_c, args.min_sims_c, args.max_sims_c, task_rows, step_sink, progress=progress
        )
        if args.agent_tasks_c > 0:
            generate_per_agent_tasks(
                "C", "C",
                SCENARIO_C_AGENTS, SCENARIO_C_OBS_SPACE, SCENARIO_C_ACTION_ITEMS,
                _simulate_recon,
                args.agent_tasks_c, args.min_sims_c, args.max_sims_c,
                task_rows, step_sink, progress=progress,
            )

    n_steps_written = step_writer.close()
    task_file = OUT_DIR / "inference_task.json"
    _write_task_json(task_file, task_rows, compact=args.compact)

    elapsed = time.time() - t0
    n_tasks = len({r["task_id"] for r in task_rows})
    total_decisions = sum(r["total_steps"] for r in task_rows)

    print(f"\n完成！耗时 {elapsed:.1f}s")
    print(f"  推理数据表：{task_file}  ({len(task_rows):,} 行 / {n_tasks} 个推理任务)")
    print(f"  步骤流水表：{OUT_DIR / 'inference_step.json'}  ({n_steps_written:,} 行)")
    print(f"  总局决策步数：{total_decisions:,}（所有仿真局 total_steps 之和）")
    print("\n示例调用：")
    print("  py main.py --mode explain_a --inference_task_id INF_A_001 --agent_id 1")
    print("  py main.py --mode explain_a --inference_task_id INF_A_AG1_001 --agent_id 1")
    print("  py main.py --mode explain_a --inference_task_id INF_A_MULTI --agent_id 1")


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    返回:
        配置好各场景规模选项的 ``ArgumentParser`` 实例。
    """
    p = argparse.ArgumentParser(description="Generate mock inference data.")
    p.add_argument("--preset", choices=sorted(PRESETS.keys()), default=None,
                   help="规模预设：dev / medium / large / huge")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-steps", type=int, default=None, help="每局最小决策步数")
    p.add_argument("--max-steps", type=int, default=None, help="每局最大决策步数")

    p.add_argument("--n-a", type=int, default=None, help="场景A共享推理任务数")
    p.add_argument("--min-sims-a", type=int, default=None, help="场景A每任务最少局数")
    p.add_argument("--max-sims-a", type=int, default=None, help="场景A每任务最多局数")
    p.add_argument("--sims-a", type=int, default=None, help="场景A固定局数（等同 min=max）")
    p.add_argument("--agent-tasks-a", type=int, default=None, help="场景A每智能体独立任务数")

    p.add_argument("--n-b", type=int, default=None)
    p.add_argument("--min-sims-b", type=int, default=None)
    p.add_argument("--max-sims-b", type=int, default=None)
    p.add_argument("--sims-b", type=int, default=None)
    p.add_argument("--agent-tasks-b", type=int, default=None)

    p.add_argument("--n-c", type=int, default=None)
    p.add_argument("--min-sims-c", type=int, default=None)
    p.add_argument("--max-sims-c", type=int, default=None)
    p.add_argument("--sims-c", type=int, default=None)
    p.add_argument("--agent-tasks-c", type=int, default=None)

    p.add_argument("--multi-min-sims", type=int, default=None, help="INF_A_MULTI 最少局数")
    p.add_argument("--multi-max-sims", type=int, default=None, help="INF_A_MULTI 最多局数")
    p.add_argument("--multi-sims", type=int, default=None, help="INF_A_MULTI 固定局数")
    p.add_argument("--with-boundary", action="store_true", help="生成 INF_A_SINGLE")
    p.add_argument("--no-boundary", action="store_true", help="不生成 INF_A_SINGLE")
    p.add_argument("--compact", action="store_true", help="紧凑 JSON（大规模推荐）")
    p.add_argument("--only-a", action="store_true", help="只生成场景A")
    p.add_argument("--only-b", action="store_true", help="只生成场景B")
    p.add_argument("--only-c", action="store_true", help="只生成场景C")
    return p


def _resolve_args(argv: Optional[List[str]] = None):
    """解析命令行参数并与 preset 默认值合并。

    参数:
        argv: 命令行参数列表；为 ``None`` 时使用 ``sys.argv``。

    返回:
        合并 preset 与 CLI 覆盖后的命名空间。
    """
    p = _build_parser()
    ns = p.parse_args(argv)

    base = dict(PRESETS["dev"])
    if ns.preset:
        base.update(PRESETS[ns.preset])

    def pick(name: str, cli_val):
        """CLI 显式值优先，否则回退到 preset 基线。

        参数:
            name: preset 字典中的键名。
            cli_val: 命令行传入值（可为 ``None``）。

        返回:
            最终采用的配置值。
        """
        return cli_val if cli_val is not None else base.get(name)

    ns.min_steps = pick("min_steps", ns.min_steps)
    ns.max_steps = pick("max_steps", ns.max_steps)
    ns.n_a = pick("n_a", ns.n_a)
    ns.agent_tasks_a = pick("agent_tasks_a", ns.agent_tasks_a)
    ns.n_b = pick("n_b", ns.n_b)
    ns.agent_tasks_b = pick("agent_tasks_b", ns.agent_tasks_b)
    ns.n_c = pick("n_c", ns.n_c)
    ns.agent_tasks_c = pick("agent_tasks_c", ns.agent_tasks_c)

    def _resolve_sims_range(tag: str) -> None:
        """解析并写回某场景（a/b/c）的 min/max 仿真局数。

        参数:
            tag: 场景标签（``"a"``、``"b"`` 或 ``"c"``）。
        """
        fixed = getattr(ns, f"sims_{tag}")
        min_v = getattr(ns, f"min_sims_{tag}")
        max_v = getattr(ns, f"max_sims_{tag}")
        if min_v is None:
            min_v = base.get(f"min_sims_{tag}")
        if max_v is None:
            max_v = base.get(f"max_sims_{tag}")
        if fixed is not None:
            min_v = max_v = fixed
        elif min_v is None and base.get(f"sims_{tag}") is not None:
            min_v = max_v = base[f"sims_{tag}"]
        if min_v is None:
            min_v = 3
        if max_v is None:
            max_v = min_v
        setattr(ns, f"min_sims_{tag}", min_v)
        setattr(ns, f"max_sims_{tag}", max_v)

    for _tag in ("a", "b", "c"):
        _resolve_sims_range(_tag)

    multi_fixed = ns.multi_sims
    multi_min = ns.multi_min_sims if ns.multi_min_sims is not None else base.get("multi_min_sims")
    multi_max = ns.multi_max_sims if ns.multi_max_sims is not None else base.get("multi_max_sims")
    if multi_fixed is not None:
        multi_min = multi_max = multi_fixed
    elif multi_min is None and base.get("multi_sims") is not None:
        multi_min = multi_max = base["multi_sims"]
    if multi_min is None:
        multi_min = 2
    if multi_max is None:
        multi_max = multi_min
    ns.multi_min_sims = multi_min
    ns.multi_max_sims = multi_max

    if ns.with_boundary:
        ns.with_boundary = True
    elif ns.no_boundary:
        ns.with_boundary = False
    else:
        ns.with_boundary = bool(base.get("with_boundary", False))

    if not ns.compact:
        ns.compact = bool(base.get("compact", False))

    if ns.min_steps > ns.max_steps:
        p.error("--min-steps 不能大于 --max-steps")

    random.seed(ns.seed)
    return ns


if __name__ == "__main__":
    run_generate(_resolve_args())
