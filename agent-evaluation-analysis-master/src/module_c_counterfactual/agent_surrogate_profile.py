"""
按智能体 + schema 在本地持久化 SurrogateBundle / PolicySurrogate（不入业务库）。

Profile 存转移 reservoir 与训练参数；joblib 存拟合好的 π/T/R 对象。
增量 = reservoir 合并 + 决策树重训，非神经网络权重热更新。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from src.module_a_rules.agent_profile import profile_id_for, schema_fingerprint
from src.module_c_counterfactual.inference_record import InferenceRecord
from src.module_c_counterfactual.policy_model import PolicySurrogate
from src.module_c_counterfactual.surrogate_bundle import SurrogateBundle
from src.module_c_counterfactual.training_data import iter_transitions

_MAX_RESERVOIR = int(os.environ.get("AGENT_SURROGATE_MAX_RESERVOIR", "2000"))


def is_surrogate_profile_enabled() -> bool:
    """
    是否启用磁盘 surrogate profile 持久化。

    返回:
        环境变量 ANALYSIS_CF_SURROGATE_PROFILE 非 0/false 时为 True。
    """
    flag = os.environ.get("ANALYSIS_CF_SURROGATE_PROFILE", "1").strip().lower()
    return flag not in ("0", "false", "no", "off")


def _profiles_dir() -> Path:
    """返回 agent surrogate profile 根目录路径。"""
    base = os.environ.get("ANALYSIS_OUTPUT_DIR", "./output")
    return Path(base) / "agent_surrogate_profiles"


def profile_json_path(profile_id: str) -> Path:
    """
    参数:
        profile_id: profile 唯一标识。

    返回:
        profile JSON 文件路径。
    """
    return _profiles_dir() / f"{profile_id}.json"


def bundle_joblib_path(profile_id: str) -> Path:
    """
    参数:
        profile_id: profile 唯一标识。

    返回:
        SurrogateBundle joblib 文件路径。
    """
    return _profiles_dir() / f"{profile_id}_bundle.joblib"


def policy_joblib_path(profile_id: str) -> Path:
    """
    参数:
        profile_id: profile 唯一标识。

    返回:
        PolicySurrogate joblib 文件路径。
    """
    return _profiles_dir() / f"{profile_id}_policy.joblib"


def fingerprint_to_str(fp: Tuple[Tuple[str, int], ...]) -> str:
    """
    将训练数据指纹元组序列化为 JSON 字符串。

    参数:
        fp: (sim_id, total_steps) 元组的有序序列。

    返回:
        JSON 字符串。
    """
    return json.dumps(list(fp), ensure_ascii=False)


def str_to_fingerprint(s: str) -> Tuple[Tuple[str, int], ...]:
    """将 JSON 字符串反序列化为训练数据指纹元组。"""
    raw = json.loads(s)
    return tuple((str(a), int(b)) for a, b in raw)


@dataclass
class AgentSurrogateProfile:
    """
    智能体代理模型本地 profile（转移 reservoir + 训练元数据）。

    用于跨会话增量合并训练样本并重训 π/T/R，不入业务数据库。
    """

    profile_id: str
    agent_id: int
    schema_fingerprint: str
    feature_names: List[str]
    policy_mode: str
    tree_params: Dict[str, Any]
    n_transitions: int
    version: int = 1
    updated_at: str = ""
    reservoir: List[Dict[str, Any]] = field(default_factory=list)
    seen_fingerprints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可 JSON 持久化的字典。"""
        return {
            "profile_id": self.profile_id,
            "agent_id": self.agent_id,
            "schema_fingerprint": self.schema_fingerprint,
            "feature_names": self.feature_names,
            "policy_mode": self.policy_mode,
            "tree_params": dict(self.tree_params),
            "n_transitions": self.n_transitions,
            "version": self.version,
            "updated_at": self.updated_at,
            "reservoir": self.reservoir,
            "seen_fingerprints": list(self.seen_fingerprints),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSurrogateProfile":
        """
        从字典反序列化 profile。

        参数:
            data: 由 to_dict() 或 JSON 加载得到的字典。

        返回:
            AgentSurrogateProfile 实例。
        """
        return cls(
            profile_id=str(data["profile_id"]),
            agent_id=int(data["agent_id"]),
            schema_fingerprint=str(data["schema_fingerprint"]),
            feature_names=list(data["feature_names"]),
            policy_mode=str(data.get("policy_mode", "joint")),
            tree_params=dict(data.get("tree_params") or {}),
            n_transitions=int(data.get("n_transitions", 0)),
            version=int(data.get("version", 1)),
            updated_at=str(data.get("updated_at", "")),
            reservoir=list(data.get("reservoir") or []),
            seen_fingerprints=list(data.get("seen_fingerprints") or []),
        )


def load_profile(profile_id: str) -> Optional[AgentSurrogateProfile]:
    """
    从磁盘加载 profile JSON。

    参数:
        profile_id: profile 唯一标识。

    返回:
        存在则返回实例，否则 None。
    """
    path = profile_json_path(profile_id)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return AgentSurrogateProfile.from_dict(json.load(f))


def save_profile(profile: AgentSurrogateProfile) -> Path:
    """
    将 profile 写入磁盘并更新 updated_at。

    参数:
        profile: 待保存的 profile 实例。

    返回:
        写入的文件路径。
    """
    _profiles_dir().mkdir(parents=True, exist_ok=True)
    profile.updated_at = datetime.now(timezone.utc).isoformat()
    path = profile_json_path(profile.profile_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)
    return path


def save_bundle(profile_id: str, bundle: SurrogateBundle) -> Path:
    """
    将 SurrogateBundle 序列化为 joblib 文件。

    参数:
        profile_id: profile 唯一标识。
        bundle: 拟合好的三模型 bundle。

    返回:
        joblib 文件路径。
    """
    _profiles_dir().mkdir(parents=True, exist_ok=True)
    path = bundle_joblib_path(profile_id)
    joblib.dump(bundle, path)
    return path


def load_bundle(profile_id: str) -> Optional[SurrogateBundle]:
    """
    从磁盘加载 SurrogateBundle。

    参数:
        profile_id: profile 唯一标识。

    返回:
        存在则返回 bundle，否则 None。
    """
    path = bundle_joblib_path(profile_id)
    if not path.is_file():
        return None
    return joblib.load(path)


def save_policy(profile_id: str, policy: PolicySurrogate) -> Path:
    """
    将 PolicySurrogate 序列化为 joblib 文件。

    参数:
        profile_id: profile 唯一标识。
        policy: 拟合好的策略近似模型。

    返回:
        joblib 文件路径。
    """
    _profiles_dir().mkdir(parents=True, exist_ok=True)
    path = policy_joblib_path(profile_id)
    joblib.dump(policy, path)
    return path


def load_policy(profile_id: str) -> Optional[PolicySurrogate]:
    """
    从磁盘加载 PolicySurrogate。

    参数:
        profile_id: profile 唯一标识。

    返回:
        存在则返回 policy，否则 None。
    """
    path = policy_joblib_path(profile_id)
    if not path.is_file():
        return None
    return joblib.load(path)


def tree_params_match(existing: Dict[str, Any], current: Dict[str, Any]) -> bool:
    """
    判断两份树训练参数是否完全一致（决定是否可增量合并 reservoir）。

    参数:
        existing: 已保存的 tree_params。
        current: 当前请求的 tree_params。

    返回:
        字典内容相等则为 True。
    """
    return dict(existing) == dict(current)


def extract_transition_rows(
    records: List[InferenceRecord],
    agent_id: int,
) -> List[Dict[str, Any]]:
    """
    从推理记录中提取转移样本行，供 reservoir 存储。

    参数:
        records: 推理记录列表。
        agent_id: 目标智能体编号。

    返回:
        每行含 sim_id、obs、action、next_obs、reward 的字典列表。
    """
    rows: List[Dict[str, Any]] = []
    for record in records:
        for obs_t, action, obs_t1, reward in iter_transitions([record], agent_id):
            rows.append(
                {
                    "sim_id": record.sim_id,
                    "obs": [float(x) for x in obs_t],
                    "action": action,
                    "next_obs": [float(x) for x in obs_t1],
                    "reward": float(reward),
                }
            )
    return rows


def _subsample_reservoir(rows: List[Dict[str, Any]], max_rows: int) -> List[Dict[str, Any]]:
    """
    对 reservoir 行做无放回随机下采样。

    参数:
        rows: 转移样本行列表。
        max_rows: 最大保留行数。

    返回:
        下采样后的行列表。
    """
    if len(rows) <= max_rows:
        return rows
    rng = np.random.default_rng(42)
    idx = rng.choice(len(rows), size=max_rows, replace=False)
    return [rows[int(i)] for i in idx]


def merge_reservoir(
    old: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
    max_rows: int = _MAX_RESERVOIR,
) -> List[Dict[str, Any]]:
    """
    合并旧 reservoir 与新转移行并下采样至上限。

    参数:
        old: 已有 reservoir 行。
        new_rows: 新增转移行。
        max_rows: reservoir 容量上限。

    返回:
        合并并下采样后的行列表。
    """
    return _subsample_reservoir(old + new_rows, max_rows)


def compute_obs_means_from_rows(rows: List[Dict[str, Any]]) -> List[float]:
    """
    从转移行计算观测与下一观测的逐维均值。

    参数:
        rows: 含 obs、next_obs 字段的转移行列表。

    返回:
        特征均值向量；无数据时返回空列表。
    """
    if not rows:
        return []
    dim = len(rows[0]["obs"])
    sums = np.zeros(dim, dtype=float)
    count = 0
    for row in rows:
        for vec in (row["obs"], row["next_obs"]):
            if len(vec) == dim:
                sums += np.array(vec, dtype=float)
                count += 1
    if count == 0:
        return []
    return (sums / count).tolist()


def profile_id_for_record(agent_id: int, record: InferenceRecord) -> str:
    """
    根据智能体 id 与记录的 schema 指纹生成 profile_id。

    参数:
        agent_id: 智能体编号。
        record: 推理记录（用于计算 schema 指纹）。

    返回:
        profile 唯一标识字符串。
    """
    return profile_id_for(agent_id, schema_fingerprint(record, agent_id))
