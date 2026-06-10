"""
按智能体 + 观测/动作 schema 在本地持久化 Preprocessor 标尺（不入业务库）。

Profile 存归一化参数与相对分档边界；可选样本 reservoir 用于跨任务合并 refit。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.module_a_rules.preprocess import Preprocessor
from src.module_c_counterfactual.inference_record import InferenceRecord

_MAX_RESERVOIR = int(os.environ.get("AGENT_PROFILE_MAX_RESERVOIR", "500"))


def schema_fingerprint(record: InferenceRecord, agent_id: int) -> str:
    """
    计算观测空间、动作项与装备个体列表的稳定指纹哈希。

    参数:
        record: 推理数据记录。
        agent_id: 目标智能体 ID。

    返回:
        16 位十六进制 SHA256 摘要，用于区分不同 schema。
    """
    from src.module_c_counterfactual.agent_schema import AgentSchema

    schema = AgentSchema.from_record(record, agent_id)
    action_sig = [
        {"name": it.name, "values": list(it.possible_values), "continuous": it.is_continuous}
        for it in record.action_items
    ]
    payload = json.dumps(
        {
            "observation_space": list(schema.observation_space),
            "action_items": action_sig,
            "equipment_units": list(schema.equipment_units),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def profile_id_for(agent_id: int, fingerprint: str) -> str:
    """
    根据智能体 ID 与 schema 指纹生成 profile 唯一标识。

    参数:
        agent_id: 智能体 ID。
        fingerprint: schema 指纹字符串。

    返回:
        形如 ``agent{id}_{fingerprint}`` 的 profile ID。
    """
    return f"agent{agent_id}_{fingerprint}"


def _profiles_dir() -> Path:
    """
    返回智能体预处理器 profile 的本地存储目录。

    返回:
        由环境变量 ``ANALYSIS_OUTPUT_DIR`` 决定的 ``agent_profiles`` 子目录路径。
    """
    base = os.environ.get("ANALYSIS_OUTPUT_DIR", "./output")
    return Path(base) / "agent_profiles"


def profile_path(profile_id: str) -> Path:
    """
    根据 profile ID 构造对应的 JSON 文件路径。

    参数:
        profile_id: profile 唯一标识。

    返回:
        profile JSON 文件的完整路径。
    """
    return _profiles_dir() / f"{profile_id}.json"


@dataclass
class AgentPreprocessorProfile:
    """
    智能体预处理器标尺的持久化数据结构。

    保存归一化参数、自动分档边界、累计样本数及可选的样本 reservoir，
    供跨任务合并 refit 时复用，不写入业务数据库。
    """

    profile_id: str
    agent_id: int
    schema_fingerprint: str
    feature_names: List[str]
    mean: List[float]
    std: List[float]
    auto_bins: Dict[str, Tuple[List[float], List[str]]]
    n_samples: int
    version: int = 1
    updated_at: str = ""
    n_quantiles: int = 3
    discretize_config: Dict[str, Tuple[List[float], List[str]]] = field(default_factory=dict)
    reservoir: List[List[float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        将 profile 序列化为可 JSON 存储的字典。

        返回:
            包含归一化参数、分箱配置与 reservoir 的字典。
        """
        return {
            "profile_id": self.profile_id,
            "agent_id": self.agent_id,
            "schema_fingerprint": self.schema_fingerprint,
            "feature_names": self.feature_names,
            "mean": self.mean,
            "std": self.std,
            "auto_bins": {
                k: [list(edges), list(labels)]
                for k, (edges, labels) in self.auto_bins.items()
            },
            "n_samples": self.n_samples,
            "version": self.version,
            "updated_at": self.updated_at,
            "n_quantiles": self.n_quantiles,
            "discretize_config": {
                k: [list(edges), list(labels)]
                for k, (v_edges, v_labels) in self.discretize_config.items()
            },
            "reservoir": self.reservoir,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentPreprocessorProfile":
        """
        从字典反序列化 profile 对象。

        参数:
            data: 由 ``to_dict`` 或 JSON 文件加载的字典。

        返回:
            ``AgentPreprocessorProfile`` 实例。
        """
        auto_bins = {
            k: (list(v[0]), list(v[1]))
            for k, v in (data.get("auto_bins") or {}).items()
        }
        disc = {
            k: (list(v[0]), list(v[1]))
            for k, v in (data.get("discretize_config") or {}).items()
        }
        return cls(
            profile_id=str(data["profile_id"]),
            agent_id=int(data["agent_id"]),
            schema_fingerprint=str(data["schema_fingerprint"]),
            feature_names=list(data["feature_names"]),
            mean=[float(x) for x in data["mean"]],
            std=[float(x) for x in data["std"]],
            auto_bins=auto_bins,
            n_samples=int(data.get("n_samples", 0)),
            version=int(data.get("version", 1)),
            updated_at=str(data.get("updated_at", "")),
            n_quantiles=int(data.get("n_quantiles", 3)),
            discretize_config=disc,
            reservoir=[list(row) for row in (data.get("reservoir") or [])],
        )


def load_profile(profile_id: str) -> Optional[AgentPreprocessorProfile]:
    """
    从本地 JSON 文件加载智能体预处理器 profile。

    参数:
        profile_id: profile 唯一标识。

    返回:
        加载成功时返回 profile 对象；文件不存在时返回 ``None``。
    """
    path = profile_path(profile_id)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return AgentPreprocessorProfile.from_dict(json.load(f))


def save_profile(profile: AgentPreprocessorProfile) -> Path:
    """
    将 profile 写入本地 JSON 文件并更新 ``updated_at`` 时间戳。

    参数:
        profile: 待持久化的 profile 对象。

    返回:
        写入文件的完整路径。
    """
    _profiles_dir().mkdir(parents=True, exist_ok=True)
    profile.updated_at = datetime.now(timezone.utc).isoformat()
    path = profile_path(profile.profile_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)
    return path


def _subsample_reservoir(rows: List[List[float]], max_rows: int) -> List[List[float]]:
    """
    对样本 reservoir 做无放回随机下采样，控制内存占用。

    参数:
        rows: 原始特征行列表。
        max_rows: 允许保留的最大行数。

    返回:
        下采样后的特征行列表。
    """
    if len(rows) <= max_rows:
        return rows
    rng = np.random.default_rng(42)
    idx = rng.choice(len(rows), size=max_rows, replace=False)
    return [rows[int(i)] for i in idx]


def _merge_reservoir(
    old: List[List[float]],
    X_new: np.ndarray,
    max_rows: int = _MAX_RESERVOIR,
) -> List[List[float]]:
    """
    将历史 reservoir 与新样本合并并下采样。

    参数:
        old: 已有 reservoir 特征行。
        X_new: 本次新增样本矩阵。
        max_rows: 合并后允许的最大行数，默认由环境变量控制。

    返回:
        合并并下采样后的 reservoir。
    """
    combined = old + X_new.tolist()
    return _subsample_reservoir(combined, max_rows)


def fit_preprocessor_with_profile(
    X_raw: np.ndarray,
    feature_names: List[str],
    agent_id: int,
    record: InferenceRecord,
    *,
    n_quantiles: int = 3,
    discretize_config: Optional[Dict[str, Tuple]] = None,
    update_profile: bool = True,
) -> Tuple[Preprocessor, Optional[AgentPreprocessorProfile], Optional[Path]]:
    """
    加载同 agent+schema 的本地 profile（若有），在 reservoir+新数据上 fit，并写回 profile。

    参数:
        X_raw: 原始（未归一化）特征矩阵。
        feature_names: 与 ``X_raw`` 列对应的特征名列表。
        agent_id: 智能体 ID。
        record: 推理数据记录，用于计算 schema 指纹。
        n_quantiles: 自动分箱使用的分位数个数。
        discretize_config: 手动覆盖的分箱配置，键为特征名。
        update_profile: 是否在 fit 后更新并保存本地 profile。

    返回:
        三元组 ``(Preprocessor, profile, saved_path)``；
        未更新 profile 时后两项为 ``None``。
    """
    fp = schema_fingerprint(record, agent_id)
    pid = profile_id_for(agent_id, fp)
    existing = load_profile(pid)

    pre = Preprocessor(
        feature_names=feature_names,
        n_quantiles=n_quantiles,
        discretize_config=dict(discretize_config or {}),
    )

    X_fit = np.array(X_raw, dtype=float)
    if existing and existing.feature_names == feature_names:
        if existing.reservoir:
            X_res = np.array(existing.reservoir, dtype=float)
            if X_res.shape[1] == X_fit.shape[1]:
                X_fit = np.vstack([X_res, X_fit])

    pre.fit(X_fit)

    saved_path: Optional[Path] = None
    profile: Optional[AgentPreprocessorProfile] = None
    if update_profile:
        reservoir = _merge_reservoir(
            existing.reservoir if existing else [],
            np.array(X_raw, dtype=float),
        )
        fit_state = pre.export_fit_state()
        profile = AgentPreprocessorProfile(
            profile_id=pid,
            agent_id=agent_id,
            schema_fingerprint=fp,
            feature_names=list(feature_names),
            mean=list(fit_state["mean"]),  # type: ignore[arg-type]
            std=list(fit_state["std"]),  # type: ignore[arg-type]
            auto_bins={
                k: (list(edges), list(labels))
                for k, (edges, labels) in fit_state["auto_bins"].items()  # type: ignore[union-attr]
            },
            n_samples=int(
                (existing.n_samples if existing else 0) + len(X_raw)
            ),
            version=int(existing.version + 1 if existing else 1),
            n_quantiles=n_quantiles,
            discretize_config={
                k: (list(v[0]), list(v[1]))
                for k, v in (discretize_config or {}).items()
            },
            reservoir=reservoir,
        )
        saved_path = save_profile(profile)

    return pre, profile, saved_path
