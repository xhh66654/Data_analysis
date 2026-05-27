"""
Top-K 关键邻居确定后，按异构群体划分对「保留邻居」做平均场（Mean Field）聚合：
  对每个群体 G：s̄_G = mean_j s_j,  ā_G = mean_j a_j（j 为属于 G 且在本步仍保留的邻居）
再与目标智能体状态/动作拼接为低维向量，供后续因果/SCM 使用。

轨迹列与 JSON 解析方式与 trajectory_maddpg 一致；群体标签由外部 JSON 配置（agent_id -> 群体名），
未出现在配置中的保留邻居归入默认池名称（见 default_pool_name，默认 pooled_neighbors，即单一均值场）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .trajectory_maddpg import META_COLS, _parse_agent_cell

logger = logging.getLogger(__name__)


def load_group_map_json(path: str | Path | None) -> dict[str, str]:
    """JSON 格式：{\"agent_1\": \"enemy_attack\", \"agent_2\": \"ally_support\", ...}"""
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"群体映射文件不存在: {p.resolve()}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("群体映射须为 JSON 对象：agent_id -> 群体名字符串")
    out: dict[str, str] = {}
    for k, v in data.items():
        out[str(k)] = str(v)
    return out


def _bucket_neighbors_by_group(
    kept_neighbors: list[str],
    group_map: dict[str, str],
    default_pool_name: str,
) -> tuple[list[str], dict[str, list[str]]]:
    """返回按名字排序的群体列表，以及 群体 -> 成员 agent id 列表。"""
    buckets: dict[str, list[str]] = {}
    for aid in kept_neighbors:
        g = group_map.get(aid, default_pool_name)
        buckets.setdefault(g, []).append(aid)
    ordered = sorted(buckets.keys())
    return ordered, buckets


def _infer_dims_from_row(df: pd.DataFrame, target_agent: str, row_idx: int = 0) -> tuple[int, int]:
    rec = _parse_agent_cell(df.iloc[row_idx][target_agent])
    o = np.asarray(rec["obs"], dtype=np.float32).reshape(-1)
    a = np.asarray(rec["action"], dtype=np.float32).reshape(-1)
    return int(o.size), int(a.size)


def build_mean_field_features(
    df: pd.DataFrame,
    *,
    target_agent: str,
    kept_neighbors: list[str],
    group_map: dict[str, str],
    default_pool_name: str = "pooled_neighbors",
    include_obs_std: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    每行输出拼接向量（浮点）：
      [ s_target, a_target,
        对每个群体 G（按名称排序）:  s_mean_G, a_mean_G,  (可选) std_obs_G ]
    若 kept_neighbors 为空，则仅 [s_target, a_target]。
    """
    n = len(df)
    if n == 0:
        raise ValueError("空轨迹")

    obs_dim, act_dim = _infer_dims_from_row(df, target_agent, 0)
    groups_sorted, buckets = _bucket_neighbors_by_group(kept_neighbors, group_map, default_pool_name)

    blocks: list[tuple[str, int]] = [("s_target", obs_dim), ("a_target", act_dim)]
    for g in groups_sorted:
        blocks.append((f"s_mean__{g}", obs_dim))
        blocks.append((f"a_mean__{g}", act_dim))
        if include_obs_std:
            blocks.append((f"s_std__{g}", obs_dim))

    feat_dim = sum(sz for _, sz in blocks)
    X = np.zeros((n, feat_dim), dtype=np.float32)
    name_to_slice: dict[str, slice] = {}
    offset = 0
    for name, sz in blocks:
        name_to_slice[name] = slice(offset, offset + sz)
        offset += sz

    agent_cols = [c for c in df.columns if c.startswith("agent_")]

    for i in range(n):
        row = df.iloc[i]
        tgt = _parse_agent_cell(row[target_agent])
        s_t = np.asarray(tgt["obs"], dtype=np.float32).reshape(-1)
        a_t = np.asarray(tgt["action"], dtype=np.float32).reshape(-1)
        if s_t.size != obs_dim or a_t.size != act_dim:
            raise ValueError(f"行{i} 目标观测/动作维度异常")
        X[i, name_to_slice["s_target"]] = s_t
        X[i, name_to_slice["a_target"]] = a_t

        for g in groups_sorted:
            members = buckets[g]
            if not members:
                continue
            obs_stack = []
            act_stack = []
            for aid in members:
                if aid not in agent_cols:
                    raise ValueError(f"群体 {g} 中 {aid} 非 CSV agent 列")
                cell = _parse_agent_cell(row[aid])
                obs_stack.append(np.asarray(cell["obs"], dtype=np.float32).reshape(-1))
                act_stack.append(np.asarray(cell["action"], dtype=np.float32).reshape(-1))
            os_ = np.stack(obs_stack, axis=0)
            as_ = np.stack(act_stack, axis=0)
            s_mean = os_.mean(axis=0)
            a_mean = as_.mean(axis=0)
            X[i, name_to_slice[f"s_mean__{g}"]] = s_mean
            X[i, name_to_slice[f"a_mean__{g}"]] = a_mean
            if include_obs_std:
                sd = os_.std(axis=0)
                np.nan_to_num(sd, nan=0.0, copy=False)
                X[i, name_to_slice[f"s_std__{g}"]] = sd

    meta = df[list(META_COLS)].reset_index(drop=True)
    mf_cols = [f"mf_{j}" for j in range(feat_dim)]
    out_df = pd.concat([meta, pd.DataFrame(X, columns=mf_cols)], axis=1)

    # 附加标量（便于 SCM 直接使用）
    r_tgt = []
    done_any = []
    for i in range(n):
        row = df.iloc[i]
        tg = _parse_agent_cell(row[target_agent])
        r_tgt.append(float(tg["reward"]))
        dg = False
        for c in agent_cols:
            rc = _parse_agent_cell(row[c])
            dg = dg or bool(rc.get("terminated")) or bool(rc.get("truncated")) or bool(rc.get("done"))
        done_any.append(1.0 if dg else 0.0)

    out_df["reward_target"] = r_tgt
    out_df["done_any_agent"] = done_any

    block_specs: list[dict[str, Any]] = []
    pos = 0
    for nm, ln in blocks:
        block_specs.append({"name": nm, "length": ln, "start": pos})
        pos += ln

    schema: dict[str, Any] = {
        "interpretation": "目标体 + 分群体平均观测/平均动作 (+ 可选观测标准差)，用于均值场近似",
        "target_agent": target_agent,
        "kept_neighbors_for_mf": list(kept_neighbors),
        "group_order": groups_sorted,
        "group_buckets": {k: list(v) for k, v in buckets.items()},
        "default_pool_if_unmapped": default_pool_name,
        "obs_dim": obs_dim,
        "action_dim_each": act_dim,
        "include_obs_std": include_obs_std,
        "blocks": block_specs,
        "aggregate_formulas": {"s_mean_group": "(1/N) sum s_j", "a_mean_group": "(1/N) sum a_j"},
    }

    logger.info(
        "平均场特征 n=%s dim=%s 群体数=%s (目标+群体统计，用于 1-vs-均值场近似)",
        n,
        feat_dim,
        len(groups_sorted),
    )
    return out_df, schema


def _validate_obs_blocks(obs_blocks: dict[str, list[int]], obs_dim: int) -> None:
    """验证 obs_blocks 完整覆盖 [0, obs_dim) 且块间不重叠。"""
    covered = np.zeros(obs_dim, dtype=bool)
    for name, (start, end) in obs_blocks.items():
        if start < 0 or end > obs_dim or start >= end:
            raise ValueError(f"obs_block '{name}' 范围 [{start},{end}) 越界 (obs_dim={obs_dim})")
        if covered[start:end].any():
            raise ValueError(f"obs_block '{name}' [{start},{end}) 与其他块重叠")
        covered[start:end] = True
    if not covered.all():
        missing = np.where(~covered)[0]
        raise ValueError(f"obs_blocks 未覆盖观测维度: {missing.tolist()}")


def build_self_obs_features(
    df: pd.DataFrame,
    *,
    target_agent: str,
    obs_blocks: dict[str, list[int]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    单智能体决策约简：将 s_target 按 obs_blocks 拆为语义子块，输出与 SCM 兼容的格式。

    obs_blocks 示例:
        {"s_self_pos": [0,2], "s_self_vel": [2,4], "s_landmarks": [4,16], ...}
    必须完整覆盖 [0, obs_dim) 且块间不重叠。

    输出每行: [s_block1, s_block2, ..., a_target]
    schema.blocks 可直接被 build_scm_tensors 消费。
    """
    n = len(df)
    if n == 0:
        raise ValueError("空轨迹")

    obs_dim, act_dim = _infer_dims_from_row(df, target_agent, 0)
    _validate_obs_blocks(obs_blocks, obs_dim)

    # 按 start 排序构建块列表
    sorted_blocks = sorted(obs_blocks.items(), key=lambda x: x[1][0])
    block_list: list[tuple[str, int]] = []
    for name, (start, end) in sorted_blocks:
        block_list.append((name, end - start))
    block_list.append(("a_target", act_dim))

    feat_dim = sum(sz for _, sz in block_list)
    X = np.zeros((n, feat_dim), dtype=np.float32)
    name_to_slice: dict[str, slice] = {}
    offset = 0
    for name, sz in block_list:
        name_to_slice[name] = slice(offset, offset + sz)
        offset += sz

    for i in range(n):
        row = df.iloc[i]
        tgt = _parse_agent_cell(row[target_agent])
        s_t = np.asarray(tgt["obs"], dtype=np.float32).reshape(-1)
        a_t = np.asarray(tgt["action"], dtype=np.float32).reshape(-1)

        for name, (start, end) in sorted_blocks:
            sl = name_to_slice[name]
            X[i, sl] = s_t[start:end]
        X[i, name_to_slice["a_target"]] = a_t

    mf_cols = [f"mf_{j}" for j in range(feat_dim)]
    meta_cols = ["episode", "global_step", "episode_step"]
    available_meta = [c for c in meta_cols if c in df.columns]
    out_df = df[available_meta].copy() if available_meta else pd.DataFrame(index=df.index)
    for j, col in enumerate(mf_cols):
        out_df[col] = X[:, j]

    schema_blocks: list[dict[str, Any]] = []
    for name, sz in block_list:
        sl = name_to_slice[name]
        schema_blocks.append({"name": name, "start": int(sl.start), "length": int(sl.stop - sl.start)})

    schema: dict[str, Any] = {
        "blocks": schema_blocks,
        "csv_columns": list(out_df.columns),
        "mode": "single_agent_self_obs",
        "obs_blocks_used": obs_blocks,
        "obs_dim": obs_dim,
        "action_dim_each": act_dim,
        "target_agent": target_agent,
    }

    logger.info(
        "自身观测分块 n=%d dim=%d 块数=%d (s_target 拆分为 %d 个语义块 + a_target)",
        n, feat_dim, len(block_list), len(obs_blocks),
    )
    return out_df, schema
