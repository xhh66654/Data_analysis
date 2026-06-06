"""
dim_reduce — 状态维约简（反事实块归因）。

核心思路：
  对状态 s 的每个「语义块」（若干维度的组合），构造反事实状态 s_cf：
    将该块内所有维度替换为基线值（0 或训练集均值），
    对比 Q(s, a) 与 Q(s_cf, a)，差值绝对值越大表示该块越重要。

  重要度排序后取 Top-K 块，作为「主要决策因素」写入解释。

主要类型与函数：
  StateBlock        — 一个语义块（名称、维度索引、描述）
  BlockImportance   — 单块归因结果（名称、delta_Q、baseline 类型）
  load_block_map()  — 从 YAML 文件读取块配置
  compute_block_importance() — 对一个状态做全量块归因
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

from causal.decision_tree.q_network import QHatNetwork

logger = logging.getLogger(__name__)


@dataclass
class StateBlock:
    name: str
    dims: List[int]
    desc: str = ""


@dataclass
class BlockImportance:
    block_name: str
    block_desc: str
    dims: List[int]
    q_original: float
    q_counterfactual: float
    delta_q: float           # q_cf - q_orig（负值：该块贡献正 Q）
    abs_delta: float
    baseline: str            # "zero" | "mean"


_DEFAULT_BLOCKS: List[StateBlock] = [
    StateBlock("本机位置", [0, 1], "自身位置坐标 (s_0, s_1)"),
    StateBlock("本机姿态", [2, 3], "自身姿态角与偏差 (s_2, s_3)"),
    StateBlock("本机速度", [4, 5], "速度分量 (s_4, s_5)"),
    StateBlock("目标状态", [6, 7], "目标/威胁离散标志 (s_6, s_7)"),
]


def load_block_map(yaml_path: str | Path) -> List[StateBlock]:
    """
    从 YAML 文件读取块配置。

    YAML 格式示例::

        blocks:
          本机位置:
            dims: [0, 1]
            desc: "自身位置坐标"
          本机姿态:
            dims: [2, 3]
            desc: "姿态角与偏差"
    """
    path = Path(yaml_path)
    if not path.is_file():
        logger.warning("block_map YAML 不存在: %s，使用默认分块", path)
        return list(_DEFAULT_BLOCKS)

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or "blocks" not in data:
        logger.warning("YAML 中未找到 'blocks' 字段，使用默认分块")
        return list(_DEFAULT_BLOCKS)

    blocks: List[StateBlock] = []
    for name, cfg in data["blocks"].items():
        dims = cfg.get("dims", [])
        desc = cfg.get("desc", "")
        blocks.append(StateBlock(name=name, dims=dims, desc=desc))

    logger.info("已加载块配置: %s  共 %d 块", path.name, len(blocks))
    return blocks


def compute_block_importance(
    q_net: QHatNetwork,
    state: np.ndarray,
    action: int,
    blocks: List[StateBlock],
    baseline: str = "zero",
    state_means: Optional[np.ndarray] = None,
    device: str | torch.device = "cpu",
) -> List[BlockImportance]:
    """
    对 ``state`` 下选择的 ``action``，逐块做反事实：
    将该块维度替换为 baseline（zero 或 mean），计算 Q 值变化。

    返回按 abs_delta 降序排列的 BlockImportance 列表。

    参数
    ----
    q_net       : 已加载、eval 模式的 QHatNetwork
    state       : 当前步状态，shape (state_dim,)
    action      : 智能体实际执行的动作 id
    blocks      : 块配置列表
    baseline    : "zero"（全零）| "mean"（训练集均值，需传 state_means）
    state_means : baseline="mean" 时必填，shape (state_dim,)
    device      : 计算设备
    """
    if baseline == "mean" and state_means is None:
        logger.warning("baseline=mean 但未提供 state_means，自动回退到 zero")
        baseline = "zero"

    dev = torch.device(device)
    q_net = q_net.to(dev)
    q_net.eval()

    def _q_val(s: np.ndarray) -> float:
        t = torch.from_numpy(s).unsqueeze(0).float().to(dev)
        with torch.no_grad():
            q_all = q_net(t)
        return float(q_all[0, action].item())

    q_orig = _q_val(state)

    results: List[BlockImportance] = []
    for blk in blocks:
        if not blk.dims:
            continue
        s_cf = state.copy()
        for d in blk.dims:
            if d >= len(s_cf):
                logger.warning("块 %s 维度 %d 超出状态维度 %d，跳过", blk.name, d, len(s_cf))
                continue
            s_cf[d] = 0.0 if baseline == "zero" else float(state_means[d])

        q_cf = _q_val(s_cf)
        delta = q_cf - q_orig

        results.append(BlockImportance(
            block_name=blk.name,
            block_desc=blk.desc,
            dims=blk.dims,
            q_original=q_orig,
            q_counterfactual=q_cf,
            delta_q=delta,
            abs_delta=abs(delta),
            baseline=baseline,
        ))

    results.sort(key=lambda x: x.abs_delta, reverse=True)
    return results
