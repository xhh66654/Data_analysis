"""
data_flow — 记录 decision_tree 流水线中「样本行数」如何变化。

用于回答：从 CSV 全量（如 S0_5 约 49.7 万行）到决策树训练/生成，中间哪些步骤
**裁剪了行**、哪些步骤**只改列值或做有放回抽样（行数不变或抽样规模可配）**。

写出 data_flow_report.json，并在日志中打印与 decision_tree 类似的汇总表。
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 统一标注前缀，便于在代码中搜索：DATA-CROP
TAG = "DATA-CROP"


@dataclass
class DataFlowStage:
    """单步数据流转记录。"""
    id: str
    name_zh: str
    n_in: int
    n_out: int
    rows_delta: int
    crop_ratio: float          # n_out / n_in（相对上一步）
    crop_ratio_from_origin: float  # n_out / 第一步 n_in
    reduces_rows: bool         # 是否减少了「可用于下游」的行池规模
    module: str
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataFlowTracker:
    """贯穿 run_pipeline 的样本行数追踪器。"""
    origin_n: int = 0
    stages: List[DataFlowStage] = field(default_factory=list)

    def set_origin(self, n: int, *, source: str) -> None:
        self.origin_n = int(n)
        self.record(
            "00",
            "读取轨迹 CSV（全量基线）",
            n_in=0,
            n_out=n,
            reduces_rows=False,
            module="trajectory_io.load_trajectory_csv",
            note=f"数据源: {source}；此后所有 crop_ratio_from_origin 均相对此行数",
        )

    def record(
        self,
        stage_id: str,
        name_zh: str,
        *,
        n_in: int,
        n_out: int,
        reduces_rows: bool,
        module: str,
        note: str = "",
    ) -> None:
        n_in = int(n_in)
        n_out = int(n_out)
        prev = self.stages[-1].n_out if self.stages else self.origin_n
        if n_in <= 0 and prev > 0:
            n_in = prev
        base = self.origin_n if self.origin_n > 0 else max(n_in, 1)
        stage = DataFlowStage(
            id=stage_id,
            name_zh=name_zh,
            n_in=n_in,
            n_out=n_out,
            rows_delta=n_out - n_in,
            crop_ratio=round(n_out / max(n_in, 1), 6),
            crop_ratio_from_origin=round(n_out / base, 6),
            reduces_rows=reduces_rows,
            module=module,
            note=note,
        )
        self.stages.append(stage)
        flag = "【裁剪】" if reduces_rows else "【保留行数】"
        logger.info(
            "%s %s %s: %d → %d (Δ=%+d, 占全量=%.1f%%) %s",
            TAG,
            stage_id,
            name_zh,
            n_in,
            n_out,
            stage.rows_delta,
            100.0 * stage.crop_ratio_from_origin,
            note,
        )

    def build_report(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "tag": TAG,
            "summary_zh": self._summary_zh(),
            "origin_n": self.origin_n,
            "stages": [s.to_dict() for s in self.stages],
        }
        if extra:
            report.update(extra)
        return report

    def _summary_zh(self) -> str:
        if not self.stages:
            return "无流转记录"
        last = self.stages[-1]
        crops = [s for s in self.stages if s.reduces_rows]
        if not crops:
            return f"全流水线行数保持 {self.origin_n}，未做行级子集裁剪（决策树训练池见 VIPER 划分步骤）"
        parts = [f"{s.id}:{s.name_zh}({s.n_out})" for s in crops]
        return (
            f"全量 {self.origin_n} 行 → 末阶段记录 {last.n_out} 行；"
            f"发生行数缩减的步骤: {' → '.join(parts)}"
        )

    def log_table(self) -> None:
        if not self.stages:
            return
        logger.info("======== %s 样本流转汇总（相对 CSV 全量） ========", TAG)
        for s in self.stages:
            kind = "裁剪" if s.reduces_rows else "全量/池内"
            logger.info(
                "  [%s] %s %-6s  %8d → %8d  (占全量 %5.1f%%)  %s",
                kind,
                s.id,
                s.name_zh[:20],
                s.n_in,
                s.n_out,
                100.0 * s.crop_ratio_from_origin,
                s.note[:80] if s.note else s.module,
            )
        logger.info("  说明: 决策树 fit 用的是「VIPER 训练池」每轮 bootstrap 的 %d 条样本，见步骤 07",
                    self._tree_fit_sample_count())
        logger.info("====================================================")

    def _tree_fit_sample_count(self) -> int:
        for s in reversed(self.stages):
            if s.id == "07":
                return s.n_out
        for s in reversed(self.stages):
            if s.id == "06":
                return s.n_out
        return 0

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.build_report(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("已写入样本流转报告: %s", path.resolve())
        return path


def record_reward_norm(
    flow: DataFlowTracker,
    n: int,
    *,
    clipped_count: int,
    clip_range: tuple[float, float],
) -> None:
    """[DATA-CROP-01] 奖励裁剪：不改行数，只改 reward 列。"""
    flow.record(
        "01",
        "奖励裁剪+标准化",
        n_in=n,
        n_out=n,
        reduces_rows=False,
        module="trajectory_io.normalize_rewards",
        note=f"裁剪 {clipped_count} 个 reward 到 {clip_range}；行数不变",
    )


def record_episode_split(
    flow: DataFlowTracker,
    n_full: int,
    train_n: int,
    val_n: int,
    test_n: int,
    *,
    val_frac: float,
    test_frac: float,
    n_episodes: int,
) -> None:
    """[DATA-CROP-06] 按 episode 划分：决策树只在 train 池上训练（主要行数缩减）。"""
    train_frac = train_n / max(n_full, 1)
    flow.record(
        "06",
        "按 episode 划分 train/val/test",
        n_in=n_full,
        n_out=train_n,
        reduces_rows=True,
        module="viper_cart.split_by_episode",
        note=(
            f"val_frac={val_frac} test_frac={test_frac} episodes={n_episodes}；"
            f"train={train_n} val={val_n} test={test_n}；"
            f"仅 train 进入 VIPER 重采样与 CART fit（val/test 不参与训练）"
        ),
    )
    flow.record(
        "06b",
        "验证集/测试集（不参与树训练）",
        n_in=n_full,
        n_out=val_n + test_n,
        reduces_rows=True,
        module="viper_cart.split_by_episode",
        note=f"held-out val={val_n} test={test_n}，仅用于选轮与 metrics",
    )


def record_viper_bootstrap(
    flow: DataFlowTracker,
    train_pool_n: int,
    bootstrap_n: int,
    *,
    resample_size: int | None,
    n_round: int,
) -> None:
    """[DATA-CROP-07] VIPER 有放回重采样：每轮 CART 实际 fit 的样本数。"""
    if resample_size is not None and resample_size > 0:
        note = f"resample_size={resample_size} 显式上限；每轮 fit {bootstrap_n} 行（有放回）"
    else:
        note = f"resample_size=0 → m=训练池行数；每轮 fit {bootstrap_n} 行（有放回，可重复抽同一条）"
    flow.record(
        "07",
        "VIPER 重采样 → CART.fit",
        n_in=train_pool_n,
        n_out=bootstrap_n,
        reduces_rows=bootstrap_n < train_pool_n,
        module="viper_cart.resample_xy",
        note=note + f"；共 {n_round} 轮",
    )


def record_full_refit(
    flow: DataFlowTracker,
    n_full: int,
    bootstrap_m: int,
    *,
    selected_round: int,
) -> None:
    """[DATA-CROP-08] 在 train/val 上选轮后，用全表重训并导出最终树。"""
    flow.record(
        "08",
        "全量重训最终决策树（导出用）",
        n_in=n_full,
        n_out=n_full,
        reduces_rows=False,
        module="viper_cart.refit_final_tree_on_full_data",
        note=(
            f"选用第 {selected_round} 轮超参；在全部 {n_full} 行上有放回 bootstrap {bootstrap_m} 条后 fit；"
            "rules.txt / policy_tree.pdf 对应此全量树，非仅 train 子集"
        ),
    )


def record_rule_ensemble(
    flow: DataFlowTracker,
    n_rules_in: int,
    n_rules_out: int,
    *,
    max_rules: int,
) -> None:
    """[DATA-CROP-10] 规则条数裁剪（不裁轨迹行）。"""
    flow.record(
        "10",
        "规则集成 Top-K",
        n_in=n_rules_in,
        n_out=n_rules_out,
        reduces_rows=n_rules_out < n_rules_in,
        module="rule_ensemble.ensemble_rules_from_rounds",
        note=f"ensemble_max_rules={max_rules}；裁的是规则条数，不是 CSV 行",
    )
