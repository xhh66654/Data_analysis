"""
基于观测数据的 VIPER 策略抽取（近似 VIPER，无需真实环境）。

================================================================================
说明（小白友好版）：
================================================================================

原版 VIPER（Bastani et al. 2018）需要一个可以反复交互的环境。
但我们只有推理记录数据（InferenceRecord），所以做近似：

    把多条推理记录当做"专家轨迹"，
    每次迭代对决策树错误分类的样本加权，重新训练，
    让决策树逐步修正在困难样本上的表现。

================================================================================
动态适配说明：
================================================================================

观测空间和动作空间因智能体/装备类型而异，不做任何硬编码：
    - 特征名从 collect_from_record 返回的 feature_names 中获取（展平后的复合键，
      如 "自身状态.血量"、"敌机距离.水平距离_km"）
    - 动作空间和动作项名称从 InferenceRecord 中读取
    - 每次训练需指定 action_item（要解释的动作维度），
      例如 "机动控制"、"雷达开关控制"

使用方式：
    viper = VIPERData.from_record(record, agent_id=1, action_item="机动控制")
    result = viper.run(n_iters=5)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from src.module_a_rules.collect_data import compute_return_to_go
from src.module_a_rules.preprocess import Preprocessor
from src.module_c_counterfactual.inference_record import InferenceRecord


@dataclass
class VIPERResult:
    """
    VIPER 训练结果。

    Attributes
    ----------
    best_tree     : 准确率最高的那棵决策树
    best_accuracy : 最佳决策树在训练数据上的准确率（0~1）
    preprocessor  : 与 best_tree 配套的预处理器（归一化参数）
    feature_names : 展平后的特征名列表（如 "自身状态.血量"），与决策树列对应
    action_item   : 本次训练针对的动作项名称（如 "机动控制"）
    action_space  : 该动作项的所有可选值列表
    history       : 每轮迭代的 (轮次, 加权准确率) 记录
    loss_history  : 每轮迭代的 (轮次, 加权损失=1-加权准确率)
    acc_orig_history : 每轮在原始样本集上的准确率
    augmentation_history : 每轮增广后的样本数等信息
    """
    best_tree: DecisionTreeClassifier
    best_accuracy: float
    best_loss: float
    preprocessor: Preprocessor
    feature_names: List[str]
    action_item: str
    action_space: List[str]
    history: List[Tuple[int, float]] = field(default_factory=list)
    loss_history: List[Tuple[int, float]] = field(default_factory=list)
    acc_orig_history: List[Tuple[int, float]] = field(default_factory=list)
    augmentation_history: List[Dict[str, object]] = field(default_factory=list)


class VIPERData:
    """
    基于推理数据的近似 VIPER，支持任意观测空间和动作空间结构。

    使用方式：
        # 从单条推理记录训练（指定要分析的动作项）
        viper = VIPERData.from_record(record, agent_id=1, action_item="机动控制")
        result = viper.run(n_iters=5)

        # 从多条推理记录训练（样本更多，效果更好）
        viper = VIPERData.from_records([r1, r2, r3], agent_id=1, action_item="武器控制")
        result = viper.run(n_iters=5)
    """

    def __init__(
        self,
        X_raw: np.ndarray,
        y: np.ndarray,
        rewards: np.ndarray,
        feature_names: List[str],
        action_item: str,
        action_space: List[str],
        max_depth: int = 6,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        episode_lengths: Optional[List[int]] = None,
        preprocessor: Optional[Preprocessor] = None,
        uniform_base_weights: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        X_raw         : (N, n_features) 原始（未归一化）展平观测特征矩阵
        y             : (N,) 动作标签数组（某动作项的选定值）
        rewards       : (N,) 每步奖励值（用于计算 return-to-go 权重）
        feature_names : 展平后的特征名列表，顺序与 X_raw 列对应
                        e.g. ["自身状态.血量", "自身状态.速度_马赫", "敌机距离.水平距离_km", ...]
        action_item   : 要训练的动作项名称，e.g. "机动控制"
        action_space  : 该动作项的可选值列表，e.g. ["规避", "追击", "保持"]
        max_depth     : 决策树最大深度
        min_samples_leaf : 叶节点最少样本数
        random_state  : 随机种子，控制训练可复现性
        episode_lengths : 各条记录贡献的样本段长度，用于分段计算 return-to-go
        preprocessor  : 可选的已 fit 预处理器；为 None 时自动创建并 fit
        uniform_base_weights : 为 True 时使用均等初始权重，跳过分段 RTG
        """
        self.feature_names = feature_names
        self.action_item = action_item
        self.action_space = action_space
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self._X_raw = np.array(X_raw, dtype=float)
        self.y = np.asarray(y)
        if uniform_base_weights:
            self._base_weights = np.ones(len(self.y), dtype=float)
        else:
            self._base_weights = _compute_segmented_return_to_go(
                np.array(rewards, dtype=float),
                episode_lengths=episode_lengths,
            )
        self._pre = preprocessor if preprocessor is not None else Preprocessor(
            feature_names=feature_names
        )
        if preprocessor is None or preprocessor._mean is None:
            self.X = self._pre.fit_transform(self._X_raw)
        else:
            self.X = self._pre.transform(self._X_raw)
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # 工厂方法：从 InferenceRecord 直接构建
    # ------------------------------------------------------------------
    @classmethod
    def from_record(
        cls,
        record: InferenceRecord,
        agent_id: int,
        action_item: str,
        max_depth: int = 6,
        min_samples_leaf: int = 2,
        random_state: int = 42,
    ) -> "VIPERData":
        """
        从单条推理记录构建 VIPERData。

        Parameters
        ----------
        record      : 推理数据记录（从数据库加载）
        agent_id    : 要训练哪个智能体的决策树
        action_item : 要分析的动作项名称，e.g. "机动控制"
                      必须是 record.action_space 中的一个值
        max_depth     : 决策树最大深度
        min_samples_leaf : 叶节点最少样本数
        random_state  : 随机种子

        返回:
            配置好数据的 ``VIPERData`` 实例。
        """
        from src.module_a_rules.collect_data import collect_from_record
        X_raw, y, rewards, feature_names = collect_from_record(record, agent_id, action_item)

        # 从动作项定义中找到该动作项的可选值
        action_space = _get_action_possible_values(record, action_item)

        return cls(
            X_raw=X_raw,
            y=y,
            rewards=rewards,
            feature_names=feature_names,
            action_item=action_item,
            action_space=action_space,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            episode_lengths=[int(len(y))] if len(y) else [],
        )

    @classmethod
    def from_records(
        cls,
        records: List[InferenceRecord],
        agent_id: int,
        action_item: str,
        max_depth: int = 6,
        min_samples_leaf: int = 2,
        random_state: int = 42,
    ) -> "VIPERData":
        """
        从多条推理记录合并后构建 VIPERData。

        注意：所有 records 必须来自同一种智能体/装备类型，
              即 observation_space 和 action_items 结构一致。

        Parameters
        ----------
        records     : 多条推理数据记录（同一智能体/装备类型）
        agent_id    : 要训练哪个智能体的决策树
        action_item : 要分析的动作项名称
        max_depth     : 决策树最大深度
        min_samples_leaf : 叶节点最少样本数
        random_state  : 随机种子

        返回:
            合并多条记录后的 ``VIPERData`` 实例。
        """
        from src.module_a_rules.collect_data import collect_from_records_with_segments
        X_raw, y, rewards, feature_names, segment_lengths = collect_from_records_with_segments(
            records, agent_id, action_item
        )

        # 以第一条记录为准获取动作项可选值（已校验结构一致性）
        record0 = records[0]
        _assert_same_structure(records, agent_id)
        action_space = _get_action_possible_values(record0, action_item)

        return cls(
            X_raw=X_raw,
            y=y,
            rewards=rewards,
            feature_names=feature_names,
            action_item=action_item,
            action_space=action_space,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            episode_lengths=segment_lengths,
        )

    # ------------------------------------------------------------------
    # 步骤4：训练一棵带权重的 CART 决策树
    # ------------------------------------------------------------------
    def train_once(
        self,
        sample_weights: Optional[np.ndarray] = None,
    ) -> DecisionTreeClassifier:
        """
        用当前数据集训练一棵 CART 决策树（带基尼指数 + 样本权重）。

        Parameters
        ----------
        sample_weights : 样本权重数组；为 None 时使用 return-to-go 初始权重

        返回:
            训练完成的 ``DecisionTreeClassifier`` 实例。
        """
        weights = sample_weights if sample_weights is not None else self._base_weights

        tree = DecisionTreeClassifier(
            criterion="gini",
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        tree.fit(self.X, self.y, sample_weight=weights)
        return tree

    # ------------------------------------------------------------------
    # 迭代优化主流程
    # ------------------------------------------------------------------
    def run(
        self,
        n_iters: int = 5,
        penalty_factor: float = 2.0,
        *,
        resample_augment: bool = True,
        augment_size_factor: float = 1.5,
    ) -> VIPERResult:
        """
        迭代训练决策树，每轮对错误样本加权，逐步提升准确率。

        离线 VIPER 增广：在 penalty 加权后，按权重有放回重采样扩充训练集 D'，
        使错分/高 RTG 样本在下一轮出现多次（近似论文重采样数据集）。

        Parameters
        ----------
        n_iters        : 迭代轮数，默认 5 轮
        penalty_factor : 错误样本权重的惩罚倍率，默认 2.0（即错误样本权重翻倍）
        resample_augment : 是否在每轮（末轮除外）按权重重采样扩充 D
        augment_size_factor : 重采样后目标样本数 ≈ factor * 当前样本数

        Returns
        -------
        VIPERResult，包含最佳决策树和训练历史
        """
        history: List[Tuple[int, float]] = []
        loss_history: List[Tuple[int, float]] = []
        acc_orig_history: List[Tuple[int, float]] = []
        aug_history: List[Dict[str, object]] = []
        best_tree: Optional[DecisionTreeClassifier] = None
        best_acc: float = -1.0
        current_weights = self._base_weights.copy()
        # 在增广前固定一份原始训练集，用于选最优树（避免在扩增/加权集上过拟合）
        X_orig = self.X.copy()
        y_orig = self.y.copy()

        for i in range(n_iters):
            tree = self.train_once(sample_weights=current_weights)

            y_pred_aug = tree.predict(self.X)
            acc_weighted = float(
                accuracy_score(self.y, y_pred_aug, sample_weight=current_weights)
            )
            history.append((i, acc_weighted))
            loss_history.append((i, 1.0 - acc_weighted))

            y_pred_orig = tree.predict(X_orig)
            acc_orig = float(accuracy_score(y_orig, y_pred_orig))
            acc_orig_history.append((i, acc_orig))
            if acc_orig > best_acc:
                best_acc = acc_orig
                best_tree = tree

            error_mask = (y_pred_aug != self.y).astype(float)
            n_errors = int(error_mask.sum())
            penalty = 1.0 + error_mask * (penalty_factor - 1.0)
            current_weights = current_weights * penalty
            w_sum = current_weights.sum()
            if w_sum > 0:
                current_weights = current_weights / w_sum * len(current_weights)

            if resample_augment and i < n_iters - 1 and len(self.y) > 0:
                before_n = len(self.y)
                current_weights = self._resample_dataset_inplace(
                    current_weights, augment_size_factor
                )
                aug_history.append(
                    {
                        "iter": i,
                        "n_before": before_n,
                        "n_after": len(self.y),
                        "n_errors": n_errors,
                        "augment_size_factor": augment_size_factor,
                    }
                )

        return VIPERResult(
            best_tree=best_tree,
            best_accuracy=best_acc,
            best_loss=1.0 - best_acc if best_acc >= 0 else 1.0,
            preprocessor=self._pre,
            feature_names=self.feature_names,
            action_item=self.action_item,
            action_space=self.action_space,
            history=history,
            loss_history=loss_history,
            acc_orig_history=acc_orig_history,
            augmentation_history=aug_history,
        )

    def _resample_dataset_inplace(
        self,
        weights: np.ndarray,
        size_factor: float,
    ) -> np.ndarray:
        """
        按样本权重有放回重采样，就地扩充训练集（VIPER 增广步骤）。

        参数:
            weights: 当前轮样本权重数组。
            size_factor: 目标样本数相对当前样本数的倍率。

        返回:
            重采样后重新归一化的权重数组。
        """
        n = len(self.y)
        target_n = max(n, int(round(n * size_factor)))
        probs = np.asarray(weights, dtype=float)
        if probs.sum() <= 0:
            probs = np.ones(n, dtype=float)
        probs = probs / probs.sum()
        idx = self._rng.choice(n, size=target_n, replace=True, p=probs)
        self.X = self.X[idx]
        self.y = self.y[idx]
        self._X_raw = self._X_raw[idx]
        new_w = np.asarray(weights, dtype=float)[idx]
        s = new_w.sum()
        if s > 0:
            new_w = new_w / s * len(new_w)
        return new_w


# ==============================================================================
# 内部工具函数
# ==============================================================================

def _get_action_possible_values(record: InferenceRecord, action_item: str) -> List[str]:
    """
    从 InferenceRecord 的 action_items 中取指定动作项的可选值列表。

    参数:
        record: 推理数据记录。
        action_item: 动作项名称。

    返回:
        可选值字符串列表；动作项不存在或为连续型时返回空列表。
    """
    for item in record.action_items:
        if item.name == action_item:
            return [str(v) for v in item.possible_values]
    return []


def _assert_same_structure(records: List[InferenceRecord], agent_id: int) -> None:
    """
    断言多条记录对同一 agent 的 schema 一致。

    参数:
        records: 推理记录列表。
        agent_id: 目标智能体 ID。

    抛出:
        ValueError: schema 不一致时。
    """
    from src.module_c_counterfactual.agent_schema import assert_same_agent_schema
    assert_same_agent_schema(records, agent_id)


def _compute_segmented_return_to_go(
    rewards: np.ndarray,
    episode_lengths: Optional[List[int]],
    gamma: float = 1.0,
) -> np.ndarray:
    """
    按 episode 分段计算 RTG，避免把不同仿真记录串成一条轨迹。

    参数:
        rewards: 拼接后的每步奖励数组。
        episode_lengths: 各条记录的有效步长列表；为空时整体计算。
        gamma: 折扣因子。

    返回:
        与 ``rewards`` 等长的归一化 return-to-go 权重数组。
    """
    if rewards.size == 0:
        return rewards.astype(float)
    if not episode_lengths:
        return compute_return_to_go(rewards, gamma=gamma)

    out = np.zeros_like(rewards, dtype=float)
    pos = 0
    for seg_len in episode_lengths:
        n = int(seg_len)
        if n <= 0:
            continue
        seg = rewards[pos: pos + n]
        if seg.size == 0:
            break
        out[pos: pos + n] = compute_return_to_go(seg, gamma=gamma)
        pos += n

    if pos < len(rewards):
        out[pos:] = compute_return_to_go(rewards[pos:], gamma=gamma)
    return out
