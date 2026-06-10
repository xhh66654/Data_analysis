"""
奖励模型（Reward Model）。

从历史推理数据中学习：
    (s_t, a_t, s_(t+1)) → r_t
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .model_validation import (
    ModelValidationResult,
    compute_regression_loss,
    evaluate_regression,
    split_data_for_validation,
)
from src.module_c_counterfactual.action_encoding import JointActionEncoder
from src.module_c_counterfactual.inference_record import InferenceRecord
from src.module_c_counterfactual.training_data import iter_transitions


class RewardModel:
    """奖励预测模型：(s_t, joint_action, s_{t+1}) → r_t。"""

    def __init__(
        self,
        n_estimators: int = 80,
        max_depth: int = 5,
        learning_rate: float = 0.08,
        random_state: int = 42,
    ) -> None:
        """
        初始化奖励回归模型。

        参数:
            n_estimators: 梯度提升树数量。
            max_depth: 单棵树最大深度。
            learning_rate: 学习率。
            random_state: 随机种子。
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self._regressor: Optional[GradientBoostingRegressor] = None
        self._action_encoder = JointActionEncoder()
        self._is_fitted = False

    def fit(self, records: List[InferenceRecord], agent_id: int) -> "RewardModel":
        """
        从推理记录中学习 (s_t, a_t, s_{t+1}) → r_t 映射。

        参数:
            records: 推理记录列表。
            agent_id: 目标智能体编号。

        返回:
            自身实例（支持链式调用）。

        抛出:
            ValueError: 无可用奖励样本时。
        """
        X_rows: List[List[float]] = []
        y_vals: List[float] = []
        action_labels: List[str] = []
        pending: List[tuple] = []

        for obs_t, action_label, obs_t1, r_t in iter_transitions(records, agent_id):
            action_labels.append(action_label)
            pending.append((obs_t, obs_t1))
            y_vals.append(r_t)

        if not y_vals:
            raise ValueError(f"agent_id={agent_id} 无可用奖励样本，无法训练奖励模型。")

        self._action_encoder.fit(action_labels)

        for (obs_t, obs_t1), action_label in zip(pending, action_labels):
            row = np.concatenate(
                [
                    np.array(obs_t, dtype=float),
                    self._action_encoder.transform_row(action_label),
                    np.array(obs_t1, dtype=float),
                ]
            )
            X_rows.append(row.tolist())

        self._regressor = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        self._regressor.fit(np.array(X_rows, dtype=float), np.array(y_vals, dtype=float))
        self._is_fitted = True
        return self

    def fit_reward_tuples(
        self,
        pending: List[tuple],
    ) -> "RewardModel":
        """
        从 (obs_t, action_label, obs_t1, r_t) 元组列表训练，供 surrogate profile 增量 refit。

        参数:
            pending: 转移样本元组列表。

        返回:
            自身实例。

        抛出:
            ValueError: pending 为空时。
        """
        if not pending:
            raise ValueError("无可用奖励样本，无法训练奖励模型。")

        action_labels = [a for _, a, _, _ in pending]
        self._action_encoder.fit(action_labels)

        X_rows: List[List[float]] = []
        y_vals: List[float] = []
        for obs_t, action_label, obs_t1, r_t in pending:
            row = np.concatenate(
                [
                    np.array(obs_t, dtype=float),
                    self._action_encoder.transform_row(action_label),
                    np.array(obs_t1, dtype=float),
                ]
            )
            X_rows.append(row.tolist())
            y_vals.append(float(r_t))

        self._regressor = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        self._regressor.fit(np.array(X_rows, dtype=float), np.array(y_vals, dtype=float))
        self._is_fitted = True
        return self

    def predict(
        self,
        obs_t: List[Any],
        action_label: str,
        obs_t1: List[Any],
    ) -> float:
        """
        预测给定转移的标量奖励。

        参数:
            obs_t: 当前步观测向量。
            action_label: 联合动作标签字符串。
            obs_t1: 下一步观测向量。

        返回:
            预测的标量奖励值。
        """
        assert self._is_fitted and self._regressor is not None
        row = np.concatenate(
            [
                np.array(obs_t, dtype=float),
                self._action_encoder.transform_row(action_label),
                np.array(obs_t1, dtype=float),
            ]
        ).reshape(1, -1)
        return float(self._regressor.predict(row)[0])

    def validate(
        self,
        records: List[InferenceRecord],
        agent_id: int,
        val_size: float = 0.2,
    ) -> ModelValidationResult:
        """
        验证奖励模型的拟合程度。

        参数:
            records: 推理记录列表。
            agent_id: 目标智能体编号。
            val_size: 验证集比例。

        返回:
            包含训练损失、验证损失和评估指标的验证结果。
        """
        result = ModelValidationResult()

        # 提取验证数据
        X_rows: List[List[float]] = []
        y_vals: List[float] = []

        for obs_t, action_label, obs_t1, r_t in iter_transitions(records, agent_id):
            row = np.concatenate(
                [
                    np.array(obs_t, dtype=float),
                    self._action_encoder.transform_row(action_label),
                    np.array(obs_t1, dtype=float),
                ]
            )
            X_rows.append(row.tolist())
            y_vals.append(r_t)

        if not y_vals:
            return result

        result.sample_count = len(y_vals)

        X_np = np.array(X_rows, dtype=float)
        y_np = np.array(y_vals, dtype=float)

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = split_data_for_validation(
            X_np, y_np, val_size=val_size
        )
        result.train_sample_count = len(X_train)
        result.val_sample_count = len(X_val)

        # 预测
        y_train_pred = self._regressor.predict(X_train)
        y_val_pred = self._regressor.predict(X_val)

        # 计算损失（MSE）
        result.train_loss = compute_regression_loss(y_train, y_train_pred)
        result.val_loss = compute_regression_loss(y_val, y_val_pred)

        # 计算评估指标
        result.train_metrics = evaluate_regression(y_train, y_train_pred)
        result.val_metrics = evaluate_regression(y_val, y_val_pred)

        return result
