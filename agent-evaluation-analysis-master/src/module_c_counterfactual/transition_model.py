"""
状态转移模型（Transition Model）。

从历史推理数据中学习：
    (s_t, a_t) -> s_(t+1)
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .model_validation import (
    ModelValidationResult,
    compute_regression_loss,
    evaluate_regression,
    split_data_for_validation,
)
from src.module_c_counterfactual.action_encoding import JointActionEncoder
from src.module_c_counterfactual.inference_record import InferenceRecord
from src.module_c_counterfactual.training_data import iter_transitions


class TransitionModel:
    """状态转移模型：(obs_vector, joint_action) → next_obs_vector。"""

    def __init__(
        self,
        n_estimators: int = 120,
        max_depth: int = 10,
        min_samples_leaf: int = 3,
        random_state: int = 42,
        model_variant: str = "multioutput",
    ) -> None:
        """
        初始化状态转移回归模型。

        参数:
            n_estimators: 随机森林树数量。
            max_depth: 单棵树最大深度。
            min_samples_leaf: 叶节点最小样本数。
            random_state: 随机种子。
            model_variant: "multioutput" 联合输出或 "per_feature" 分维建模。
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model_variant = model_variant

        self._regressor: Optional[RandomForestRegressor] = None
        self._regressors_per_dim: List[RandomForestRegressor] = []
        self._action_encoder = JointActionEncoder()
        self._is_fitted = False
        self._obs_dim = 0

    def fit(self, records: List[InferenceRecord], agent_id: int) -> "TransitionModel":
        """
        从推理记录中学习 (s_t, a_t) → s_{t+1} 映射。

        参数:
            records: 推理记录列表。
            agent_id: 目标智能体编号。

        返回:
            自身实例。

        抛出:
            ValueError: 无可用转移样本时。
        """
        pending: List[tuple] = []
        for obs_t, action_label, obs_t1, _ in iter_transitions(records, agent_id):
            pending.append((obs_t, action_label, obs_t1))

        if not pending:
            raise ValueError(f"agent_id={agent_id} 无可用转移样本，无法训练状态转移模型。")

        action_labels = [a for _, a, _ in pending]
        self._action_encoder.fit(action_labels)
        self._obs_dim = len(pending[0][2])

        X = []
        y_rows = []
        for obs_t, action_label, obs_t1 in pending:
            row = np.concatenate(
                [np.array(obs_t, dtype=float), self._action_encoder.transform_row(action_label)]
            )
            X.append(row)
            y_rows.append(obs_t1)

        X_np = np.array(X, dtype=float)
        y_np = np.array(y_rows, dtype=float)
        if self.model_variant == "per_feature":
            self._regressor = None
            self._regressors_per_dim = []
            for d in range(y_np.shape[1]):
                reg_d = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state + d,
                    n_jobs=-1,
                )
                reg_d.fit(X_np, y_np[:, d])
                self._regressors_per_dim.append(reg_d)
        else:
            self._regressors_per_dim = []
            self._regressor = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self._regressor.fit(X_np, y_np)
        self._is_fitted = True
        return self

    def fit_transition_tuples(
        self,
        pending: List[tuple],
    ) -> "TransitionModel":
        """
        从 (obs_t, action_label, obs_t1) 元组列表训练，供 surrogate profile 增量 refit。

        参数:
            pending: 不含奖励的转移样本元组列表。

        返回:
            自身实例。

        抛出:
            ValueError: pending 为空时。
        """
        if not pending:
            raise ValueError("无可用转移样本，无法训练状态转移模型。")

        action_labels = [a for _, a, _ in pending]
        self._action_encoder.fit(action_labels)
        self._obs_dim = len(pending[0][2])

        X = []
        y_rows = []
        for obs_t, action_label, obs_t1 in pending:
            row = np.concatenate(
                [np.array(obs_t, dtype=float), self._action_encoder.transform_row(action_label)]
            )
            X.append(row)
            y_rows.append(obs_t1)

        X_np = np.array(X, dtype=float)
        y_np = np.array(y_rows, dtype=float)
        if self.model_variant == "per_feature":
            self._regressor = None
            self._regressors_per_dim = []
            for d in range(y_np.shape[1]):
                reg_d = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state + d,
                    n_jobs=-1,
                )
                reg_d.fit(X_np, y_np[:, d])
                self._regressors_per_dim.append(reg_d)
        else:
            self._regressors_per_dim = []
            self._regressor = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self._regressor.fit(X_np, y_np)
        self._is_fitted = True
        return self

    def predict(self, obs_vector: List[Any], action_label: str) -> List[float]:
        """
        预测执行动作后的下一帧观测向量。

        参数:
            obs_vector: 当前步观测向量。
            action_label: 联合动作标签字符串。

        返回:
            预测的下一观测向量（与训练时维度一致）。
        """
        assert self._is_fitted
        row = np.concatenate(
            [
                np.array(obs_vector, dtype=float),
                self._action_encoder.transform_row(action_label),
            ]
        ).reshape(1, -1)
        if self.model_variant == "per_feature":
            assert self._regressors_per_dim
            pred = np.array([reg.predict(row)[0] for reg in self._regressors_per_dim], dtype=float)
        else:
            assert self._regressor is not None
            pred = self._regressor.predict(row)[0]
        return [float(v) for v in pred]

    def validate(
        self,
        records: List[InferenceRecord],
        agent_id: int,
        val_size: float = 0.2,
    ) -> ModelValidationResult:
        """
        验证转移模型的拟合程度。

        参数:
            records: 推理记录列表。
            agent_id: 目标智能体编号。
            val_size: 验证集比例。

        返回:
            包含训练损失、验证损失和评估指标的验证结果。
        """
        result = ModelValidationResult()

        # 提取验证数据
        pending: List[tuple] = []
        for obs_t, action_label, obs_t1, _ in iter_transitions(records, agent_id):
            pending.append((obs_t, action_label, obs_t1))

        if not pending:
            return result

        result.sample_count = len(pending)

        X = []
        y_rows = []
        for obs_t, action_label, obs_t1 in pending:
            row = np.concatenate(
                [np.array(obs_t, dtype=float), self._action_encoder.transform_row(action_label)]
            )
            X.append(row)
            y_rows.append(obs_t1)

        X_np = np.array(X, dtype=float)
        y_np = np.array(y_rows, dtype=float)

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = split_data_for_validation(
            X_np, y_np, val_size=val_size
        )
        result.train_sample_count = len(X_train)
        result.val_sample_count = len(X_val)

        # 预测
        if self.model_variant == "per_feature":
            y_train_pred = np.column_stack([
                reg.predict(X_train) for reg in self._regressors_per_dim
            ])
            y_val_pred = np.column_stack([
                reg.predict(X_val) for reg in self._regressors_per_dim
            ])
        else:
            y_train_pred = self._regressor.predict(X_train)
            y_val_pred = self._regressor.predict(X_val)

        # 计算损失（MSE）
        result.train_loss = compute_regression_loss(y_train, y_train_pred)
        result.val_loss = compute_regression_loss(y_val, y_val_pred)

        # 计算评估指标
        result.train_metrics = evaluate_regression(y_train, y_train_pred)
        result.val_metrics = evaluate_regression(y_val, y_val_pred)

        return result
