"""
策略近似模型（Policy Surrogate Model）。

================================================================================
这个模型的作用（小白友好版）：
================================================================================

我们手头只有智能体跑出来的推理数据（每步的观测 + 动作 + 奖励），
没有办法直接访问智能体内部的神经网络。

所以我们要做一件事：
    用一棵"决策树"来模仿智能体的决策行为。

决策树你可以理解为一套 if-else 规则：
    如果 敌机距离 < 500 且 自身血量 > 0.5：→ 选择 "开启雷达"
    否则如果 敌机距离 >= 500：              → 选择 "远离敌机"
    ...

这棵树训练好之后，我们就可以用它来做"局部反事实推理"：
    "如果当时的敌机距离不是 300，而是 800，智能体还会选择开启雷达吗？"
    → 把 敌机距离 改成 800，喂给决策树，看它预测什么动作。

================================================================================
算法步骤（对应图中的规则抽取算法）：
================================================================================

步骤1：找到训练粒度，从 InferenceRecord 提取数据，整理成 (观测特征, 动作) 格式
步骤2：构造特征 X（观测向量）和标签 y（动作）
步骤3：根据数据进行权重赋值
        ——用 return-to-go（从当前步到结束的累计奖励）作为样本权重
        ——权重越高，说明这条决策越"重要"（后续收益越大），训练时多关注
步骤4：训练一棵带权重的 CART 决策树（基尼指数作为分裂标准）
步骤5：从根节点到叶节点提取规则集，每条路径是一条 if-else 规则

================================================================================
在局部反事实推理中的使用方式：
================================================================================

    # 1. 准备好推理数据（从数据库加载）
    record = load_record_from_doris(task_id)

    # 2. 训练决策树（拟合智能体行为）
    model = PolicySurrogate()
    model.fit(record, agent_id=1)

    # 3. 做反事实预测
    #    原始观测：敌机距离=300, 自身血量=0.8, 敌机状态=存活
    #    现在假设：如果敌机距离改成 800，智能体会怎么做？
    cf_obs = [800, 0.8, 1.0]             # 修改了敌机距离
    cf_action = model.predict(cf_obs)    # 预测新动作

    # 4. 比较：原来的动作 vs 反事实动作
    #    如果不一样，说明"敌机距离"就是影响这次决策的关键原因
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, TYPE_CHECKING

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, _tree

from .model_validation import (
    ModelValidationResult,
    compute_classification_loss,
    evaluate_classification,
    split_data_for_validation,
)

from src.module_c_counterfactual.inference_record import InferenceRecord

if TYPE_CHECKING:
    from src.module_a_rules.preprocess import Preprocessor


def _is_policy_preprocess_enabled() -> bool:
    """
    是否对策略训练特征做 z-score 归一化（与 Module A 一致，默认开启）。

    返回:
        环境变量 ANALYSIS_CF_POLICY_PREPROCESS 非 0/false 时为 True。
    """
    import os

    v = os.environ.get("ANALYSIS_CF_POLICY_PREPROCESS", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _viper_iters_from_env() -> int:
    """
    holistic joint 模式下 VIPER 错分加权迭代次数。

    返回:
        环境变量 ANALYSIS_CF_POLICY_VIPER_ITERS 解析值（默认 0，0 表示关闭）。
    """
    import os

    raw = os.environ.get("ANALYSIS_CF_POLICY_VIPER_ITERS", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _class_weight_from_env() -> Optional[str]:
    """
    读取决策树类别权重配置。

    返回:
        "balanced" 或 None（关闭类别权重）。
    """
    import os

    v = os.environ.get("ANALYSIS_CF_POLICY_CLASS_WEIGHT", "balanced").strip().lower()
    if v in ("0", "none", "off", "false"):
        return None
    if v == "balanced":
        return "balanced"
    return None


def _n_estimators_from_env() -> int:
    """读取集成策略估计器的树数量（默认 150）。"""
    import os

    raw = os.environ.get("ANALYSIS_CF_POLICY_N_ESTIMATORS", "150").strip()
    try:
        return max(10, int(raw))
    except ValueError:
        return 150


def _policy_estimator_kind() -> str:
    """读取策略估计器类型（tree / et / rf，默认 tree）。"""
    import os

    return os.environ.get("ANALYSIS_CF_POLICY_ESTIMATOR", "tree").strip().lower()


def _make_policy_classifier(*, max_depth: int, min_samples_leaf: int, class_weight: Optional[str]):
    """
    构造策略分类估计器（joint/composed 子树共用）。

    参数:
        max_depth: 树最大深度。
        min_samples_leaf: 叶节点最小样本数。
        class_weight: sklearn 类别权重参数。

    返回:
        DecisionTreeClassifier、ExtraTreesClassifier 或 RandomForestClassifier。
    """
    kind = _policy_estimator_kind()
    common = dict(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=42,
    )
    if kind in ("et", "extra_trees", "extratrees", "extra"):
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=_n_estimators_from_env(),
            n_jobs=-1,
            **common,
        )
    if kind in ("rf", "forest", "random_forest"):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=_n_estimators_from_env(),
            n_jobs=-1,
            **common,
        )
    return DecisionTreeClassifier(criterion="gini", **common)


class PolicySurrogate:
    """
    CART 决策树策略近似模型（holistic / joint）。

    学习映射：状态 X → **一步完整 decision_content（整体动作类）**。
    每个 agent_id 单独 fit；不同智能体的观测/动作 schema 互不共享。
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_leaf: int = 5,
        *,
        mode: Literal["joint", "per_item", "composed"] = "joint",
    ) -> None:
        """
        mode="joint"：单棵树预测整体决策类（标签极度不均衡时 joint-exact 可能接近多数类基线）。
        mode="composed"：各动作项各一棵树，预测后组装为 holistic JSON 标签（推荐，准确率高且与 CF 一致）。
        mode="per_item"：调试用，输出旧 tuple 字符串标签。
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.mode = mode

        # 以下属性在 fit() 后才有值
        self._clf: Optional[DecisionTreeClassifier] = None  # joint: 训练好的决策树
        self._item_clfs: Dict[str, DecisionTreeClassifier] = {}  # per_item/composed
        self._feature_names: List[str] = []                 # 观测特征名列表
        self._action_space: List[str] = []                  # 动作空间名列表
        self._preprocessor: Optional["Preprocessor"] = None
        self._holistic_schema: Any = None                   # composed: AgentSchema
        self._is_fitted: bool = False                       # 是否已经训练过

    def _uses_item_trees(self) -> bool:
        """当前模式是否使用各动作项独立子树（per_item / composed）。"""
        return self.mode in ("per_item", "composed")

    def _label_from_item_predictions(self, pred: Dict[str, Any]) -> str:
        """
        将各动作项预测值组装为统一标签字符串。

        参数:
            pred: 动作项名 → 预测值 的字典。

        返回:
            holistic JSON 或旧 tuple 字符串格式标签。
        """
        if self.mode == "composed" and self._holistic_schema is not None:
            from src.module_c_counterfactual.agent_schema import holistic_decision_label

            return holistic_decision_label(pred, self._holistic_schema)
        return str(sorted(pred.items()))

    def _prepare_features(
        self,
        X_np: np.ndarray,
        feature_names: List[str],
        *,
        fit: bool,
        preprocessor: Optional["Preprocessor"] = None,
    ) -> np.ndarray:
        """
        对特征矩阵应用 z-score 预处理（fit 或 transform）。

        参数:
            X_np: 原始特征矩阵。
            feature_names: 特征名列表。
            fit: True 时拟合预处理器，False 时仅变换。
            preprocessor: 可选外部预处理器实例。

        返回:
            预处理后的特征矩阵。
        """
        if not _is_policy_preprocess_enabled():
            return X_np
        if preprocessor is not None:
            self._preprocessor = preprocessor
        if self._preprocessor is None:
            from src.module_a_rules.preprocess import Preprocessor

            self._preprocessor = Preprocessor(feature_names=feature_names)
        elif feature_names and not self._preprocessor.feature_names:
            self._preprocessor.feature_names = list(feature_names)
        if fit or self._preprocessor._mean is None:
            return self._preprocessor.fit_transform(X_np)
        return self._preprocessor.transform(X_np)

    def _fit_joint_classifier(
        self,
        X_np: np.ndarray,
        y: List[Any],
        sample_weights: np.ndarray,
        feature_names: List[str],
        *,
        preprocessor: Optional["Preprocessor"] = None,
        refit_preprocessor: bool = True,
    ) -> None:
        """
        训练 joint 模式下的单棵（或集成）策略分类器，可选 VIPER 迭代加权。

        参数:
            X_np: 特征矩阵。
            y: holistic 标签列表。
            sample_weights: 样本权重数组。
            feature_names: 特征名列表。
            preprocessor: 可选预处理器。
            refit_preprocessor: 是否重新拟合预处理器。
        """
        X_fit = self._prepare_features(
            X_np,
            feature_names,
            fit=refit_preprocessor or preprocessor is None,
            preprocessor=preprocessor,
        )
        y_arr = np.array(y)
        cw = _class_weight_from_env()
        n_iters = _viper_iters_from_env()
        use_viper = n_iters > 0 and _policy_estimator_kind() in ("tree", "dt", "cart", "")
        if not use_viper:
            self._clf = _make_policy_classifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                class_weight=cw,
            )
            self._clf.fit(X_fit, y_arr, sample_weight=sample_weights)
            return

        penalty = 2.0
        weights = np.array(sample_weights, dtype=float)
        best_tree = None
        best_acc = -1.0
        X_orig = X_fit.copy()
        y_orig = y_arr.copy()
        w_orig = weights.copy()
        for _ in range(n_iters):
            clf = _make_policy_classifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                class_weight=cw,
            )
            clf.fit(X_fit, y_arr, sample_weight=weights)
            acc_orig = float(accuracy_score(y_orig, clf.predict(X_orig), sample_weight=w_orig))
            if acc_orig > best_acc:
                best_acc = acc_orig
                best_tree = clf
            wrong = clf.predict(X_fit) != y_arr
            weights = weights * np.where(wrong, penalty, 1.0)
            s = float(weights.sum())
            if s > 0:
                weights = weights / s * len(weights)
        self._clf = best_tree

    # ------------------------------------------------------------------
    # 步骤1-4：从推理数据中训练决策树
    # ------------------------------------------------------------------
    def fit(self, record: InferenceRecord, agent_id: int) -> "PolicySurrogate":
        """
        从 InferenceRecord 中提取样本并训练 CART 决策树。

        整个流程：
            1. 遍历每一个时间步，提取 (观测特征向量, 动作) 作为一条训练样本
            2. 计算每条样本的 return-to-go（从该步到末尾的累计奖励）作为权重
            3. 用带权重的 CART 决策树拟合 s → a 的映射关系

        Parameters
        ----------
        record   : 推理数据记录（从 Doris 数据库加载的完整数据）
        agent_id : 要为哪个智能体训练决策树（一个智能体训练一棵树）
        """
        # ---- 步骤1&2：构造特征 X 和 holistic 标签 y（一步完整 decision_content） ----
        from src.module_a_rules.collect_data import compute_return_to_go
        from src.module_c_counterfactual.training_data import joint_action_label

        X: List[List[float]] = []
        y: List[Any] = []
        step_rewards: List[float] = []
        all_rewards = getattr(record, "rewards", [])
        action_items = list(record.action_space)

        for t in range(record.total_steps):
            obs_vec = record.get_obs_vector(t, agent_id)
            if not obs_vec:
                continue
            dec = record.get_decision_at(t, agent_id)
            if dec is None:
                continue
            X.append(obs_vec)
            if self.mode == "joint":
                y.append(joint_action_label(dec.content, record, agent_id))
            else:
                y.append(dict(dec.content))
            step_rewards.append(all_rewards[t] if t < len(all_rewards) else 0.0)

        # ---- 步骤3：计算 return-to-go 作为样本权重 ----
        sample_weights = compute_return_to_go(np.array(step_rewards))

        # ---- 步骤4：训练带权重的 CART 决策树 ----
        cw = _class_weight_from_env()
        feat_names = record.get_flat_feature_names(agent_id)
        X_np = np.array(X, dtype=float)
        if self.mode == "composed":
            from src.module_c_counterfactual.agent_schema import AgentSchema

            self._holistic_schema = AgentSchema.from_record(record, agent_id)
        if self.mode == "joint":
            self._fit_joint_classifier(X_np, y, sample_weights, feat_names)
        elif self._uses_item_trees():
            # per_item / composed：各动作项一棵树，composed 预测时组装 holistic JSON
            self._item_clfs = {}
            X_fit = self._prepare_features(X_np, feat_names, fit=True)
            y_dicts: List[Dict[str, Any]] = [v for v in y if isinstance(v, dict)]
            if len(y_dicts) != len(X):
                raise ValueError("per_item 模式训练数据对齐失败：y_dicts 与 X 长度不一致。")
            for item in action_items:
                y_item = [d.get(item) for d in y_dicts]
                keep_idx = [i for i, v in enumerate(y_item) if v is not None]
                if not keep_idx:
                    continue
                clf_i = _make_policy_classifier(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    class_weight=cw,
                )
                clf_i.fit(
                    X_fit[keep_idx],
                    [y_item[i] for i in keep_idx],
                    sample_weight=np.array(sample_weights, dtype=float)[keep_idx],
                )
                self._item_clfs[item] = clf_i

        # 记录展平后的特征名（格式：观测项.子字段，如 "自身状态.血量"）
        # 注意：必须用展平键，不能用顶层键（observation_space 只有 3 项，而 X 可能有 8 列）
        self._feature_names = record.get_flat_feature_names(agent_id)
        self._action_space = record.action_space
        self._is_fitted = True

        return self

    def fit_records(
        self,
        records: List[InferenceRecord],
        agent_id: int,
        *,
        preprocessor: Optional["Preprocessor"] = None,
        refit_preprocessor: bool = True,
    ) -> "PolicySurrogate":
        """
        合并多条 InferenceRecord（同一 inference_task 下多局 sim）训练策略近似树。

        参数:
            records: 推理记录列表。
            agent_id: 目标智能体编号。
            preprocessor: 可选外部特征预处理器。
            refit_preprocessor: 是否重新拟合预处理器。

        返回:
            自身实例。

        抛出:
            ValueError: 无可用训练样本时。
        """
        from src.module_a_rules.collect_data import compute_return_to_go
        from src.module_c_counterfactual.training_data import joint_action_label

        X: List[List[float]] = []
        y: List[Any] = []
        sample_weights_list: List[float] = []
        feature_names: List[str] = []

        action_items = list(records[0].action_space) if records else []
        for record in records:
            all_rewards = getattr(record, "rewards", [])
            rec_rewards_for_weights: List[float] = []
            names = record.get_flat_feature_names(agent_id)
            if names and not feature_names:
                feature_names = names

            for t in range(record.total_steps):
                obs_vec = record.get_obs_vector(t, agent_id)
                if not obs_vec:
                    continue
                dec = record.get_decision_at(t, agent_id)
                if dec is None:
                    continue
                X.append(obs_vec)
                if self.mode == "joint":
                    y.append(joint_action_label(dec.content, record, agent_id))
                else:
                    y.append(dict(dec.content))
                rec_rewards_for_weights.append(all_rewards[t] if t < len(all_rewards) else 0.0)
            if rec_rewards_for_weights:
                # 分 record 计算 RTG，避免跨仿真串联权重。
                w_rec = compute_return_to_go(np.array(rec_rewards_for_weights, dtype=float))
                sample_weights_list.extend([float(v) for v in w_rec.tolist()])

        if not X:
            raise ValueError(f"agent_id={agent_id} 在 records 中无可用策略训练样本。")

        sample_weights = np.array(sample_weights_list, dtype=float)
        X_np = np.array(X, dtype=float)
        feat_names = feature_names or records[0].get_flat_feature_names(agent_id)
        if self.mode == "composed":
            from src.module_c_counterfactual.agent_schema import assert_same_agent_schema

            self._holistic_schema = assert_same_agent_schema(records, agent_id)
        cw = _class_weight_from_env()
        if self.mode == "joint":
            self._fit_joint_classifier(
                X_np,
                y,
                sample_weights,
                feat_names,
                preprocessor=preprocessor,
                refit_preprocessor=refit_preprocessor,
            )
        elif self._uses_item_trees():
            X_fit = self._prepare_features(
                X_np,
                feat_names,
                fit=refit_preprocessor or preprocessor is None,
                preprocessor=preprocessor,
            )
            self._item_clfs = {}
            y_dicts: List[Dict[str, Any]] = [v for v in y if isinstance(v, dict)]
            if len(y_dicts) != len(X):
                raise ValueError("per_item/composed 模式训练数据对齐失败：y_dicts 与 X 长度不一致。")
            w_np = np.array(sample_weights, dtype=float)
            for item in action_items:
                y_item = [d.get(item) for d in y_dicts]
                keep_idx = [i for i, v in enumerate(y_item) if v is not None]
                if not keep_idx:
                    continue
                clf_i = _make_policy_classifier(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    class_weight=cw,
                )
                clf_i.fit(
                    X_fit[keep_idx],
                    [y_item[i] for i in keep_idx],
                    sample_weight=w_np[keep_idx],
                )
                self._item_clfs[item] = clf_i
        self._feature_names = feat_names
        self._action_space = records[0].action_space
        self._is_fitted = True
        return self

    def fit_transition_rows(
        self,
        rows: List[Dict[str, Any]],
        *,
        feature_names: List[str],
        action_space: List[str],
    ) -> "PolicySurrogate":
        """
        从转移 reservoir 行训练策略近似（均匀权重），供 surrogate profile 增量 refit。

        参数:
            rows: 含 obs、action 字段的转移样本行列表。
            feature_names: 观测特征名列表。
            action_space: 动作项名称列表。

        返回:
            自身实例。

        抛出:
            ValueError: rows 为空或无有效样本时。
        """
        import ast

        if not rows:
            raise ValueError("无可用策略训练样本。")

        X: List[List[float]] = []
        y: List[Any] = []
        for row in rows:
            obs = row["obs"]
            action_label = row["action"]
            X.append(obs)
            if self.mode == "joint":
                y.append(action_label)
            else:
                try:
                    pairs = ast.literal_eval(action_label)
                    y.append(dict(pairs) if isinstance(pairs, list) else {})
                except (ValueError, SyntaxError):
                    y.append({})

        if not X:
            raise ValueError("无可用策略训练样本。")

        sample_weights = np.ones(len(X), dtype=float)
        X_np = np.array(X, dtype=float)
        cw = _class_weight_from_env()
        if self.mode == "joint":
            self._fit_joint_classifier(X_np, y, sample_weights, feature_names)
        elif self._uses_item_trees():
            X_fit = self._prepare_features(X_np, feature_names, fit=True)
            self._item_clfs = {}
            y_dicts: List[Dict[str, Any]] = [v for v in y if isinstance(v, dict)]
            for item in action_space:
                y_item = [d.get(item) for d in y_dicts]
                keep_idx = [i for i, v in enumerate(y_item) if v is not None]
                if not keep_idx:
                    continue
                clf_i = _make_policy_classifier(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    class_weight=cw,
                )
                clf_i.fit(
                    X_fit[keep_idx],
                    [y_item[i] for i in keep_idx],
                    sample_weight=sample_weights[keep_idx],
                )
                self._item_clfs[item] = clf_i
        self._feature_names = feature_names
        self._action_space = action_space
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # 反事实预测：给定一个（可能修改过的）观测，预测动作
    # ------------------------------------------------------------------
    def predict(self, obs_vector: List[Any]) -> Any:
        """
        给定观测特征向量，返回决策树预测的动作。

        在局部反事实推理中的使用：
            传入修改过某个特征的反事实观测，看预测动作是否发生变化。
            如果变化了，说明被修改的特征是影响决策的关键原因。

        Parameters
        ----------
        obs_vector : 观测特征向量，顺序必须与训练时一致（按 observation_space）

        Returns
        -------
        预测的动作（动作名字符串或动作索引）
        """
        assert self._is_fitted, "请先调用 fit() 训练决策树"
        x = np.array(obs_vector, dtype=float).reshape(1, -1)
        if self._preprocessor is not None and _is_policy_preprocess_enabled():
            x = self._preprocessor.transform(x)
        if self.mode == "joint":
            assert self._clf is not None
            return self._clf.predict(x)[0]
        # per_item：逐动作项预测，再拼成联合动作标签
        pred: Dict[str, Any] = {}
        for item, clf_i in self._item_clfs.items():
            v = clf_i.predict(x)[0]
            # numpy 标量转 Python 标量，确保联合动作标签字符串稳定
            if hasattr(v, "item"):
                try:
                    v = v.item()
                except Exception:
                    pass
            pred[item] = v
        return self._label_from_item_predictions(pred)

    def predict_proba(self, obs_vector: List[Any]) -> Dict[Any, float]:
        """
        返回动作类别概率分布（用于“变化幅度”评分）。

        返回示例：
            {"机动控制=规避 ...": 0.72, "机动控制=追击 ...": 0.28}
        """
        assert self._is_fitted, "请先调用 fit() 训练决策树"
        x = np.array(obs_vector, dtype=float).reshape(1, -1)
        if self._preprocessor is not None and _is_policy_preprocess_enabled():
            x = self._preprocessor.transform(x)
        if self.mode == "joint":
            probs = self._clf.predict_proba(x)[0]
            classes = list(self._clf.classes_)
            return {classes[i]: float(probs[i]) for i in range(len(classes))}

        # per_item：近似联合动作分布（假设各动作项条件独立）。
        # 目的：支持 prob_delta_l1 的“分布变化幅度”评分，避免退化到 0/1。
        items = sorted(self._item_clfs.keys(), key=str)
        if not items:
            return {}

        per_item_classes: List[List[Any]] = []
        per_item_probs: List[np.ndarray] = []
        for it in items:
            clf_i = self._item_clfs[it]
            p = clf_i.predict_proba(x)[0]
            per_item_probs.append(np.array(p, dtype=float))
            cls_list: List[Any] = []
            for c in list(clf_i.classes_):
                if hasattr(c, "item"):
                    try:
                        c = c.item()
                    except Exception:
                        pass
                cls_list.append(c)
            per_item_classes.append(cls_list)

        dist: Dict[str, float] = {}
        # 逐项扩展（笛卡尔积）
        partial: List[Tuple[Dict[str, Any], float]] = [({}, 1.0)]
        for it, classes_i, probs_i in zip(items, per_item_classes, per_item_probs):
            next_partial: List[Tuple[Dict[str, Any], float]] = []
            for d_prev, p_prev in partial:
                for cls_val, p_val in zip(classes_i, probs_i):
                    d_new = dict(d_prev)
                    d_new[it] = cls_val
                    next_partial.append((d_new, p_prev * float(p_val)))
            partial = next_partial

        for d_joint, p_joint in partial:
            label = self._label_from_item_predictions(d_joint)
            dist[label] = dist.get(label, 0.0) + float(p_joint)

        # 归一化（防止数值误差）
        s = float(sum(dist.values()))
        if s > 0:
            dist = {k: float(v / s) for k, v in dist.items()}
        return dist

    # ------------------------------------------------------------------
    # 步骤5：从决策树提取规则集
    # ------------------------------------------------------------------
    def extract_rules(self) -> List[str]:
        """
        从训练好的决策树中提取规则集（步骤5）。

        从根节点到每个叶节点的路径就是一条规则：
            "如果 敌机距离 <= 500.0 且 自身血量 > 0.5：→ 动作：开启雷达（置信度: 0.85）"

        Returns
        -------
        rules : 每条规则是一个可读的字符串列表，可以直接展示给用户
        """
        assert self._is_fitted, "请先调用 fit() 训练决策树"
        rules: List[str] = []
        _collect_rules(
            tree=self._clf,
            feature_names=self._feature_names,
            rules=rules,
        )
        return rules

    def feature_importances(self) -> Optional[np.ndarray]:
        """
        返回决策树的特征重要性向量。

        重要性越高，说明该特征在决策树分裂时被用到的越多、越关键。
        可以用来快速判断哪些观测特征最影响智能体的决策。
        """
        if not self._is_fitted:
            return None
        if self.mode == "joint" and self._clf is not None:
            return self._clf.feature_importances_
        return None

    def validate(
        self,
        record: InferenceRecord,
        agent_id: int,
        val_size: float = 0.2,
    ) -> ModelValidationResult:
        """
        验证策略模型的拟合程度。

        参数:
            record: 推理记录。
            agent_id: 目标智能体编号。
            val_size: 验证集比例。

        返回:
            包含训练损失、验证损失和评估指标的验证结果。
        """
        result = ModelValidationResult()

        # 提取验证数据
        X: List[List[float]] = []
        y: List[str] = []
        for t in range(record.total_steps):
            obs_vec = record.get_obs_vector(t, agent_id)
            if not obs_vec:
                continue
            dec = record.get_decision_at(t, agent_id)
            if dec is None:
                continue
            X.append(obs_vec)
            # 使用与预测相同的方式转换为标签字符串
            y.append(self._decision_to_label(dec.content, record, agent_id))

        if not X:
            return result

        result.sample_count = len(X)
        X_np = np.array(X, dtype=float)
        y_arr = np.array(y)

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = split_data_for_validation(
            X_np, y_arr, val_size=val_size
        )
        result.train_sample_count = len(X_train)
        result.val_sample_count = len(X_val)

        # 预处理特征
        X_train_fit = self._prepare_features(
            X_train, self._feature_names, fit=False
        )
        X_val_fit = self._prepare_features(
            X_val, self._feature_names, fit=False
        )

        # 预测（使用统一的标签格式）
        y_train_pred = [self.predict(x) for x in X_train.tolist()]
        y_val_pred = [self.predict(x) for x in X_val.tolist()]

        # 计算损失
        result.train_loss = compute_classification_loss(y_train, y_train_pred)
        result.val_loss = compute_classification_loss(y_val, y_val_pred)

        # 计算评估指标
        result.train_metrics = evaluate_classification(y_train, y_train_pred)
        result.val_metrics = evaluate_classification(y_val, y_val_pred)

        return result

    def _decision_to_label(self, decision_content: dict, record: InferenceRecord, agent_id: int) -> str:
        """
        将决策内容转换为统一的标签字符串（与预测格式一致）。
        """
        return self._label_from_item_predictions(dict(decision_content))


# ==============================================================================
# 工具函数
# ==============================================================================

def _compute_return_to_go(rewards: List[float]) -> np.ndarray:
    """
    计算每个时间步的 return-to-go（从当前步到末尾的累计奖励）。

    例如：rewards = [1, 2, 3, 4]
          return_to_go = [10, 9, 7, 4]
              第0步：1+2+3+4=10
              第1步：2+3+4=9
              第2步：3+4=7
              第3步：4

    为了让权重都是正数且归一化到合理范围，会进行 min-max 归一化后加 1e-6 防止全0。

    Parameters
    ----------
    rewards : 每步奖励值列表

    Returns
    -------
    weights : 归一化后的 return-to-go 权重数组，长度与 rewards 相同
    """
    T = len(rewards)
    rtg = np.zeros(T)
    # 从后往前累加：rtg[t] = rewards[t] + rtg[t+1]
    for t in reversed(range(T)):
        rtg[t] = rewards[t] + (rtg[t + 1] if t + 1 < T else 0.0)

    # 归一化到 [1e-6, 1]，确保权重全为正数
    r_min, r_max = rtg.min(), rtg.max()
    if r_max > r_min:
        rtg = (rtg - r_min) / (r_max - r_min)
    rtg = rtg + 1e-6   # 防止出现零权重
    return rtg


def _collect_rules(
    tree: DecisionTreeClassifier,
    feature_names: List[str],
    rules: List[str],
    node_id: int = 0,
    conditions: Optional[List[str]] = None,
) -> None:
    """
    递归遍历决策树，从根节点到叶节点收集 if-else 规则。

    这是一个辅助函数，用户直接调用 PolicySurrogate.extract_rules() 即可。

    Parameters
    ----------
    tree          : 训练好的决策树对象
    feature_names : 特征名列表（用于把特征索引翻译成名字）
    rules         : 用于收集规则字符串的列表（in-place 追加）
    node_id       : 当前处理的节点编号（递归用，外部调用传默认值即可）
    conditions    : 到达当前节点的条件列表（递归用，外部调用传默认值即可）
    """
    if conditions is None:
        conditions = []

    tree_ = tree.tree_
    feature_idx = tree_.feature        # 每个节点用于分裂的特征索引
    threshold = tree_.threshold        # 每个节点的分裂阈值

    if feature_idx[node_id] == _tree.TREE_UNDEFINED:
        # 叶节点：记录这条完整的规则
        # 取该叶节点中样本数最多的类别作为预测动作
        class_idx = int(np.argmax(tree_.value[node_id]))
        predicted_action = tree.classes_[class_idx]
        # 计算置信度（叶节点纯度）
        node_samples = tree_.value[node_id][0]
        confidence = node_samples[class_idx] / node_samples.sum() if node_samples.sum() > 0 else 0.0

        if conditions:
            rule_body = " 且 ".join(conditions)
            rules.append(f"如果 {rule_body}：→ 动作：{predicted_action}（置信度: {confidence:.2f}）")
        else:
            rules.append(f"无条件：→ 动作：{predicted_action}（置信度: {confidence:.2f}）")
        return

    # 非叶节点：向左子树（满足条件）和右子树（不满足条件）分别递归
    feat_name = feature_names[feature_idx[node_id]] if feature_idx[node_id] < len(feature_names) else f"特征{feature_idx[node_id]}"
    thresh_val = threshold[node_id]

    # 左子树：特征值 <= 阈值
    _collect_rules(
        tree, feature_names, rules,
        node_id=tree_.children_left[node_id],
        conditions=conditions + [f"{feat_name} <= {thresh_val:.4f}"],
    )
    # 右子树：特征值 > 阈值
    _collect_rules(
        tree, feature_names, rules,
        node_id=tree_.children_right[node_id],
        conditions=conditions + [f"{feat_name} > {thresh_val:.4f}"],
    )
