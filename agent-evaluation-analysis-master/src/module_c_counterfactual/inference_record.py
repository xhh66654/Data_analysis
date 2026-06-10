"""
推理数据结构定义（重构版）。

================================================================================
说明（小白友好版）：
================================================================================

一条推理数据 = 一次完整仿真过程的记录。

根据实际数据格式，有几个重要变化：

1. 智能体控制的是「一类装备」（可能是一架飞机，也可能是一个编队），
   每个智能体有自己的装备类型名称。

2. 动作空间是「动作项列表」，每个动作项有多个可选维度值。
   e.g. 动作空间 = ["雷达开关控制", "雷达方向控制"]
        雷达开关控制 的可选值 = ["开", "关"]
        雷达方向控制 的可选值 = ["左", "右", "正前方"]

3. 每步决策内容是字典格式，灵活支持：
   - 单装备单动作：{"雷达开关控制": "开"}
   - 单装备多动作：{"雷达开关控制": "开", "雷达方向控制": "左"}
   - 多装备多动作：{"装备1": {"雷达开关控制": "开"}, "装备2": {"雷达开关控制": "关"}}

4. 观测空间字段是嵌套字典，字段本身包含子字段：
   e.g. "自身状态": {"血量": 0.8, "速度": 1.2, "高度": 8000}
        "敌机距离": {"水平距离": 40.0, "高度差": 500}
        "敌机状态": {"存活": 1, "锁定中": 0}

5. 数据以「数据库表行」的形式存储，每行是一个时间步的记录。
   （模拟 Doris 数据库的表结构）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from src.module_c_counterfactual.agent_schema import AgentSchema


# ==============================================================================
# 智能体元信息
# ==============================================================================

@dataclass
class AgentMeta:
    """
    智能体的基本信息。

    Attributes
    ----------
    agent_id        : 智能体唯一编号，e.g. 1
    agent_name      : 智能体名称，e.g. "己方编队Alpha"
    equipment_type  : 控制的装备类型，e.g. "歼-20" / "无人机编队"
    equipment_units : 装备个体 id 列表；单个体为 ["__default__"]
    """
    agent_id: int
    agent_name: str
    equipment_type: str
    equipment_units: List[str] = field(default_factory=lambda: ["__default__"])


# ==============================================================================
# 动作空间定义
# ==============================================================================

@dataclass
class ActionItem:
    """
    单个动作项的定义（动作空间中的一个维度）。

    Attributes
    ----------
    name           : 动作项名称，e.g. "雷达开关控制"
    possible_values: 该动作项的所有可选值，e.g. ["开", "关"]
                     如果是连续值则为空列表，说明是浮点数
    is_continuous  : 是否为连续值（True=浮点，False=离散枚举）
    """
    name: str
    possible_values: List[Any] = field(default_factory=list)
    is_continuous: bool = False


# ==============================================================================
# 每步决策记录
# ==============================================================================

@dataclass
class AgentDecision:
    """
    单个智能体在某一步的决策内容。

    Attributes
    ----------
    agent_id : 智能体编号
    content  : 决策内容字典，格式灵活：

               单动作项单值：
                   {"雷达开关控制": "开"}

               多动作项（同时控制多个维度）：
                   {"雷达开关控制": "开", "雷达方向控制": "左"}

               多装备各自动作（编队场景）：
                   {"装备1": {"雷达开关控制": "开"}, "装备2": {"雷达开关控制": "关"}}
    """
    agent_id: int
    content: Dict[str, Any]

    def get_action_value(
        self,
        action_item: str,
        *,
        schema: Optional["AgentSchema"] = None,
        unit_id: Optional[str] = None,
    ) -> Optional[Any]:
        """
        取某个动作项的选定值（支持多装备个体嵌套结构）。

        参数:
            action_item: 动作项名称。
            schema: 可选 AgentSchema；提供时支持多装备嵌套解析。
            unit_id: 可选装备个体 id。

        返回:
            动作值；缺失时返回 None。
        """
        if schema is not None:
            from src.module_c_counterfactual.agent_schema import get_action_value_from_content
            return get_action_value_from_content(
                self.content, schema, action_item, unit_id=unit_id
            )
        # 向后兼容：扁平结构
        if unit_id and unit_id in self.content and isinstance(self.content[unit_id], dict):
            return self.content[unit_id].get(action_item)
        return self.content.get(action_item)


@dataclass
class StepDecision:
    """
    某一时间步所有智能体的决策汇总。

    Attributes
    ----------
    step      : 时间步编号（0-based）
    decisions : 该步每个智能体的决策列表
    """
    step: int
    decisions: List[AgentDecision]

    def get_decision(self, agent_id: int) -> Optional[AgentDecision]:
        """取某个智能体在本步的决策对象。找不到返回 None。"""
        for d in self.decisions:
            if d.agent_id == agent_id:
                return d
        return None

    def get_action_value(self, agent_id: int, action_item: str) -> Optional[Any]:
        """
        快捷方法：取某智能体在某动作项上的选定值。

        Parameters
        ----------
        agent_id    : 智能体编号
        action_item : 动作项名称

        Returns
        -------
        动作值；找不到则返回 None
        """
        d = self.get_decision(agent_id)
        if d is None:
            return None
        return d.get_action_value(action_item)


# ==============================================================================
# 每步观测记录
# ==============================================================================

@dataclass
class AgentObservation:
    """
    单个智能体在某一步的观测内容。

    Attributes
    ----------
    agent_id   : 智能体编号
    obs_values : 观测值字典，key 是观测项名称，value 是嵌套字典（包含子字段）

                 e.g. {
                     "自身状态": {"血量": 0.8, "速度": 1.2, "高度": 8000.0},
                     "敌机距离": {"水平距离": 40.0, "高度差": 500.0},
                     "敌机状态": {"存活": 1, "锁定中": 0}
                 }
    """
    agent_id: int
    obs_values: Dict[str, Dict[str, Any]]   # 观测项 → 子字段字典

    def get_flat_vector(
        self,
        observation_space: List[str],
        *,
        schema: Optional["AgentSchema"] = None,
    ) -> List[float]:
        """
        将嵌套观测 dict 展平为浮点向量。

        参数:
            observation_space: 观测项名称列表。
            schema: 可选 AgentSchema；提供时支持多装备个体前缀。

        返回:
            按 schema 固定顺序排列的浮点特征向量。
        """
        if schema is not None:
            from src.module_c_counterfactual.agent_schema import flatten_obs
            vec, _ = flatten_obs(self.obs_values, schema)
            return vec
        vec: List[float] = []
        for obs_item in observation_space:
            sub_dict = self.obs_values.get(obs_item, {})
            if not isinstance(sub_dict, dict):
                continue
            for sub_key in sorted(sub_dict.keys()):
                vec.append(float(sub_dict[sub_key]))
        return vec

    def get_flat_feature_names(
        self,
        observation_space: List[str],
        *,
        schema: Optional["AgentSchema"] = None,
    ) -> List[str]:
        """
        返回与 get_flat_vector 一一对应的特征名列表。

        参数:
            observation_space: 观测项名称列表。
            schema: 可选 AgentSchema。

        返回:
            展平特征名；多装备时为 unit.观测项.子字段 格式。
        """
        if schema is not None:
            from src.module_c_counterfactual.agent_schema import flatten_obs
            _, names = flatten_obs(self.obs_values, schema)
            return names
        names: List[str] = []
        for obs_item in observation_space:
            sub_dict = self.obs_values.get(obs_item, {})
            if not isinstance(sub_dict, dict):
                continue
            for sub_key in sorted(sub_dict.keys()):
                names.append(f"{obs_item}.{sub_key}")
        return names


@dataclass
class StepObservation:
    """
    某一时间步所有智能体的观测汇总。

    Attributes
    ----------
    step         : 时间步编号（0-based）
    observations : 该步每个智能体的观测列表
    """
    step: int
    observations: List[AgentObservation]

    def get_obs(self, agent_id: int) -> Optional[AgentObservation]:
        """取某个智能体在本步的观测对象。找不到返回 None。"""
        for o in self.observations:
            if o.agent_id == agent_id:
                return o
        return None


# ==============================================================================
# 主数据结构：一次完整仿真的推理记录
# ==============================================================================

@dataclass
class InferenceRecord:
    """
    一次完整仿真过程的推理数据记录。

    Attributes
    ----------
    task_id           : 关联推理任务 id（inference_task_id），e.g. "INF_A_001"
    sim_id            : 推理数据（仿真）id，e.g. "SIM_20240501_001"
    agents            : 参与本次仿真的智能体元信息列表
    observation_space : 观测项名称列表（每项内部是嵌套字典），e.g. ["自身状态", "敌机距离"]
    action_items      : 动作项定义列表（含可选值），e.g. [ActionItem("雷达开关控制", ["开","关"])]
    decisions         : 每步所有智能体的决策记录
    observations      : 每步所有智能体的观测记录
    total_steps       : 总决策次数
    rewards           : 每步奖励值列表
    """
    task_id: str
    sim_id: str
    agents: List[AgentMeta]
    observation_space: List[str]
    action_items: List[ActionItem]
    decisions: List[StepDecision]
    observations: List[StepObservation]
    total_steps: int
    rewards: List[float]
    _decision_index: Dict[Tuple[int, int], AgentDecision] = field(default_factory=dict, init=False, repr=False)
    _obs_index: Dict[Tuple[int, int], AgentObservation] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """构建 (step, agent_id) 索引，加速决策与观测的随机访问。"""
        # 构建 (step, agent_id) 索引，避免训练/推理时频繁线性扫描。
        for step_dec in self.decisions:
            t = int(step_dec.step)
            for dec in step_dec.decisions:
                self._decision_index[(t, int(dec.agent_id))] = dec
        for step_obs in self.observations:
            t = int(step_obs.step)
            for obs in step_obs.observations:
                self._obs_index[(t, int(obs.agent_id))] = obs

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------

    @property
    def agent_ids(self) -> List[int]:
        """返回所有智能体的 id 列表。"""
        return [a.agent_id for a in self.agents]

    @property
    def action_space(self) -> List[str]:
        """返回所有动作项的名称列表（向后兼容用）。"""
        return [item.name for item in self.action_items]

    def get_agent_meta(self, agent_id: int) -> Optional[AgentMeta]:
        """
        根据 agent_id 取智能体元信息。

        参数:
            agent_id: 智能体编号。

        返回:
            AgentMeta；不存在时 None。
        """
        for a in self.agents:
            if a.agent_id == agent_id:
                return a
        return None

    def get_agent_schema(self, agent_id: int) -> "AgentSchema":
        """
        获取指定智能体的 AgentSchema。

        参数:
            agent_id: 智能体编号。

        返回:
            从本记录派生的 AgentSchema。
        """
        from src.module_c_counterfactual.agent_schema import AgentSchema
        return AgentSchema.from_record(self, agent_id)

    # ------------------------------------------------------------------
    # 数据取用接口
    # ------------------------------------------------------------------

    def get_obs_at(self, t: int, agent_id: int) -> Optional[AgentObservation]:
        """
        取第 t 步、agent_id 的观测对象。

        参数:
            t: 时间步索引（0-based）。
            agent_id: 智能体编号。

        返回:
            AgentObservation；不存在时 None。
        """
        return self._obs_index.get((int(t), int(agent_id)))

    def get_decision_at(self, t: int, agent_id: int) -> Optional[AgentDecision]:
        """
        取第 t 步、agent_id 的决策对象。

        参数:
            t: 时间步索引（0-based）。
            agent_id: 智能体编号。

        返回:
            AgentDecision；不存在时 None。
        """
        return self._decision_index.get((int(t), int(agent_id)))

    def get_obs_vector(self, t: int, agent_id: int) -> List[float]:
        """
        返回第 t 步 agent_id 的展平观测向量（含多装备个体前缀）。

        参数:
            t: 时间步索引（0-based）。
            agent_id: 智能体编号。

        返回:
            浮点特征向量；无观测时返回空列表。
        """
        obs = self.get_obs_at(t, agent_id)
        if obs is None:
            return []
        schema = self.get_agent_schema(agent_id)
        return obs.get_flat_vector(self.observation_space, schema=schema)

    def get_flat_feature_names(self, agent_id: int) -> List[str]:
        """
        返回展平特征名列表（与 get_obs_vector 顺序一致）。

        参数:
            agent_id: 智能体编号。

        返回:
            特征名字符串列表；多装备时为 unit.观测项.子字段 格式。
        """
        schema = self.get_agent_schema(agent_id)
        for t in range(self.total_steps):
            obs = self.get_obs_at(t, agent_id)
            if obs is not None:
                return obs.get_flat_feature_names(self.observation_space, schema=schema)
        return []

    def get_action_value(
        self,
        t: int,
        agent_id: int,
        action_item: str,
        unit_id: Optional[str] = None,
    ) -> Optional[Any]:
        """
        取第 t 步、agent_id 在某动作项上的选定值（可多装备）。

        参数:
            t: 时间步索引（0-based）。
            agent_id: 智能体编号。
            action_item: 动作项名称。
            unit_id: 可选装备个体 id。

        返回:
            动作值；无决策时返回 None。
        """
        d = self.get_decision_at(t, agent_id)
        if d is None:
            return None
        schema = self.get_agent_schema(agent_id)
        return d.get_action_value(action_item, schema=schema, unit_id=unit_id)

    # ------------------------------------------------------------------
    # 前端定位接口
    # ------------------------------------------------------------------

    def locate_decision_step(
        self,
        agent_id: int,
        decision_content: Dict[str, Any],
        query_step: Optional[int] = None,
    ) -> Optional[int]:
        """
        根据前端传入的完整 decision_content 定位时间步。

        匹配规则：该步智能体决策中，decision_content 的每个键值对与记录完全一致。
        同一组合在多条时间步重复出现时，应传 query_step（0-based）指定步号。

        Parameters
        ----------
        agent_id         : 智能体 id
        decision_content : 要解释的决策组合，e.g.
                           {"雷达开关控制": "开", "雷达方向控制": "正前方",
                            "武器控制": "不发射", "机动控制": "追击"}
        query_step       : 可选，只在该步上匹配

        Returns
        -------
        t_query : 时间步索引（0-based），找不到则返回 None
        """
        if not decision_content:
            return None

        schema = self.get_agent_schema(agent_id)
        from src.module_c_counterfactual.agent_schema import deep_equal_decision

        for step_dec in self.decisions:
            if query_step is not None and step_dec.step != query_step:
                continue
            d = step_dec.get_decision(agent_id)
            if d is None:
                continue
            if deep_equal_decision(d.content, decision_content, schema):
                return step_dec.step

        return None

    def list_decision_snapshots(
        self,
        agent_id: int,
        *,
        limit: int = 10,
    ) -> List[Tuple[int, str]]:
        """
        列出记录中该智能体的 (step, 决策组合文案)，用于定位失败时的提示。

        参数:
            agent_id: 智能体编号。
            limit: 最多返回条数。

        返回:
            (时间步, 可读决策文案) 元组列表。
        """
        out: List[Tuple[int, str]] = []
        for step_dec in self.decisions:
            d = step_dec.get_decision(agent_id)
            if d is None or not d.content:
                continue
            label = "、".join(
                f"{k}={v}" for k, v in sorted(d.content.items(), key=lambda kv: str(kv[0]))
            )
            out.append((step_dec.step, label))
            if len(out) >= limit:
                break
        return out
