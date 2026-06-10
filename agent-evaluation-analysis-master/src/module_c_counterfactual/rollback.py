"""
基于观测数据的决策回溯。

================================================================================
说明（小白友好版）：
================================================================================

这个模块负责"定位 + 构建反事实上下文"，是数据加载和反事实推理之间的桥梁。

完整的数据流：
    前端输入：(inference_task_id, sim_id, agent_id, decision_content[, query_step])
                ↓
    【数据层】data_loader.load_inference_record(inference_task_id, sim_id)
        根据 (inference_task_id, sim_id) 去 Doris/Mock 查询 → 得到 InferenceRecord
                ↓
    【本模块】ObservationRollback(record)
        .from_frontend_input(agent_id, decision_content, query_step=...)
        根据完整 decision_content 在记录中找到对应的时间步 t_query
        取出 t_query 时刻的真实观测和真实动作
        → 返回 CFContext（反事实上下文）
                ↓
    【反事实推理】local_counterfactual(ctx, policy_model)
        逐一修改每个观测特征 → 决策树预测 → 比较动作是否变化

注意：sim_id 在数据加载阶段（data_loader）就已经使用，
      本模块只接收已加载好的 InferenceRecord，不重复使用 sim_id。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.module_c_counterfactual.inference_record import InferenceRecord
from src.module_c_counterfactual.counterfactual import CFContext


class ObservationRollback:
    """
    基于观测数据的决策回溯控制器。

    职责：
        1. 接收前端输入的 (agent_id, decision_content[, query_step])
        2. 在 InferenceRecord 中定位到对应的时间步 t_query
        3. 取出该时刻的真实观测向量和真实动作
        4. 构建 CFContext，交给局部反事实推理使用

    典型使用：
        # 第一步：数据层加载（见 data_loader.py）
        record = load_inference_record("INF_A_001")

        # 第二步：回溯定位
        rb  = ObservationRollback(record)
        ctx = rb.from_frontend_input(
                  agent_id=1,
                  decision_content={
                      "雷达开关控制": "开",
                      "武器控制": "不发射",
                      "机动控制": "追击",
                  },
                  query_step=3,
              )
        if ctx is None:
            print("未找到对应决策")
        else:
            print(f"定位到第 {ctx.t_query} 步，真实动作：{ctx.action_t}")
    """

    def __init__(self, record: InferenceRecord) -> None:
        """
        初始化回溯控制器。

        参数:
            record: 已从数据库或 Mock 加载的推理数据记录。
        """
        self.record = record

    # ------------------------------------------------------------------
    # 主入口：对应前端完整输入
    # ------------------------------------------------------------------
    def from_frontend_input(
        self,
        agent_id: int,
        decision_content: Dict[str, Any],
        query_step: Optional[int] = None,
    ) -> Optional[CFContext]:
        """
        根据前端传入的完整 decision_content 定位并构建反事实上下文。

        参数:
            agent_id: 被解释的智能体编号。
            decision_content: 要解释的决策组合（与记录 decision_json 字段一致）。
            query_step: 可选；同一组合多次出现时指定 0-based 步号。

        返回:
            定位成功时返回 ``CFContext``；未找到匹配决策时返回 None。

        注意:
            sim_id 已在调用方（data_loader）加载 InferenceRecord 时使用，此处不再传入。
        """
        # 步骤1：定位时间步
        t_query = self.record.locate_decision_step(
            agent_id, decision_content, query_step=query_step
        )
        if t_query is None:
            return None   # 没找到匹配的决策，返回 None

        # 步骤2：构建上下文
        return self.build_context(agent_id, t_query)

    # ------------------------------------------------------------------
    # 内部：根据已知时间步构建上下文
    # ------------------------------------------------------------------
    def build_context(self, agent_id: int, t_query: int) -> CFContext:
        """
        根据 (agent_id, t_query) 构建反事实推理上下文。

        从 InferenceRecord 中取出 t_query 时刻的真实观测和真实动作，
        打包成 CFContext 供后续反事实推理算法使用。

        参数:
            agent_id: 被解释的智能体编号。
            t_query: 要解释的时间步索引（0-based）。

        返回:
            含该时刻真实观测向量（obs_t）与真实动作（action_t）的 ``CFContext``。
        """
        # 取出 t_query 时刻的观测特征向量（展平后的数值列表，按 observation_space 顺序）
        obs_t = self.record.get_obs_vector(t_query, agent_id)

        # 与 policy_model / iter_transitions 一致：一步完整 decision_content 的整体标签
        from src.module_c_counterfactual.training_data import joint_action_label

        dec = self.record.get_decision_at(t_query, agent_id)
        if dec is not None:
            action_t = joint_action_label(dec.content, self.record, agent_id)
        else:
            action_t = ""

        return CFContext(
            record=self.record,
            agent_id=agent_id,
            t_query=t_query,
            obs_t=obs_t,
            action_t=action_t,
        )
