"""模块 C 测试：反事实推理（local counterfactual）。"""

import json
import os
import pytest

from src.module_c_counterfactual.agent_surrogate_profile import (
    load_profile as load_surrogate_profile,
    profile_id_for_record,
)
from src.module_c_counterfactual.data_loader import load_inference_record, load_inference_records
from src.module_c_counterfactual.surrogate_cache import clear_surrogate_bundle_cache
from src.service import counterfactual_service


@pytest.fixture(autouse=True)
def _isolate_profile_dirs(monkeypatch, tmp_path):
    """各测试使用独立 output 目录，避免 agent/surrogate profile 串扰。

    参数:
        monkeypatch: pytest 环境变量补丁 fixture。
        tmp_path: pytest 临时目录 fixture。
    """
    out = tmp_path / "out"
    monkeypatch.setenv("ANALYSIS_OUTPUT_DIR", str(out))


def _maybe_print(title: str, payload) -> None:
    """在设置 ``SHOW_TEST_OUTPUT=1`` 时将调试载荷打印到控制台。

    参数:
        title: 打印区块标题。
        payload: 字符串或可 JSON 序列化的对象。
    """
    if os.environ.get("SHOW_TEST_OUTPUT", "") not in ("1", "true", "True", "yes", "YES"):
        return
    print(f"\n\n===== {title} =====")
    if isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def _pick_decision_content(
    record,
    *,
    agent_id: int,
    step: int | None = None,
) -> tuple[int, dict]:
    """从记录中取一步完整 decision_content（与前端 decision_json 一致）。

    参数:
        record: 单局推理记录对象。
        agent_id: 智能体 ID。
        step: 指定步号；为 ``None`` 时取第一个有效决策步。

    返回:
        ``(step, decision_content)`` 元组。
    """
    if step is not None:
        dec = record.get_decision_at(step, agent_id)
        assert dec is not None and dec.content, f"step={step} 无 agent_id={agent_id} 的决策"
        return step, dict(dec.content)
    for t in range(record.total_steps):
        dec = record.get_decision_at(t, agent_id)
        if dec is not None and dec.content:
            return t, dict(dec.content)
    raise AssertionError(f"记录中无 agent_id={agent_id} 的决策。")


def test_counterfactual_service_end_to_end_smoke():
    """
    端到端验证：直接调用 `counterfactual_service`，应能成功训练近似策略并输出解释。

    断言点：
    - 返回的 mechanistic / teleological 为非空字符串
    - key_features 中每项包含必要字段
    - t_query 落在记录的时间步范围内
    """
    inference_task_id = os.environ.get("TEST_TASK_ID_C", "INF_A_001")
    agent_id = int(os.environ.get("TEST_AGENT_ID_C", "1"))
    sim_id = os.environ.get("TEST_SIM_ID_C", "")

    if not sim_id:
        records = load_inference_records(inference_task_id)
        assert records
        record = records[0]
        sim_id = record.sim_id
    else:
        record = load_inference_record(inference_task_id, sim_id=sim_id)
        assert record is not None

    dc_json = os.environ.get("TEST_DECISION_JSON_C", "").strip()
    if dc_json:
        decision_content = json.loads(dc_json)
        qs = os.environ.get("TEST_QUERY_STEP_C", "").strip()
        t_query = int(qs) if qs else record.locate_decision_step(agent_id, decision_content)
    else:
        t_query, decision_content = _pick_decision_content(record, agent_id=agent_id)

    assert t_query == record.locate_decision_step(agent_id, decision_content, query_step=t_query)

    result = counterfactual_service(
        agent_id=agent_id,
        inference_task_id=inference_task_id,
        sim_id=sim_id,
        decision_content=decision_content,
        query_step=t_query,
        top_k=5,
    )
    _maybe_print(
        "ModuleC.counterfactual_service (SAFE)",
        {
            "task_id": result.get("task_id"),
            "sim_id": result.get("sim_id"),
            "agent_id": result.get("agent_id"),
            "decision_content": result.get("decision_content"),
            "t_query": result.get("t_query"),
            "original_action": result.get("original_action"),
            "n_key_features_changed": result.get("n_key_features_changed"),
            "key_features": result.get("key_features"),
            "mechanistic": result.get("mechanistic"),
            "teleological": result.get("teleological"),
        },
    )

    assert result["t_query"] == t_query
    assert isinstance(result["mechanistic"], str) and result["mechanistic"]
    assert isinstance(result["teleological"], str) and result["teleological"]

    assert result["mechanistic"].startswith("【机械性解释】")
    assert result["teleological"].startswith("【目的性解释】")
    assert isinstance(result.get("nl_explanation"), str) and result["nl_explanation"]
    assert "为什么" in result["nl_explanation"]
    assert "或者回答" in result["nl_explanation"]

    assert isinstance(result["key_features"], list) and result["key_features"]
    for f in result["key_features"]:
        assert "feature" in f
        assert "value" in f
        assert "label" in f
        assert "changed" in f
        assert "change_score" in f
        assert "change_score_mode" in f
        assert "cf_action" in f
    assert isinstance(result.get("key_factors"), list)
    assert result.get("n_key_factors_changed") == result.get("n_key_features_changed")


def test_counterfactual_service_prob_delta_score_mode():
    """
    变化评分模式可切换为概率分布差值（为后续奖励变化模型做接口预留）。
    """
    inference_task_id = "INF_A_001"
    agent_id = 1
    records = load_inference_records(inference_task_id)
    assert records
    record = records[0]
    sim_id = record.sim_id
    t_query, decision_content = _pick_decision_content(record, agent_id=agent_id)

    result = counterfactual_service(
        agent_id=agent_id,
        inference_task_id=inference_task_id,
        sim_id=sim_id,
        decision_content=decision_content,
        query_step=t_query,
        change_score_mode="prob_delta_l1",
    )
    assert result["change_score_mode"] == "prob_delta_l1"
    assert result["key_features"]
    for f in result["key_features"]:
        assert f["change_score_mode"] == "prob_delta_l1"
        assert f["change_score"] >= 0.0



def test_counterfactual_service_one_step_smoke():
    """一步反事实：π+T+R 合并训练 + train_mean 扰动。"""
    inference_task_id = "INF_A_001"
    agent_id = 1
    records = load_inference_records(inference_task_id)
    assert records
    record = records[0]
    sim_id = record.sim_id
    t_query, decision_content = _pick_decision_content(record, agent_id=agent_id)
    if t_query >= record.total_steps - 1:
        pytest.skip("唯一可用决策在最后一步，无法做一步反事实")

    result = counterfactual_service(
        agent_id=agent_id,
        inference_task_id=inference_task_id,
        sim_id=sim_id,
        decision_content=decision_content,
        query_step=t_query,
        cf_level="one_step",
        top_k=5,
        use_k_sampling=False,
    )

    assert result["cf_level"] == "one_step"
    assert result["perturb_strategy"] == "train_mean"
    assert result["n_training_records"] >= 2
    assert isinstance(result["mechanistic"], str) and "一步反事实" in result["mechanistic"]
    assert isinstance(result["teleological"], str) and "一步反事实" in result["teleological"]
    assert result["key_features"]
    for f in result["key_features"]:
        assert "reward_delta" in f
        assert "cf_reward" in f


def test_counterfactual_service_multi_step_smoke():
    """多步反事实：单特征扰动 + 3～5 步累计奖励对比。"""
    inference_task_id = "INF_A_001"
    agent_id = 1
    records = load_inference_records(inference_task_id)
    assert records
    record = records[0]
    sim_id = record.sim_id
    t_query, decision_content = _pick_decision_content(record, agent_id=agent_id)
    if t_query > record.total_steps - 4:
        pytest.skip("当前记录末尾步数不足 3 步，跳过多步反事实测试。")

    result = counterfactual_service(
        agent_id=agent_id,
        inference_task_id=inference_task_id,
        sim_id=sim_id,
        decision_content=decision_content,
        query_step=t_query,
        cf_level="multi_step",
        horizon=3,
        explain_with_llm=False,
        use_k_sampling=False,
    )

    assert result["cf_level"] == "multi_step"
    assert result["horizon"] == 3
    assert "多步反事实" in result["mechanistic"]
    assert result.get("original_cumulative_reward") is not None
    assert result["key_features"]
    assert "reward_delta" in result["key_features"][0]
    assert "horizon" in result["key_features"][0]
    assert isinstance(result.get("nl_explanation"), str) and "为什么" in result["nl_explanation"]
    assert isinstance(result.get("original_action_seq"), list)
    assert isinstance(result.get("cf_action_seq"), list)
    assert len(result["original_action_seq"]) == result["horizon"]
    assert result.get("disclaimer")
    assert result["key_features"][0].get("cf_action_seq")


def test_nl_explanation_focuses_on_multi_action_decision_content():
    """nl_explanation 应以用户传入的 decision_content 为解释对象（可多动作组合）。"""
    inference_task_id = "INF_A_001"
    agent_id = 1
    records = load_inference_records(inference_task_id)
    record = records[0]
    sim_id = record.sim_id

    t_query, decision_content = _pick_decision_content(record, agent_id=agent_id, step=0)
    assert len(decision_content) >= 2

    result = counterfactual_service(
        agent_id=agent_id,
        inference_task_id=inference_task_id,
        sim_id=sim_id,
        decision_content=decision_content,
        query_step=t_query,
        cf_level="one_step",
        explain_with_llm=False,
    )

    explained = "、".join(f"{k}={v}" for k, v in sorted(decision_content.items()))
    assert result.get("explained_decision") == explained
    assert explained in result["nl_explanation"]
    assert "完整动作" not in result["nl_explanation"]
    assert "为什么" in result["nl_question"]
    assert explained in result["nl_question"]
    assert explained in result["nl_answer_mechanistic"]
    assert result["mechanistic_factors"][0]["behavior"] == explained


def test_counterfactual_service_k_sampling_smoke():
    """K 次采样 + 表 2 标量目的论（one_step）。"""
    inference_task_id = "INF_A_001"
    agent_id = 1
    records = load_inference_records(inference_task_id)
    record = records[0]
    t_query, decision_content = _pick_decision_content(record, agent_id=agent_id)
    if t_query >= record.total_steps - 1:
        pytest.skip("末尾步无法做一步反事实")

    result = counterfactual_service(
        agent_id=agent_id,
        inference_task_id=inference_task_id,
        sim_id=record.sim_id,
        decision_content=decision_content,
        query_step=t_query,
        cf_level="one_step",
        use_k_sampling=True,
        k_samples=30,
        k_seed=42,
    )
    assert result.get("use_k_sampling") is True
    assert result.get("k_sampling_meta", {}).get("n_samples") == 30
    assert "K 采样" in result.get("mechanistic", "")
    assert result.get("teleological_effect_scalar") is not None
    assert result.get("disclaimer")


def test_counterfactual_service_invalid_action_content_raises():
    """
    如果 decision_content 不匹配记录中的决策，ObservationRollback 应返回 None，
    counterfactual_service 则应抛出 ValueError。
    """
    inference_task_id = "INF_A_001"
    agent_id = 1
    records = load_inference_records(inference_task_id)
    assert records
    sim_id = records[0].sim_id
    _, base_dc = _pick_decision_content(records[0], agent_id=agent_id)
    bad_dc = dict(base_dc)
    bad_dc["机动控制"] = "不存在的动作值"

    with pytest.raises(ValueError, match="找不到匹配"):
        counterfactual_service(
            agent_id=agent_id,
            inference_task_id=inference_task_id,
            sim_id=sim_id,
            decision_content=bad_dc,
        )


def test_build_fact_bundle_multi_step_narrative_no_crash():
    """多步反事实构建 LLM 事实稿时，累计奖励字段为 None 不应报错。"""
    from src.module_c_counterfactual.llm_explain import build_fact_bundle

    explanation = {
        "original_action": "机动控制=规避",
        "original_cumulative_reward": 0.2,
        "horizon": 4,
        "original_action_seq": ["机动控制=规避", "机动控制=追击"],
        "mechanistic": "【多步反事实·机械性解释】模板",
        "teleological": "【多步反事实·目的性解释】模板",
        "key_features": [
            {
                "feature": "敌机距离.水平距离_km",
                "value": 40.0,
                "label": "低",
                "changed": True,
                "reward_delta": 0.05,
                "original_cumulative_reward": 0.2,
                "cf_cumulative_reward": 0.25,
                "horizon": 4,
                "cf_action": "[('机动控制', '追击')]",
                "cf_action_seq": ["机动控制=追击", "机动控制=追击"],
            }
        ],
    }
    bundle = build_fact_bundle(
        explanation,
        cf_level="multi_step",
        inference_task_id="INF_A_001",
        sim_id="SIM_A_0001",
        agent_id=1,
        t_query=0,
        decision_content={"机动控制": "规避"},
        perturb_strategy="train_mean",
    )
    assert "narrative_for_llm" in bundle
    assert "累计奖励" in bundle["narrative_for_llm"]


def test_agent_preprocessor_profile_version_increments():
    """同一 agent 连续两次 CF 应递增 agent_profile_version。"""
    clear_surrogate_bundle_cache()
    inference_task_id = "INF_A_001"
    agent_id = 1
    records = load_inference_records(inference_task_id)
    sim_id = records[0].sim_id
    t_query, dc = _pick_decision_content(records[0], agent_id=agent_id)

    r1 = counterfactual_service(
        agent_id=agent_id,
        inference_task_id=inference_task_id,
        sim_id=sim_id,
        decision_content=dc,
        query_step=t_query,
        cf_level="local",
        explain_with_llm=False,
    )
    r2 = counterfactual_service(
        agent_id=agent_id,
        inference_task_id=inference_task_id,
        sim_id=sim_id,
        decision_content=dc,
        query_step=t_query,
        cf_level="local",
        explain_with_llm=False,
    )
    assert r1.get("agent_profile_version") is not None
    assert r2.get("agent_profile_version") is not None
    assert r2["agent_profile_version"] >= r1["agent_profile_version"]
    feats1 = [f["feature"] for f in r1["key_features"]]
    feats2 = [f["feature"] for f in r2["key_features"]]
    assert feats1 == feats2


def test_update_agent_profile_false_skips_disk_write(tmp_path):
    """update_agent_profile=False 时不写 agent_profiles。"""
    clear_surrogate_bundle_cache()
    out = tmp_path / "no_write"
    os.environ["ANALYSIS_OUTPUT_DIR"] = str(out)
    inference_task_id = "INF_A_001"
    agent_id = 1
    records = load_inference_records(inference_task_id)
    sim_id = records[0].sim_id
    t_query, dc = _pick_decision_content(records[0], agent_id=agent_id)

    counterfactual_service(
        agent_id=agent_id,
        inference_task_id=inference_task_id,
        sim_id=sim_id,
        decision_content=dc,
        query_step=t_query,
        cf_level="local",
        explain_with_llm=False,
        update_agent_profile=False,
    )
    profiles_dir = out / "agent_profiles"
    assert not profiles_dir.exists() or not any(profiles_dir.glob("*.json"))


def test_policy_holdout_val_accuracy_in_train_debug():
    """train_debug 应包含按 sim 划分的验证集准确率。"""
    from src.module_c_counterfactual.surrogate_bundle import SurrogateBundle

    records = load_inference_records("INF_A_001")
    bundle = SurrogateBundle.fit(records, agent_id=1)
    td = bundle.training_debug or {}
    assert td.get("primary_metric") == "policy_val_weighted_accuracy"
    assert td.get("policy_learning_target") in (
        "holistic_decision_content",
        "holistic_decision_content_composed",
    )
    assert td.get("agent_id") == 1
    assert td.get("holistic_action_space_size", 0) >= 1
    assert td.get("policy_val_weighted_accuracy") is not None
    assert td.get("policy_val_per_item_accuracy") is not None
    assert td.get("policy_val_accuracy") is not None
    assert td.get("n_val_records", 0) >= 1
    assert isinstance(td.get("val_sim_ids"), list)


def test_surrogate_profile_disk_roundtrip():
    """Surrogate profile 写盘；清空内存后第二次应 profile_hit。"""
    clear_surrogate_bundle_cache()
    inference_task_id = "INF_A_001"
    agent_id = 1
    records = load_inference_records(inference_task_id)
    sim_id = records[0].sim_id
    t_query, dc = _pick_decision_content(records[0], agent_id=agent_id)
    if t_query >= records[0].total_steps - 1:
        pytest.skip("末尾步无法做一步反事实")

    r1 = counterfactual_service(
        agent_id=agent_id,
        inference_task_id=inference_task_id,
        sim_id=sim_id,
        decision_content=dc,
        query_step=t_query,
        cf_level="one_step",
        explain_with_llm=False,
        use_k_sampling=False,
    )
    assert r1.get("surrogate_profile_version") is not None
    pid = profile_id_for_record(agent_id, records[0])
    assert load_surrogate_profile(pid) is not None

    clear_surrogate_bundle_cache()
    r2 = counterfactual_service(
        agent_id=agent_id,
        inference_task_id=inference_task_id,
        sim_id=sim_id,
        decision_content=dc,
        query_step=t_query,
        cf_level="one_step",
        explain_with_llm=False,
        use_k_sampling=False,
    )
    assert r2.get("surrogate_cache_hit") is False
    assert r2.get("surrogate_profile_hit") is True


def test_surrogate_profile_tree_params_change_misses_old():
    """树参数变化时不应静默复用旧 surrogate profile。"""
    clear_surrogate_bundle_cache()
    inference_task_id = "INF_A_001"
    agent_id = 1
    records = load_inference_records(inference_task_id)
    sim_id = records[0].sim_id
    t_query, dc = _pick_decision_content(records[0], agent_id=agent_id)
    if t_query >= records[0].total_steps - 1:
        pytest.skip("末尾步无法做一步反事实")

    counterfactual_service(
        agent_id=agent_id,
        inference_task_id=inference_task_id,
        sim_id=sim_id,
        decision_content=dc,
        query_step=t_query,
        cf_level="one_step",
        explain_with_llm=False,
        use_k_sampling=False,
        max_depth=5,
    )
    pid = profile_id_for_record(agent_id, records[0])
    prof = load_surrogate_profile(pid)
    assert prof is not None
    assert prof.tree_params.get("policy_max_depth") == 5

    clear_surrogate_bundle_cache()
    r2 = counterfactual_service(
        agent_id=agent_id,
        inference_task_id=inference_task_id,
        sim_id=sim_id,
        decision_content=dc,
        query_step=t_query,
        cf_level="one_step",
        explain_with_llm=False,
        use_k_sampling=False,
        max_depth=8,
    )
    prof2 = load_surrogate_profile(pid)
    assert prof2 is not None
    assert prof2.tree_params.get("policy_max_depth") == 8
    assert r2.get("surrogate_profile_hit") in (True, False)


def test_policy_surrogate_cache_local_cf():
    """local 模式第二次调用应命中 policy-only 缓存。"""
    clear_surrogate_bundle_cache()
    inference_task_id = "INF_A_001"
    agent_id = 1
    records = load_inference_records(inference_task_id)
    sim_id = records[0].sim_id
    t_query, dc = _pick_decision_content(records[0], agent_id=agent_id)

    r1 = counterfactual_service(
        agent_id=agent_id,
        inference_task_id=inference_task_id,
        sim_id=sim_id,
        decision_content=dc,
        query_step=t_query,
        cf_level="local",
        explain_with_llm=False,
    )
    r2 = counterfactual_service(
        agent_id=agent_id,
        inference_task_id=inference_task_id,
        sim_id=sim_id,
        decision_content=dc,
        query_step=t_query,
        cf_level="local",
        explain_with_llm=False,
    )
    assert r1.get("policy_cache_hit") is False
    assert r2.get("policy_cache_hit") is True


def test_surrogate_bundle_cache_second_call_hits():
    """同一 task+agent 第二次解释应命中 SurrogateBundle 缓存。"""
    from src.module_c_counterfactual.surrogate_cache import clear_surrogate_bundle_cache

    clear_surrogate_bundle_cache()
    inference_task_id = "INF_A_001"
    agent_id = 1
    records = load_inference_records(inference_task_id)
    sim_id = records[0].sim_id
    t_query, dc = _pick_decision_content(records[0], agent_id=agent_id)

    r1 = counterfactual_service(
        agent_id=agent_id,
        inference_task_id=inference_task_id,
        sim_id=sim_id,
        decision_content=dc,
        query_step=t_query,
        cf_level="one_step",
        explain_with_llm=False,
    )
    r2 = counterfactual_service(
        agent_id=agent_id,
        inference_task_id=inference_task_id,
        sim_id=sim_id,
        decision_content=dc,
        query_step=t_query,
        cf_level="one_step",
        explain_with_llm=False,
    )
    assert r1.get("surrogate_cache_hit") is False
    assert r2.get("surrogate_cache_hit") is True


def test_frontend_entry_manual_request_and_print_result():
    """
    手动调试入口：模拟前端反事实解释请求并打印完整回传结果。

    流程与线上一致：全任务训练 π/T/R（有缓存），再在指定 sim 上解释。

    环境变量：
       - FRONT_TASK_ID / FRONT_SIM_ID / FRONT_AGENT_ID
       - FRONT_DECISION_JSON（完整 decision_content，与 decision_json 一致）
       - FRONT_QUERY_STEP（0-based 步号；同一组合多次出现时必须指定，如 3）
       - FRONT_CF_LEVEL（local / one_step / multi_step，默认 multi_step）
       - FRONT_HORIZON（仅 multi_step，3～5，默认 4）
       - FRONT_PERTURB_STRATEGY
       - ANALYSIS_LLM_EXPLAIN=0 关闭 LLM
       - ANALYSIS_CF_BUNDLE_CACHE=0 关闭代理模型缓存
       - ANALYSIS_CF_TRAIN_DEBUG=1 显示训练/验证准确率（手动测试默认开启）
       - ANALYSIS_CF_VAL_RATIO=0.2  验证集占比（按 sim 划分，默认 20%）

    运行示例（PowerShell）：
        py -m pytest -s tests/test_module_c.py::test_frontend_entry_manual_request_and_print_result -q

    默认使用 mock 中已有的 INF_A_001；sim / 步号 / decision 未指定时从记录自动选取。
    指定其他任务请设环境变量，例如：
        $env:FRONT_TASK_ID="INF_A_002"; $env:FRONT_SIM_ID="SIM_A_0006"
    """
    # ===== 方式A：直接改这里（留空则自动从 mock 记录解析）=====
    inference_task_id = "INF_A_001"
    agent_id = 1
    sim_id = ""
    query_step: int | None = None
    decision_content: dict | None = None
    cf_level = "multi_step"
    horizon = 4

    # ===== 方式B：环境变量覆盖 =====
    inference_task_id = os.environ.get("FRONT_TASK_ID", inference_task_id)
    sim_id = os.environ.get("FRONT_SIM_ID", sim_id).strip()
    agent_id = int(os.environ.get("FRONT_AGENT_ID", str(agent_id)))
    qs_env = os.environ.get("FRONT_QUERY_STEP", "").strip()
    if qs_env != "":
        query_step = int(qs_env)
    dc_json = os.environ.get("FRONT_DECISION_JSON", "").strip()
    if dc_json:
        decision_content = json.loads(dc_json)
    cf_level = os.environ.get("FRONT_CF_LEVEL", cf_level)
    horizon = int(os.environ.get("FRONT_HORIZON", str(horizon)))

    records = load_inference_records(inference_task_id)
    if not records:
        pytest.skip(
            f"任务 {inference_task_id} 在 data/mock_records 中无记录。"
            f"请改用已有任务（如 INF_A_001）或设置 FRONT_TASK_ID。"
        )
    if not sim_id:
        sim_id = records[0].sim_id
    record = next((r for r in records if r.sim_id == sim_id), None)
    if record is None:
        available = [r.sim_id for r in records]
        pytest.fail(
            f"sim_id={sim_id} 不在任务 {inference_task_id} 中。"
            f"可用 sim：{available[:8]}{'...' if len(available) > 8 else ''}"
        )

    if cf_level == "multi_step":
        last_ok = record.total_steps - horizon - 1
    elif cf_level == "one_step":
        last_ok = record.total_steps - 2
    else:
        last_ok = record.total_steps - 1

    if query_step is None:
        picked = None
        for t in range(max(0, last_ok + 1)):
            dec = record.get_decision_at(t, agent_id)
            if dec is not None and dec.content:
                picked = (t, dict(dec.content))
                break
        if picked is None:
            pytest.skip(f"agent_id={agent_id} 在 {sim_id} 中无可解释决策步。")
        query_step, decision_content = picked
    elif decision_content is None:
        dec = record.get_decision_at(query_step, agent_id)
        if dec is None or not dec.content:
            pytest.fail(f"step={query_step} 无 agent_id={agent_id} 的决策。")
        decision_content = dict(dec.content)

    perturb_strategy = os.environ.get("FRONT_PERTURB_STRATEGY", "train_mean")

    # 手动调试默认打开训练指标（policy_train_accuracy 等）；关闭：$env:ANALYSIS_CF_TRAIN_DEBUG=0
    if os.environ.get("ANALYSIS_CF_TRAIN_DEBUG", "").strip() == "":
        os.environ["ANALYSIS_CF_TRAIN_DEBUG"] = "1"

    llm_flag = os.environ.get("ANALYSIS_LLM_EXPLAIN", "").strip().lower()
    use_llm = not (llm_flag in ("0", "false", "no", "off"))
    from src.module_c_counterfactual.llm_explain import is_local_llm_model_ready

    if use_llm:
        use_llm = is_local_llm_model_ready()

    request_payload = {
        "inference_task_id": inference_task_id,
        "sim_id": sim_id,
        "agent_id": agent_id,
        "decision_content": decision_content,
        "query_step": query_step,
        "cf_level": cf_level,
        "horizon": horizon,
        "perturb_strategy": perturb_strategy,
        "explain_with_llm": use_llm,
    }

    if use_llm:
        print("\n[LLM] 本地模型已就绪，默认开启解释润色\n")
    elif is_local_llm_model_ready() is False and llm_flag not in ("0", "false", "no", "off"):
        print(
            "\n[LLM] 未检测到 model.safetensors，使用模板解释。"
            "可先运行: py scripts/verify_llm_model.py\n"
        )

    result = counterfactual_service(
        agent_id=request_payload["agent_id"],
        inference_task_id=request_payload["inference_task_id"],
        sim_id=request_payload["sim_id"],
        decision_content=request_payload["decision_content"],
        query_step=query_step,
        cf_level=cf_level,
        horizon=horizon,
        perturb_strategy=perturb_strategy,
        top_k=5,
        explain_with_llm=use_llm,
        use_k_sampling=False,
    )

    print(f"\n===== 定位步 t_query={result.get('t_query')}（请求 query_step={query_step}）=====")
    assert result.get("t_query") == query_step
    assert result.get("sim_id") == sim_id

    # 无条件打印：这个测试就是给你手动看“前端回传结果”用的
    print("\n\n===== FRONTEND REQUEST (SIMULATED) =====")
    print(json.dumps(request_payload, ensure_ascii=False, indent=2))
    print("\n===== FRONTEND RESPONSE (SERVICE RESULT) =====")
    print(
        json.dumps(
            {
                "task_id": result.get("task_id"),
                "sim_id": result.get("sim_id"),
                "agent_id": result.get("agent_id"),
                "decision_content": result.get("decision_content"),
                "cf_level": result.get("cf_level"),
                "perturb_strategy": result.get("perturb_strategy"),
                "n_training_records": result.get("n_training_records"),
                "n_training_transitions": result.get("n_training_transitions"),
                "train_debug": result.get("train_debug"),
                "t_query": result.get("t_query"),
                "original_action": result.get("original_action"),
                "original_reward": result.get("original_reward"),
                "horizon": result.get("horizon"),
                "original_cumulative_reward": result.get("original_cumulative_reward"),
                "original_action_seq": result.get("original_action_seq"),
                "cf_action_seq": result.get("cf_action_seq"),
                "top_feature": result.get("top_feature"),
                "surrogate_cache_hit": result.get("surrogate_cache_hit"),
                "disclaimer": result.get("disclaimer"),
                "n_key_features_changed": result.get("n_key_features_changed"),
                "n_features_total": result.get("n_features_total"),
                "key_features": result.get("key_features"),
                "explanation_backend": result.get("explanation_backend"),
                "headline": result.get("headline"),
                "summary": result.get("summary"),
                "llm_error": result.get("llm_error"),
                "nl_explanation": result.get("nl_explanation"),
                "teleological_factors": result.get("teleological_factors"),
                "mechanistic_factors": result.get("mechanistic_factors"),
                "mechanistic": result.get("mechanistic"),
                "teleological": result.get("teleological"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if result.get("headline"):
        print("\n===== 一句话结论 =====")
        print(result["headline"])
    if result.get("summary"):
        print("\n===== 综合摘要 =====")
        print(result["summary"])
    print(f"\n===== explanation_backend = {result.get('explanation_backend', 'template')} =====")
    if result.get("llm_error"):
        print(f"===== llm_error = {result['llm_error']} =====")
    print("\n===== 自然语言因果解释（主展示） =====")
    print(result.get("nl_explanation"))
    if result.get("cf_level") == "multi_step":
        print("\n===== 多步轨迹摘要 =====")
        print(f"horizon={result.get('horizon')}  top_feature={result.get('top_feature')}")
        print(f"累计奖励(事实)={result.get('original_cumulative_reward')}")
        print(f"事实动作序列: {result.get('original_action_seq')}")
        print(f"反事实动作序列: {result.get('cf_action_seq')}")
    print(f"\n===== surrogate_cache_hit = {result.get('surrogate_cache_hit')} =====")
    print(f"===== surrogate_profile_hit = {result.get('surrogate_profile_hit')} =====")
    td = result.get("train_debug")
    if td:
        print("\n===== TRAIN / VAL 指标（按 sim 划分，主看验证集）=====")
        print(json.dumps(td, ensure_ascii=False, indent=2))
        val_pi = td.get("policy_val_per_item_accuracy")
        val_acc = td.get("policy_val_accuracy")
        val_base = td.get("majority_baseline_val_accuracy")
        train_pi = td.get("policy_train_per_item_accuracy")
        ratio = td.get("val_split_ratio")
        val_sims = td.get("val_sim_ids")
        if val_pi is not None:
            print(f"\n【主指标】验证集动作项平均准确率 policy_val_per_item_accuracy = {val_pi:.4f}")
        if val_acc is not None:
            print(f"（参考）联合标签完全匹配 policy_val_accuracy = {val_acc:.4f}")
        if val_base is not None:
            print(f"（参考）多数类基线 majority_baseline_val_accuracy = {val_base:.4f}")
        if train_pi is not None:
            print(f"（参考）训练集动作项平均准确率 policy_train_per_item_accuracy = {train_pi:.4f}")
        if ratio is not None:
            print(f"划分比例 val_split_ratio={ratio}  验证局 val_sim_ids={val_sims}")
        if td.get("policy_val_per_item_breakdown"):
            print(f"验证集分项: {td.get('policy_val_per_item_breakdown')}")
        if td.get("accuracy_note"):
            print(td.get("accuracy_note"))
    elif cf_level == "local":
        print(
            "\n[train_debug] local 模式仅训练 π，当前 service 不向 train_debug 写入准确率；"
            "请改用 one_step / multi_step，或看 eval_training_effect.py。"
        )
    else:
        print(
            "\n[train_debug] 为空：请确认 ANALYSIS_CF_TRAIN_DEBUG=1；"
            "若 surrogate 从磁盘 profile 增量 refit，training_debug 可能未计算。"
        )
    if result.get("disclaimer"):
        print(result["disclaimer"])
    print("\n===== MECHANISTIC (FULL TEXT) =====")
    print(result.get("mechanistic"))
    print("\n===== TELEOLOGICAL (FULL TEXT) =====")
    print(result.get("teleological"))

    assert result.get("cf_level") == cf_level
    assert result.get("n_training_records", 0) >= 1
    assert isinstance(result.get("nl_explanation"), str) and "为什么" in result["nl_explanation"]
    assert "回答：" in result["nl_explanation"] and "或者回答：" in result["nl_explanation"]
    assert result["key_features"]

    backend = result.get("explanation_backend", "template")

    if cf_level == "one_step":
        assert "reward_delta" in result["key_features"][0]
        assert "一步反事实" in result.get("mechanistic", "") or "一步反事实" in (
            result.get("mechanistic_raw") or ""
        )
    elif cf_level == "multi_step":
        assert result.get("horizon") == horizon
        assert isinstance(result.get("original_action_seq"), list)
        assert "reward_delta" in result["key_features"][0]
        # 开启 LLM 时 mechanistic 会被改写；模板原文在 mechanistic_raw
        assert "多步反事实" in result.get("mechanistic", "") or "多步反事实" in (
            result.get("mechanistic_raw") or ""
        ) or backend.startswith("llm_")
    else:
        assert "change_score" in result["key_features"][0]

    assert isinstance(result.get("mechanistic"), str) and result["mechanistic"]
    assert isinstance(result.get("teleological"), str) and result["teleological"]
