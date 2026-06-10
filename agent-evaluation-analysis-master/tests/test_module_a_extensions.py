"""模块 A 扩展：VIPER 增广、agent profile、规则匹配、单动作项抽取。"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.module_a_rules.agent_profile import (
    fit_preprocessor_with_profile,
    load_profile,
    profile_id_for,
    schema_fingerprint,
)
from src.module_a_rules.merge_rules import merge_rules
from src.module_a_rules.extract_rules import extract_rules_from_tree
from src.module_a_rules.preprocess import Preprocessor
from src.module_a_rules.rule_match import (
    load_rules_json,
    match_rules,
    predict_from_rules,
    save_rules_json,
)
from src.module_a_rules.viper import VIPERData
from src.module_c_counterfactual.data_loader import load_inference_records
from src.service import rule_extraction_service


@pytest.fixture(autouse=True)
def _tmp_output(monkeypatch, tmp_path):
    """将输出目录重定向到 pytest 临时路径，避免污染项目 output/。

    参数:
        monkeypatch: pytest 环境变量补丁 fixture。
        tmp_path: pytest 临时目录 fixture。
    """
    monkeypatch.setenv("ANALYSIS_OUTPUT_DIR", str(tmp_path / "out"))


def test_viper_resample_augment_increases_dataset():
    """VIPER 重采样增广应扩大训练样本量并记录增广历史。"""
    records = load_inference_records("INF_A_001")
    X, y, r, fn, seg = __import__(
        "src.module_a_rules.collect_data", fromlist=["collect_from_records_with_segments"]
    ).collect_from_records_with_segments(records, 1, action_item="机动控制")
    if len(y) < 4:
        pytest.skip("样本过少")
    v = VIPERData(X, y, r, fn, "机动控制", [], episode_lengths=seg)
    res = v.run(n_iters=3, resample_augment=True, augment_size_factor=1.5)
    assert res.augmentation_history
    assert res.augmentation_history[0]["n_after"] >= res.augmentation_history[0]["n_before"]


def test_agent_profile_save_and_reload(tmp_path):
    """拟合 Preprocessor profile 后应能写盘并按 profile_id 重新加载。

    参数:
        tmp_path: pytest 临时目录（本测试未直接使用，保留 fixture 一致性）。
    """
    records = load_inference_records("INF_A_001")
    r0 = records[0]
    X, _, _, fn = __import__(
        "src.module_a_rules.collect_data", fromlist=["collect_from_records"]
    ).collect_from_records(records, 1, action_item=None)
    pre, profile, path = fit_preprocessor_with_profile(
        X, fn, 1, r0, update_profile=True
    )
    assert profile is not None
    assert path is not None and path.is_file()
    loaded = load_profile(profile.profile_id)
    assert loaded is not None
    assert loaded.feature_names == fn
    assert schema_fingerprint(r0, 1) == profile.schema_fingerprint


def test_match_rules_on_training_sample():
    """训练集样本应能命中至少一条规则并得到非空预测。"""
    result = rule_extraction_service(
        agent_id=1,
        inference_task_id="INF_A_001",
        update_agent_profile=False,
    )
    rules = result["rules"]
    from src.module_a_rules.collect_data import collect_from_records

    records = load_inference_records("INF_A_001")
    X_raw, y, _, fn = collect_from_records(records, 1, action_item=None)
    pre, _, _ = fit_preprocessor_with_profile(
        X_raw, fn, 1, records[0], update_profile=False
    )
    X_pre = pre.transform(X_raw)
    hits = match_rules(rules, X_pre[0], all_matching=True)
    assert hits
    pred = predict_from_rules(rules, X_pre[0])
    assert pred is not None


def test_rule_json_roundtrip(tmp_path):
    """规则 JSON 保存与加载应保留条数与元数据。

    参数:
        tmp_path: 写入临时 rules.json 的目录。
    """
    records = load_inference_records("INF_A_001")
    result = rule_extraction_service(
        agent_id=1,
        inference_task_id="INF_A_001",
        update_agent_profile=False,
    )
    path = tmp_path / "rules.json"
    save_rules_json(path, result["rules"], feature_names=result["feature_names"], metadata={"t": 1})
    rules2, fn2, meta = load_rules_json(path)
    assert meta["t"] == 1
    assert len(rules2) == result["n_rules"]


def test_action_item_single_tree():
    """单动作项模式应返回 mode=single 且标签名为该动作项。"""
    records = load_inference_records("INF_A_001")
    items = [it.name for it in records[0].action_items]
    if not items:
        pytest.skip("无动作项")
    result = rule_extraction_service(
        agent_id=1,
        inference_task_id="INF_A_001",
        action_item=items[0],
        update_agent_profile=False,
    )
    assert result["mode"] == "single"
    assert result["label_name"] == items[0]
    assert result["n_rules"] > 0


def test_action_items_multi_tree():
    """多动作项模式应为每个动作项各生成一棵树。"""
    records = load_inference_records("INF_A_001")
    items = [it.name for it in records[0].action_items][:2]
    if len(items) < 2:
        pytest.skip("动作项不足")
    result = rule_extraction_service(
        agent_id=1,
        inference_task_id="INF_A_001",
        action_items=items,
        update_agent_profile=False,
    )
    assert result["mode"] == "multi"
    assert set(result["trees"].keys()) == set(items)
