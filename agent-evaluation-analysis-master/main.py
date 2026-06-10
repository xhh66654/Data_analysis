"""
命令行入口。

================================================================================
用法示例：
================================================================================

【基于规则的策略提取】
    py main.py --mode explain_a --inference_task_id INF_A_001 --agent_id 1

【反事实推理】
    py main.py --mode explain_c --inference_task_id INF_A_001 --sim_id SIM_A_0001 --agent_id 1 ^
        --decision "雷达开关控制=开,雷达方向控制=正前方,武器控制=不发射,机动控制=追击"

参数说明：
    --inference_task_id  推理任务 id（规则抽取），e.g. INF_A_001
    --task_id            推理任务 id（已废弃别名，优先用 --inference_task_id）
    --sim_id             仿真 id（反事实推理必填）
    --agent_id    智能体编号，e.g. 1
    --decision    完整决策组合，格式 "动作项=值" 或逗号分隔多项（与 decision_json 一致）
    --query_step  可选，0-based 步号；同一组合多次出现时必填
    --autotune_a  explain_a 时启用自动调参（按 sim 做 CV 网格搜索）
    --out         输出路径（规则抽取时指定 PDF 路径前缀，不含扩展名）
"""
from __future__ import annotations

import argparse
import json

from src.utils import get_logger, load_config, set_seed

log = get_logger("main")


def build_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器。

    支持 ``train``、``explain_a``、``explain_b``、``explain_c``、``render`` 等运行模式，
    以及规则抽取与反事实推理的专用参数。

    返回
    ----
    argparse.ArgumentParser
        已注册全部子参数的命令行解析器。
    """
    p = argparse.ArgumentParser(description="智能体溯因分析项目")
    p.add_argument("--mode", required=True,
                   choices=["train", "explain_a", "explain_b", "explain_c", "render"])
    # 通用参数
    p.add_argument("--config",  default="config.yaml")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--out",     default=None, help="输出路径")
    # 旧参数（保留兼容）
    p.add_argument("--agent",   default="dqn", choices=["dqn", "ppo", "mfrl", "scripted"])
    p.add_argument("--model",   default=None)
    p.add_argument("--query",   default=None)
    p.add_argument("--steps",   type=int, default=None)
    # 新参数：explain_a / explain_c 专用
    p.add_argument(
        "--inference_task_id",
        default=None,
        help="推理任务 id（规则抽取），e.g. INF_A_001",
    )
    p.add_argument(
        "--task_id",
        default=None,
        help="推理任务 id（已废弃别名，优先用 --inference_task_id）",
    )
    p.add_argument("--sim_id", default=None, help="仿真 id（反事实推理专用）")
    p.add_argument("--agent_id",    type=int, default=1, help="智能体编号，默认 1")
    p.add_argument("--decision",    default=None,
                   help="完整决策组合，如 '机动控制=追击,武器控制=不发射,...'（explain_c 专用）")
    p.add_argument("--query_step", type=int, default=None,
                   help="要解释的时间步（0-based）；同一 decision 重复出现时指定")
    p.add_argument(
        "--autotune_a",
        action="store_true",
        help="explain_a 启用自动调参（ANALYSIS_A_AUTOTUNE=1）",
    )
    p.add_argument(
        "--action-item",
        default=None,
        help="规则抽取：单动作项标签，如 机动控制（默认联合动作）",
    )
    p.add_argument(
        "--action-items",
        default=None,
        help="规则抽取：逗号分隔多个动作项，各训练一棵树",
    )
    p.add_argument(
        "--cf_level",
        default="local",
        choices=["local", "one_step", "multi_step"],
        help="反事实层级：local / one_step / multi_step（3～5 步滚动）",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="多步反事实滚动步数（仅 multi_step），3～5，默认 5",
    )
    p.add_argument(
        "--perturb_strategy",
        default=None,
        help="特征扰动：zero / train_mean（默认 local=zero, one_step=train_mean）",
    )
    p.add_argument(
        "--llm_explain",
        action="store_true",
        help="用预训练模型润色 mechanistic/teleological（需 ANALYSIS_LLM_EXPLAIN 或本地模型）",
    )
    p.add_argument(
        "--optimize_c",
        action="store_true",
        help="explain_c 开启模块C自动优化（policy_mode=auto + T模型自动调参）",
    )
    p.add_argument(
        "--grouped_t",
        action="store_true",
        help="explain_c 允许 T 模型评估分维建模候选（需配合 --optimize_c）",
    )
    p.add_argument(
        "--strict_conservative",
        action="store_true",
        help="严格保守模式：仅当CV显著优于默认基线才采用调参结果",
    )
    return p


# ==============================================================================
# 各模式处理函数
# ==============================================================================

def cmd_train(args: argparse.Namespace, cfg: dict) -> None:
    """
    执行 ``train`` 模式：训练强化学习智能体。

    参数
    ----
    args : argparse.Namespace
        命令行解析结果。
    cfg : dict
        自 YAML 加载的项目配置。

    异常
    ----
    NotImplementedError
        当前尚未实现。
    """
    raise NotImplementedError("train 模式尚未实现")


def cmd_explain_a(args: argparse.Namespace, cfg: dict) -> None:
    """
    执行 ``explain_a`` 模式：模块 A 基于规则的策略提取。

    从推理任务数据训练 CART 决策树，提取 IF-THEN 规则集并导出 PDF，
    结果打印至标准输出。

    参数
    ----
    args : argparse.Namespace
        需含 ``inference_task_id``（或 ``task_id``）、``agent_id`` 等字段。
    cfg : dict
        项目配置字典（当前未直接使用，保留接口一致性）。
    """
    inference_task_id = args.inference_task_id or args.task_id
    if inference_task_id is None:
        print("[错误] --inference_task_id 为必填参数，e.g. --inference_task_id INF_A_001")
        return

    from src.module_a_rules.run_report import print_terminal_summary
    from src.service import rule_extraction_service

    log.info(
        "规则抽取启动: inference_task_id=%s agent_id=%s",
        inference_task_id,
        args.agent_id,
    )
    if args.autotune_a:
        import os
        os.environ["ANALYSIS_A_AUTOTUNE"] = "1"
        log.info("自动调参 = 开启")
    if args.strict_conservative:
        import os
        os.environ["ANALYSIS_STRICT_CONSERVATIVE"] = "1"
        log.info("严格保守模式 = 开启")
    log.info("正在训练决策树……")

    action_items = None
    if getattr(args, "action_items", None):
        action_items = [s.strip() for s in args.action_items.split(",") if s.strip()]

    result = rule_extraction_service(
        agent_id=args.agent_id,
        inference_task_id=inference_task_id,
        pdf_path=args.out,
        action_item=getattr(args, "action_item", None),
        action_items=action_items,
    )

    if result.get("mode") == "multi":
        print("=" * 60)
        print("规则抽取流水线 · 多动作项模式完成")
        print("=" * 60)
        for name, sub in (result.get("trees") or {}).items():
            vm = sub.get("viper_metrics") or {}
            arts = sub.get("run_artifacts") or {}
            print(
                f"  [{name}] acc={sub.get('accuracy'):.4f}  "
                f"loss={vm.get('best_loss', 'N/A')}  "
                f"rules={sub.get('n_rules')}  → {arts.get('rules_txt', '')}"
            )
        print("=" * 60)
        return

    print_terminal_summary(
        result,
        artifact_paths=result.get("run_artifacts") or {},
        viper_metrics=result.get("viper_metrics") or {},
        holdout=result.get("holdout_eval") or {},
        preprocessor=result.get("_preprocessor"),
    )


def cmd_explain_b(args: argparse.Namespace, cfg: dict) -> None:
    """
    执行 ``explain_b`` 模式：模块 B 博弈约简溯因。

    参数
    ----
    args : argparse.Namespace
        命令行解析结果。
    cfg : dict
        项目配置字典。

    异常
    ----
    NotImplementedError
        当前尚未实现。
    """
    raise NotImplementedError("explain_b 模式尚未实现")


def cmd_explain_c(args: argparse.Namespace, cfg: dict) -> None:
    """
    执行 ``explain_c`` 模式：模块 C 反事实推理。

    给定推理任务、仿真局与完整决策组合，调用反事实服务并打印
    机械性/目的性解释；可选将 JSON 结果写入 ``--out`` 路径。

    参数
    ----
    args : argparse.Namespace
        需含 ``inference_task_id``、``sim_id``、``decision`` 等字段。
    cfg : dict
        项目配置字典。
    """
    inference_task_id = args.inference_task_id or args.task_id
    if inference_task_id is None:
        print("[错误] --inference_task_id 为必填参数，e.g. --inference_task_id INF_A_001")
        return
    if args.sim_id is None:
        print("[错误] --sim_id 为必填参数，e.g. --sim_id SIM_A_0001")
        return
    if args.decision is None:
        print("[错误] --decision 为必填参数，传入完整决策组合，见 --help")
        return

    # 解析 --decision 参数：支持 "键=值" 或 "键=值,键=值" 格式
    decision_content = _parse_decision(args.decision)
    if not decision_content:
        print(f"[错误] --decision 格式错误（应为 '键=值' 或 '键=值,键=值'），收到：{args.decision}")
        return

    from src.service import counterfactual_service

    print(
        f"\n[反事实推理] inference_task_id={inference_task_id}  sim_id={args.sim_id}  "
        f"agent_id={args.agent_id}"
    )
    print(f"  目标决策：{decision_content}")
    print("  正在推理……")

    if args.llm_explain:
        import os
        os.environ["ANALYSIS_LLM_EXPLAIN"] = "1"
    if args.optimize_c:
        import os
        os.environ["ANALYSIS_CF_POLICY_MODE"] = "auto"
        os.environ["ANALYSIS_CF_T_AUTOTUNE"] = "1"
        print("  自动优化     = 开启（policy_mode=auto + transition_autotune）")
    if args.grouped_t:
        import os
        os.environ["ANALYSIS_CF_T_GROUPED"] = "1"
        print("  T分维候选    = 开启（在自动调参中评估 per_feature）")
    if args.strict_conservative:
        import os
        os.environ["ANALYSIS_STRICT_CONSERVATIVE"] = "1"
        print("  严格保守模式 = 开启（仅显著优于基线才采用）")

    result = counterfactual_service(
        agent_id=args.agent_id,
        inference_task_id=inference_task_id,
        sim_id=args.sim_id,
        decision_content=decision_content,
        query_step=args.query_step,
        cf_level=args.cf_level,
        horizon=args.horizon,
        perturb_strategy=args.perturb_strategy,
        explain_with_llm=args.llm_explain or None,
    )

    print(f"\n  推理完成：")
    print(f"    解释来源     = {result.get('explanation_backend', 'template')}")
    if result.get("llm_error"):
        print(f"    LLM 提示     = {result['llm_error']}")
    print(f"    反事实层级   = {result.get('cf_level', 'local')}")
    print(f"    扰动策略     = {result.get('perturb_strategy', 'zero')}")
    print(f"    定位时间步   = t={result['t_query']}")
    print(f"    真实动作     = {result['original_action']}")
    if result.get("cf_level") == "one_step":
        print(f"    真实一步奖励 = {result.get('original_reward')}")
    if result.get("cf_level") == "multi_step":
        print(f"    滚动步数     = {result.get('horizon')}")
        print(f"    真实累计奖励 = {result.get('original_cumulative_reward')}")
    print(f"    关键特征数   = {result['n_key_features_changed']} / {result['n_features_total']}")
    if result.get("summary"):
        print(f"\n{'='*60}")
        print("【综合摘要】")
        print(result["summary"])
    print(f"\n{'='*60}")
    print(result["mechanistic"])
    print(f"\n{'='*60}")
    print(result["teleological"])

    if args.out:
        out_file = args.out if args.out.endswith(".json") else args.out + ".json"
        # 过滤掉不可序列化的对象
        serializable = {k: v for k, v in result.items() if k != "key_features"}
        serializable["key_features"] = result["key_features"]
        import json
        import pathlib
        pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out_file).write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"\n  结果已保存至：{out_file}")


def cmd_render(args: argparse.Namespace, cfg: dict) -> None:
    """
    执行 ``render`` 模式：加载已训练模型运行一局并渲染动画。

    参数
    ----
    args : argparse.Namespace
        命令行解析结果。
    cfg : dict
        项目配置字典。

    异常
    ----
    NotImplementedError
        当前尚未实现。
    """
    raise NotImplementedError("render 模式尚未实现")


# ==============================================================================
# 工具函数
# ==============================================================================

def _parse_decision(decision_str: str) -> dict:
    """
    将命令行 ``--decision`` 参数字符串解析为决策字典。

    支持格式：
    - ``"机动控制=规避"`` → ``{"机动控制": "规避"}``
    - ``"机动控制=规避,武器控制=不发射"`` → 多键值对

    参数
    ----
    decision_str : str
        逗号分隔的 ``键=值`` 对字符串。

    返回
    ----
    dict
        解析后的决策内容字典；格式错误时返回空字典或部分键值。
    """
    result = {}
    try:
        for pair in decision_str.split(","):
            pair = pair.strip()
            if "=" not in pair:
                continue
            k, v = pair.split("=", 1)
            result[k.strip()] = v.strip()
    except Exception:
        pass
    return result


# ==============================================================================
# 主入口
# ==============================================================================

HANDLERS = {
    "train":     cmd_train,
    "explain_a": cmd_explain_a,
    "explain_b": cmd_explain_b,
    "explain_c": cmd_explain_c,
    "render":    cmd_render,
}


def main() -> None:
    """
    程序主入口：解析命令行、加载配置、设置随机种子并分发至对应模式处理函数。
    """
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    log.info("启动模式：%s", args.mode)
    HANDLERS[args.mode](args, cfg)


if __name__ == "__main__":
    main()
