# PDF 论文中文导读

本目录整理了 `pdf` 文件夹下论文/综述的中文内容介绍、关键知识点讲解，以及部分论文的算法实现说明。

## 文件、论文与算法对应关系

1. [01_cema_multi_agent_causal_explanations.md](E:\工作\analysis\paper\01_cema_multi_agent_causal_explanations.md)
   - 对应论文：`2302.10809v4.pdf`
   - 论文标题：`Causal Explanations for Sequential Decision-Making in Multi-Agent Systems`
   - 对应算法/框架：`CEMA`
   - 核心内容：多智能体序列决策中的反事实因果解释

2. [02_seven_tools_of_causal_inference.md](E:\工作\analysis\paper\02_seven_tools_of_causal_inference.md)
   - 对应论文：`3241036.pdf`
   - 论文标题：`The Seven Tools of Causal Inference, with Reflections on Machine Learning`
   - 对应算法/框架：不是单一算法，而是 `SCM + 因果阶梯 + 七大因果工具`
   - 核心内容：Pearl 因果推断总框架与机器学习反思

3. [03_causal_mean_field_marl.md](E:\工作\analysis\paper\03_causal_mean_field_marl.md)
   - 对应论文：`4408_causal_mean_field_multi_agent_.pdf`
   - 论文标题：`Causal Mean Field Multi-Agent Reinforcement Learning`
   - 对应算法/框架：`CMFQ`
   - 核心内容：因果均值场多智能体强化学习

4. [04_explainable_rl_via_model_transforms.md](E:\工作\analysis\paper\04_explainable_rl_via_model_transforms.md)
   - 对应论文：`NeurIPS-2022-explainable-reinforcement-learning-via-model-transforms-Paper-Conference.pdf`
   - 论文标题：`Explainable Reinforcement Learning via Model Transforms`
   - 对应算法/框架：`RLPE / MDP Model Transforms`
   - 核心内容：基于模型变换的可解释强化学习

## 与新增材料的关系

- `溯因分析方案.pdf`
  - 是一个整合性方案，不是单篇原始论文。
  - 其中三个主要算法大致对应：
    - 基于规则抽取的智能溯因技术 -> 主要对应 `硕士论文-李萱露5.24(1).pdf` 中的 `VIPER + 决策树规则提取`
    - 基于博弈约简的智能溯因分析技术 -> 主要对应 `CMFQ`
    - 基于反事实推理的智能溯因分析技术 -> 主要对应 `CEMA`

- `硕士论文-李萱露5.24(1).pdf`
  - 与当前讲解文件没有单独对应的 `md` 文件。
  - 但它与 `溯因分析方案.pdf` 中第一类算法关系最紧密，主要提供：
    - `VIPER` 策略提取
    - `CART` 决策树训练
    - `DFS` 规则提取
    - 规则合并与策略规则库构建

## 建议阅读顺序

1. 先看 `02`，建立因果推断的总框架；
2. 再看 `04`，理解强化学习中的“解释”可以如何形式化；
3. 然后看 `01`，理解多智能体场景中的反事实因果解释；
4. 最后看 `03`，理解因果思想如何直接进入多智能体强化学习算法设计。
