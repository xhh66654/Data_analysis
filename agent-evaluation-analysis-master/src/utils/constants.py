"""
项目全局常量。

集中定义动作编码、阵营标识、奖励维度等枚举值，避免魔法数字散落到各模块。
"""

# ---------- 动作编码 ----------
# 对应方案文档表 4
ACTION_NOOP = 0      # 无操作
ACTION_TURN = 1      # 转向规避
ACTION_ACCEL = 2     # 加速推进
ACTION_FIRE = 3      # 发射导弹
ACTION_RADAR = 4     # 切换雷达开关

ACTION_NAMES = {
    ACTION_NOOP: "无操作",
    ACTION_TURN: "转向规避",
    ACTION_ACCEL: "加速推进",
    ACTION_FIRE: "发射导弹",
    ACTION_RADAR: "切换雷达",
}

# ---------- 阵营 ----------
SIDE_BLUE = 0
SIDE_RED = 1

# ---------- 奖励维度（多维奖励向量，对应方案文档） ----------
REWARD_DIMS = [
    "kill_rate",       # 击毁率
    "survival",        # 自身存活率
    "exposure_risk",   # 被发现风险（负向）
    "ammo_cost",       # 弹药消耗（负向）
    "tactic_score",    # 总体战术评分
]
