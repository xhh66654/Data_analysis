"""空战环境渲染：静态帧绘制与轨迹动画导出。"""
from __future__ import annotations

from typing import Any, List

import numpy as np


def render_frame(env: Any, ax: Any = None) -> Any:
    """
    在 matplotlib 坐标轴上绘制一帧空战场景。

    绘制内容包括飞机位置、航向、雷达探测圈及已发射导弹等要素。

    参数
    ----
    env : Any
        空战环境实例，需提供实体状态查询接口。
    ax : Any
        可选 matplotlib Axes；为 ``None`` 时自动创建。

    返回
    ----
    Any
        绘制所用的 matplotlib Axes 对象。
    """
    # TODO
    raise NotImplementedError


def animate_trajectory(
    snapshots: List[Any],
    out_path: str = "trajectory.mp4",
    fps: int = 10,
) -> None:
    """
    将一系列环境快照渲染为视频或 GIF 动画。

    参数
    ----
    snapshots : List[Any]
        按时间顺序排列的环境状态快照列表。
    out_path : str
        输出文件路径，扩展名决定格式（如 ``.mp4``、``.gif``）。
    fps : int
        帧率（每秒帧数），默认 10。
    """
    # TODO: 用 matplotlib.animation.FuncAnimation
    raise NotImplementedError
