"""
随机种子工具。

统一设置 Python、NumPy 及 PyTorch（若已安装）的随机种子，保证实验可复现。
"""
import random

import numpy as np


def set_seed(seed: int) -> None:
    """
    设置全局随机种子。

    依次设置 ``random``、``numpy`` 及 ``torch``（含 CUDA）的种子；
    未安装 PyTorch 时静默跳过。

    参数
    ----
    seed : int
        随机种子整数值。
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
