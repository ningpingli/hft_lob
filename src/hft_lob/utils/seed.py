"""随机种子（需求文档 §29）：Python / NumPy / PyTorch / DataLoader 全种子。"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

_MAX_SEED = 2**32


def set_seed(seed: int) -> None:
    """设置全局随机种子（§29 可复现：Python/NumPy/PyTorch/CUDA + cuDNN 确定性）。

    DataLoader 侧由根 seed 派生 worker 和 shuffle seed，bootstrap 等其他随机流
    也必须从同一根 seed 按稳定命名空间派生。

    Args:
        seed: 全局种子。

    Raises:
        ValueError: seed 不是 ``[0, 2**32)`` 范围内的整数。
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < _MAX_SEED:
        raise ValueError("seed must be an integer in [0, 2**32)")

    # PYTHONHASHSEED 对当前进程已建立的 hash secret 无法追溯生效，但会被后续
    # spawn 的 DataLoader/子进程继承。CUBLAS 配置同样应在首次 CUDA 调用前设置。
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
