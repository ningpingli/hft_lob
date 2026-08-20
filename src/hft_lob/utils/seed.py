"""随机种子（需求文档 §29）：Python / NumPy / PyTorch / DataLoader 全种子。"""

from __future__ import annotations


def set_seed(seed: int) -> None:
    """设置全局随机种子（§29 可复现：Python/NumPy/PyTorch/CUDA + cuDNN 确定性）。

    DataLoader 侧由根 seed 派生 worker 和 shuffle seed，bootstrap 等其他随机流
    也必须从同一根 seed 按稳定命名空间派生。

    Args:
        seed: 全局种子。
    """
    raise NotImplementedError("set_seed not implemented")
