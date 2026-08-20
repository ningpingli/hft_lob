"""随机种子（需求文档 §29）：Python / NumPy / PyTorch / DataLoader 全种子。"""

from __future__ import annotations


def set_seed(seed: int) -> None:
    """设置全局随机种子（§29 可复现：Python/NumPy/PyTorch/CUDA + cuDNN 确定性）。

    DataLoader 侧的确定性由 ``systems.lob_data_module._seed_worker``（worker
    种子）与 ``loader.seed`` 播种的 generator（shuffle 顺序）共同保证。

    Args:
        seed: 全局种子。
    """
    raise NotImplementedError("set_seed not implemented")
