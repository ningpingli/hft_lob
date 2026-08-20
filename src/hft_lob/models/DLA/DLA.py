"""DLA：双重注意力（时间与特征维 soft attention）GRU 回归模型。"""

from __future__ import annotations

import lightning.pytorch as pl
import torch


class DLA(pl.LightningModule):
    """DLA：双重注意力加权 + 双层 GRU 的回归模型。"""

    def __init__(
        self,
        num_features: int | None = None,
        num_snapshots: int = 100,
        hidden_size: int = 128,
    ) -> None:
        """初始化 DLA。

        Args:
            num_features: 每快照特征数（None 时为 40）。
            num_snapshots: 每样本的快照数。
            hidden_size: GRU 隐藏层大小。
        """
        raise NotImplementedError("DLA.__init__ not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量（形状 ``(N, 1, num_snapshots, num_features)``）。

        Returns:
            模型输出。
        """
        raise NotImplementedError("DLA.forward not implemented")
