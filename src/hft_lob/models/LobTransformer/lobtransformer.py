"""LobTransformer：卷积特征提取 + Transformer 编码器的 LOB 回归模型。"""

from __future__ import annotations

import lightning.pytorch as pl
import torch


class LobTransformer(pl.LightningModule):
    """LobTransformer：CNN 卷积 + Inception + Transformer 编码器回归模型。"""

    def __init__(
        self,
        num_features: int | None = None,
        levels: int | None = None,
        hidden: int | None = None,
        d_model: int | None = None,
        nhead: int | None = None,
        num_layers: int | None = None,
    ) -> None:
        """初始化 LobTransformer。

        Args:
            num_features: 每快照特征数（None 时为 40）。
            levels: 盘口档位数（None 时为 10）。
            hidden: 卷积隐藏通道数。
            d_model: Transformer 模型维度（None 时按 hidden*2*3 推导）。
            nhead: 注意力头数。
            num_layers: 编码器层数。
        """
        raise NotImplementedError("LobTransformer.__init__ not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量。

        Returns:
            模型输出。
        """
        raise NotImplementedError("LobTransformer.forward not implemented")
