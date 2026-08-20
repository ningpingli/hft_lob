"""iTransformer：倒置 Transformer——把时间序列视为特征维进行嵌入的回归模型。"""

from __future__ import annotations

import lightning.pytorch as pl
import torch


class ITransformer(pl.LightningModule):
    """ITransformer：沿特征维嵌入历史序列的倒置 Transformer 回归模型。"""

    def __init__(
        self,
        num_features: int | None = None,
        d_model: int | None = None,
        dim_feedforward: int | None = None,
        nhead: int | None = None,
        num_layers: int | None = None,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_first: bool = False,
        history_length: int = 100,
    ) -> None:
        """初始化 ITransformer。

        Args:
            num_features: 每快照特征数（None 时为 40）。
            d_model: 模型维度（None 时为 64）。
            dim_feedforward: 前馈维度（None 时为 256）。
            nhead: 注意力头数（None 时为 8）。
            num_layers: 编码器层数（None 时为 2）。
            dropout: dropout 概率。
            activation: 前馈激活函数名。
            norm_first: 是否先做层归一化。
            history_length: 历史窗口长度。
        """
        raise NotImplementedError("ITransformer.__init__ not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量。

        Returns:
            模型输出。
        """
        raise NotImplementedError("ITransformer.forward not implemented")
