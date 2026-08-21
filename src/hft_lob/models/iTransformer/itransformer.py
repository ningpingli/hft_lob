"""iTransformer：倒置 Transformer——把时间序列视为特征维进行嵌入的回归模型。"""

from __future__ import annotations

import torch
from torch import nn


class ITransformer(nn.Module):
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
        super().__init__()
        d_model = 64 if d_model is None else d_model
        dim_feedforward = 256 if dim_feedforward is None else dim_feedforward
        nhead = 8 if nhead is None else nhead
        num_layers = 2 if num_layers is None else num_layers

        self.history_length = history_length
        # 嵌入宽度绑定历史长度：每条特征的时间序列被嵌入为 d_model 维。
        self.embed = nn.Linear(history_length, d_model, bias=False)
        layer_norm_eps: float = 1e-5
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            batch_first=True,
        )
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=encoder_norm
        )
        self.regression_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 统一时序输入 ``[B, T, F]``。

        Returns:
            模型输出 ``(N, 1)``。

        Raises:
            ValueError: 时间维与构造契约不一致。
        """
        if x.ndim != 3:
            raise ValueError(f"ITransformer expects [B, T, F], got shape {tuple(x.shape)}")
        # 嵌入宽度绑定 history_length，时间维不匹配会崩溃。
        if x.shape[1] != self.history_length:
            raise ValueError(
                f"ITransformer expects {self.history_length} snapshots per "
                f"sample, got {x.shape[1]}. 请核对 ExperimentConfig 的 "
                f"window.history_snapshots 契约。"
            )
        # 转置：沿特征维（每条特征一个 token）嵌入历史序列。
        x = x.permute(0, 2, 1)
        x = self.embed(x)

        # Transformer 编码器
        x = self.transformer_encoder(x)

        # 回归读出头（均值池化）
        x = torch.mean(x, dim=1)

        prediction = self.regression_head(x)
        return prediction
