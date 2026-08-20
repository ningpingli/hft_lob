"""Transformer：正弦位置嵌入 + Transformer 编码器的 LOB 回归模型。"""

from __future__ import annotations

import torch
from torch import nn


class SinusoidalPositionalEmbedding(nn.Embedding):
    """生成任意长度的正弦位置嵌入（cos 特征位于向量后半部分）。"""

    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: int | None = None
    ) -> None:
        """初始化正弦位置嵌入。

        Args:
            num_positions: 位置数量。
            embedding_dim: 嵌入维度。
            padding_idx: padding 下标（可选）。
        """
        raise NotImplementedError("SinusoidalPositionalEmbedding.__init__ not implemented")

    @torch.no_grad()
    def forward(
        self, input_ids_shape: torch.Size, past_key_values_length: int = 0
    ) -> torch.Tensor:
        """生成 ``input_ids_shape``（[bsz, seqlen]）对应长度的位置嵌入。

        Args:
            input_ids_shape: 输入形状，前两维为 [bsz, seqlen]。
            past_key_values_length: 过去键值长度（位置偏移）。

        Returns:
            位置嵌入张量。
        """
        raise NotImplementedError("SinusoidalPositionalEmbedding.forward not implemented")


class Transformer(nn.Module):
    """Transformer：嵌入 + 正弦位置编码 + Transformer 编码器回归模型。"""

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
        """初始化 Transformer。

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
        raise NotImplementedError("Transformer.__init__ not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量。

        Returns:
            模型输出。
        """
        raise NotImplementedError("Transformer.forward not implemented")
