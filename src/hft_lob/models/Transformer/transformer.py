"""Transformer：正弦位置嵌入 + Transformer 编码器的 LOB 回归模型。"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SinusoidalPositionalEmbedding(nn.Module):
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
        super().__init__()
        self.embedding = nn.Embedding(
            num_positions, embedding_dim, padding_idx=padding_idx
        )
        self.embedding.weight = self._init_weight(self.embedding.weight)

    @staticmethod
    def _init_weight(out: torch.Tensor) -> torch.Tensor:
        """与 XLM 的 create_sinusoidal_embeddings 相同，只是特征不交错排列。

        cos 特征位于向量的后半部分，即 ``[dim // 2:]``。
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # 提前置 False，避免 pytorch-1.8+ 报错
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.tensor(
            np.sin(position_enc[:, 0::2]), dtype=torch.float32
        )
        out[:, sentinel:] = torch.tensor(
            np.cos(position_enc[:, 1::2]), dtype=torch.float32
        )
        out.detach_()
        return out

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
        _, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.embedding.weight.device,
        )
        return self.embedding(positions)


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
        output_dim: int = 1,
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
        super().__init__()
        self.num_features = 40 if num_features is None else num_features
        d_model = 64 if d_model is None else d_model
        dim_feedforward = 256 if dim_feedforward is None else dim_feedforward
        nhead = 8 if nhead is None else nhead
        num_layers = 2 if num_layers is None else num_layers

        # ``num_features`` 是每快照特征数（5 档 20 / 10 档 40）。
        self.embed = nn.Linear(self.num_features, d_model, bias=False)

        # 位置嵌入长度必须覆盖实际 history_length（曾硬编码 100）。
        self.embed_positions = SinusoidalPositionalEmbedding(history_length, d_model)

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
        self.regression_head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量 ``(N, 1, history_length, num_features)``。

        Returns:
            模型输出 ``(N, 1)``。

        Raises:
            ValueError: 特征维度与构造契约不一致。
        """
        # 输入维度契约（5 档数据 = 20 特征，10 档 = 40）。
        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"Transformer expects {self.num_features} features per snapshot, "
                f"got {x.shape[-1]}. 请核对 ExperimentConfig 的 features 特征列 "
                f"与 data.levels 契约。"
            )
        x = self.embed(x.squeeze(1))

        embed_pos = self.embed_positions(x.shape)

        # Transformer 编码器
        x = self.transformer_encoder(x + embed_pos)

        # 回归读出头（均值池化）
        x = torch.mean(x, dim=1)

        prediction = self.regression_head(x)
        return prediction
