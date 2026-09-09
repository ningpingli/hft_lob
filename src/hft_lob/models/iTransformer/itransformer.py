"""iTransformer：倒置 Transformer——把时间序列视为特征维进行嵌入的回归模型。"""

from __future__ import annotations

import torch
from torch import nn


class ITransformer(nn.Module):
    """ITransformer：沿特征维嵌入历史序列的倒置 Transformer 回归模型。"""

    #: 输入特征截断界(默认 ±1000)。causal_rolling 归一化在盘口近常数段(rolling std→0)
    #: 会产出 1e3-1e6 级有限特征,经 embed 线性放大后 attention logits 在 fp32 softmax
    #: (exp 上限≈88)溢出 → epoch0 即 NaN(688009、688981 fold50-59 同因)。
    #: 正常特征 |f| 的 p99.99 ≤ ~31;±1000 仅截断 >1e3 病态尾,对健康股近似 no-op
    #: (判别实验:688008 上 clamp 100/1000/无 三档差异在轨迹噪声带内,见 EXPERIMENT_REPORT)。
    #: 可经 model.feature_clip 配置覆盖(≤0 无效,None 用本默认)。
    FEATURE_CLIP: float = 1000.0

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
        feature_clip: float | None = None,
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
        if feature_clip is not None:
            self.FEATURE_CLIP = float(feature_clip)
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
        self.regression_head = nn.Linear(d_model, output_dim)

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
        x = torch.clamp(x, min=-self.FEATURE_CLIP, max=self.FEATURE_CLIP)
        # 转置：沿特征维（每条特征一个 token）嵌入历史序列。
        x = x.permute(0, 2, 1)
        x = self.embed(x)

        # Transformer 编码器
        x = self.transformer_encoder(x)

        # 回归读出头（均值池化）
        x = torch.mean(x, dim=1)

        prediction = self.regression_head(x)
        return prediction
