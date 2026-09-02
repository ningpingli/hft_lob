"""LobTransformer：卷积特征提取 + Transformer 编码器的 LOB 回归模型。"""

from __future__ import annotations

import torch
from torch import nn


class LobTransformer(nn.Module):
    """LobTransformer：CNN 卷积 + Inception + Transformer 编码器回归模型。"""

    def __init__(
        self,
        num_features: int | None = None,
        levels: int | None = None,
        hidden: int | None = None,
        d_model: int | None = None,
        nhead: int | None = None,
        num_layers: int | None = None,
        output_dim: int = 1,
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
        super().__init__()
        self.num_features = 40 if num_features is None else num_features
        self.levels = 10 if levels is None else levels
        hidden = 32 if hidden is None else hidden
        nhead = 8 if nhead is None else nhead
        num_layers = 2 if num_layers is None else num_layers
        # d_model 遵循架构关系 hidden*2*3，除非配置显式声明。
        d_model = hidden * 2 * 3 if d_model is None else d_model

        # 卷积块。
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=hidden, kernel_size=(1, 2), stride=(1, 2)
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden, out_channels=hidden, kernel_size=(1, 2), stride=(1, 2)
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden),
        )

        # conv3 kernel 宽度跨全部特征轴：5 档 -> 5，10 档 -> 10（来自配置 levels）。
        conv3_kernel_size = self.levels

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden, out_channels=hidden, kernel_size=(1, conv3_kernel_size)
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden),
        )

        # Inception 模块。
        self.inp1 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden, out_channels=hidden * 2, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden * 2),
            nn.Conv2d(
                in_channels=hidden * 2, out_channels=hidden * 2, kernel_size=(3, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden * 2),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden, out_channels=hidden * 2, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden * 2),
            nn.Conv2d(
                in_channels=hidden * 2, out_channels=hidden * 2, kernel_size=(5, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden * 2),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(
                in_channels=hidden, out_channels=hidden * 2, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden * 2),
        )

        # Transformer 编码器。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.regression_head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 统一时序输入 ``[B, T, F]``；卷积通道维由本模型内部增加。

        Returns:
            模型输出 ``(N, 1)``。

        Raises:
            ValueError: 特征维度与构造契约不一致，或卷积后特征轴未坍缩为 1。
        """
        if x.ndim != 3:
            raise ValueError(f"LobTransformer expects [B, T, F], got shape {tuple(x.shape)}")
        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"LobTransformer expects {self.num_features} features per "
                f"snapshot, got {x.shape[-1]}. 请核对 ExperimentConfig 的 "
                f"features 特征列与 data.levels 契约。"
            )
        x = self.conv1(x.unsqueeze(1))
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        # (B, d_model, H', W') -> (B, H', d_model, W') -> (B*W', H', d_model)。
        # conv3 kernel 坍缩特征轴后 W' 恒为 1；断言以捕获未来几何变化（否则
        # reshape 会静默合并 batch 维）。
        if x.shape[-1] != 1:
            raise ValueError(
                f"LobTransformer: unexpected feature width {x.shape[-1]} after "
                f"convolutions; expected 1. 请核对 ExperimentConfig 的 "
                f"data.levels 与特征列契约。"
            )
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))

        x = self.transformer_encoder(x)
        # 回归读出头（均值池化）
        x = torch.mean(x, dim=1)

        prediction = self.regression_head(x)
        return prediction
