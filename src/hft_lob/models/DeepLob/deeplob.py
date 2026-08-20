"""DeepLOB：双流 CNN + Inception + LSTM 的 LOB 回归模型。"""

from __future__ import annotations

import torch
from torch import nn


class DeepLOB(nn.Module):
    """DeepLOB：卷积块 + Inception 模块 + LSTM 回归模型。"""

    def __init__(
        self,
        num_features: int | None = None,
        levels: int | None = None,
    ) -> None:
        """初始化 DeepLOB。

        Args:
            num_features: 每快照特征数（None 时为 40）。
            levels: 盘口档位数（None 时为 10）。
        """
        super().__init__()
        self.num_features = 40 if num_features is None else num_features
        self.levels = 10 if levels is None else levels

        # 卷积块。
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # conv3 kernel 宽度跨全部特征轴：5 档 -> 5，10 档 -> 10（来自配置 levels）。
        conv3_kernel_size = self.levels

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, conv3_kernel_size)
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # Inception 模块。
        self.inp1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(3, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(5, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # LSTM 层（Inception 输出通道 64*3 作为输入维）。
        self.lstm = nn.LSTM(
            input_size=192, hidden_size=64, num_layers=1, batch_first=True
        )
        self.fc1 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量 ``(N, 1, history_length, num_features)``。

        Returns:
            模型输出 ``(N, 1)``。

        Raises:
            ValueError: 特征维度与构造契约不一致。
        """
        # 输入维度契约：conv3 宽度 kernel 必须适配特征轴（5 档数据 = 20 特征，
        # 10 档数据 = 40 特征）。
        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"DeepLOB expects {self.num_features} features per snapshot, got "
                f"{x.shape[-1]}. 请核对 ExperimentConfig 的 features 特征列与 "
                f"data.levels 契约。"
            )
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))

        x, _ = self.lstm(x)
        x = x[:, -1, :]
        logits = self.fc1(x)

        return logits
