"""CNN1：卷积神经网络 LOB 回归模型。"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class CNN1(nn.Module):
    """CNN1：双层卷积 + 池化 + 全连接回归模型。"""

    def __init__(
        self,
        num_features: int = 20,
        history_length: int = 100,
        temp: int | None = None,
    ) -> None:
        """初始化 CNN1。

        Args:
            num_features: 每快照特征数。
            history_length: 历史窗口长度。
            temp: 卷积池化堆叠后的时间长度；None 时由 history_length 推导。
        """
        super().__init__()
        self.num_features = num_features

        # ``temp`` 是卷积/池化堆叠之后的时间长度。历史上硬编码 26（仅对
        # history_length=100 正确），现由实际 history_length 推导：
        #   conv2 (k=4) -> H-3；pool1 (k=2) -> (H-3)//2；conv3/conv4 (k=3,
        #   padding=2) 各 +2；pool2 (k=2) -> //2。
        if temp is None:
            temp = (((history_length - 4 + 1) // 2) + 2 + 2) // 2

        # 卷积 1（kernel 宽度覆盖全部 num_features 列）
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(4, num_features),
            padding=(3, 0),
            dilation=(2, 1),
        )
        self.relu1 = nn.LeakyReLU()

        # 卷积 2
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(4,))
        self.relu2 = nn.LeakyReLU()

        # 最大池化 1
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)

        # 卷积 3
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=(3,), padding=2)
        self.relu3 = nn.LeakyReLU()

        # 卷积 4
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=(3,), padding=2)
        self.relu4 = nn.LeakyReLU()

        # 最大池化 2
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        # 全连接 1
        self.fc1 = nn.Linear(temp * 64, 64)
        self.relu5 = nn.LeakyReLU()

        # 全连接 2（回归读出头）
        self.regression_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 统一时序输入 ``[B,T,F]``；卷积通道维由本模型内部增加。

        Returns:
            无界回归预测，形状为 ``(N, 1)``。

        Raises:
            ValueError: 特征维度与构造契约不一致。
        """
        if x.ndim != 3:
            raise ValueError(
                f"CNN1 expects [B, T, F], got shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"CNN1 expects {self.num_features} features per snapshot, got "
                f"{x.shape[-1]}. 请核对 ExperimentConfig 的 features 特征列与 "
                f"data.levels 契约。"
            )
        # 统一契约不暴露卷积通道维；仅在模型内部转换为 [B, 1, T, F]。
        x = x.unsqueeze(1)

        # 卷积 1
        out = self.conv1(x)
        out = self.relu1(out)
        out = out.reshape(out.shape[0], out.shape[1], -1)

        # 卷积 2
        out = self.conv2(out)
        out = self.relu2(out)

        # 最大池化 1
        out = self.maxpool1(out)

        # 卷积 3
        out = self.conv3(out)
        out = self.relu3(out)

        # 卷积 4
        out = self.conv4(out)
        out = self.relu4(out)

        # 最大池化 2
        out = self.maxpool2(out)

        # 展平
        out = out.view(out.size(0), -1)

        # 全连接 1
        out = self.fc1(out)
        out = self.relu5(out)

        # 回归读出头
        out = self.regression_head(out)

        return cast(torch.Tensor, out)
