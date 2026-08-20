"""CNN2：五层卷积神经网络 LOB 回归模型。"""

from __future__ import annotations

import torch
from torch import nn


class CNN2(nn.Module):
    """CNN2：五层卷积（含 BatchNorm/PReLU）+ 全连接回归模型。"""

    def __init__(
        self,
        num_features: int = 20,
        history_length: int = 100,
        temp: int | None = None,
    ) -> None:
        """初始化 CNN2。

        Args:
            num_features: 每快照特征数。
            history_length: 历史窗口长度。
            temp: 卷积池化堆叠后的时间长度；None 时由 history_length 推导。
        """
        super().__init__()
        self.num_features = num_features

        # ``temp`` 是卷积堆叠之后的时间长度。历史上硬编码 249（仅对
        # history_length=100 正确），现由实际 history_length 推导：
        #   conv1 (2d, k=10) -> H-9，宽度坍缩为 3；conv2 (k=10)、conv3 (k=8)、
        #   conv4 (k=6)、conv5 (k=4) 各收缩 k-1，故 temp = 3*(H-9)-24。
        if temp is None:
            temp = 3 * (history_length - 9) - 24

        # 卷积 1
        # kernel 宽度覆盖全部 ``num_features``（10 档 40 / 5 档 20）加 2 列 padding。
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(10, num_features + 2),
            padding=(0, 2),
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.prelu1 = nn.PReLU()

        # 卷积 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=(10,))
        self.bn2 = nn.BatchNorm1d(16)
        self.prelu2 = nn.PReLU()

        # 卷积 3
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(8,))
        self.bn3 = nn.BatchNorm1d(32)
        self.prelu3 = nn.PReLU()

        # 卷积 4
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(6,))
        self.bn4 = nn.BatchNorm1d(32)
        self.prelu4 = nn.PReLU()

        # 卷积 5
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(4,))
        self.bn5 = nn.BatchNorm1d(32)
        self.prelu5 = nn.PReLU()

        # 全连接 1
        self.fc1 = nn.Linear(temp * 32, 32)
        self.prelu6 = nn.PReLU()

        # 全连接 2（回归读出头）
        self.regression_head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量 ``(N, 1, history_length, num_features)``。

        Returns:
            无界回归预测，形状为 ``(N, 1)``。

        Raises:
            ValueError: 特征维度与构造契约不一致。
        """
        # 输入维度契约：conv1 kernel 宽度为 num_features + 2，宽度不匹配会在
        # 运行时崩溃，此处提前报错。
        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"CNN2 expects {self.num_features} features per snapshot, got "
                f"{x.shape[-1]}. 请核对 ExperimentConfig 的 features 特征列与 "
                f"data.levels 契约。"
            )
        # 卷积 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)
        out = out.reshape(out.shape[0], out.shape[1], -1)

        # 卷积 2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu2(out)

        # 卷积 3
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.prelu3(out)

        # 卷积 4
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.prelu4(out)

        # 卷积 5
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.prelu5(out)

        # 展平
        out = out.view(out.size(0), -1)

        # 全连接 1
        out = self.fc1(out)
        out = self.prelu6(out)

        # 回归读出头
        out = self.regression_head(out)

        return out
