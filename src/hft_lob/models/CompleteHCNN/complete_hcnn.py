"""CompleteHCNN：基于完整同调结构（四面体/三角形/边）的 HCNN 回归模型。"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class Complete_HCNN(nn.Module):
    """Complete_HCNN：对同调结构分别卷积后经 LSTM 读出的回归模型。"""

    def __init__(
        self,
        homological_structures: dict[str, Any],
        num_features: int = 20,
    ) -> None:
        """初始化 Complete_HCNN。

        Args:
            homological_structures: 同调结构字典（tetrahedra / triangles / edges）。
            num_features: 每快照特征数。
        """
        super().__init__()
        self.homological_structures = homological_structures
        self.tetrahedra = self.homological_structures["tetrahedra"]
        self.triangles = self.homological_structures["triangles"]
        self.edges = self.homological_structures["edges"]
        self.num_features = num_features

        # 同调引用的所有特征索引必须落在输入特征轴内（A 股五档为 20）；若上游
        # 40 列同调遇上 20 特征数据会 IndexError，此处提前报错。保存的结构为
        # 扁平索引列表（退化日可能为空），同时兼容扁平/嵌套布局。
        def _max_index(structures: list) -> int:
            flat: list[int] = []
            for item in structures:
                if isinstance(item, (list, tuple)):
                    flat.extend(item)
                else:
                    flat.append(item)
            return max(flat) if flat else -1

        self.max_feature_index = max(
            _max_index(self.tetrahedra),
            _max_index(self.triangles),
            _max_index(self.edges),
        )

        # ------------ #

        self.conv1_tetrahedra = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)
            ),
            nn.ReLU(),
        )

        self.conv1_triangles = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)
            ),
            nn.ReLU(),
        )

        self.conv1_edges = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)
            ),
            nn.ReLU(),
        )

        # ------------ #

        self.conv2_tetrahedra = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, 4), stride=(1, 4)
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.ReLU(),
        )

        self.conv2_triangles = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 3)
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.ReLU(),
        )

        self.conv2_edges = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.ReLU(),
        )

        # ------------ #

        self.conv3_tetrahedra = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, int(len(self.tetrahedra) / 8)),
            ),
            nn.Dropout(0.35),
            nn.ReLU(),
        )

        self.conv3_triangles = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, int(len(self.triangles) / 6)),
            ),
            nn.Dropout(0.35),
            nn.ReLU(),
        )

        self.conv3_edges = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, int(len(self.edges) / 4)),
            ),
            nn.Dropout(0.35),
            nn.ReLU(),
        )

        # ------------ #

        self.lstm = nn.LSTM(
            input_size=96, hidden_size=32, num_layers=1, batch_first=True
        )
        self.regression_head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 统一时序输入 ``[B, T, F]``；卷积通道维由本模型内部增加。

        Returns:
            模型输出 ``(N, 1)``。

        Raises:
            ValueError: 同调特征索引超出输入特征轴。
        """
        if x.ndim != 3:
            raise ValueError(f"Complete_HCNN expects [B, T, F], got shape {tuple(x.shape)}")
        if x.shape[-1] <= self.max_feature_index:
            raise ValueError(
                f"Complete_HCNN: homological feature indices reach "
                f"{self.max_feature_index} but the input has only "
                f"{x.shape[-1]} features. 同调在另一特征布局上计算，请核对 "
                f"complete_homological_utils 与 ExperimentConfig 特征列契约。"
            )
        x = x.unsqueeze(1)
        x_tetrahedra = x[:, :, :, self.tetrahedra]
        x_triangles = x[:, :, :, self.triangles]
        x_edges = x[:, :, :, self.edges]

        x_tetrahedra = self.conv1_tetrahedra(x_tetrahedra)
        x_triangles = self.conv1_triangles(x_triangles)
        x_edges = self.conv1_edges(x_edges)

        x_tetrahedra = self.conv2_tetrahedra(x_tetrahedra)
        x_triangles = self.conv2_triangles(x_triangles)
        x_edges = self.conv2_edges(x_edges)

        x_tetrahedra = self.conv3_tetrahedra(x_tetrahedra)
        x_triangles = self.conv3_triangles(x_triangles)
        x_edges = self.conv3_edges(x_edges)

        x = torch.cat((x_tetrahedra, x_triangles, x_edges), dim=1)

        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))

        x, _ = self.lstm(x)
        x = x[:, -1, :]
        prediction = self.regression_head(x)

        return prediction
