"""DLA：双重注意力（时间与特征维 soft attention）GRU 回归模型。"""

from __future__ import annotations

import torch
from torch import nn


class DLA(nn.Module):
    """DLA：双重注意力加权 + 双层 GRU 的回归模型。"""

    def __init__(
        self,
        num_features: int | None = None,
        num_snapshots: int = 100,
        hidden_size: int = 128,
        output_dim: int = 1,
    ) -> None:
        """初始化 DLA。

        Args:
            num_features: 每快照特征数（None 时为 40）。
            num_snapshots: 每样本的快照数。
            hidden_size: GRU 隐藏层大小。
        """
        super().__init__()
        num_features = 40 if num_features is None else num_features
        self.num_features = num_features
        self.num_snapshots = num_snapshots

        self.W1 = nn.Linear(num_features, num_features, bias=False)

        self.softmax = nn.Softmax(dim=1)

        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
        )

        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W3 = nn.Linear(num_snapshots * hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量（形状 ``(N, 1, num_snapshots, num_features)``）。

        Returns:
            模型输出 ``(N, 1)``。

        Raises:
            ValueError: 输入维度与构造契约不一致。
        """
        if x.ndim != 3:
            raise ValueError(f"DLA expects [B, T, F], got shape {tuple(x.shape)}")
        # GRU 输入维度与展平后的 W3 宽度依赖快照数 / 特征数。
        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"DLA expects {self.num_features} features per snapshot, got "
                f"{x.shape[-1]}. 请核对 ExperimentConfig 的 features 特征列契约。"
            )
        if x.shape[1] != self.num_snapshots:
            raise ValueError(
                f"DLA expects {self.num_snapshots} snapshots per sample, got "
                f"{x.shape[1]}. 请核对 ExperimentConfig 的 window.history_snapshots 契约。"
            )
        # x.shape = [batch_size, num_snapshots, num_features]
        X_tilde = self.W1(x)

        alpha = self.softmax(X_tilde)
        # alpha.shape = [batch_size, num_snapshots, num_features]

        alpha = torch.mean(alpha, dim=2)
        # alpha.shape = [batch_size, num_snapshots]

        x_tilde = torch.einsum("ij,ijk->ijk", alpha, x)
        # x_tilde.shape = [batch_size, num_snapshots, num_features]

        H, _ = self.gru(x_tilde)
        # H.shape = [batch_size, num_snapshots, hidden_size]

        H_tilde = self.W2(H)
        # H_tilde.shape = [batch_size, num_snapshots, hidden_size]

        beta = self.softmax(H_tilde)
        # beta.shape = [batch_size, num_snapshots, hidden_size]

        beta = torch.mean(beta, dim=2)
        # beta.shape = [batch_size, num_snapshots]

        h_tilde = torch.einsum("ij,ijk->ijk", beta, H)
        # h_tilde.shape = [batch_size, num_snapshots, hidden_size]

        h_tilde = torch.flatten(h_tilde, start_dim=1)
        # h_tilde.shape = [batch_size, hidden_size * num_snapshots]

        prediction = self.W3(h_tilde)
        # prediction.shape = [batch_size, 1]（回归读出头）

        return prediction
