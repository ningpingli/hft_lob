"""AxialLOB：门控轴向注意力 LOB 模型。"""

from __future__ import annotations

import math

import torch
from torch import nn


def _conv1d1x1(in_channels: int, out_channels: int) -> nn.Sequential:
    """1x1 卷积 + BatchNorm 辅助构造器。"""
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm1d(out_channels),
    )


class GatedAxialAttention(nn.Module):
    """多头上轴自注意力（沿 LOB 帧的高度或宽度方向），带门控与相对位置嵌入。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        dim: int,
        flag: bool,
    ) -> None:
        """初始化门控轴向注意力层。

        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
            heads: 注意力头数。
            dim: 轴向维度长度。
            flag: 为 True 时沿宽度方向计算注意力，否则沿高度方向。
        """
        assert (in_channels % heads == 0) and (out_channels % heads == 0)
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dim_head_v = out_channels // heads
        self.flag = flag  # flag 为 True 时沿宽度方向计算注意力
        self.dim = dim
        self.dim_head_qk = self.dim_head_v // 2
        self.qkv_channels = self.dim_head_v + self.dim_head_qk * 2

        # 多头自注意力
        self.to_qkv = _conv1d1x1(in_channels, self.heads * self.qkv_channels)
        self.bn_qkv = nn.BatchNorm1d(self.heads * self.qkv_channels)
        self.bn_similarity = nn.BatchNorm2d(heads * 3)
        self.bn_output = nn.BatchNorm1d(self.heads * self.qkv_channels)

        # 门控机制
        self.f_qr = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.f_kr = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.f_sve = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.f_sv = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        # 相对位置嵌入
        self.relative = nn.Parameter(
            torch.randn(self.dim_head_v * 2, dim * 2 - 1), requires_grad=True
        )
        query_index = torch.arange(dim).unsqueeze(0)
        key_index = torch.arange(dim).unsqueeze(1)
        relative_index = key_index - query_index + dim - 1
        self.register_buffer("flatten_index", relative_index.view(-1))

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量。

        Returns:
            轴向注意力输出张量。
        """
        if self.flag:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # (N, W, C, H)
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # 变换
        x = self.to_qkv(x)

        qkv = self.bn_qkv(x)
        q, k, v = torch.split(
            qkv.reshape(N * W, self.heads, self.dim_head_v * 2, H),
            [self.dim_head_v // 2, self.dim_head_v // 2, self.dim_head_v],
            dim=2,
        )

        # 相对位置嵌入
        all_embeddings = torch.index_select(
            self.relative, 1, self.flatten_index
        ).view(self.dim_head_v * 2, self.dim, self.dim)
        q_embedding, k_embedding, v_embedding = torch.split(
            all_embeddings,
            [self.dim_head_qk, self.dim_head_qk, self.dim_head_v],
            dim=0,
        )
        qr = torch.einsum("bgci,cij->bgij", q, q_embedding)
        kr = torch.einsum("bgci,cij->bgij", k, k_embedding).transpose(2, 3)
        qk = torch.einsum("bgci, bgcj->bgij", q, k)

        # 乘以门控因子
        qr = torch.mul(qr, self.f_qr)
        kr = torch.mul(kr, self.f_kr)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = (
            self.bn_similarity(stacked_similarity)
            .view(N * W, 3, self.heads, H, H)
            .sum(dim=1)
        )
        similarity = torch.softmax(stacked_similarity, dim=3)
        sv = torch.einsum("bgij,bgcj->bgci", similarity, v)
        sve = torch.einsum("bgij,cij->bgci", similarity, v_embedding)

        # 乘以门控因子
        sv = torch.mul(sv, self.f_sv)
        sve = torch.mul(sve, self.f_sve)

        stacked_output = torch.cat([sv, sve], dim=-1).view(
            N * W, self.out_channels * 2, H
        )
        output = (
            self.bn_output(stacked_output)
            .view(N, W, self.out_channels, 2, H)
            .sum(dim=-2)
        )

        if self.flag:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        return output

    def reset_parameters(self) -> None:
        """重置相对位置嵌入参数。"""
        nn.init.normal_(self.relative, 0.0, math.sqrt(1.0 / self.dim_head_v))


class AxialLOB(nn.Module):
    """AxialLOB：CNN 卷积 + 门控轴向注意力 + 残差 + 池化回归模型。"""

    def __init__(
        self,
        W: int = 40,
        H: int = 100,
        c_in: int = 32,
        c_out: int = 32,
        c_final: int = 4,
        n_heads: int = 4,
        pool_kernel: tuple[int, int] = (1, 4),
        pool_stride: tuple[int, int] = (1, 4),
    ) -> None:
        """初始化 AxialLOB。

        Args:
            W: 输入帧宽度（特征数，40 或 20）。
            H: 输入帧高度（时间快照数）。
            c_in: CNN 输入通道数。
            c_out: 轴向层输出通道数。
            c_final: 最终输出通道数。
            n_heads: 注意力头数。
            pool_kernel: 平均池化核。
            pool_stride: 平均池化步长。
        """
        super().__init__()
        self.W = W
        self.H = H

        # CNN_in 的通道输出即轴向层的通道输入
        self.c_in = c_in
        self.c_out = c_out
        self.c_final = c_final

        self.CNN_in = nn.Conv2d(in_channels=1, out_channels=c_in, kernel_size=1)
        self.CNN_out = nn.Conv2d(in_channels=c_out, out_channels=c_final, kernel_size=1)
        self.CNN_res2 = nn.Conv2d(in_channels=c_out, out_channels=c_final, kernel_size=1)
        self.CNN_res1 = nn.Conv2d(in_channels=1, out_channels=c_out, kernel_size=1)

        self.norm = nn.BatchNorm2d(c_in)
        self.res_norm2 = nn.BatchNorm2d(c_final)
        self.res_norm1 = nn.BatchNorm2d(c_out)
        self.norm2 = nn.BatchNorm2d(c_final)

        self.axial_height_1 = GatedAxialAttention(c_out, c_out, n_heads, H, flag=False)
        self.axial_width_1 = GatedAxialAttention(c_out, c_out, n_heads, W, flag=True)

        self.axial_height_2 = GatedAxialAttention(c_out, c_out, n_heads, H, flag=False)
        self.axial_width_2 = GatedAxialAttention(c_out, c_out, n_heads, W, flag=True)

        self.activation = nn.ReLU()
        # 展平宽度 = 池化后的 W（pool_stride[1] 作用于特征轴）；W=40 -> 4000，
        # W=20 -> 2000。
        self.linear = nn.Linear(c_final * H * (W // pool_stride[1]), 1)
        self.pooling = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量（形状 ``(N, 1, H, W)``）。

        Returns:
            模型输出 ``(N, 1)``。

        Raises:
            ValueError: 输入帧尺寸与构造契约不一致。
        """
        # 输入维度契约：轴向注意力的 dim 与最终 Linear 宽度在构造时由 (H, W) 固定。
        if x.shape[-1] != self.W or x.shape[-2] != self.H:
            raise ValueError(
                f"AxialLOB expects samples of shape (1, {self.H}, {self.W}), "
                f"got {tuple(x.shape[1:])}. 请核对 ExperimentConfig 的 "
                f"window.history_snapshots 与 features 特征列契约。"
            )
        # 注意力前的首次卷积
        y = self.CNN_in(x)
        y = self.norm(y)
        y = self.activation(y)

        # 门控多头轴向注意力
        y = self.axial_width_1(y)
        y = self.axial_height_1(y)

        # 下分支
        x = self.CNN_res1(x)
        x = self.res_norm1(x)
        x = self.activation(x)

        # 第一次残差连接
        y = y + x
        # 注：detach() 阻止第二个轴向块的梯度流入第一个轴向块，与上游
        # LOBFrame 实现保持一致（若需双块梯度流动请对照 AxialLOB 论文验证）。
        z = y.detach().clone()

        # 第二个轴向层
        y = self.axial_width_2(y)
        y = self.axial_height_2(y)

        # 第二次卷积
        y = self.CNN_out(y)
        y = self.res_norm2(y)
        y = self.activation(y)

        # 下分支
        z = self.CNN_res2(z)
        z = self.norm2(z)
        z = self.activation(z)

        # 第二次残差连接
        y = y + z

        # 最终部分
        y = self.pooling(y)
        y = torch.flatten(y, 1)
        y = self.linear(y)

        return y
