"""回归任务的可配置损失函数（mse / mae / huber）。"""

from __future__ import annotations

from torch import nn

#: 支持的损失名（mse 为默认，保持框架历史行为）。
LOSS_NAMES = ("mse", "mae", "huber")


def build_loss(
    name: str = "mse",
    *,
    huber_delta: float = 1.0,
) -> nn.Module:
    """构建由 name 选定的回归损失模块。

    Args:
        name: ``"mse"``（默认）、``"mae"``、``"huber"`` 三者之一（不区分大小写）。
        huber_delta: Huber 损失的阈值参数（仅 huber 使用）。

    Returns:
        可按 ``loss(preds, targets) -> 标量 Tensor`` 调用的损失模块。

    Raises:
        ValueError: name 不是 LOSS_NAMES 之一时。
    """
    raise NotImplementedError("build_loss not implemented")
