"""损失函数（需求文档 §20）：MSE / MAE / Huber；primary 推荐 Huber。"""

from __future__ import annotations

from torch import nn

#: 支持的损失名（huber 为 MVP primary，§20）。
LOSS_NAMES: tuple[str, ...] = ("mse", "mae", "huber")


def build_loss(name: str = "huber", *, huber_delta: float = 1.0) -> nn.Module:
    """构建由 name 选定的回归损失模块（大小写不敏感）。

    Args:
        name: ``"mse"`` / ``"mae"`` / ``"huber"``（默认 huber，§20）。
        huber_delta: Huber 损失的阈值参数（仅 huber 使用）。

    Returns:
        可按 ``loss(preds, targets) -> 标量 Tensor`` 调用的损失模块。

    Raises:
        ValueError: name 不是 LOSS_NAMES 之一时。
    """
    raise NotImplementedError("build_loss not implemented")
