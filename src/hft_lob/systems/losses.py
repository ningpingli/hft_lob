"""损失函数（需求文档 §20）：MSE / MAE / Huber；primary 推荐 Huber。"""

from __future__ import annotations

from typing import cast

import torch
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
        ValueError: name 不是 LOSS_NAMES 之一，或 huber_delta 不为正数时。
    """
    normalized_name = name.strip().lower()

    if normalized_name == "mse":
        return nn.MSELoss()
    if normalized_name == "mae":
        return nn.L1Loss()
    if normalized_name == "huber":
        if huber_delta <= 0:
            raise ValueError("huber_delta must be > 0")
        return nn.HuberLoss(delta=huber_delta)

    raise ValueError(f"unsupported loss name: {name!r}; expected one of {LOSS_NAMES}")


def masked_loss(
    loss_fn: nn.Module,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_valid: torch.Tensor,
) -> torch.Tensor:
    """在每个样本的有效 target 元素上计算损失。"""
    if predictions.shape != targets.shape or predictions.shape != target_valid.shape:
        raise ValueError("predictions, targets and target_valid must have the same shape")
    if target_valid.dtype is not torch.bool:
        raise TypeError("target_valid must have dtype torch.bool")
    valid = target_valid
    if not valid.any():
        return predictions.sum() * 0.0
    return cast(torch.Tensor, loss_fn(predictions[valid], targets[valid]))
