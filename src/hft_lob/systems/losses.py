"""损失函数（需求文档 §20）：MSE / MAE / Huber；primary 推荐 Huber。"""

from __future__ import annotations

from torch import nn

#: 支持的损失名（huber 为 MVP primary，§20）。
LOSS_NAMES: tuple[str, ...] = ("mse", "mae", "huber")


def build_loss(name: str = "huber", *, huber_delta: float = 1.0) -> nn.Module:
    """构建多标签回归损失，按所有标签元素求均值。"""
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
