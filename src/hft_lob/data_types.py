"""跨数据集、模型、baseline 与报告层共享的底层数据类型。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SampleMeta:
    """单个样本的市场时间与盘口元数据。"""

    ticker: str
    trade_date: str
    session_id: str
    anchor_timestamp: str
    mid_t: float
    bid1: float
    ask1: float
    spread: float


@dataclass(frozen=True)
class LOBBatch:
    """DataLoader 交给模型或 baseline 的一个批次。"""

    features: torch.Tensor
    targets: torch.Tensor
    metadata: tuple[SampleMeta, ...]
