"""模型、DataLoader、预测 artifact 共用的数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class SampleMeta:
    ticker: str
    trade_date: str
    session_id: str
    anchor_timestamp: str
    mid_t: float
    future_mid: float
    bid1: float
    ask1: float
    spread: float


@dataclass(frozen=True)
class LOBBatch:
    features: torch.Tensor
    targets: torch.Tensor
    metadata: tuple[SampleMeta, ...]
    targets_by_horizon: dict[int, torch.Tensor] = field(default_factory=dict)
