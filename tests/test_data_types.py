"""底层数据类型契约测试。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from hft_lob.data_types import LOBBatch, SampleMeta


def _sample_meta() -> SampleMeta:
    return SampleMeta(
        ticker="TEST",
        trade_date="2026-01-05",
        session_id="morning",
        anchor_timestamp="2026-01-05T09:30:00",
        mid_t=10.0,
        bid1=9.99,
        ask1=10.01,
        spread=0.02,
    )


def test_sample_meta_preserves_market_identity_and_quote_fields() -> None:
    metadata = _sample_meta()

    assert metadata.ticker == "TEST"
    assert metadata.trade_date == "2026-01-05"
    assert metadata.session_id == "morning"
    assert metadata.anchor_timestamp == "2026-01-05T09:30:00"
    assert metadata.mid_t == 10.0
    assert metadata.bid1 == 9.99
    assert metadata.ask1 == 10.01
    assert metadata.spread == 0.02


def test_lob_batch_preserves_tensors_and_metadata() -> None:
    features = torch.randn(2, 8, 20)
    targets = torch.randn(2, 2)
    metadata = (_sample_meta(), _sample_meta())
    batch = LOBBatch(features=features, targets=targets, metadata=metadata)

    assert batch.features is features
    assert batch.targets is targets
    assert batch.metadata == metadata
    assert batch.features.shape == (2, 8, 20)
    assert batch.targets.shape == (2, 2)


def test_shared_data_types_are_immutable() -> None:
    metadata = _sample_meta()
    batch = LOBBatch(
        features=torch.zeros(1, 1, 1),
        targets=torch.zeros(1, 1),
        metadata=(metadata,),
    )

    with pytest.raises(FrozenInstanceError):
        metadata.spread = 0.03  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        batch.metadata = ()  # type: ignore[misc]
