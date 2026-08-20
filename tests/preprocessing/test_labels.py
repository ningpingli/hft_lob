from __future__ import annotations

import math
from datetime import datetime, timedelta

import polars as pl
import pytest

from hft_lob.configs.experiment import TargetConfig
from hft_lob.preprocessing.clean import SessionSegment
from hft_lob.preprocessing.labels import LabelTransformer, label_column


def _segment() -> SessionSegment:
    start = datetime(2026, 1, 5, 9, 30)
    frame = pl.DataFrame(
        {
            "trade_date": ["2026-01-05"] * 3,
            "session_id": ["AM"] * 3,
            "timestamp": [start, start + timedelta(seconds=62), start + timedelta(seconds=120)],
            "mid_price": [100.0, 110.0, 121.0],
            "book_valid": [True, True, True],
        },
        schema_overrides={"timestamp": pl.Datetime("us")},
    )
    return SessionSegment("2026-01-05", "AM", frame)


def test_transform_matches_nearest_future_with_bounded_tolerance() -> None:
    transformer = LabelTransformer(
        TargetConfig(type="log_mid_return", horizon_seconds=60, tolerance_seconds=3)
    )

    result = transformer.transform(_segment()).frame

    assert result.get_column("future_mid").to_list() == [110.0, 121.0, None]
    assert result.get_column("target_valid").to_list() == [True, True, False]
    assert math.isclose(result["Target_60s_log"][0], math.log(1.1))
    assert math.isclose(result["Target_60s_simple"][0], 0.1)
    assert result["Target_60s_log"][2] is None


def test_transform_does_not_match_beyond_tolerance() -> None:
    transformer = LabelTransformer(
        TargetConfig(type="simple_mid_return", horizon_seconds=60, tolerance_seconds=1)
    )

    result = transformer.transform(_segment()).frame

    assert result.get_column("future_mid").to_list() == [None, None, None]
    assert result.get_column("target_valid").to_list() == [False, False, False]
    assert label_column(transformer.config) == "Target_60s_simple"


def test_transform_rejects_segment_metadata_mismatch() -> None:
    transformer = LabelTransformer(TargetConfig())

    with pytest.raises(ValueError, match="session_id"):
        transformer.transform(SessionSegment("2026-01-05", "PM", _segment().frame))


def test_target_config_rejects_tolerance_that_can_reach_anchor() -> None:
    with pytest.raises(ValueError, match="smaller than horizon"):
        LabelTransformer(TargetConfig(horizon_seconds=60, tolerance_seconds=60))
