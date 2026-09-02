from __future__ import annotations

import math
from datetime import datetime, timedelta

import polars as pl
import pytest

from hft_lob.configs.experiment import TargetConfig
from hft_lob.data_pipeline.clean import SessionSegment
from hft_lob.data_pipeline.labels import LabelTransformer, label_columns, target_column


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


def test_transform_generates_selected_labels_without_validity_columns() -> None:
    transformer = LabelTransformer(
        TargetConfig(type="log_mid_return", labels=[60, 120], tolerance_seconds=3)
    )
    result = transformer.transform(_segment()).frame

    assert set(result.columns[-2:]) == {"Target_60s_log", "Target_120s_log"}
    assert result.get_column("Target_60s_log").to_list()[2] is None
    assert result.get_column("Target_120s_log").to_list()[1:] == [None, None]
    assert math.isclose(result["Target_60s_log"][0], math.log(1.1))
    assert result.get_column("Target_120s_log")[0] == pytest.approx(math.log(1.21))
    assert "future_mid" not in result.columns
    assert all(not name.startswith("_") for name in result.columns)


def test_transform_does_not_match_beyond_tolerance() -> None:
    transformer = LabelTransformer(
        TargetConfig(type="simple_mid_return", labels=[60], tolerance_seconds=1)
    )
    result = transformer.transform(_segment()).frame

    assert result.get_column("Target_60s_simple").to_list() == [None, None, None]
    assert label_columns(transformer.config) == ("Target_60s_simple",)


def test_label_columns_follow_configured_type_and_label_order() -> None:
    config = TargetConfig(type="simple_mid_return", labels=[120, 60])
    assert (config.labels, target_column(config, 60)) == ([120, 60], "Target_60s_simple")
    assert label_columns(config) == ("Target_120s_simple", "Target_60s_simple")


def test_transform_rejects_unsorted_timestamps() -> None:
    transformer = LabelTransformer(TargetConfig(labels=[60]))
    segment = _segment()

    with pytest.raises(ValueError, match="timestamps must be sorted"):
        transformer.transform(
            SessionSegment(segment.trade_date, segment.session_id, segment.frame.reverse())
        )


def test_transform_rejects_segment_metadata_mismatch() -> None:
    transformer = LabelTransformer(TargetConfig())

    with pytest.raises(ValueError, match="session_id"):
        transformer.transform(SessionSegment("2026-01-05", "PM", _segment().frame))


def test_target_config_rejects_tolerance_that_can_reach_label() -> None:
    with pytest.raises(ValueError, match="smaller than label"):
        LabelTransformer(TargetConfig(labels=[60], tolerance_seconds=60))
