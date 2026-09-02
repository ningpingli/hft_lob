from __future__ import annotations

import json
from datetime import datetime, timedelta

import polars as pl
import pytest

from hft_lob.data_pipeline.processor import CausalRollingStandardizer


def _frame(values: list[float], *, valid: list[bool] | None = None) -> pl.DataFrame:
    start = datetime(2026, 1, 5, 9, 30)
    return pl.DataFrame(
        {
            "trade_date": ["2026-01-05"] * len(values),
            "session_id": ["AM"] * len(values),
            "timestamp": [start + timedelta(seconds=3 * i) for i in range(len(values))],
            "feature_valid": valid or [True] * len(values),
            "price": values,
        },
        schema_overrides={"timestamp": pl.Datetime("us")},
    )


def test_rolling_statistics_exclude_current_row() -> None:
    standardizer = CausalRollingStandardizer(("price",), normalize_window=2)

    result = standardizer.transform_frame(_frame([1.0, 3.0, 5.0, 7.0]))

    assert result.get_column("normalized__price").to_list() == [None, None, 3.0, 3.0]
    assert result.get_column("normalization_valid").to_list() == [False, False, True, True]
    assert result.get_column("price").to_list() == [1.0, 3.0, 5.0, 7.0]


def test_modifying_future_does_not_change_past_normalized_values() -> None:
    standardizer = CausalRollingStandardizer(("price",), normalize_window=2)
    original = standardizer.transform_frame(_frame([1.0, 3.0, 5.0, 7.0]))
    modified = standardizer.transform_frame(_frame([1.0, 3.0, 5.0, 7000.0]))

    assert original.get_column("normalized__price")[:3].to_list() == modified.get_column(
        "normalized__price"
    )[:3].to_list()
    assert modified["normalized__price"][3] == 6996.0


def test_invalid_history_breaks_consecutive_normalization_window() -> None:
    standardizer = CausalRollingStandardizer(("price",), normalize_window=2)

    result = standardizer.transform_frame(
        _frame([1.0, 3.0, 5.0, 7.0], valid=[True, False, True, True])
    )

    assert result.get_column("normalization_valid").to_list() == [False, False, False, False]


def test_constant_history_uses_unit_standard_deviation() -> None:
    standardizer = CausalRollingStandardizer(("price",), normalize_window=2)

    result = standardizer.transform_frame(_frame([3.0, 3.0, 5.0]))

    assert result["normalized__price"][2] == 2.0


def test_state_round_trip_preserves_window_and_feature_order() -> None:
    standardizer = CausalRollingStandardizer(("second", "first"), normalize_window=180)

    state = json.loads(json.dumps(standardizer.state_dict()))
    restored = CausalRollingStandardizer.from_state_dict(state)

    assert restored == standardizer
    assert restored.output_feature_cols == ["normalized__second", "normalized__first"]


def test_standardizer_rejects_mixed_sessions() -> None:
    frame = _frame([1.0, 2.0]).with_columns(
        pl.Series("session_id", ["AM", "PM"])
    )

    with pytest.raises(ValueError, match="one session_id"):
        CausalRollingStandardizer(("price",), 2).transform_frame(frame)
