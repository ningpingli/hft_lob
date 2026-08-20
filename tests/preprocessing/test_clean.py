from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl

from hft_lob.configs.experiment import RAW_FEATURE_COLUMNS, SessionConfig
from hft_lob.preprocessing.clean import DataCleaner


def _book_row(timestamp: datetime, *, bid: float | None, ask: float | None) -> dict[str, object]:
    row: dict[str, object] = {"raw_time": timestamp}
    for level in range(1, 6):
        row[f"ASKp{level}"] = ask + 0.01 * (level - 1) if ask is not None else None
        row[f"ASKs{level}"] = 100.0 if ask is not None else None
        row[f"BIDp{level}"] = bid - 0.01 * (level - 1) if bid is not None else None
        row[f"BIDs{level}"] = 100.0 if bid is not None else None
    row.update(last=10.0, volume=1_000.0, amount=10_000.0)
    return row


def test_clean_day_splits_sessions_and_bounds_forward_fill(tmp_path: Path) -> None:
    path = tmp_path / "20260105.parquet"
    rows = [
        _book_row(datetime(2026, 1, 5, 9, 30, 0), bid=9.99, ask=10.01),
        _book_row(datetime(2026, 1, 5, 9, 30, 0), bid=10.00, ask=10.02),
        _book_row(datetime(2026, 1, 5, 9, 30, 9), bid=None, ask=None),
        _book_row(datetime(2026, 1, 5, 13, 0, 0), bid=10.09, ask=None),
    ]
    pl.DataFrame(rows).select("raw_time", *RAW_FEATURE_COLUMNS).write_parquet(path)

    result = DataCleaner(
        SessionConfig(),
        snapshot_interval_seconds=3,
        max_ffill_gap_seconds=6,
        column_mapping={"raw_time": "timestamp"},
    ).clean_day(str(path), ticker="000001.SZ")

    assert [segment.session_id for segment in result.sessions] == ["AM", "PM"]
    morning = result.sessions[0].frame
    assert morning.height == 4
    assert morning.get_column("BIDp1").to_list() == [10.0, 10.0, 10.0, None]
    assert morning.get_column("is_ffilled").to_list() == [False, True, True, False]
    assert morning.get_column("book_valid").to_list() == [True, True, True, False]
    assert morning.get_column("ticker").unique().to_list() == ["000001.SZ"]

    afternoon = result.sessions[1].frame
    assert afternoon.get_column("mid_price").to_list() == [10.09]
    assert afternoon.get_column("book_valid").to_list() == [True]
    assert result.quality_report.duplicate_count == 1
    assert result.quality_report.row_count == 5


def test_clean_day_reads_datetime_index_as_timestamp(tmp_path: Path) -> None:
    """原始时间是 DatetimeIndex，不要求它同时作为普通字段存在。"""
    path = tmp_path / "20260105.parquet"
    timestamps = [
        datetime(2026, 1, 5, 9, 30, 0),
        datetime(2026, 1, 5, 9, 30, 3),
    ]
    rows = [_book_row(timestamp, bid=9.99, ask=10.01) for timestamp in timestamps]
    pandas_frame = pd.DataFrame(rows).drop(columns="raw_time")
    pandas_frame.index = pd.DatetimeIndex(timestamps, name="event_time")
    pandas_frame.to_parquet(path)

    result = DataCleaner(
        SessionConfig(),
        snapshot_interval_seconds=3,
        max_ffill_gap_seconds=6,
        column_mapping={},
    ).clean_day(str(path), ticker="000001.SZ")

    morning = result.sessions[0].frame
    assert morning.schema["timestamp"] == pl.Datetime("us")
    assert morning.get_column("timestamp").to_list() == timestamps
    assert "event_time" not in morning.columns
