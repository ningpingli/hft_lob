from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
import torch

from hft_lob.datasets.lob_dataset import LOBWindowDataset
from hft_lob.preprocessing.normalize import CausalRollingStandardizer


def _frame(*, session_ids: list[str] | None = None) -> pl.DataFrame:
    size = 8
    start = datetime(2026, 1, 5, 9, 30)
    return pl.DataFrame(
        {
            "ticker": ["TEST"] * size,
            "trade_date": ["2026-01-05"] * size,
            "session_id": session_ids or ["AM"] * size,
            "timestamp": [start + timedelta(seconds=3 * index) for index in range(size)],
            "seconds": [(start + timedelta(seconds=3 * index)).time() for index in range(size)],
            "book_valid": [True] * size,
            "feature_valid": [True] * size,
            "target_valid": [True, True, True, True, True, True, True, False],
            "f1": [float(index) for index in range(1, size + 1)],
            "f2": [10.0] * size,
            "Target_60s_log": [index / 100.0 for index in range(size)],
            "mid_price": [10.0 + index / 100.0 for index in range(size)],
            "future_mid": [10.1 + index / 100.0 for index in range(size)],
            "ASKp1": [10.01 + index / 100.0 for index in range(size)],
            "BIDp1": [9.99 + index / 100.0 for index in range(size)],
        },
        schema_overrides={"timestamp": pl.Datetime("us")},
    )


def _write(path: Path, frame: pl.DataFrame | None = None) -> str:
    (frame if frame is not None else _frame()).write_parquet(path)
    return str(path)


def test_dataset_uses_causal_columns_and_anchor_inclusive_window(tmp_path: Path) -> None:
    path = _write(tmp_path / "session.parquet")
    standardizer = CausalRollingStandardizer(("f1", "f2"), normalize_window=2)
    dataset = LOBWindowDataset(
        [path],
        ticker="TEST",
        window_size=2,
        feature_cols=["f1", "f2"],
        target_col="Target_60s_log",
        standardizer=standardizer,
    )

    features, target, metadata = dataset[0]

    assert len(dataset) == 4
    assert dataset.n_features == 2
    assert dataset.feature_cols == ["normalized__f1", "normalized__f2"]
    assert features.shape == (1, 2, 2)
    assert features.dtype == torch.float32
    torch.testing.assert_close(features, torch.tensor([[[3.0, 0.0], [3.0, 0.0]]]))
    torch.testing.assert_close(target, torch.tensor([0.03]))
    assert metadata.anchor_timestamp == "2026-01-05T09:30:09"
    assert metadata.trade_date == "2026-01-05"
    assert metadata.session_id == "AM"
    assert metadata.bid1 == pytest.approx(10.02)
    assert metadata.ask1 == pytest.approx(10.04)
    assert metadata.spread == pytest.approx(0.02)


def test_invalid_history_row_breaks_windows_without_dropping_rows(tmp_path: Path) -> None:
    frame = _frame().with_columns(
        pl.Series("feature_valid", [True, True, True, True, False, True, True, True])
    )
    dataset = LOBWindowDataset(
        [_write(tmp_path / "invalid.parquet", frame)],
        ticker="TEST",
        window_size=2,
        feature_cols=["f1", "f2"],
        target_col="Target_60s_log",
    )

    assert len(dataset) == 4
    assert dataset.feature_cols == ["f1", "f2"]


def test_dataset_rejects_mixed_sessions(tmp_path: Path) -> None:
    frame = _frame(session_ids=["AM"] * 4 + ["PM"] * 4)

    with pytest.raises(ValueError, match="one session_id"):
        LOBWindowDataset(
            [_write(tmp_path / "mixed.parquet", frame)],
            ticker="TEST",
            window_size=2,
            feature_cols=["f1", "f2"],
            target_col="Target_60s_log",
        )


def test_dataset_supports_negative_index_and_checks_bounds(tmp_path: Path) -> None:
    dataset = LOBWindowDataset(
        [_write(tmp_path / "bounds.parquet")],
        ticker="TEST",
        window_size=2,
        feature_cols=["f1", "f2"],
        target_col="Target_60s_log",
    )

    assert dataset[-1][2].anchor_timestamp == "2026-01-05T09:30:18"
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]
