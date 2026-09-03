from __future__ import annotations

import math

import numpy as np
import pytest

from hft_lob.configs.experiment import EvaluationConfig
from hft_lob.data_types import SampleMeta
from hft_lob.metrics.metrics import (
    build_evaluation_report,
    daily_ic_records,
    evaluate,
    mae,
    mean_daily_ic,
    mse,
    positive_ic_day_ratio,
    prediction_quantile_bins,
    ts_ic,
)
from hft_lob.reporting.artifact import PredictionArtifact


def test_error_metrics_ignore_non_finite_pairs_and_reject_length_mismatch() -> None:
    predictions = np.array([1.0, 2.0, np.nan])
    targets = np.array([1.0, 4.0, 3.0])
    assert mse(predictions, targets) == pytest.approx(2.0)
    assert mae(predictions, targets) == pytest.approx(1.0)
    assert evaluate(predictions, targets) == pytest.approx({"mse": 2.0, "mae": 1.0})
    with pytest.raises(ValueError, match="same length"):
        mse(np.array([1.0]), np.array([1.0, 2.0]))


def test_daily_ic_statistics_use_finite_days_and_valid_sample_counts() -> None:
    records = daily_ic_records(
        np.array([1.0, 2.0, 1.0, 2.0, 1.0, np.nan]),
        np.array([2.0, 1.0, 1.0, 2.0, 3.0, 4.0]),
        np.array(["2025-01-02", "2025-01-02", "2025-01-01", "2025-01-01", "2025-01-03", "2025-01-03"]),
    )
    assert [record.trade_date for record in records] == ["2025-01-01", "2025-01-02", "2025-01-03"]
    assert [record.sample_count for record in records] == [2, 2, 1]
    assert records[0].ic == pytest.approx(1.0)
    assert records[1].ic == pytest.approx(-1.0)
    assert math.isnan(records[2].ic)
    values = np.asarray([record.ic for record in records])
    assert mean_daily_ic(values) == pytest.approx(0.0)
    assert positive_ic_day_ratio(values) == pytest.approx(0.5)
    assert math.isnan(ts_ic(np.ones(3), np.arange(3)))


def test_prediction_bins_are_equal_count_and_deterministic_with_ties() -> None:
    records = prediction_quantile_bins(np.array([2.0, 1.0, 1.0, 4.0]), np.array([20.0, 10.0, 11.0, 40.0]), n_bins=2)
    assert [record.sample_count for record in records] == [2, 2]
    assert records[0].mean_realized_return == pytest.approx(10.5)
    assert records[1].mean_realized_return == pytest.approx(30.0)


def test_build_evaluation_report_contains_per_label_metrics() -> None:
    predictions = np.array([-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0])
    metadata = tuple(
        SampleMeta(
            ticker="TEST",
            trade_date="2025-01-01" if index < 4 else "2025-01-02",
            session_id="am" if index % 4 < 2 else "pm",
            anchor_timestamp=f"2025-01-{1 if index < 4 else 2:02d}T09:{30 + index % 4:02d}:00",
            mid_t=100.0,
            bid1=99.0,
            ask1=101.0,
            spread=2.0,
        )
        for index in range(8)
    )
    matrix = np.column_stack((predictions, predictions + 1))
    artifact = PredictionArtifact(
        predictions=matrix,
        targets=matrix.copy(),
        labels=(60, 120),
        metadata=metadata,
        model_name="model",
        model_version="v1",
        dataset_version="v1",
        fold_index=1,
        split="test",
    )
    report = build_evaluation_report(artifact, EvaluationConfig(prediction_bins=2))
    assert report.labels == (60, 120)
    assert report.sample_count == 8
    assert report.valid_sample_count == 16
    assert report.overall == {"mse": 0.0, "mae": 0.0}
    assert set(report.per_label) == {60, 120}
    assert report.per_label[60].overall == {"mse": 0.0, "mae": 0.0}


def test_evaluation_config_rejects_invalid_bin_count() -> None:
    with pytest.raises(ValueError, match="integer >= 2"):
        EvaluationConfig(prediction_bins=1)
