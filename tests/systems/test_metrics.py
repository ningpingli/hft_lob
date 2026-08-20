from __future__ import annotations

import math

import numpy as np
import pytest

from hft_lob.configs.experiment import EvaluationConfig
from hft_lob.datasets.lob_dataset import SampleMeta
from hft_lob.systems.artifact import PredictionArtifact
from hft_lob.systems.metrics import (
    block_bootstrap_confidence_interval,
    build_evaluation_report,
    direction_accuracy,
    directional_precision_recall,
    evaluate,
    icir,
    mae,
    positive_ic_day_ratio,
    prediction_quantile_bins,
    rank_ic,
    rmse,
    ts_ic,
)


def test_regression_and_correlation_metrics() -> None:
    predictions = np.array([1.0, 2.0, 3.0])
    targets = np.array([1.0, 3.0, 2.0])

    assert mae(predictions, targets) == pytest.approx(2 / 3)
    assert rmse(predictions, targets) == pytest.approx(math.sqrt(2 / 3))
    assert ts_ic(predictions, targets) == pytest.approx(0.5)
    assert rank_ic(predictions, targets) == pytest.approx(0.5)


def test_metrics_ignore_non_finite_pairs_and_reject_length_mismatch() -> None:
    assert mae(np.array([1.0, np.nan]), np.array([3.0, 2.0])) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="same length"):
        mae(np.array([1.0]), np.array([1.0, 2.0]))


def test_degenerate_correlations_are_nan() -> None:
    assert math.isnan(ts_ic(np.ones(3), np.arange(3)))
    assert math.isnan(rank_ic(np.arange(3), np.ones(3)))


def test_direction_metrics() -> None:
    predictions = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    targets = np.array([1.0, -1.0, -1.0, 1.0, 0.0])

    assert direction_accuracy(predictions, targets) == pytest.approx(0.5)
    assert directional_precision_recall(predictions, targets, direction="up") == pytest.approx(
        (1 / 3, 1 / 2)
    )
    assert directional_precision_recall(predictions, targets, direction="down") == pytest.approx(
        (1 / 2, 1 / 2)
    )
    with pytest.raises(ValueError, match="direction"):
        directional_precision_recall(predictions, targets, direction="flat")


def test_daily_ic_stability() -> None:
    daily = np.array([0.2, 0.4, np.nan])
    assert icir(daily) == pytest.approx(3.0)
    assert positive_ic_day_ratio(daily) == pytest.approx(1.0)


def test_evaluate_has_all_metric_names() -> None:
    assert set(evaluate(np.array([1.0, -1.0]), np.array([1.0, -1.0]))) == {
        "mae",
        "rmse",
        "ts_ic",
        "rank_ic",
        "direction_accuracy",
        "up_precision",
        "up_recall",
        "down_precision",
        "down_recall",
    }


def test_prediction_bins_are_equal_count_and_deterministic_with_ties() -> None:
    records = prediction_quantile_bins(
        np.array([2.0, 1.0, 1.0, 4.0]),
        np.array([20.0, 10.0, 11.0, 40.0]),
        n_bins=2,
    )

    assert [record.sample_count for record in records] == [2, 2]
    assert records[0].mean_realized_return == pytest.approx(10.5)
    assert records[1].mean_realized_return == pytest.approx(30.0)


def test_block_bootstrap_is_reproducible() -> None:
    predictions = np.arange(8, dtype=float)
    targets = predictions + 1
    dates = np.array(["2025-01-01"] * 4 + ["2025-01-02"] * 4)
    sessions = np.array(["am"] * 2 + ["pm"] * 2 + ["am"] * 2 + ["pm"] * 2)
    kwargs = dict(
        metric=mae,
        n_resamples=20,
        block_size=2,
        seed=7,
    )

    first = block_bootstrap_confidence_interval(predictions, targets, dates, sessions, **kwargs)
    second = block_bootstrap_confidence_interval(predictions, targets, dates, sessions, **kwargs)
    assert first == second
    assert first.estimate == pytest.approx(1.0)


def test_build_evaluation_report() -> None:
    predictions = np.array([-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0])
    metadata = tuple(
        SampleMeta(
            ticker="TEST",
            trade_date="2025-01-01" if index < 4 else "2025-01-02",
            session_id="am" if index % 4 < 2 else "pm",
            anchor_timestamp=(f"2025-01-0{1 if index < 4 else 2}T09:{30 + index % 4:02d}:00"),
            mid_t=100.0,
            future_mid=101.0,
            bid1=99.0,
            ask1=101.0,
            spread=2.0,
        )
        for index in range(8)
    )
    artifact = PredictionArtifact(
        predictions=predictions,
        targets=predictions.copy(),
        metadata=metadata,
        model_name="model",
        model_version="v1",
        dataset_version="v1",
        fold_index=1,
        split="test",
    )
    config = EvaluationConfig(
        metrics=("mae", "ts_ic"),
        prediction_bins=2,
        bootstrap_samples=10,
        bootstrap_block_size=2,
    )

    report = build_evaluation_report(artifact, config, seed=42)

    assert report.sample_count == 8
    assert report.overall["mae"] == 0.0
    assert report.overall["ts_ic"] == pytest.approx(1.0)
    assert len(report.daily) == 2
    assert report.daily_summary["positive_ic_day_ratio"] == 1.0
    assert set(report.confidence_intervals) == {"mae", "ts_ic"}
    assert len(report.prediction_bins) == 2
