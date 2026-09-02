"""Single-stock time-series evaluation metrics and report construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hft_lob.configs.experiment import EvaluationConfig
from hft_lob.reporting.artifact import PredictionArtifact

METRIC_NAMES: tuple[str, ...] = ("mse", "mae")


@dataclass(frozen=True)
class DailyICRecord:
    """Pearson TS-IC for one trading day."""

    trade_date: str
    sample_count: int
    ic: float


@dataclass(frozen=True)
class PredictionBinRecord:
    """One prediction-sorted temporal bin and its realized return."""

    bin_index: int
    lower_quantile: float
    upper_quantile: float
    sample_count: int
    mean_prediction: float
    mean_realized_return: float


@dataclass(frozen=True)
class LabelEvaluation:
    """One complete evaluation view for one configured label."""

    label: int
    valid_sample_count: int
    valid_day_count: int
    overall: dict[str, float]
    daily_ic: tuple[DailyICRecord, ...]
    mean_daily_ic: float
    positive_ic_day_ratio: float
    prediction_bins: tuple[PredictionBinRecord, ...]


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate metrics plus an independently computed report per label."""

    labels: tuple[int, ...]
    sample_count: int
    valid_sample_count: int
    valid_day_count: int
    overall: dict[str, float]
    daily_ic: tuple[DailyICRecord, ...]
    mean_daily_ic: float
    positive_ic_day_ratio: float
    prediction_bins: tuple[PredictionBinRecord, ...]
    per_label: dict[int, LabelEvaluation]

def mse(preds: np.ndarray, targets: np.ndarray) -> float:
    """Mean squared error over finite prediction-target pairs."""
    prediction, target = _valid_pairs(preds, targets)
    return float(np.mean(np.square(prediction - target))) if prediction.size else float("nan")


def mae(preds: np.ndarray, targets: np.ndarray) -> float:
    """Mean absolute error over finite prediction-target pairs."""
    prediction, target = _valid_pairs(preds, targets)
    return float(np.mean(np.abs(prediction - target))) if prediction.size else float("nan")


def ts_ic(preds: np.ndarray, targets: np.ndarray) -> float:
    """Pearson time-series IC; degenerate inputs produce NaN."""
    prediction, target = _valid_pairs(preds, targets)
    if prediction.size < 2 or _is_constant(prediction) or _is_constant(target):
        return float("nan")
    return float(np.corrcoef(prediction, target)[0, 1])


def mean_daily_ic(daily_ics: np.ndarray) -> float:
    """Equal-weight mean of finite daily Pearson TS-IC values."""
    values = _finite_vector(daily_ics)
    return float(np.mean(values)) if values.size else float("nan")


def positive_ic_day_ratio(daily_ics: np.ndarray) -> float:
    """Fraction of finite daily IC values that are strictly positive."""
    values = _finite_vector(daily_ics)
    return float(np.mean(values > 0)) if values.size else float("nan")


def evaluate(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Calculate the complete configured scalar error metric set."""
    return {
        "mse": mse(preds, targets),
        "mae": mae(preds, targets),
    }


def daily_ic_records(
    preds: np.ndarray,
    targets: np.ndarray,
    trade_dates: np.ndarray,
) -> tuple[DailyICRecord, ...]:
    """Calculate daily Pearson TS-IC in chronological date order."""
    prediction, target = _paired_vectors(preds, targets)
    dates = _metadata_vector(trade_dates, field="trade_dates", size=prediction.size)
    finite_pairs = np.isfinite(prediction) & np.isfinite(target)
    records: list[DailyICRecord] = []
    for trade_date in sorted(set(dates.tolist())):
        date_mask = dates == trade_date
        valid_mask = date_mask & finite_pairs
        records.append(
            DailyICRecord(
                trade_date=str(trade_date),
                sample_count=int(np.count_nonzero(valid_mask)),
                ic=ts_ic(prediction[date_mask], target[date_mask]),
            )
        )
    return tuple(records)


def prediction_quantile_bins(
    preds: np.ndarray,
    targets: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[PredictionBinRecord, ...]:
    """Sort all temporal samples by prediction and split them into equal-count bins."""
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("n_bins must be an integer >= 2")
    prediction, target = _valid_pairs(preds, targets)
    if prediction.size < n_bins:
        raise ValueError("valid sample count must be >= n_bins")
    order = np.argsort(prediction, kind="stable")
    groups = np.array_split(order, n_bins)
    return tuple(
        PredictionBinRecord(
            bin_index=index,
            lower_quantile=(index - 1) / n_bins,
            upper_quantile=index / n_bins,
            sample_count=int(group.size),
            mean_prediction=float(np.mean(prediction[group])),
            mean_realized_return=float(np.mean(target[group])),
        )
        for index, group in enumerate(groups, start=1)
    )

def build_evaluation_report(
    artifact: PredictionArtifact,
    config: EvaluationConfig,
) -> EvaluationReport:
    """Build aggregate and per-label reports from the prediction matrix."""
    trade_dates = np.asarray([meta.trade_date for meta in artifact.metadata])
    per_label: dict[int, LabelEvaluation] = {}
    flattened_predictions: list[np.ndarray] = []
    flattened_targets: list[np.ndarray] = []
    flattened_dates: list[np.ndarray] = []
    for position, label in enumerate(artifact.labels):
        predictions = artifact.predictions[:, position]
        targets = artifact.targets[:, position]
        dates = trade_dates
        daily_ic = daily_ic_records(predictions, targets, dates)
        daily_values = np.asarray([record.ic for record in daily_ic], dtype=np.float64)
        bins = prediction_quantile_bins(
            predictions,
            targets,
            n_bins=config.prediction_bins,
        )
        per_label[label] = LabelEvaluation(
            label=label,
            valid_sample_count=int(predictions.size),
            valid_day_count=int(np.count_nonzero(np.isfinite(daily_values))),
            overall=evaluate(predictions, targets),
            daily_ic=daily_ic,
            mean_daily_ic=mean_daily_ic(daily_values),
            positive_ic_day_ratio=positive_ic_day_ratio(daily_values),
            prediction_bins=bins,
        )
        flattened_predictions.append(predictions)
        flattened_targets.append(targets)
        flattened_dates.append(dates)

    predictions = np.concatenate(flattened_predictions)
    targets = np.concatenate(flattened_targets)
    dates = np.concatenate(flattened_dates)
    daily_ic = daily_ic_records(predictions, targets, dates)
    daily_values = np.asarray([record.ic for record in daily_ic], dtype=np.float64)
    return EvaluationReport(
        labels=artifact.labels,
        sample_count=int(artifact.predictions.shape[0]),
        valid_sample_count=int(predictions.size),
        valid_day_count=int(np.count_nonzero(np.isfinite(daily_values))),
        overall=evaluate(predictions, targets),
        daily_ic=daily_ic,
        mean_daily_ic=mean_daily_ic(daily_values),
        positive_ic_day_ratio=positive_ic_day_ratio(daily_values),
        prediction_bins=prediction_quantile_bins(
            predictions,
            targets,
            n_bins=config.prediction_bins,
        ),
        per_label=per_label,
    )


def _paired_vectors(preds: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction = _as_vector(preds, field="preds")
    target = _as_vector(targets, field="targets")
    if prediction.size != target.size:
        raise ValueError("preds and targets must have the same length")
    return prediction, target


def _valid_pairs(preds: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction, target = _paired_vectors(preds, targets)
    valid = np.isfinite(prediction) & np.isfinite(target)
    return prediction[valid], target[valid]


def _as_vector(values: np.ndarray, *, field: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"{field} must have shape [N] or [N, 1], got {array.shape}")
    return array


def _finite_vector(values: np.ndarray) -> np.ndarray:
    vector = _as_vector(values, field="values")
    return vector[np.isfinite(vector)]


def _is_constant(values: np.ndarray) -> bool:
    return bool(np.all(values == values[0]))


def _metadata_vector(values: np.ndarray, *, field: str, size: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size != size:
        raise ValueError(f"{field} must be one-dimensional with length {size}")
    if any(value is None or str(value) == "" for value in array):
        raise ValueError(f"{field} must not contain empty values")
    return array
