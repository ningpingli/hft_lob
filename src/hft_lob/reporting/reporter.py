"""Evaluation report persistence and diagnostic plots."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from hft_lob.metrics.metrics import (
    METRIC_NAMES,
    DailyICRecord,
    EvaluationReport,
    LabelEvaluation,
    PredictionBinRecord,
    mean_daily_ic,
    positive_ic_day_ratio,
)
from hft_lob.utils._yaml_io import atomic_dump_yaml

_REPORT_FIELDS = {
    "labels",
    "sample_count",
    "valid_sample_count",
    "valid_day_count",
    "overall",
    "daily_ic",
    "mean_daily_ic",
    "positive_ic_day_ratio",
    "prediction_bins",
    "per_label",
}


def save_evaluation_outputs(report: EvaluationReport, output_dir: str | Path) -> dict[str, str]:
    """Persist the complete test evaluation report and diagnostic plots."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "evaluation.yaml"
    atomic_dump_yaml(report_path, asdict(report))
    daily_ic_path = plot_daily_ic_curve(report, destination / "daily_ic_curve.png")
    grouped_return_path = plot_time_series_grouped_return_curve(
        report, destination / "time_series_grouped_return_curve.png"
    )
    return {
        "evaluation_report": str(report_path.resolve()),
        "daily_ic_curve": str(daily_ic_path.resolve()),
        "time_series_grouped_return_curve": str(grouped_return_path.resolve()),
    }


def load_evaluation_report(path: str | Path) -> EvaluationReport:
    """Load and validate one persisted evaluation report."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid evaluation report: {source}") from exc
    if not isinstance(value, dict) or set(value) != _REPORT_FIELDS:
        raise ValueError("evaluation report has an invalid root schema")
    labels_raw = value["labels"]
    if (
        not isinstance(labels_raw, list)
        or not labels_raw
        or any(not isinstance(label, int) or isinstance(label, bool) or label <= 0 for label in labels_raw)
        or len(set(labels_raw)) != len(labels_raw)
    ):
        raise ValueError("evaluation labels must be a non-empty unique positive list")
    overall_raw = value["overall"]
    daily_raw = value["daily_ic"]
    bins_raw = value["prediction_bins"]
    per_label_raw = value["per_label"]
    if not isinstance(overall_raw, dict) or set(overall_raw) != set(METRIC_NAMES):
        raise ValueError("evaluation overall metrics must be exactly mse and mae")
    if not isinstance(daily_raw, list) or not isinstance(bins_raw, list):
        raise ValueError("evaluation daily_ic and prediction_bins must be lists")
    if not isinstance(per_label_raw, dict):
        raise ValueError("evaluation per_label must be a mapping")

    per_label = {
        int(label): _label_report(item, expected_label=int(label))
        for label, item in per_label_raw.items()
    }
    if set(per_label) != set(labels_raw):
        raise ValueError("evaluation per_label keys must match labels")
    report = EvaluationReport(
        labels=tuple(labels_raw),
        sample_count=_integer(value["sample_count"], field="sample_count", minimum=1),
        valid_sample_count=_integer(value["valid_sample_count"], field="valid_sample_count", minimum=0),
        valid_day_count=_integer(value["valid_day_count"], field="valid_day_count", minimum=0),
        overall={name: _number(overall_raw[name], field=name) for name in METRIC_NAMES},
        daily_ic=tuple(_daily_record(item) for item in daily_raw),
        mean_daily_ic=_number(value["mean_daily_ic"], field="mean_daily_ic"),
        positive_ic_day_ratio=_number(value["positive_ic_day_ratio"], field="positive_ic_day_ratio"),
        prediction_bins=tuple(_prediction_bin(item) for item in bins_raw),
        per_label=per_label,
    )
    _validate_report(report)
    return report


def _label_report(value: object, *, expected_label: int) -> LabelEvaluation:
    if not isinstance(value, dict):
        raise ValueError("per_label entries must be mappings")
    required = {
        "label",
        "valid_sample_count",
        "valid_day_count",
        "overall",
        "daily_ic",
        "mean_daily_ic",
        "positive_ic_day_ratio",
        "prediction_bins",
    }
    if set(value) != required or value["label"] != expected_label:
        raise ValueError("per_label entry has an invalid schema")
    overall = value["overall"]
    if not isinstance(overall, dict) or set(overall) != set(METRIC_NAMES):
        raise ValueError("per_label overall metrics must be exactly mse and mae")
    daily = value["daily_ic"]
    bins = value["prediction_bins"]
    if not isinstance(daily, list) or not isinstance(bins, list):
        raise ValueError("per_label daily_ic and prediction_bins must be lists")
    return LabelEvaluation(
        label=expected_label,
        valid_sample_count=_integer(value["valid_sample_count"], field="label sample_count", minimum=0),
        valid_day_count=_integer(value["valid_day_count"], field="label day_count", minimum=0),
        overall={name: _number(overall[name], field=name) for name in METRIC_NAMES},
        daily_ic=tuple(_daily_record(item) for item in daily),
        mean_daily_ic=_number(value["mean_daily_ic"], field="label mean_daily_ic"),
        positive_ic_day_ratio=_number(value["positive_ic_day_ratio"], field="label positive_ic_day_ratio"),
        prediction_bins=tuple(_prediction_bin(item) for item in bins),
    )


def plot_daily_ic_curve(report: EvaluationReport, output_path: str | Path) -> Path:
    """Plot the chronological daily Pearson TS-IC series and its mean."""
    path = _prepare_output_path(output_path)
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(10, 5))
    dates = [record.trade_date for record in report.daily_ic]
    values = np.asarray([record.ic for record in report.daily_ic], dtype=np.float64)
    positions = np.arange(len(dates))
    finite = np.isfinite(values)
    if np.any(finite):
        axis.plot(
            positions[finite],
            values[finite],
            marker="o",
            linewidth=1.5,
            label="Daily TS-IC",
        )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    if np.isfinite(report.mean_daily_ic):
        axis.axhline(
            report.mean_daily_ic,
            color="tab:orange",
            linestyle="--",
            linewidth=1.2,
            label=f"Mean daily IC = {report.mean_daily_ic:.4f}",
        )
    axis.set_title("Daily TS-IC Curve")
    axis.set_xlabel("Trade date")
    axis.set_ylabel("TS-IC")
    axis.set_xticks(positions)
    axis.set_xticklabels(dates, rotation=45, ha="right")
    axis.grid(alpha=0.25)
    if np.any(finite) or np.isfinite(report.mean_daily_ic):
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    pyplot.close(figure)
    return path


def plot_time_series_grouped_return_curve(
    report: EvaluationReport, output_path: str | Path
) -> Path:
    """Plot realized returns for prediction-sorted temporal bins."""
    path = _prepare_output_path(output_path)
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(8, 5))
    bins = report.prediction_bins
    positions = np.asarray([record.bin_index for record in bins], dtype=np.int64)
    returns = np.asarray(
        [record.mean_realized_return for record in bins],
        dtype=np.float64,
    )
    axis.plot(
        positions,
        returns,
        marker="o",
        linewidth=1.8,
        color="tab:blue",
        label="Mean realized return",
    )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title("Time-Series Grouped Return Curve")
    axis.set_xlabel("Prediction-sorted bin (low → high)")
    axis.set_ylabel("Mean realized return")
    axis.set_xticks(positions)
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    pyplot.close(figure)
    return path


def _daily_record(value: object) -> DailyICRecord:
    if not isinstance(value, dict) or set(value) != {"trade_date", "sample_count", "ic"}:
        raise ValueError("daily_ic record has an invalid schema")
    trade_date = value["trade_date"]
    if not isinstance(trade_date, str) or not trade_date:
        raise ValueError("daily_ic trade_date must be non-empty")
    return DailyICRecord(
        trade_date=trade_date,
        sample_count=_integer(value["sample_count"], field="daily sample_count", minimum=0),
        ic=_number(value["ic"], field="daily ic"),
    )


def _prediction_bin(value: object) -> PredictionBinRecord:
    fields = {
        "bin_index",
        "lower_quantile",
        "upper_quantile",
        "sample_count",
        "mean_prediction",
        "mean_realized_return",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("prediction bin has an invalid schema")
    return PredictionBinRecord(
        bin_index=_integer(value["bin_index"], field="bin_index", minimum=1),
        lower_quantile=_number(value["lower_quantile"], field="lower_quantile"),
        upper_quantile=_number(value["upper_quantile"], field="upper_quantile"),
        sample_count=_integer(value["sample_count"], field="bin sample_count", minimum=1),
        mean_prediction=_number(value["mean_prediction"], field="mean_prediction"),
        mean_realized_return=_number(
            value["mean_realized_return"], field="mean_realized_return"
        ),
    )


def _validate_report(report: EvaluationReport) -> None:
    if report.valid_sample_count > report.sample_count * len(report.labels):
        raise ValueError("valid_sample_count exceeds total label cells")
    dates = [record.trade_date for record in report.daily_ic]
    if not dates or dates != sorted(dates) or len(set(dates)) != len(dates):
        raise ValueError("daily_ic dates must be non-empty, unique and chronological")
    daily_values = np.asarray([record.ic for record in report.daily_ic], dtype=np.float64)
    if report.valid_day_count != int(np.count_nonzero(np.isfinite(daily_values))):
        raise ValueError("valid_day_count does not match daily_ic")
    if not _same_number(report.mean_daily_ic, mean_daily_ic(daily_values)):
        raise ValueError("mean_daily_ic does not match daily_ic")
    if not _same_number(report.positive_ic_day_ratio, positive_ic_day_ratio(daily_values)):
        raise ValueError("positive_ic_day_ratio does not match daily_ic")
    if not report.prediction_bins:
        raise ValueError("prediction_bins must not be empty")
    if [record.bin_index for record in report.prediction_bins] != list(
        range(1, len(report.prediction_bins) + 1)
    ):
        raise ValueError("prediction bin indices must be consecutive")
    if sum(record.sample_count for record in report.prediction_bins) != report.valid_sample_count:
        raise ValueError("prediction bin samples must cover all valid samples")


def _integer(value: object, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _same_number(left: float, right: float) -> bool:
    return (math.isnan(left) and math.isnan(right)) or math.isclose(
        left, right, rel_tol=1e-12, abs_tol=1e-15
    )


def _prepare_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot
