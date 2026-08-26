"""Evaluation report persistence and diagnostic plots."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from hft_lob.systems.metrics import EvaluationReport


def save_evaluation_outputs(report: EvaluationReport, output_dir: str | Path) -> dict[str, str]:
    """Persist one evaluation report and its required diagnostic plots."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "evaluation.yaml"
    report_path.write_text(
        yaml.safe_dump(asdict(report), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    daily_ic_path = plot_daily_ic_curve(report, destination / "daily_ic_curve.png")
    grouped_return_path = plot_time_series_grouped_return_curve(
        report, destination / "time_series_grouped_return_curve.png"
    )
    horizon_decay_path = plot_horizon_decay_curve(
        report, destination / "horizon_pearson_decay_curve.png"
    )
    return {
        "evaluation_report": str(report_path.resolve()),
        "daily_ic_curve": str(daily_ic_path.resolve()),
        "time_series_grouped_return_curve": str(grouped_return_path.resolve()),
        "horizon_pearson_decay_curve": str(horizon_decay_path.resolve()),
    }


def plot_daily_ic_curve(report: EvaluationReport, output_path: str | Path) -> Path:
    """Plot the chronological daily TS-IC series and its mean."""
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


def plot_horizon_decay_curve(report: EvaluationReport, output_path: str | Path) -> Path:
    """Plot Mean Daily Pearson Corr versus future horizon."""
    path = _prepare_output_path(output_path)
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(8, 5))
    records = report.horizon_decay
    horizons = np.asarray([record.horizon_seconds / 60 for record in records], dtype=np.float64)
    values = np.asarray(
        [record.mean_daily_pearson_corr for record in records], dtype=np.float64
    )
    finite = np.isfinite(values)
    if np.any(finite):
        axis.plot(
            horizons[finite],
            values[finite],
            marker="o",
            linewidth=1.8,
            color="tab:green",
            label="Mean Daily Pearson Corr",
        )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title("Future Return Pearson Corr Decay")
    axis.set_xlabel("Future horizon (minutes)")
    axis.set_ylabel("Mean Daily Pearson Corr")
    axis.grid(alpha=0.25)
    if np.any(finite):
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    pyplot.close(figure)
    return path


def plot_time_series_grouped_return_curve(
    report: EvaluationReport, output_path: str | Path
) -> Path:
    """Plot realized returns for prediction-sorted temporal bins.

    This is deliberately not a cross-sectional portfolio grouping plot: all valid samples
    from the temporal evaluation window are sorted by prediction before binning.
    """
    path = _prepare_output_path(output_path)
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(8, 5))
    bins = report.prediction_bins
    positions = np.asarray([record.bin_index for record in bins], dtype=np.int64)
    returns = np.asarray(
        [record.mean_realized_return for record in bins],
        dtype=np.float64,
    )
    if positions.size:
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
    if positions.size:
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    pyplot.close(figure)
    return path


def _prepare_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot
