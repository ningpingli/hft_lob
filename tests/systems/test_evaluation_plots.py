from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import yaml

from hft_lob.systems.evaluation_plots import load_evaluation_report, save_evaluation_outputs
from hft_lob.systems.metrics import DailyICRecord, EvaluationReport, PredictionBinRecord


def _report() -> EvaluationReport:
    return EvaluationReport(
        sample_count=6,
        valid_sample_count=6,
        valid_day_count=2,
        overall={"mse": 0.1, "mae": 0.2},
        daily_ic=(
            DailyICRecord(trade_date="2025-01-01", sample_count=3, ic=0.4),
            DailyICRecord(trade_date="2025-01-02", sample_count=3, ic=0.6),
        ),
        mean_daily_ic=0.5,
        positive_ic_day_ratio=1.0,
        prediction_bins=(
            PredictionBinRecord(1, 0.0, 0.5, 3, -0.2, -0.1),
            PredictionBinRecord(2, 0.5, 1.0, 3, 0.3, 0.2),
        ),
    )


def test_save_evaluation_outputs_round_trips_report_and_curves(tmp_path: Path) -> None:
    report = _report()

    paths = save_evaluation_outputs(report, tmp_path)

    assert set(paths) == {
        "evaluation_report",
        "daily_ic_curve",
        "time_series_grouped_return_curve",
    }
    for path in paths.values():
        assert Path(path).is_file()
        assert Path(path).stat().st_size > 0
    assert load_evaluation_report(paths["evaluation_report"]) == report


def test_load_evaluation_report_rejects_inconsistent_daily_summary(tmp_path: Path) -> None:
    report = replace(_report(), positive_ic_day_ratio=0.5)
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump(asdict(report)), encoding="utf-8")

    with pytest.raises(ValueError, match="positive_ic_day_ratio does not match"):
        load_evaluation_report(path)
