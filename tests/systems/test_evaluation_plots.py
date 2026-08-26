from __future__ import annotations

from pathlib import Path

import yaml

from hft_lob.systems.evaluation_plots import save_evaluation_outputs
from hft_lob.systems.metrics import (
    ConfidenceInterval,
    DailyICRecord,
    EvaluationReport,
    PredictionBinRecord,
)


def test_save_evaluation_outputs_writes_report_and_required_curves(tmp_path: Path) -> None:
    report = EvaluationReport(
        sample_count=6,
        overall={"ts_ic": 0.5},
        daily=(),
        daily_ic=(
            DailyICRecord(trade_date="2025-01-01", sample_count=3, ic=0.4),
            DailyICRecord(trade_date="2025-01-02", sample_count=3, ic=0.6),
        ),
        mean_daily_ic=0.5,
        daily_summary={"mean_daily_ic": 0.5},
        confidence_intervals={
            "ts_ic": ConfidenceInterval(
                estimate=0.5, lower=0.2, upper=0.8, confidence_level=0.95
            )
        },
        prediction_bins=(
            PredictionBinRecord(1, 0.0, 0.5, 3, -0.2, -0.1),
            PredictionBinRecord(2, 0.5, 1.0, 3, 0.3, 0.2),
        ),
    )

    paths = save_evaluation_outputs(report, tmp_path)

    assert set(paths) == {
        "evaluation_report",
        "daily_ic_curve",
        "time_series_grouped_return_curve",
        "horizon_pearson_decay_curve",
    }
    for path in paths.values():
        assert Path(path).is_file()
        assert Path(path).stat().st_size > 0
    saved = yaml.safe_load((tmp_path / "evaluation.yaml").read_text(encoding="utf-8"))
    assert saved["mean_daily_ic"] == 0.5
    assert saved["daily_ic"][0]["trade_date"] == "2025-01-01"
