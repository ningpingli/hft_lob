from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hft_lob.configs.experiment import EvaluationConfig, FoldSelectionConfig
from hft_lob.data_types import SampleMeta
from hft_lob.metrics.metrics import mean_daily_ic
from hft_lob.reporting.artifact import PredictionArtifact
from hft_lob.reporting.reporter import load_evaluation_report
from hft_lob.trainner.walk_forward import CandidateFoldRun, run_walk_forward


class _Executor:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def run_candidate(
        self,
        *,
        package: object,
        config: object,
        fold_index: int,
        candidate_name: str,
    ) -> CandidateFoldRun:
        trade_date = f"2026-01-0{fold_index}"
        metadata = tuple(
            SampleMeta(
                ticker="TEST",
                trade_date=trade_date,
                session_id="AM",
                anchor_timestamp=f"{trade_date}T09:30:0{index}",
                mid_t=10.0 + index,
                bid1=9.99 + index,
                ask1=10.01 + index,
                spread=0.02,
            )
            for index in range(2)
        )
        predictions = np.asarray([[1.0], [2.0]])
        targets = np.asarray([[1.0], [2.0]]) if fold_index == 1 else np.asarray([[2.0], [1.0]])
        artifact = PredictionArtifact(
            predictions=predictions,
            targets=targets,
            labels=(60,),
            metadata=metadata,
            model_name=candidate_name,
            model_version=f"model-fold-{fold_index}",
            dataset_version="dataset",
            fold_index=fold_index,
            split="test",
        )
        return CandidateFoldRun(
            artifact=artifact,
            dataset_metadata_path="dataset.json",
            predictions_path=str(
                self.output_root / f"fold_{fold_index:03d}" / candidate_name / "predictions.parquet"
            ),
        )


def test_walk_forward_builds_one_continuous_evaluation_for_all_folds(tmp_path: Path) -> None:
    (tmp_path / "folds" / "fold_001").mkdir(parents=True)
    (tmp_path / "folds" / "fold_002").mkdir(parents=True)
    package = SimpleNamespace(
        root=tmp_path,
        metadata=SimpleNamespace(dataset_id="dataset"),
    )
    config = SimpleNamespace(
        model=SimpleNamespace(name="model"),
        folds=FoldSelectionConfig(),
        evaluation=EvaluationConfig(prediction_bins=2),
    )

    report = run_walk_forward(
        package,
        config,
        executor=_Executor(tmp_path / "walk_forward"),
    )

    assert len(report.fold_results) == 2
    aggregate_dir = tmp_path / "walk_forward" / "model"
    aggregate = load_evaluation_report(aggregate_dir / "evaluation.yaml")
    assert [record.trade_date for record in aggregate.daily_ic] == [
        "2026-01-01",
        "2026-01-02",
    ]
    assert aggregate.mean_daily_ic == mean_daily_ic(
        np.asarray([1.0, -1.0], dtype=np.float64)
    )
    assert report.summary["model"]["mean_daily_ic_mean"] == aggregate.mean_daily_ic
    assert not (tmp_path / "walk_forward" / "fold_001" / "model" / "evaluation.yaml").exists()
    assert not (tmp_path / "walk_forward" / "fold_002" / "model" / "evaluation.yaml").exists()
