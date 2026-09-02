from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hft_lob.reporting.artifact import (
    PredictionArtifact,
    load_prediction_artifact,
    save_prediction_artifact,
)
from hft_lob.systems.contracts import SampleMeta


def _metadata() -> tuple[SampleMeta, ...]:
    return tuple(
        SampleMeta(
            ticker="TEST",
            trade_date="2026-01-05",
            session_id="AM",
            anchor_timestamp=f"2026-01-05T09:30:0{index}",
            mid_t=10.0 + index * 0.01,
            bid1=9.99 + index * 0.01,
            ask1=10.01 + index * 0.01,
            spread=0.02,
        )
        for index in range(2)
    )


def _artifact() -> PredictionArtifact:
    return PredictionArtifact(
        predictions=np.asarray([[0.01, 0.02], [0.02, 0.03]]),
        targets=np.asarray([[0.015, 0.025], [0.025, 0.035]]),
        labels=(60, 120),
        metadata=_metadata(),
        model_name="cnn1",
        model_version="model-v1",
        dataset_version="dataset-v1",
        fold_index=2,
        split="test",
    )


def test_prediction_artifact_freezes_aligned_matrices() -> None:
    artifact = _artifact()
    assert artifact.predictions.shape == (2, 2)
    assert artifact.targets.shape == (2, 2)
    assert artifact.labels == (60, 120)
    assert not artifact.predictions.flags.writeable
    assert not artifact.targets.flags.writeable


def test_save_and_load_prediction_artifact_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "fold-2" / "predictions.parquet"
    output = save_prediction_artifact(artifact=_artifact(), path=str(path))
    assert Path(output).is_file()
    columns = pl.read_parquet(path).columns
    assert "target_60s" in columns
    assert "prediction_120s" in columns
    assert not any(column.startswith("target_valid_") for column in columns)
    assert "future_mid" not in columns
    loaded = load_prediction_artifact(str(path))
    np.testing.assert_allclose(loaded.predictions, _artifact().predictions)
    np.testing.assert_allclose(loaded.targets, _artifact().targets)
    assert loaded.labels == (60, 120)


def test_prediction_artifact_rejects_non_finite_targets() -> None:
    artifact = _artifact()
    with pytest.raises(ValueError, match="targets.*finite"):
        PredictionArtifact(
            predictions=artifact.predictions,
            targets=np.asarray([[0.015, np.nan], [0.025, 0.035]]),
            labels=artifact.labels,
            metadata=artifact.metadata,
            model_name=artifact.model_name,
            model_version=artifact.model_version,
            dataset_version=artifact.dataset_version,
            fold_index=artifact.fold_index,
            split=artifact.split,
        )


def test_save_requires_parquet_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".parquet"):
        save_prediction_artifact(artifact=_artifact(), path=str(tmp_path / "predictions.pt"))
