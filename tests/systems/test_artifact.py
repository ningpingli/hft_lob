from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hft_lob.systems.artifact import (
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
            future_mid=10.1 + index * 0.01,
            bid1=9.99 + index * 0.01,
            ask1=10.01 + index * 0.01,
            spread=0.02,
        )
        for index in range(2)
    )


def _artifact() -> PredictionArtifact:
    return PredictionArtifact(
        predictions=np.asarray([[0.01, 0.02], [0.02, 0.03]], dtype=np.float32),
        targets=np.asarray([[0.015, 0.025], [0.025, 0.035]]),
        target_valid=np.asarray([[True, True], [True, False]]),
        labels=(60, 120),
        metadata=_metadata(),
        model_name="cnn1",
        model_version="model-v1",
        dataset_version="dataset-v1",
        fold_index=2,
        split="test",
    )


def test_prediction_artifact_normalizes_matrices_and_freezes_arrays() -> None:
    artifact = _artifact()

    assert artifact.predictions.shape == (2, 2)
    assert artifact.targets.shape == (2, 2)
    assert artifact.predictions.dtype == np.float64
    assert not artifact.predictions.flags.writeable
    assert not artifact.target_valid.flags.writeable
    with pytest.raises(ValueError):
        artifact.predictions[0, 0] = 99.0


def test_save_prediction_artifact_writes_typed_label_columns(tmp_path: Path) -> None:
    path = tmp_path / "fold-2" / "predictions.parquet"

    output = save_prediction_artifact(artifact=_artifact(), path=str(path))
    frame = pl.read_parquet(output)

    assert Path(output) == path.resolve()
    assert frame.height == 2
    assert frame.columns[-6:] == [
        "target_60s",
        "prediction_60s",
        "target_valid_60s",
        "target_120s",
        "prediction_120s",
        "target_valid_120s",
    ]
    assert frame.schema["anchor_timestamp"] == pl.Datetime("us")
    assert frame.get_column("prediction_120s").to_list() == pytest.approx([0.02, 0.03])
    assert frame.get_column("target_valid_120s").to_list() == [True, False]


def test_prediction_artifact_round_trips_complete_target_matrix(tmp_path: Path) -> None:
    artifact = _artifact()
    path = tmp_path / "fold-2" / "predictions.parquet"

    save_prediction_artifact(artifact=artifact, path=str(path))
    loaded = load_prediction_artifact(str(path))

    assert loaded.labels == (60, 120)
    np.testing.assert_allclose(loaded.predictions, artifact.predictions)
    np.testing.assert_allclose(loaded.targets, artifact.targets, equal_nan=True)
    np.testing.assert_array_equal(loaded.target_valid, artifact.target_valid)


def test_prediction_artifact_rejects_mismatched_or_duplicate_samples() -> None:
    with pytest.raises(ValueError, match="same shape"):
        PredictionArtifact(
            predictions=np.asarray([[0.1]]),
            targets=np.asarray([[0.1, 0.2]]),
            target_valid=np.asarray([[True]]),
            labels=(60,),
            metadata=_metadata()[:1],
            model_name="cnn1",
            model_version="v1",
            dataset_version="d1",
            fold_index=1,
            split="test",
        )
    duplicate = (_metadata()[0], _metadata()[0])
    with pytest.raises(ValueError, match="duplicate prediction sample"):
        PredictionArtifact(
            predictions=np.asarray([[0.1], [0.2]]),
            targets=np.asarray([[0.1], [0.2]]),
            target_valid=np.asarray([[True], [True]]),
            labels=(60,),
            metadata=duplicate,
            model_name="cnn1",
            model_version="v1",
            dataset_version="d1",
            fold_index=1,
            split="test",
        )


def test_save_requires_parquet_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".parquet"):
        save_prediction_artifact(artifact=_artifact(), path=str(tmp_path / "predictions.pt"))
