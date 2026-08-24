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
        predictions=np.asarray([[0.01], [0.02]], dtype=np.float32),
        targets=np.asarray([0.015, 0.025]),
        metadata=_metadata(),
        model_name="cnn1",
        model_version="model-v1",
        dataset_version="dataset-v1",
        fold_index=2,
        split="test",
    )


def test_prediction_artifact_normalizes_vectors_and_freezes_arrays() -> None:
    artifact = _artifact()

    assert artifact.predictions.shape == (2,)
    assert artifact.targets.shape == (2,)
    assert artifact.predictions.dtype == np.float64
    assert not artifact.predictions.flags.writeable
    with pytest.raises(ValueError):
        artifact.predictions[0] = 99.0


def test_save_prediction_artifact_writes_typed_complete_parquet(tmp_path: Path) -> None:
    path = tmp_path / "fold-2" / "predictions.parquet"

    output = save_prediction_artifact(artifact=_artifact(), path=str(path))
    frame = pl.read_parquet(output)

    assert Path(output) == path.resolve()
    assert frame.height == 2
    assert frame.columns == [
        "model_name",
        "model_version",
        "dataset_version",
        "fold_index",
        "split",
        "ticker",
        "trade_date",
        "session_id",
        "anchor_timestamp",
        "mid_t",
        "future_mid",
        "target",
        "prediction",
        "bid1",
        "ask1",
        "spread",
    ]
    assert frame.schema["anchor_timestamp"] == pl.Datetime("us")
    assert frame.get_column("prediction").to_list() == pytest.approx([0.01, 0.02])
    assert frame.get_column("fold_index").unique().to_list() == [2]


def test_load_prediction_artifact_round_trips_saved_artifact(tmp_path: Path) -> None:
    path = tmp_path / "fold-2" / "predictions.parquet"
    save_prediction_artifact(artifact=_artifact(), path=str(path))

    loaded = load_prediction_artifact(str(path))

    assert loaded.model_name == "cnn1"
    assert loaded.model_version == "model-v1"
    assert loaded.fold_index == 2
    assert loaded.predictions.tolist() == pytest.approx([0.01, 0.02])
    assert loaded.metadata == _metadata()


def test_prediction_artifact_rejects_mismatched_or_duplicate_samples() -> None:
    with pytest.raises(ValueError, match="same sample count"):
        PredictionArtifact(
            predictions=np.asarray([0.1]),
            targets=np.asarray([0.1, 0.2]),
            metadata=_metadata(),
            model_name="cnn1",
            model_version="v1",
            dataset_version="d1",
            fold_index=1,
            split="test",
        )
    duplicate = (_metadata()[0], _metadata()[0])
    with pytest.raises(ValueError, match="duplicate prediction sample"):
        PredictionArtifact(
            predictions=np.asarray([0.1, 0.2]),
            targets=np.asarray([0.1, 0.2]),
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
