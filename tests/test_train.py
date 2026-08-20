from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from hft_lob import train
from hft_lob.configs.experiment import EvaluationConfig
from hft_lob.datasets.lob_dataset import SampleMeta
from hft_lob.systems.artifact import PredictionArtifact


class FakeTrainer:
    def __init__(self, *, predictions: list[PredictionArtifact] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.predictions = predictions or []

    def fit(self, **kwargs: Any) -> None:
        self.calls.append(("fit", kwargs))

    def test(self, **kwargs: Any) -> None:
        self.calls.append(("test", kwargs))

    def predict(self, **kwargs: Any) -> list[PredictionArtifact]:
        self.calls.append(("predict", kwargs))
        return self.predictions


def _artifact(*, offset: int = 0, split: str = "test") -> PredictionArtifact:
    metadata = tuple(
        SampleMeta(
            ticker="TEST",
            trade_date="2026-01-05",
            session_id="AM",
            anchor_timestamp=(
                datetime(2026, 1, 5, 9, 30) + timedelta(seconds=offset + index)
            ).isoformat(),
            mid_t=10.0,
            future_mid=10.1,
            bid1=9.9,
            ask1=10.1,
            spread=0.2,
        )
        for index in range(2)
    )
    return PredictionArtifact(
        predictions=np.asarray([0.1 + offset, 0.2 + offset]),
        targets=np.asarray([0.2 + offset, 0.3 + offset]),
        metadata=metadata,
        model_name="cnn1",
        model_version="v1",
        dataset_version="data-v1",
        fold_index=1,
        split=split,
    )


def test_callback_factories(tmp_path: Path) -> None:
    checkpoint = train.build_checkpoint_callback(
        str(tmp_path), monitor="val/ts_ic", mode="max"
    )
    early_stopping = train.build_early_stopping_callback(
        monitor="val/ts_ic", mode="max", patience=3
    )

    assert isinstance(checkpoint, ModelCheckpoint)
    assert checkpoint.monitor == "val/ts_ic"
    assert checkpoint.dirpath == str(tmp_path.resolve())
    assert isinstance(early_stopping, EarlyStopping)
    assert early_stopping.patience == 3


def test_build_trainer_adds_default_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(train.L, "Trainer", lambda **kwargs: captured.update(kwargs) or kwargs)

    result = train.build_trainer(str(tmp_path), epochs=2, patience=4)

    assert result == captured
    assert captured["max_epochs"] == 2
    assert captured["logger"] is False
    assert sum(isinstance(item, ModelCheckpoint) for item in captured["callbacks"]) == 1
    assert sum(isinstance(item, EarlyStopping) for item in captured["callbacks"]) == 1


def test_run_training_delegates_to_lightning() -> None:
    trainer = FakeTrainer()
    module = SimpleNamespace()
    datamodule = SimpleNamespace()

    train.run_training(trainer, module, datamodule, ckpt_path="resume.ckpt")  # type: ignore[arg-type]

    assert trainer.calls == [
        (
            "fit",
            {"model": module, "datamodule": datamodule, "ckpt_path": "resume.ckpt"},
        )
    ]


def test_run_test_builds_evaluation_report() -> None:
    trainer = FakeTrainer()
    module = SimpleNamespace(
        test_artifact=_artifact(),
        config=SimpleNamespace(
            evaluation=EvaluationConfig(
                metrics=("mae",),
                report_daily=False,
                prediction_bins=2,
                bootstrap_samples=2,
            ),
            seed=7,
        ),
    )

    report = train.run_test(trainer, module, SimpleNamespace(), "best.ckpt")  # type: ignore[arg-type]

    assert report.sample_count == 2
    assert report.overall["mae"] == pytest.approx(0.1)
    assert trainer.calls[0][0] == "test"


def test_run_predict_merges_batch_artifacts() -> None:
    trainer = FakeTrainer(predictions=[_artifact(offset=0), _artifact(offset=2)])
    module = SimpleNamespace(prediction_split="validation")

    artifact = train.run_predict(
        trainer, module, SimpleNamespace(), "best.ckpt", split="test"  # type: ignore[arg-type]
    )

    assert artifact.predictions.shape == (4,)
    assert len(artifact.metadata) == 4
    assert module.prediction_split == "test"


def test_run_predict_rejects_empty_outputs() -> None:
    with pytest.raises(RuntimeError, match="PredictionArtifact"):
        train.run_predict(
            FakeTrainer(),
            SimpleNamespace(prediction_split="test"),
            SimpleNamespace(),
            "best.ckpt",  # type: ignore[arg-type]
        )
