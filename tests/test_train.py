from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from hft_lob.reporting.artifact import PredictionArtifact
from hft_lob.systems import executor as train
from hft_lob.systems.contracts import SampleMeta


class FakeTrainer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def fit(self, **kwargs: Any) -> None:
        self.calls.append(("fit", kwargs))

    def test(self, **kwargs: Any) -> None:
        self.calls.append(("test", kwargs))


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
    checkpoint = train.build_checkpoint_callback(str(tmp_path), monitor="val/mse", mode="min")
    early_stopping = train.build_early_stopping_callback(
        monitor="val/mse", mode="min", patience=3
    )

    assert isinstance(checkpoint, ModelCheckpoint)
    assert checkpoint.monitor == "val/mse"
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




def test_run_test_returns_module_artifact() -> None:
    trainer = FakeTrainer()
    module = SimpleNamespace(test_artifact=_artifact())

    artifact = train.run_test(
        trainer,
        module,
        SimpleNamespace(),
        "best.ckpt",  # type: ignore[arg-type]
    )

    assert artifact.predictions.shape == (2,)
    assert trainer.calls[0][0] == "test"


def test_run_test_rejects_missing_artifact() -> None:
    with pytest.raises(RuntimeError, match="PredictionArtifact"):
        train.run_test(
            FakeTrainer(),
            SimpleNamespace(test_artifact=None),
            SimpleNamespace(),
            "best.ckpt",  # type: ignore[arg-type]
        )
