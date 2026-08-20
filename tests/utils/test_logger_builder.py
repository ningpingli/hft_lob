from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hft_lob.utils import logger_builder
from hft_lob.utils.logger_builder import build_logger


class FakeLogger:
    def __init__(self, *, fail_hyperparams: bool = False) -> None:
        self.fail_hyperparams = fail_hyperparams
        self.hyperparams: dict[str, Any] | None = None

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        if self.fail_hyperparams:
            raise RuntimeError("cannot log hyperparameters")
        self.hyperparams = params


def test_build_logger_uses_tensorboard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tensorboard = FakeLogger()
    captured: dict[str, str] = {}

    def fake_tensorboard(**kwargs: str) -> FakeLogger:
        captured.update(kwargs)
        return tensorboard

    monkeypatch.setattr(logger_builder, "_build_tensorboard_logger", fake_tensorboard)

    result = build_logger(
        "run-1",
        str(tmp_path),
        hyperparams={"seed": 42},
    )

    assert result is tensorboard
    assert captured == {"experiment_id": "run-1", "log_dir": str(tmp_path)}
    assert tensorboard.hyperparams == {"seed": 42}


def test_build_logger_creates_log_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_dir = tmp_path / "nested" / "logs"
    monkeypatch.setattr(logger_builder, "_build_tensorboard_logger", lambda **kwargs: FakeLogger())

    build_logger("run-1", str(log_dir))

    assert log_dir.is_dir()


def test_build_logger_returns_none_when_tensorboard_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        logger_builder,
        "_build_tensorboard_logger",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("tensorboard unavailable")),
    )

    with pytest.warns(RuntimeWarning, match="TensorBoard"):
        result = build_logger("run-1", str(tmp_path))

    assert result is None


def test_hyperparameter_failure_does_not_discard_logger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = FakeLogger(fail_hyperparams=True)
    monkeypatch.setattr(logger_builder, "_build_tensorboard_logger", lambda **kwargs: logger)

    with pytest.warns(RuntimeWarning, match="hyperparameter logging"):
        result = build_logger("run-1", str(tmp_path), hyperparams={"seed": 42})

    assert result is logger


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("experiment_id", {"experiment_id": ""}),
        ("log_dir", {"log_dir": " "}),
    ],
)
def test_build_logger_rejects_empty_required_values(
    tmp_path: Path, field: str, kwargs: dict[str, str]
) -> None:
    arguments = {
        "experiment_id": "run-1",
        "log_dir": str(tmp_path),
        **kwargs,
    }
    with pytest.raises(ValueError, match=field):
        build_logger(**arguments)
