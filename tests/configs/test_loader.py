from __future__ import annotations

from pathlib import Path

import pytest

from hft_lob.configs.loader import load_baseline_config, load_data_config, load_model_config


def test_loads_two_stage_repository_configs() -> None:
    data = load_data_config("configs/dataset.yaml")
    model = load_model_config("configs/train.yaml", experiment_id="loader-test")

    assert data.target.labels == [60]
    assert data.sessions.morning == ("09:30:00", "11:30:00") and data.sessions.afternoon == ("13:00:00", "14:57:00")
    assert model.experiment_id == "loader-test"
    assert model.training.betas == (0.9, 0.95)
    assert model.training.learning_rate == pytest.approx(3e-4)
    assert model.training.min_learning_rate == pytest.approx(1e-5)
    assert model.training.scheduler == "cosine"
    assert model.training.warmup_ratio == pytest.approx(0.1)
    assert model.training.gradient_clip_val == pytest.approx(1.0)
    baseline = load_baseline_config("configs/baselines.yaml", experiment_id="baseline-test")
    assert baseline.baselines.names == ("ridge",)


def test_rejects_unknown_config_fields(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("task:\n  ticker: TEST\nunknown: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown config sections"):
        load_data_config(str(path))


def test_rejects_unknown_nested_fields(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("task:\n  ticker: TEST\n  typo: value\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid config field"):
        load_data_config(str(path))


def test_rejects_invalid_training_stability_fields(tmp_path: Path) -> None:
    path = tmp_path / "invalid-training.yaml"
    path.write_text(
        Path("configs/train.yaml")
        .read_text(encoding="utf-8")
        .replace("scheduler: cosine", "scheduler: linear"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="training.scheduler must be 'cosine'"):
        load_model_config(str(path), experiment_id="invalid-training")
