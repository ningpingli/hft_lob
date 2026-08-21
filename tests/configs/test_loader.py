from __future__ import annotations

from pathlib import Path

import pytest

from hft_lob.configs.loader import load_data_config, load_model_config


def test_loads_two_stage_repository_configs() -> None:
    data = load_data_config("configs/data.yaml")
    model = load_model_config("configs/model.yaml", experiment_id="loader-test")

    assert data.ticker == "000001"
    assert data.sessions.morning == ("09:30:00", "11:30:00")
    assert model.experiment_id == "loader-test"
    assert model.training.betas == (0.9, 0.95)
    assert model.baselines.names == ("zero", "imbalance", "ridge")


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
