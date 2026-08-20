from __future__ import annotations

from pathlib import Path

import pytest

from hft_lob.configs.loader import load_config


def test_loads_repository_config_into_typed_contract() -> None:
    config = load_config("configs/experiment.yaml", experiment_id="loader-test")

    assert config.experiment_id == "loader-test"
    assert config.ticker == "000001"
    assert config.sessions.morning == ("09:30:00", "11:30:00")
    assert config.training.betas == (0.9, 0.95)
    assert config.baselines.names == ("zero", "imbalance", "ridge", "mlp")


def test_rejects_unknown_config_fields(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("task:\n  ticker: TEST\nunknown: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown config sections"):
        load_config(str(path), experiment_id="test")


def test_rejects_unknown_nested_fields(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("task:\n  ticker: TEST\n  typo: value\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid config field"):
        load_config(str(path), experiment_id="test")
