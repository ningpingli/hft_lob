from __future__ import annotations

from pathlib import Path

import yaml

from hft_lob.utils.checkpoint_utils import backup_experiment_config


def test_backup_experiment_config_creates_directory_and_yaml(tmp_path: Path) -> None:
    log_dir = tmp_path / "nested" / "experiment"
    config = {
        "model": {"name": "DeepLOB", "layers": [16, 32]},
        "training": {"learning_rate": 0.001},
        "description": "高频实验",
    }

    result = backup_experiment_config(str(log_dir), config)

    destination = log_dir / "config_used.yaml"
    assert result == str(destination)
    assert yaml.safe_load(destination.read_text(encoding="utf-8")) == config
    assert list(log_dir.glob(".config_used.yaml.*.tmp")) == []


def test_backup_experiment_config_overwrites_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "custom.yaml"
    destination.write_text("stale: true\n", encoding="utf-8")

    backup_experiment_config(str(tmp_path), {"stale": False}, filename="custom.yaml")

    assert yaml.safe_load(destination.read_text(encoding="utf-8")) == {"stale": False}
