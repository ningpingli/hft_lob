from __future__ import annotations

import os
from pathlib import Path

import yaml

from hft_lob.utils.checkpoint_utils import backup_experiment_config, resolve_ckpt_path


def test_resolve_ckpt_path_prefers_requested_file(tmp_path: Path) -> None:
    requested = tmp_path / "best_val_model.ckpt"
    requested.write_bytes(b"best")
    newer = tmp_path / "newer.ckpt"
    newer.write_bytes(b"newer")
    os.utime(newer, ns=(requested.stat().st_mtime_ns + 1, requested.stat().st_mtime_ns + 1))

    assert resolve_ckpt_path(str(tmp_path)) == str(requested)


def test_resolve_ckpt_path_falls_back_to_latest_checkpoint(tmp_path: Path) -> None:
    older = tmp_path / "older.ckpt"
    latest = tmp_path / "latest.ckpt"
    older.write_bytes(b"older")
    latest.write_bytes(b"latest")
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(latest, ns=(2_000_000_000, 2_000_000_000))
    (tmp_path / "ignored.txt").write_text("not a checkpoint", encoding="utf-8")

    assert resolve_ckpt_path(str(tmp_path), filename="missing.ckpt") == str(latest)


def test_resolve_ckpt_path_can_disable_fallback(tmp_path: Path) -> None:
    (tmp_path / "available.ckpt").write_bytes(b"checkpoint")

    assert (
        resolve_ckpt_path(str(tmp_path), filename="missing.ckpt", fallback_to_latest=False) is None
    )


def test_resolve_ckpt_path_returns_none_for_missing_directory(tmp_path: Path) -> None:
    assert resolve_ckpt_path(str(tmp_path / "missing")) is None


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
