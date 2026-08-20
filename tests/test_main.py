from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
import yaml

from hft_lob.configs.experiment import RAW_FEATURE_COLUMNS
from hft_lob.main import main, parse_args


def test_parse_prepare_data_cli() -> None:
    args = parse_args(
        [
            "--config",
            "custom.yaml",
            "--experiment-id",
            "prepare-1",
            "--stages",
            "prepare-data",
            "--seed",
            "7",
            "--gpu-id",
            "2",
        ]
    )

    assert args.config == "custom.yaml"
    assert args.experiment_id == "prepare-1"
    assert args.stages == ["prepare-data"]
    assert args.seed == 7
    assert args.gpu_id == 2


def test_parse_rejects_unknown_stage() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--stages", "random-split"])


def test_prepare_data_cli_builds_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw" / "TEST"
    raw_root.mkdir(parents=True)
    for day in range(1, 5):
        start = datetime(2026, 1, day, 9, 30)
        rows = [_raw_row(start, day), _raw_row(start + timedelta(seconds=63), day + 1)]
        pl.DataFrame(rows).select("timestamp", *RAW_FEATURE_COLUMNS).write_parquet(
            raw_root / f"202601{day:02d}.parquet"
        )

    source_config = Path(__file__).parents[1] / "configs" / "experiment.yaml"
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    config["task"]["ticker"] = "TEST"
    config["data"].update(
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "processed"),
        manifest_dir=str(tmp_path / "datasets"),
        column_mapping={},
    )
    config["window"]["history_snapshots"] = 2
    config["normalization"]["normalize_window"] = 2
    config["walk_forward"].update(
        train_window_days=2,
        validation_window_days=1,
        test_window_days=1,
        step_days=1,
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hft_lob",
            "--config",
            str(config_path),
            "--experiment-id",
            "cli-integration",
            "--stages",
            "prepare-data",
        ],
    )
    main()

    manifests = list((tmp_path / "datasets" / "TEST").glob("*/manifest.parquet"))
    assert len(manifests) == 1
    assert pl.read_parquet(manifests[0]).height == 4
    assert (tmp_path / "loggers/results/cli-integration/config_used.yaml").is_file()
    assert (tmp_path / "loggers/results/cli-integration/data.yaml").is_file()


def _raw_row(timestamp: datetime, offset: int) -> dict[str, object]:
    row: dict[str, object] = {"timestamp": timestamp}
    for level in range(1, 6):
        row[f"ASKp{level}"] = 10.01 + offset * 0.001 + level * 0.01
        row[f"ASKs{level}"] = 100.0
        row[f"BIDp{level}"] = 9.99 + offset * 0.001 - level * 0.01
        row[f"BIDs{level}"] = 100.0
    row.update(last=10.0, volume=1_000.0, amount=10_000.0)
    return row
