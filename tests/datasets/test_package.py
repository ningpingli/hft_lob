from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
import torch

from hft_lob.datasets.package import (
    FOLD_INDEX_SCHEMA,
    DatasetPackageMetadata,
    compute_dataset_id,
)
from hft_lob.datasets.validation import validate_dataset_package


def _metadata() -> DatasetPackageMetadata:
    identity = {
        "ticker": "688981",
        "source_hash": "source",
        "processing_config_hash": "processing",
        "fold_plan_hash": "folds",
    }
    return DatasetPackageMetadata(
        dataset_id=compute_dataset_id(**identity),
        feature_columns=("ASKp1", "BIDp1"),
        target_column="Target_60s_log",
        feature_dtype="float32",
        target_dtype="float32",
        snapshot_interval_seconds=3,
        history_snapshots=3,
        normalization_mode="causal_rolling",
        normalization_window=3,
        **identity,
    )


def _write_package(tmp_path: Path) -> Path:
    metadata = _metadata()
    root = tmp_path / metadata.dataset_id
    session = root / "sessions" / "2020-07-16_AM.pt"
    session.parent.mkdir(parents=True)
    torch.save(
        {
            "features": torch.ones(4, 2),
            "targets": torch.ones(4, 1),
            "row_valid": torch.ones(4, dtype=torch.bool),
            "target_valid": torch.ones(4, dtype=torch.bool),
            "timestamps": [
                "2020-07-16T09:30:00",
                "2020-07-16T09:30:03",
                "2020-07-16T09:30:06",
                "2020-07-16T09:30:09",
            ],
            "mid_price": torch.ones(4),
            "future_mid": torch.ones(4),
            "bid1": torch.ones(4),
            "ask1": torch.ones(4),
            "trade_date": "2020-07-16",
            "session_id": "AM",
        },
        session,
    )
    fold_dir = root / "folds" / "fold_001"
    fold_dir.mkdir(parents=True)
    frame = pl.DataFrame(
        {
            "session_file": ["sessions/2020-07-16_AM.pt"],
            "anchor_index": [2],
            "trade_date": ["2020-07-16"],
            "session_id": ["AM"],
            "anchor_timestamp": [datetime(2020, 7, 16, 9, 30, 6)],
        },
        schema=FOLD_INDEX_SCHEMA,
    )
    for split in ("train", "validation", "test"):
        frame.write_parquet(fold_dir / f"{split}.parquet")
    (root / "dataset.json").write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False), encoding="utf-8"
    )
    pl.DataFrame({"status": ["passed"]}).write_parquet(root / "quality.parquet")
    (root / "_SUCCESS").touch()
    return root


def test_dataset_id_is_stable_and_tracks_identity() -> None:
    first = _metadata()
    assert DatasetPackageMetadata.from_dict(first.to_dict()) == first
    assert first.dataset_id != compute_dataset_id(
        ticker=first.ticker,
        source_hash=first.source_hash,
        processing_config_hash="changed",
        fold_plan_hash=first.fold_plan_hash,
    )


def test_validate_complete_dataset_package(tmp_path: Path) -> None:
    root = _write_package(tmp_path)

    assert validate_dataset_package(root) == _metadata()


def test_validation_rejects_unpublished_or_missing_session(tmp_path: Path) -> None:
    root = _write_package(tmp_path)
    (root / "_SUCCESS").unlink()
    with pytest.raises(ValueError, match="not published"):
        validate_dataset_package(root)

    (root / "_SUCCESS").touch()
    (root / "sessions" / "2020-07-16_AM.pt").unlink()
    with pytest.raises(ValueError, match="missing session"):
        validate_dataset_package(root)


def test_metadata_rejects_identity_mismatch() -> None:
    value = _metadata().to_dict()
    value["dataset_id"] = "wrong"

    with pytest.raises(ValueError, match="does not match"):
        DatasetPackageMetadata.from_dict(value)
