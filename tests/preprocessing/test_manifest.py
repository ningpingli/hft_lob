from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from hft_lob.configs.experiment import FeatureConfig
from hft_lob.preprocessing.manifest import (
    build_manifest,
    dataset_version,
    feature_version,
    raw_file_hash,
    read_manifest,
    stable_config_hash,
    write_manifest,
)


def _record(*, session_id: str = "AM") -> dict[str, object]:
    return {
        "trade_date": "2026-01-05",
        "session_id": session_id,
        "source_file": "raw/20260105.parquet",
        "processed_file": f"processed/20260105_{session_id}.parquet",
        "raw_hash": "a" * 64,
        "processing_config_hash": "b" * 64,
        "dataset_version": "c" * 64,
        "row_count": 100,
        "valid_row_count": 90,
        "data_start": datetime(2026, 1, 5, 9, 30),
        "data_end": datetime(2026, 1, 5, 11, 29, 57),
        "feature_version": "d" * 64,
        "label_version": "log_mid_return_60s_tol3",
        "quality_status": "passed",
    }


def test_hashes_are_content_addressed_and_order_stable(tmp_path: Path) -> None:
    source = tmp_path / "raw.bin"
    source.write_bytes(b"lob-data")

    assert raw_file_hash(str(source)) == hashlib.sha256(b"lob-data").hexdigest()
    assert stable_config_hash({"b": 2, "a": {"y": 1, "x": 0}}) == stable_config_hash(
        {"a": {"x": 0, "y": 1}, "b": 2}
    )
    assert dataset_version(
        "000001.SZ", ["raw-b", "raw-a"], processing_config_hash="config"
    ) == dataset_version(
        "000001.SZ", ["raw-a", "raw-b"], processing_config_hash="config"
    )


def test_feature_version_tracks_enabled_feature_order() -> None:
    first = feature_version(
        FeatureConfig(use_derived=True, derived_features=("spread", "mid_price"))
    )
    reordered = feature_version(
        FeatureConfig(use_derived=True, derived_features=("mid_price", "spread"))
    )
    disabled = feature_version(
        FeatureConfig(use_derived=False, derived_features=("spread", "mid_price"))
    )

    assert first != reordered
    assert first != disabled


def test_manifest_round_trip_preserves_fixed_schema(tmp_path: Path) -> None:
    manifest = build_manifest(
        ticker="000001.SZ", records=[_record(session_id="PM"), _record(session_id="AM")]
    )
    path = tmp_path / "manifest.parquet"
    write_manifest(manifest, str(path))
    restored = read_manifest(str(path))

    assert restored.equals(manifest)
    assert restored.get_column("session_id").to_list() == ["AM", "PM"]
    assert restored.get_column("ticker").unique().to_list() == ["000001.SZ"]
    assert build_manifest(ticker="000001.SZ", records=[]).height == 0
    assert build_manifest(ticker="000001.SZ", records=[]).schema == manifest.schema


def test_manifest_rejects_invalid_counts() -> None:
    record = _record()
    record["valid_row_count"] = 101

    with pytest.raises(ValueError, match="valid_row_count"):
        build_manifest(ticker="000001.SZ", records=[record])


def test_read_manifest_rejects_uncontracted_schema(tmp_path: Path) -> None:
    path = tmp_path / "invalid.parquet"
    pl.DataFrame({"ticker": ["000001.SZ"]}).write_parquet(path)

    with pytest.raises(ValueError, match="invalid manifest columns"):
        read_manifest(str(path))
