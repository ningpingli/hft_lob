"""从现有数据工程产物构建连续、可 memory-map 的训练数据包。"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from hft_lob.configs.experiment import ExperimentConfig
from hft_lob.datasets.package import FOLD_INDEX_COLUMNS, DatasetPackageMetadata, compute_dataset_id
from hft_lob.datasets.validation import validate_dataset_package
from hft_lob.preprocessing.manifest import read_manifest, stable_config_hash
from hft_lob.preprocessing.normalize import CausalRollingStandardizer
from hft_lob.preprocessing.pipeline import PreparedDataset, prepare_dataset


def build_dataset_package(config: ExperimentConfig, output_root: str | Path) -> Path:
    """构建并原子发布数据包；相同数据包已存在时直接复用。"""
    prepared = prepare_dataset(config)
    manifest = read_manifest(prepared.manifest_path)
    metadata = _metadata(config, prepared, manifest)
    root = Path(output_root).resolve()
    destination = root / metadata.dataset_id
    if destination.exists():
        validate_dataset_package(destination)
        return destination

    build_root = root / f".building-{uuid.uuid4().hex}"
    package_root = build_root / metadata.dataset_id
    try:
        _build_contents(package_root, config, prepared, manifest, metadata)
        validate_dataset_package(package_root)
        root.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(package_root, destination)
        except FileExistsError:
            validate_dataset_package(destination)
        return destination
    finally:
        shutil.rmtree(build_root, ignore_errors=True)


def _metadata(config: ExperimentConfig, prepared: PreparedDataset, manifest: pl.DataFrame) -> DatasetPackageMetadata:
    processing_hashes = manifest.get_column("processing_config_hash").unique().to_list()
    if len(processing_hashes) != 1:
        raise ValueError("manifest must contain one processing_config_hash")
    source_hash = stable_config_hash({"raw_hashes": sorted(manifest.get_column("raw_hash").unique().to_list())})
    fold_plan_hash = stable_config_hash({"folds": [asdict(fold) for fold in prepared.walk_forward_plan.folds]})
    identity = {"ticker": config.ticker, "source_hash": source_hash, "processing_config_hash": processing_hashes[0], "fold_plan_hash": fold_plan_hash}
    return DatasetPackageMetadata(
        dataset_id=compute_dataset_id(**identity),
        feature_columns=prepared.feature_columns,
        target_column=config.target_column,
        feature_dtype="float32",
        target_dtype="float32",
        snapshot_interval_seconds=config.data.snapshot_interval_seconds,
        history_snapshots=config.window.history_snapshots,
        normalization_mode=config.normalization.mode,
        normalization_window=config.normalization.normalize_window,
        **identity,
    )


def _build_contents(root: Path, config: ExperimentConfig, prepared: PreparedDataset, manifest: pl.DataFrame, metadata: DatasetPackageMetadata) -> None:
    root.mkdir(parents=True)
    row_count = int(manifest.get_column("row_count").sum())
    features = np.lib.format.open_memmap(root / "features.npy", mode="w+", dtype=np.float32, shape=(row_count, len(metadata.feature_columns)))
    targets = np.lib.format.open_memmap(root / "targets.npy", mode="w+", dtype=np.float32, shape=(row_count, 1))
    validity = np.lib.format.open_memmap(root / "validity.npy", mode="w+", dtype=np.bool_, shape=(row_count, 2))
    market = np.lib.format.open_memmap(root / "market.npy", mode="w+", dtype=np.float32, shape=(row_count, 4))
    standardizer = CausalRollingStandardizer(prepared.feature_columns, config.normalization.normalize_window)
    row_writer: pq.ParquetWriter | None = None
    anchor_writer: pq.ParquetWriter | None = None
    offset = 0
    try:
        for record in manifest.iter_rows(named=True):
            frame = standardizer.transform_frame(pl.read_parquet(record["processed_file"]))
            end = offset + frame.height
            if end > row_count:
                raise ValueError("manifest row_count is smaller than processed data")
            output_columns = [f"normalized__{name}" for name in metadata.feature_columns]
            row_valid = _row_valid(frame, output_columns)
            features[offset:end] = frame.select(output_columns).to_numpy().astype(np.float32)
            targets[offset:end] = frame.select(metadata.target_column).to_numpy().astype(np.float32)
            validity[offset:end, 0] = row_valid
            validity[offset:end, 1] = np.asarray(frame.get_column("target_valid").fill_null(False), dtype=np.bool_)
            market[offset:end] = frame.select("mid_price", "future_mid", "BIDp1", "ASKp1").to_numpy().astype(np.float32)

            rows = frame.select("trade_date", "session_id", pl.col("timestamp")).with_columns(
                pl.int_range(offset, end, dtype=pl.Int64).alias("global_index")
            ).select("global_index", "trade_date", "session_id", "timestamp")
            anchors = _anchor_frame(frame, row_valid, offset, metadata.history_snapshots, metadata.target_column)
            row_writer = _write_chunk(root / "rows.parquet", rows, row_writer)
            if not anchors.is_empty():
                anchor_writer = _write_chunk(root / "anchors.parquet", anchors, anchor_writer)
            offset = end
    finally:
        if row_writer is not None:
            row_writer.close()
        if anchor_writer is not None:
            anchor_writer.close()
        del features, targets, validity, market
    if offset != row_count or anchor_writer is None:
        raise ValueError("processed data row count mismatch or no valid anchors")

    lazy_anchors = pl.scan_parquet(root / "anchors.parquet")
    for fold in prepared.walk_forward_plan.folds:
        for split, dates in (("train", fold.train_dates), ("validation", fold.validation_dates), ("test", fold.test_dates)):
            path = root / "folds" / f"fold_{fold.index:03d}" / f"{split}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            lazy_anchors.filter(pl.col("trade_date").is_in(dates)).select(FOLD_INDEX_COLUMNS).sink_parquet(path)

    (root / "dataset.json").write_text(json.dumps(metadata.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(prepared.quality_report_path, root / "quality.parquet")
    (root / "_SUCCESS").touch()


def _write_chunk(path: Path, frame: pl.DataFrame, writer: pq.ParquetWriter | None) -> pq.ParquetWriter:
    table = frame.to_arrow()
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema)
    writer.write_table(table)
    return writer


def _row_valid(frame: pl.DataFrame, feature_columns: list[str]) -> np.ndarray:
    expression = pl.col("book_valid").fill_null(False) & pl.col("feature_valid").fill_null(False) & pl.col("normalization_valid").fill_null(False) & pl.all_horizontal(
        pl.col(name).is_not_null() & pl.col(name).is_finite() for name in feature_columns
    )
    return np.asarray(frame.select(expression.alias("valid")).get_column("valid"), dtype=np.bool_)


def _anchor_frame(frame: pl.DataFrame, row_valid: np.ndarray, session_start: int, history_snapshots: int, target_column: str) -> pl.DataFrame:
    target_valid = np.asarray(
        frame.select((pl.col("target_valid").fill_null(False) & pl.col(target_column).is_not_null() & pl.col(target_column).is_finite() & pl.col("future_mid").is_not_null() & pl.col("future_mid").is_finite()).alias("valid")).get_column("valid"),
        dtype=np.bool_,
    )
    prefix = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(row_valid)))
    local = np.arange(history_snapshots - 1, frame.height, dtype=np.int64)
    starts = local - history_snapshots + 1
    keep = (prefix[local + 1] - prefix[starts] == history_snapshots) & target_valid[local]
    local = local[keep]
    return pl.DataFrame(
        {
            "global_anchor_index": session_start + local,
            "session_start_index": np.full(local.size, session_start, dtype=np.int64),
            "anchor_index": local,
            "trade_date": [str(frame.get_column("trade_date").item(0))] * local.size,
            "session_id": [str(frame.get_column("session_id").item(0))] * local.size,
            "anchor_timestamp": frame.get_column("timestamp").gather(local.tolist()),
        }
    )
