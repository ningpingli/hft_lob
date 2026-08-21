"""从现有数据工程产物构建不可变训练数据包。"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

import polars as pl
import torch

from hft_lob.configs.experiment import ExperimentConfig
from hft_lob.datasets.package import (
    FOLD_INDEX_COLUMNS,
    FOLD_INDEX_SCHEMA,
    DatasetPackageMetadata,
    compute_dataset_id,
    validate_fold_index,
)
from hft_lob.datasets.validation import validate_dataset_package
from hft_lob.preprocessing.manifest import read_manifest, stable_config_hash
from hft_lob.preprocessing.normalize import CausalRollingStandardizer
from hft_lob.preprocessing.pipeline import PreparedDataset, prepare_dataset


def build_dataset_package(
    config: ExperimentConfig,
    output_root: str | Path,
) -> Path:
    """构建并原子发布一个数据包；相同数据包已存在时直接复用。"""
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


def _metadata(
    config: ExperimentConfig,
    prepared: PreparedDataset,
    manifest: pl.DataFrame,
) -> DatasetPackageMetadata:
    processing_hashes = manifest.get_column("processing_config_hash").unique().to_list()
    if len(processing_hashes) != 1:
        raise ValueError("manifest must contain one processing_config_hash")
    source_hash = stable_config_hash(
        {"raw_hashes": sorted(manifest.get_column("raw_hash").unique().to_list())}
    )
    fold_plan_hash = stable_config_hash(
        {"folds": [asdict(fold) for fold in prepared.walk_forward_plan.folds]}
    )
    identity = {
        "ticker": config.ticker,
        "source_hash": source_hash,
        "processing_config_hash": processing_hashes[0],
        "fold_plan_hash": fold_plan_hash,
    }
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


def _build_contents(
    root: Path,
    config: ExperimentConfig,
    prepared: PreparedDataset,
    manifest: pl.DataFrame,
    metadata: DatasetPackageMetadata,
) -> None:
    root.mkdir(parents=True)
    standardizer = CausalRollingStandardizer(
        prepared.feature_columns,
        config.normalization.normalize_window,
    )
    anchors_by_date: dict[str, list[dict[str, object]]] = {}
    for record in manifest.iter_rows(named=True):
        processed_path = Path(record["processed_file"])
        frame = standardizer.transform_frame(pl.read_parquet(processed_path))
        trade_date = str(record["trade_date"])
        session_id = str(record["session_id"])
        relative = Path("sessions") / f"{trade_date}_{session_id}.pt"
        session_path = root / relative
        session_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_session_payload(frame, metadata), session_path)
        anchors_by_date.setdefault(trade_date, []).extend(
            _anchor_records(frame, relative.as_posix(), metadata.history_snapshots)
        )

    for fold in prepared.walk_forward_plan.folds:
        for split, dates in (
            ("train", fold.train_dates),
            ("validation", fold.validation_dates),
            ("test", fold.test_dates),
        ):
            records = [record for date in dates for record in anchors_by_date.get(date, [])]
            frame = pl.DataFrame(records, schema=FOLD_INDEX_SCHEMA).select(FOLD_INDEX_COLUMNS)
            validate_fold_index(frame)
            path = root / "folds" / f"fold_{fold.index:03d}" / f"{split}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(path)

    (root / "dataset.json").write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(prepared.quality_report_path, root / "quality.parquet")
    (root / "_SUCCESS").touch()


def _session_payload(
    frame: pl.DataFrame,
    metadata: DatasetPackageMetadata,
) -> dict[str, object]:
    output_columns = [f"normalized__{name}" for name in metadata.feature_columns]

    def tensor(columns: list[str]) -> torch.Tensor:
        return torch.tensor(frame.select(columns).to_numpy(), dtype=torch.float32)

    return {
        "features": tensor(output_columns),
        "targets": tensor([metadata.target_column]),
        "row_valid": torch.tensor(_row_valid(frame, output_columns), dtype=torch.bool),
        "target_valid": torch.tensor(
            frame.get_column("target_valid").fill_null(False).to_list(), dtype=torch.bool
        ),
        "timestamps": [value.isoformat() for value in frame.get_column("timestamp")],
        "mid_price": tensor(["mid_price"])[:, 0],
        "future_mid": tensor(["future_mid"])[:, 0],
        "bid1": tensor(["BIDp1"])[:, 0],
        "ask1": tensor(["ASKp1"])[:, 0],
        "trade_date": str(frame.get_column("trade_date").item(0)),
        "session_id": str(frame.get_column("session_id").item(0)),
    }


def _row_valid(frame: pl.DataFrame, feature_columns: list[str]) -> list[bool]:
    expression = (
        pl.col("book_valid").fill_null(False)
        & pl.col("feature_valid").fill_null(False)
        & pl.col("normalization_valid").fill_null(False)
        & pl.all_horizontal(
            pl.col(name).is_not_null() & pl.col(name).is_finite()
            for name in feature_columns
        )
    )
    return frame.select(expression.alias("valid")).get_column("valid").to_list()


def _anchor_records(
    frame: pl.DataFrame,
    session_file: str,
    history_snapshots: int,
) -> list[dict[str, object]]:
    rows = _row_valid(frame, [name for name in frame.columns if name.startswith("normalized__")])
    targets = frame.select(
        (
            pl.col("target_valid").fill_null(False)
            & pl.col("future_mid").is_not_null()
            & pl.col("future_mid").is_finite()
        ).alias("valid")
    ).get_column("valid").to_list()
    prefix = [0]
    for valid in rows:
        prefix.append(prefix[-1] + int(valid))
    records: list[dict[str, object]] = []
    for anchor in range(history_snapshots - 1, frame.height):
        start = anchor - history_snapshots + 1
        if prefix[anchor + 1] - prefix[start] == history_snapshots and targets[anchor]:
            records.append(
                {
                    "session_file": session_file,
                    "anchor_index": anchor,
                    "trade_date": str(frame.get_column("trade_date").item(anchor)),
                    "session_id": str(frame.get_column("session_id").item(anchor)),
                    "anchor_timestamp": frame.get_column("timestamp").item(anchor),
                }
            )
    return records
