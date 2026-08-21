"""直接从原始行情构建连续、可 memory-map 的训练数据包。"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from hft_lob.configs.experiment import DataBuildConfig
from hft_lob.datasets.package import FOLD_INDEX_COLUMNS, DatasetPackageMetadata, compute_dataset_id
from hft_lob.datasets.validation import validate_dataset_package
from hft_lob.preprocessing.clean import DataCleaner
from hft_lob.preprocessing.features import FeatureTransformer
from hft_lob.preprocessing.labels import LabelTransformer
from hft_lob.preprocessing.manifest import dataset_version, raw_file_hash, stable_config_hash
from hft_lob.preprocessing.normalize import CausalRollingStandardizer
from hft_lob.preprocessing.split import (
    Fold,
    WalkForwardPlan,
    chronological_split,
    walk_forward_folds,
)


class _ArrayAppender:
    """在构建目录中顺序追加数组，完成后封装为标准 ``.npy``。"""

    def __init__(self, path: Path, *, dtype: Any, width: int) -> None:
        self.path = path
        self.dtype = np.dtype(dtype)
        self.width = width
        self.rows = 0
        self._file: BinaryIO = path.open("wb")

    def append(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=self.dtype)
        if array.ndim != 2 or array.shape[1] != self.width:
            raise ValueError(f"expected array [N,{self.width}], got {array.shape}")
        array.tofile(self._file)
        self.rows += array.shape[0]

    def finalize(self, destination: Path) -> None:
        self._file.close()
        output = np.lib.format.open_memmap(
            destination,
            mode="w+",
            dtype=self.dtype,
            shape=(self.rows, self.width),
        )
        source = np.memmap(self.path, mode="r", dtype=self.dtype, shape=(self.rows, self.width))
        chunk_rows = max(1, (64 * 1024 * 1024) // (self.dtype.itemsize * self.width))
        for start in range(0, self.rows, chunk_rows):
            output[start : start + chunk_rows] = source[start : start + chunk_rows]
        output.flush()
        del source, output
        self.path.unlink()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


def build_dataset_package(config: DataBuildConfig, output_root: str | Path) -> Path:
    """单遍处理 raw parquet，并原子发布最终训练数据包。

    不生成 processed parquet 或 manifest。临时追加文件只存在于
    ``.building-*`` 事务目录，成功发布后不会保留。
    """
    raw_files = _discover_raw_files(config)
    raw_hashes = [raw_file_hash(str(path)) for path in raw_files]
    processing_hash = stable_config_hash(_processing_config(config))
    data_version = dataset_version(
        config.ticker,
        raw_hashes,
        processing_config_hash=processing_hash,
    )
    root = Path(output_root).resolve()
    build_root = root / f".building-{uuid.uuid4().hex}"
    work_root = build_root / "package"
    try:
        result = _build_contents(work_root, config, raw_files)
        plan = _walk_forward_plan(result["trade_dates"], config, data_version)
        metadata = _metadata(
            config,
            raw_hashes=raw_hashes,
            processing_hash=processing_hash,
            plan=plan,
            feature_columns=result["feature_columns"],
        )
        _finish_package(work_root, metadata, plan, result["quality"])
        package_root = build_root / metadata.dataset_id
        os.replace(work_root, package_root)
        validate_dataset_package(package_root)

        destination = root / metadata.dataset_id
        root.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            validate_dataset_package(destination)
            return destination
        try:
            os.replace(package_root, destination)
        except FileExistsError:
            validate_dataset_package(destination)
        return destination
    finally:
        shutil.rmtree(build_root, ignore_errors=True)


def _build_contents(
    root: Path,
    config: DataBuildConfig,
    raw_files: list[Path],
) -> dict[str, object]:
    root.mkdir(parents=True)
    feature_transformer = FeatureTransformer(config.features)
    label_transformer = LabelTransformer(config.target)
    feature_columns = tuple(feature_transformer.feature_columns())
    standardizer = CausalRollingStandardizer(feature_columns, config.normalization.normalize_window)
    cleaner = DataCleaner(
        config.sessions,
        config.data.snapshot_interval_seconds,
        config.cleaning.max_ffill_gap_seconds,
        column_mapping=config.data.column_mapping,
    )
    arrays = {
        "features": _ArrayAppender(
            root / ".features.bin", dtype=np.float32, width=len(feature_columns)
        ),
        "targets": _ArrayAppender(root / ".targets.bin", dtype=np.float32, width=1),
        "validity": _ArrayAppender(root / ".validity.bin", dtype=np.bool_, width=2),
        "market": _ArrayAppender(root / ".market.bin", dtype=np.float32, width=4),
    }
    row_writer: pq.ParquetWriter | None = None
    anchor_writer: pq.ParquetWriter | None = None
    offset = 0
    trade_dates: list[str] = []
    quality: list[dict[str, object]] = []
    seen_dates: set[str] = set()
    seen_sessions: set[tuple[str, str]] = set()
    try:
        for source_path in raw_files:
            cleaned = cleaner.clean_day(str(source_path), ticker=config.ticker)
            report = cleaned.quality_report
            if report.trade_date in seen_dates:
                raise ValueError(f"multiple raw files resolve to trade_date {report.trade_date}")
            if not cleaned.sessions:
                raise ValueError(f"raw file {source_path} contains no configured trading session")
            seen_dates.add(report.trade_date)
            trade_dates.append(report.trade_date)
            quality.append(report.to_dict())
            for cleaned_segment in cleaned.sessions:
                key = (cleaned_segment.trade_date, cleaned_segment.session_id)
                if key in seen_sessions:
                    raise ValueError(f"duplicate session: {key}")
                seen_sessions.add(key)
                frame = label_transformer.transform(
                    feature_transformer.transform(cleaned_segment)
                ).frame
                frame = standardizer.transform_frame(frame)
                end = offset + frame.height
                output_columns = [f"normalized__{name}" for name in feature_columns]
                row_valid = _row_valid(frame, output_columns)
                target_valid = np.asarray(
                    frame.get_column("target_valid").fill_null(False), dtype=np.bool_
                )
                arrays["features"].append(frame.select(output_columns).to_numpy())
                arrays["targets"].append(frame.select(config.target_column).to_numpy())
                arrays["validity"].append(np.column_stack((row_valid, target_valid)))
                arrays["market"].append(
                    frame.select("mid_price", "future_mid", "BIDp1", "ASKp1").to_numpy()
                )

                rows = (
                    frame.select("trade_date", "session_id", pl.col("timestamp"))
                    .with_columns(pl.int_range(offset, end, dtype=pl.Int64).alias("global_index"))
                    .select("global_index", "trade_date", "session_id", "timestamp")
                )
                anchors = _anchor_frame(
                    frame, row_valid, offset, config.window.history_snapshots, config.target_column
                )
                row_writer = _write_chunk(root / "rows.parquet", rows, row_writer)
                if not anchors.is_empty():
                    anchor_writer = _write_chunk(root / "anchors.parquet", anchors, anchor_writer)
                offset = end
    finally:
        if row_writer is not None:
            row_writer.close()
        if anchor_writer is not None:
            anchor_writer.close()
        for appender in arrays.values():
            appender.close()
    if trade_dates != sorted(trade_dates):
        raise ValueError("raw files must resolve to chronologically ordered trade dates")
    if offset == 0 or anchor_writer is None:
        raise ValueError("processed data is empty or contains no valid anchors")
    if any(appender.rows != offset for appender in arrays.values()):
        raise ValueError("array row counts do not match")
    for name, appender in arrays.items():
        appender.finalize(root / f"{name}.npy")
    return {
        "trade_dates": trade_dates,
        "quality": quality,
        "feature_columns": feature_columns,
    }


def _finish_package(
    root: Path,
    metadata: DatasetPackageMetadata,
    plan: WalkForwardPlan,
    quality: object,
) -> None:
    lazy_anchors = pl.scan_parquet(root / "anchors.parquet")
    for fold in plan.folds:
        for split, dates in (
            ("train", fold.train_dates),
            ("validation", fold.validation_dates),
            ("test", fold.test_dates),
        ):
            path = root / "folds" / f"fold_{fold.index:03d}" / f"{split}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            lazy_anchors.filter(pl.col("trade_date").is_in(dates)).select(
                FOLD_INDEX_COLUMNS
            ).sink_parquet(path)
    if not isinstance(quality, list):
        raise TypeError("quality records must be a list")
    pl.DataFrame(quality).sort("trade_date").write_parquet(root / "quality.parquet")
    (root / "dataset.json").write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "_SUCCESS").touch()


def _metadata(
    config: DataBuildConfig,
    *,
    raw_hashes: list[str],
    processing_hash: str,
    plan: WalkForwardPlan,
    feature_columns: object,
) -> DatasetPackageMetadata:
    if not isinstance(feature_columns, tuple) or not all(
        isinstance(name, str) for name in feature_columns
    ):
        raise TypeError("feature_columns must be a tuple of strings")
    source_hash = stable_config_hash({"raw_hashes": sorted(raw_hashes)})
    fold_plan_hash = stable_config_hash({"folds": [asdict(fold) for fold in plan.folds]})
    return DatasetPackageMetadata(
        dataset_id=compute_dataset_id(
            ticker=config.ticker,
            source_hash=source_hash,
            processing_config_hash=processing_hash,
            fold_plan_hash=fold_plan_hash,
        ),
        ticker=config.ticker,
        feature_columns=feature_columns,
        target_column=config.target_column,
        feature_dtype="float32",
        target_dtype="float32",
        snapshot_interval_seconds=config.data.snapshot_interval_seconds,
        history_snapshots=config.window.history_snapshots,
        normalization_mode=config.normalization.mode,
        normalization_window=config.normalization.normalize_window,
        source_hash=source_hash,
        processing_config_hash=processing_hash,
        fold_plan_hash=fold_plan_hash,
    )


def _discover_raw_files(config: DataBuildConfig) -> list[Path]:
    root = Path(config.data.raw_dir)
    ticker_root = root / config.ticker
    search_root = ticker_root if ticker_root.is_dir() else root
    if not search_root.is_dir():
        raise FileNotFoundError(f"raw data directory does not exist: {search_root}")
    files = sorted(
        (path for path in search_root.glob("*.parquet") if path.is_file()),
        key=lambda path: path.name,
    )
    if not files:
        raise FileNotFoundError(f"no raw parquet files found in {search_root}")
    return files


def _processing_config(config: DataBuildConfig) -> dict[str, object]:
    return {
        "pipeline_semantics_version": 2,
        "ticker": config.ticker,
        "data": {
            "levels": config.data.levels,
            "snapshot_interval_seconds": config.data.snapshot_interval_seconds,
            "column_mapping": config.data.column_mapping,
        },
        "cleaning": asdict(config.cleaning),
        "sessions": asdict(config.sessions),
        "features": asdict(config.features),
        "target": asdict(config.target),
        "normalization": asdict(config.normalization),
    }


def _walk_forward_plan(dates: object, config: DataBuildConfig, version: str) -> WalkForwardPlan:
    if not isinstance(dates, list) or not all(isinstance(value, str) for value in dates):
        raise TypeError("trade_dates must be a list of strings")
    if config.walk_forward.enabled:
        folds = tuple(walk_forward_folds(dates, config.walk_forward))
    else:
        split = chronological_split(dates, config.split)
        folds = (Fold(1, split.train_dates, split.validation_dates, split.test_dates),)
    return WalkForwardPlan(dataset_version=version, folds=folds)


def _write_chunk(
    path: Path, frame: pl.DataFrame, writer: pq.ParquetWriter | None
) -> pq.ParquetWriter:
    table = frame.to_arrow()
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema)
    writer.write_table(table)
    return writer


def _row_valid(frame: pl.DataFrame, feature_columns: list[str]) -> np.ndarray:
    expression = (
        pl.col("book_valid").fill_null(False)
        & pl.col("feature_valid").fill_null(False)
        & pl.col("normalization_valid").fill_null(False)
        & pl.all_horizontal(
            pl.col(name).is_not_null() & pl.col(name).is_finite() for name in feature_columns
        )
    )
    return np.asarray(frame.select(expression.alias("valid")).get_column("valid"), dtype=np.bool_)


def _anchor_frame(
    frame: pl.DataFrame,
    row_valid: np.ndarray,
    session_start: int,
    history_snapshots: int,
    target_column: str,
) -> pl.DataFrame:
    target_valid = np.asarray(
        frame.select(
            (
                pl.col("target_valid").fill_null(False)
                & pl.col(target_column).is_not_null()
                & pl.col(target_column).is_finite()
                & pl.col("future_mid").is_not_null()
                & pl.col("future_mid").is_finite()
            ).alias("valid")
        ).get_column("valid"),
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
