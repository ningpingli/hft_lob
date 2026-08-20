"""预处理总流程（需求文档 §40 流水线前段）：raw → 清洗 → 特征 → 标签 → manifest → split。"""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from hft_lob.configs.experiment import ExperimentConfig
from hft_lob.preprocessing.clean import DataCleaner, SessionSegment
from hft_lob.preprocessing.features import FeatureTransformer
from hft_lob.preprocessing.labels import LabelTransformer
from hft_lob.preprocessing.manifest import (
    build_manifest,
    dataset_version,
    feature_version,
    label_version,
    raw_file_hash,
    stable_config_hash,
    write_manifest,
)
from hft_lob.preprocessing.quality import QualityReport
from hft_lob.preprocessing.split import (
    Fold,
    WalkForwardPlan,
    chronological_split,
    walk_forward_folds,
)


@dataclass(frozen=True)
class PreparedDataset:
    """数据准备阶段对训练侧的唯一交付对象。"""

    dataset_version: str
    feature_columns: tuple[str, ...]
    feature_version: str
    label_version: str
    manifest_path: str
    quality_report_path: str
    walk_forward_plan: WalkForwardPlan


def prepare_dataset(config: ExperimentConfig) -> PreparedDataset:
    """执行 raw→独立 session parquet→manifest，返回唯一训练交付对象。

    processed 文件名包含 trade_date/session_id；manifest 每行对应一个 session，
    所有 fold 仍按完整 trade_date 切分。
    """
    raw_files = _discover_raw_files(config)
    raw_hash_by_path = {path: raw_file_hash(str(path)) for path in raw_files}
    processing_hash = stable_config_hash(_processing_config(config))
    version = dataset_version(
        config.ticker,
        list(raw_hash_by_path.values()),
        processing_config_hash=processing_hash,
    )

    feature_transformer = FeatureTransformer(config.features)
    label_transformer = LabelTransformer(config.target)
    cleaner = DataCleaner(
        config.sessions,
        config.data.snapshot_interval_seconds,
        config.cleaning.max_ffill_gap_seconds,
        column_mapping=config.data.column_mapping,
    )
    feature_columns = tuple(feature_transformer.feature_columns())
    feature_id = feature_version(config.features)
    label_id = label_version(config.target)
    processed_root = Path(config.data.processed_dir) / config.ticker / version
    manifest_root = Path(config.data.manifest_dir) / config.ticker / version
    manifest_path = manifest_root / "manifest.parquet"
    quality_path = manifest_root / "quality_reports.parquet"

    records: list[dict[str, object]] = []
    quality_records: list[dict[str, object]] = []
    seen_sessions: set[tuple[str, str]] = set()
    seen_trade_dates: set[str] = set()
    for source_path in raw_files:
        cleaned = cleaner.clean_day(str(source_path), ticker=config.ticker)
        report = cleaned.quality_report
        if report.trade_date in seen_trade_dates:
            raise ValueError(f"multiple raw files resolve to trade_date {report.trade_date}")
        seen_trade_dates.add(report.trade_date)
        quality_records.append(report.to_dict())
        if not cleaned.sessions:
            raise ValueError(f"raw file {source_path} contains no configured trading session")

        quality_status = _quality_status(report)
        for cleaned_segment in cleaned.sessions:
            key = (cleaned_segment.trade_date, cleaned_segment.session_id)
            if key in seen_sessions:
                raise ValueError(f"duplicate processed session: {key}")
            seen_sessions.add(key)

            featured = feature_transformer.transform(cleaned_segment)
            labeled = label_transformer.transform(featured)
            processed_path = processed_root / (
                f"{labeled.trade_date}_{labeled.session_id}.parquet"
            )
            _write_parquet_atomic(labeled.frame, processed_path)
            records.append(
                _manifest_record(
                    segment=labeled,
                    source_path=source_path,
                    processed_path=processed_path,
                    raw_hash=raw_hash_by_path[source_path],
                    processing_hash=processing_hash,
                    dataset_id=version,
                    feature_id=feature_id,
                    label_id=label_id,
                    quality_status=quality_status,
                )
            )

    manifest = build_manifest(ticker=config.ticker, records=records)
    trade_dates = (
        manifest.get_column("trade_date").unique().sort().to_list()
    )
    if config.walk_forward.enabled:
        folds = tuple(walk_forward_folds(trade_dates, config.walk_forward))
    else:
        split = chronological_split(trade_dates, config.split)
        folds = (
            Fold(1, split.train_dates, split.validation_dates, split.test_dates),
        )
    plan = WalkForwardPlan(dataset_version=version, folds=folds)

    write_manifest(manifest, str(manifest_path))
    quality_frame = pl.DataFrame(quality_records).sort("trade_date")
    _write_parquet_atomic(quality_frame, quality_path)
    return PreparedDataset(
        dataset_version=version,
        feature_columns=feature_columns,
        feature_version=feature_id,
        label_version=label_id,
        manifest_path=str(manifest_path.resolve()),
        quality_report_path=str(quality_path.resolve()),
        walk_forward_plan=plan,
    )


def _discover_raw_files(config: ExperimentConfig) -> list[Path]:
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


def _processing_config(config: ExperimentConfig) -> dict[str, object]:
    """只纳入会改变 processed 内容的配置，排除目录、训练和执行范围。"""
    return {
        "pipeline_semantics_version": 1,
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
    }


def _manifest_record(
    *,
    segment: SessionSegment,
    source_path: Path,
    processed_path: Path,
    raw_hash: str,
    processing_hash: str,
    dataset_id: str,
    feature_id: str,
    label_id: str,
    quality_status: str,
) -> dict[str, object]:
    frame = segment.frame
    valid_row = (
        pl.col("book_valid").fill_null(False)
        & pl.col("feature_valid").fill_null(False)
        & pl.col("target_valid").fill_null(False)
    )
    valid_count = int(frame.select(valid_row.sum()).item())
    data_start = frame.get_column("timestamp").min()
    data_end = frame.get_column("timestamp").max()
    if not isinstance(data_start, datetime) or not isinstance(data_end, datetime):
        raise ValueError("processed session timestamp bounds must be datetime values")
    return {
        "trade_date": segment.trade_date,
        "session_id": segment.session_id,
        "source_file": str(source_path.resolve()),
        "processed_file": str(processed_path.resolve()),
        "raw_hash": raw_hash,
        "processing_config_hash": processing_hash,
        "dataset_version": dataset_id,
        "row_count": frame.height,
        "valid_row_count": valid_count,
        "data_start": data_start,
        "data_end": data_end,
        "feature_version": feature_id,
        "label_version": label_id,
        "quality_status": quality_status,
    }


def _quality_status(report: QualityReport) -> str:
    if (
        report.missing_ratio > 0
        or report.crossed_book_count > 0
        or report.invalid_level_order_count > 0
    ):
        return "warning"
    return "passed"


def _write_parquet_atomic(frame: pl.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        frame.write_parquet(temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
