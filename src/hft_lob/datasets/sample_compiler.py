"""把 raw 行情编译为与存储格式无关的训练样本块。"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

from hft_lob.configs.experiment import DataBuildConfig
from hft_lob.datasets.dataset_validator import stable_config_hash
from hft_lob.preprocessing.clean import DataCleaner, SessionSegment
from hft_lob.preprocessing.features import FeatureTransformer
from hft_lob.preprocessing.labels import LabelTransformer
from hft_lob.preprocessing.normalize import CausalRollingStandardizer
from hft_lob.preprocessing.quality import QualityReport


@dataclass(frozen=True)
class CompiledSession:
    """一个 session 的连续数组与索引记录。"""

    features: np.ndarray
    targets: np.ndarray
    validity: np.ndarray
    market: np.ndarray
    rows: pl.DataFrame
    anchors: pl.DataFrame


@dataclass(frozen=True)
class CompiledDay:
    """一个交易日的质量记录和全部 session。"""

    trade_date: str
    quality: QualityReport
    sessions: tuple[CompiledSession, ...]


@dataclass(frozen=True)
class SourceSet:
    files: tuple[Path, ...]
    raw_hashes: tuple[str, ...]
    processing_hash: str
    version: str


def discover_sources(config: DataBuildConfig) -> SourceSet:
    root = Path(config.data.raw_dir)
    ticker_root = root / config.ticker
    search_root = ticker_root if ticker_root.is_dir() else root
    if not search_root.is_dir():
        raise FileNotFoundError(f"raw data directory does not exist: {search_root}")
    files = tuple(sorted(search_root.glob("*.parquet"), key=lambda path: path.name))
    if not files:
        raise FileNotFoundError(f"no raw parquet files found in {search_root}")
    raw_hashes = tuple(_raw_file_hash(path) for path in files)
    processing_hash = stable_config_hash(_processing_config(config))
    version = stable_config_hash(
        {
            "ticker": config.ticker,
            "raw_hashes": sorted(raw_hashes),
            "processing_config_hash": processing_hash,
        }
    )
    return SourceSet(files, raw_hashes, processing_hash, version)


class SampleCompiler:
    """唯一的阶段一业务变换链，不负责文件写入或发布。"""

    def __init__(self, config: DataBuildConfig) -> None:
        self.config = config
        self.feature_transformer = FeatureTransformer(config.features)
        self.label_transformer = LabelTransformer(config.target)
        self.feature_columns = tuple(self.feature_transformer.feature_columns())
        self.standardizer = CausalRollingStandardizer(
            self.feature_columns,
            config.normalization.normalize_window,
        )
        self.cleaner = DataCleaner(
            config.sessions,
            config.data.snapshot_interval_seconds,
            config.cleaning.max_ffill_gap_seconds,
            column_mapping=config.data.column_mapping,
        )

    def compile(self, sources: tuple[Path, ...]) -> Iterator[CompiledDay]:
        seen_dates: set[str] = set()
        seen_sessions: set[tuple[str, str]] = set()
        dates: list[str] = []
        anchor_count = 0
        offset = 0
        for source in sources:
            cleaned = self.cleaner.clean_day(str(source), ticker=self.config.ticker)
            report = cleaned.quality_report
            if report.trade_date in seen_dates:
                raise ValueError(f"multiple raw files resolve to trade_date {report.trade_date}")
            if not cleaned.sessions:
                raise ValueError(f"raw file {source} contains no configured trading session")
            seen_dates.add(report.trade_date)
            dates.append(report.trade_date)
            sessions: list[CompiledSession] = []
            for segment in cleaned.sessions:
                key = (segment.trade_date, segment.session_id)
                if key in seen_sessions:
                    raise ValueError(f"duplicate session: {key}")
                seen_sessions.add(key)
                compiled = self._compile_session(segment, offset)
                sessions.append(compiled)
                offset += compiled.features.shape[0]
                anchor_count += compiled.anchors.height
            yield CompiledDay(
                trade_date=report.trade_date,
                quality=report,
                sessions=tuple(sessions),
            )
        if dates != sorted(dates):
            raise ValueError("raw files must resolve to chronologically ordered trade dates")
        if offset == 0 or anchor_count == 0:
            raise ValueError("processed data is empty or contains no valid anchors")

    def _compile_session(self, segment: SessionSegment, offset: int) -> CompiledSession:
        transformed = self.label_transformer.transform(
            self.feature_transformer.transform(segment)
        )
        frame = self.standardizer.transform_frame(transformed.frame)
        output_columns = [f"normalized__{name}" for name in self.feature_columns]
        row_valid = _row_valid(frame, output_columns)
        target_valid = np.asarray(
            frame.get_column("target_valid").fill_null(False), dtype=np.bool_
        )
        end = offset + frame.height
        rows = (
            frame.select("trade_date", "session_id", pl.col("timestamp"))
            .with_columns(pl.int_range(offset, end, dtype=pl.Int64).alias("global_index"))
            .select("global_index", "trade_date", "session_id", "timestamp")
        )
        return CompiledSession(
            features=np.asarray(frame.select(output_columns).to_numpy(), dtype=np.float32),
            targets=np.asarray(frame.select(self.config.target_column).to_numpy(), dtype=np.float32),
            validity=np.column_stack((row_valid, target_valid)),
            market=np.asarray(
                frame.select("mid_price", "future_mid", "BIDp1", "ASKp1").to_numpy(),
                dtype=np.float32,
            ),
            rows=rows,
            anchors=_anchor_frame(
                frame,
                row_valid,
                offset,
                self.config.window.history_snapshots,
                self.config.target_column,
            ),
        )


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


def _raw_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
