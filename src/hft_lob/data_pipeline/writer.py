"""不可变数据包写入、读取与校验。"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from hft_lob.data_pipeline.loader import CompiledDay, CompiledSession
from hft_lob.data_pipeline.splitter import (
    FOLD_INDEX_COLUMNS,
    FOLD_INDEX_SCHEMA,
    WalkForwardPlan,
    write_fold_indexes,
)
from hft_lob.utils.identity import stable_config_hash

PACKAGE_SCHEMA_VERSION = 4

SUCCESS_MARKER = "_SUCCESS"


QUALITY_COLUMNS = (
    "trade_date",
    "row_count",
    "missing_ratio",
    "duplicate_count",
    "crossed_book_count",
    "one_side_missing_count",
    "max_gap",
    "p95_gap",
    "stale_snapshot_ratio",
    "invalid_level_order_count",
)

QUALITY_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "trade_date": pl.String,
    "row_count": pl.Int64,
    "missing_ratio": pl.Float64,
    "duplicate_count": pl.Int64,
    "crossed_book_count": pl.Int64,
    "one_side_missing_count": pl.Int64,
    "max_gap": pl.Float64,
    "p95_gap": pl.Float64,
    "stale_snapshot_ratio": pl.Float64,
    "invalid_level_order_count": pl.Int64,
}


def compute_dataset_id(
    *,
    ticker: str,
    source_hash: str,
    processing_config_hash: str,
    fold_plan_hash: str,
) -> str:
    values = {
        "ticker": ticker,
        "source_hash": source_hash,
        "processing_config_hash": processing_config_hash,
        "fold_plan_hash": fold_plan_hash,
        "schema_version": PACKAGE_SCHEMA_VERSION,
    }
    empty = [name for name, value in values.items() if isinstance(value, str) and not value.strip()]
    if empty:
        raise ValueError(f"dataset identity fields must not be empty: {empty}")
    return stable_config_hash(values)

@dataclass(frozen=True)
class DatasetPackageMetadata:
    dataset_id: str
    ticker: str
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    feature_dtype: str
    target_dtype: str
    snapshot_interval_seconds: int
    history_snapshots: int
    normalization_mode: str
    normalization_window: int
    source_hash: str
    processing_config_hash: str
    fold_plan_hash: str
    labels: tuple[int, ...] = (60,)
    schema_version: int = PACKAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        text_fields = (
            "dataset_id",
            "ticker",
            "feature_dtype",
            "target_dtype",
            "normalization_mode",
            "source_hash",
            "processing_config_hash",
            "fold_plan_hash",
        )
        empty = [name for name in text_fields if not str(getattr(self, name)).strip()]
        if empty:
            raise ValueError(f"metadata fields must not be empty: {empty}")
        if self.schema_version != PACKAGE_SCHEMA_VERSION:
            raise ValueError(f"unsupported dataset schema version: {self.schema_version}")
        if not self.feature_columns or len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns must be non-empty and unique")
        if not self.target_columns or len(set(self.target_columns)) != len(self.target_columns):
            raise ValueError("target_columns must be non-empty and unique")
        labels = tuple(self.labels)
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("labels must be non-empty and unique")
        if any(not isinstance(label, int) or isinstance(label, bool) or label <= 0 for label in labels):
            raise ValueError("labels must contain positive integers")
        valid_target_columns = {tuple(f"Target_{label}s_{suffix}" for label in labels) for suffix in ("log", "simple")}
        if self.target_columns not in valid_target_columns:
            raise ValueError("target_columns must match labels in order and target type")
        if self.snapshot_interval_seconds <= 0 or self.history_snapshots <= 0:
            raise ValueError("snapshot_interval_seconds and history_snapshots must be > 0")
        if self.normalization_window < 2:
            raise ValueError("normalization_window must be >= 2")
        expected = compute_dataset_id(
            ticker=self.ticker,
            source_hash=self.source_hash,
            processing_config_hash=self.processing_config_hash,
            fold_plan_hash=self.fold_plan_hash,
        )
        if self.dataset_id != expected:
            raise ValueError("dataset_id does not match package identity fields")
        object.__setattr__(self, "labels", labels)

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["feature_columns"] = list(self.feature_columns)
        value["target_columns"] = list(self.target_columns)
        value["labels"] = list(self.labels)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> DatasetPackageMetadata:
        expected = set(cls.__dataclass_fields__)
        missing = sorted(expected.difference(value))
        unknown = sorted(set(value).difference(expected))
        if missing or unknown:
            raise ValueError(
                f"invalid dataset metadata fields: missing={missing}, unknown={unknown}"
            )
        columns = value["feature_columns"]
        targets = value["target_columns"]
        labels = value["labels"]
        if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
            raise ValueError("feature_columns must be a list of strings")
        if not isinstance(targets, list) or not all(isinstance(item, str) for item in targets):
            raise ValueError("target_columns must be a list of strings")
        if not isinstance(labels, list) or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in labels
        ):
            raise ValueError("labels must be a list of integers")
        payload: dict[str, Any] = {
            **value,
            "feature_columns": tuple(columns),
            "target_columns": tuple(targets),
            "labels": tuple(labels),
        }
        return cls(**payload)

@dataclass(frozen=True)
class DatasetPackage:
    root: Path
    metadata: DatasetPackageMetadata

    def __post_init__(self) -> None:
        resolved = self.root.resolve()
        if resolved.name != self.metadata.dataset_id:
            raise ValueError("package root name must equal dataset_id")
        object.__setattr__(self, "root", resolved)

def fold_index_path(package_dir: str | Path, fold_index: int, split: str) -> Path:
    if fold_index <= 0:
        raise ValueError("fold_index must be > 0")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be 'train', 'validation', or 'test'")
    return Path(package_dir) / "folds" / f"fold_{fold_index:03d}" / f"{split}.parquet"

def validate_fold_index(frame: pl.DataFrame) -> None:
    if frame.is_empty() or tuple(frame.columns) != FOLD_INDEX_COLUMNS:
        raise ValueError("fold index must be non-empty and use the fixed schema")
    if any(frame.schema[name] != dtype for name, dtype in FOLD_INDEX_SCHEMA.items()):
        raise ValueError("invalid fold index dtypes")
    if frame.null_count().select(pl.sum_horizontal(pl.all())).item() != 0:
        raise ValueError("fold index must not contain null values")
    if not frame.filter(
        (pl.col("global_anchor_index") < 0)
        | (pl.col("session_start_index") < 0)
        | (pl.col("anchor_index") < 0)
    ).is_empty():
        raise ValueError("fold indexes must be >= 0")
    if bool(frame.select(pl.col("global_anchor_index").is_duplicated().any()).item()):
        raise ValueError("fold index contains duplicate samples")

_ROW_COLUMNS = ("global_index", "trade_date", "session_id", "timestamp")

_ROW_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "global_index": pl.Int64,
    "trade_date": pl.String,
    "session_id": pl.String,
    "timestamp": pl.Datetime("us"),
}

def validate_dataset_package(package_dir: str | Path) -> DatasetPackageMetadata:
    """验证已发布数据包并返回 metadata；不修复、不回退构建。"""
    root = Path(package_dir).resolve()
    if not (root / SUCCESS_MARKER).is_file():
        raise ValueError(f"dataset package is not published: missing {SUCCESS_MARKER}")
    metadata = _read_metadata(root / "dataset.json")
    if root.name != metadata.dataset_id:
        raise ValueError("package directory name must equal dataset_id")
    for name in ("quality.parquet", "rows.parquet"):
        if not (root / name).is_file():
            raise ValueError(f"dataset package is missing {name}")

    features = np.load(root / "features.npy", mmap_mode="r")
    targets = np.load(root / "targets.npy", mmap_mode="r")
    validity = np.load(root / "validity.npy", mmap_mode="r")
    market = np.load(root / "market.npy", mmap_mode="r")
    row_count = features.shape[0]
    expected_shapes = {
        "features.npy": (row_count, len(metadata.feature_columns)),
        "targets.npy": (row_count, len(metadata.labels)),
        "validity.npy": (row_count, 1),
        "market.npy": (row_count, 3),
    }
    arrays = {
        "features.npy": features,
        "targets.npy": targets,
        "validity.npy": validity,
        "market.npy": market,
    }
    for name, array in arrays.items():
        if array.shape != expected_shapes[name]:
            raise ValueError(f"invalid {name} shape: {array.shape}")
    if features.dtype.name != metadata.feature_dtype or targets.dtype.name != metadata.target_dtype:
        raise ValueError("array dtype does not match dataset metadata")
    if validity.dtype != np.bool_ or market.dtype != np.float32:
        raise ValueError("validity.npy must be bool and market.npy must be float32")
    if not np.isfinite(np.asarray(targets)).all():
        raise ValueError("targets must contain only finite values")

    rows = pl.read_parquet(root / "rows.parquet")
    _validate_rows(rows, row_count)
    _validate_quality(pl.read_parquet(root / "quality.parquet"), rows, row_count)
    fold_dirs = sorted(path for path in (root / "folds").glob("fold_*") if path.is_dir())
    if not fold_dirs:
        raise ValueError("dataset package contains no fold indexes")
    for fold_dir in fold_dirs:
        expected = {fold_dir / f"{split}.parquet" for split in ("train", "validation", "test")}
        if set(fold_dir.glob("*.parquet")) != expected:
            raise ValueError(f"fold must contain exactly train/validation/test: {fold_dir.name}")
        for index_path in expected:
            frame = pl.read_parquet(index_path)
            validate_fold_index(frame)
            _validate_fold_references(frame, rows, validity, row_count, metadata.history_snapshots)
    return metadata

def open_dataset_package(package_dir: str | Path) -> DatasetPackage:
    """完整校验一次并返回供训练阶段共享的只读句柄。"""
    root = Path(package_dir).resolve()
    return DatasetPackage(root=root, metadata=validate_dataset_package(root))

def load_dataset_package(package_dir: str | Path) -> DatasetPackage:
    """轻量打开构建阶段已验证并发布的数据包。

    训练阶段只确认发布标记并读取内容寻址元数据，不重复扫描数组、rows 和 folds。
    需要重新执行完整性审计时应显式调用 ``validate_dataset_package``。
    """
    root = Path(package_dir).resolve()
    if not (root / SUCCESS_MARKER).is_file():
        raise ValueError(f"dataset package is not published: missing {SUCCESS_MARKER}")
    return DatasetPackage(root=root, metadata=_read_metadata(root / "dataset.json"))

def _read_metadata(path: Path) -> DatasetPackageMetadata:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("dataset.json is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("dataset.json root must be an object")
    return DatasetPackageMetadata.from_dict(value)

def _validate_rows(rows: pl.DataFrame, row_count: int) -> None:
    if tuple(rows.columns) != _ROW_COLUMNS or any(
        rows.schema[name] != dtype for name, dtype in _ROW_SCHEMA.items()
    ):
        raise ValueError("rows.parquet has an invalid schema")
    if rows.height != row_count or rows.get_column("global_index").to_list() != list(
        range(row_count)
    ):
        raise ValueError("rows.parquet must cover every global row exactly once")

def _validate_quality(quality: pl.DataFrame, rows: pl.DataFrame, row_count: int) -> None:
    if tuple(quality.columns) != QUALITY_COLUMNS or any(
        quality.schema[name] != dtype for name, dtype in QUALITY_SCHEMA.items()
    ):
        raise ValueError("quality.parquet has an invalid schema")
    if quality.is_empty() or quality.null_count().select(pl.sum_horizontal(pl.all())).item() != 0:
        raise ValueError("quality.parquet must be non-empty and contain no nulls")
    if quality.get_column("trade_date").is_duplicated().any():
        raise ValueError("quality.parquet must contain one row per trade date")
    if set(quality.get_column("trade_date")) != set(rows.get_column("trade_date")):
        raise ValueError("quality.parquet trade dates do not match rows.parquet")
    if quality.get_column("row_count").sum() != row_count:
        raise ValueError("quality.parquet row counts do not match arrays")
    invalid = quality.filter(
        (pl.col("row_count") <= 0)
        | ~pl.col("missing_ratio").is_between(0.0, 1.0, closed="both")
        | ~pl.col("stale_snapshot_ratio").is_between(0.0, 1.0, closed="both")
        | (pl.col("max_gap") < 0)
        | (pl.col("p95_gap") < 0)
        | (pl.col("p95_gap") > pl.col("max_gap"))
        | pl.any_horizontal(
            pl.col(name) < 0
            for name in (
                "duplicate_count",
                "crossed_book_count",
                "one_side_missing_count",
                "invalid_level_order_count",
            )
        )
    )
    if not invalid.is_empty():
        raise ValueError("quality.parquet contains invalid metric values")

def _validate_fold_references(
    frame: pl.DataFrame,
    rows: pl.DataFrame,
    validity: np.ndarray,
    row_count: int,
    history_snapshots: int,
) -> None:
    invalid = frame.filter(
        (pl.col("global_anchor_index") >= row_count)
        | (pl.col("global_anchor_index") - pl.col("session_start_index") != pl.col("anchor_index"))
        | (pl.col("anchor_index") < history_snapshots - 1)
    )
    if not invalid.is_empty():
        raise ValueError("fold index contains an invalid global or session-local anchor")
    for anchor in frame.get_column("global_anchor_index"):
        start = anchor - history_snapshots + 1
        if not validity[start : anchor + 1, 0].all():
            raise ValueError("fold index references an invalid sample window")
    referenced = frame.join(
        rows,
        left_on="global_anchor_index",
        right_on="global_index",
        suffix="_row",
    )
    mismatch = referenced.filter(
        (pl.col("trade_date") != pl.col("trade_date_row"))
        | (pl.col("session_id") != pl.col("session_id_row"))
        | (pl.col("anchor_timestamp") != pl.col("timestamp"))
    )
    if referenced.height != frame.height or not mismatch.is_empty():
        raise ValueError("fold index metadata does not match rows.parquet")


logger = logging.getLogger(__name__)

class _ArrayAppender:
    """顺序追加二维数组，完成后封装成标准 ``.npy``。"""

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
        self.close()
        output = np.lib.format.open_memmap(
            destination,
            mode="w+",
            dtype=self.dtype,
            shape=(self.rows, self.width),
        )
        source = np.memmap(
            self.path,
            mode="r",
            dtype=self.dtype,
            shape=(self.rows, self.width),
        )
        chunk_rows = max(1, (64 * 1024 * 1024) // (self.dtype.itemsize * self.width))
        for start in range(0, self.rows, chunk_rows):
            output[start : start + chunk_rows] = source[start : start + chunk_rows]
        output.flush()
        del source, output
        self.path.unlink()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

class DatasetPackageWriter:
    """持有一次构建的文件句柄、临时目录和原子发布事务。"""

    def __init__(
        self, output_root: str | Path, feature_count: int, target_count: int = 1
    ) -> None:
        if feature_count <= 0:
            raise ValueError("feature_count must be > 0")
        if target_count <= 0:
            raise ValueError("target_count must be > 0")
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.build_root = self.output_root / f".building-{uuid.uuid4().hex}"
        self.work_root = self.build_root / "package"
        self.work_root.mkdir(parents=True)
        self.arrays = {
            "features": _ArrayAppender(
                self.work_root / ".features.bin", dtype=np.float32, width=feature_count
            ),
            "targets": _ArrayAppender(
                self.work_root / ".targets.bin", dtype=np.float32, width=target_count
            ),
            "validity": _ArrayAppender(
                self.work_root / ".validity.bin", dtype=np.bool_, width=1
            ),
            "market": _ArrayAppender(self.work_root / ".market.bin", dtype=np.float32, width=3),
        }
        self.row_writer: pq.ParquetWriter | None = None
        self.anchor_writer: pq.ParquetWriter | None = None
        self.trade_dates: list[str] = []
        self.quality: list[dict[str, object]] = []
        self.anchor_count = 0
        self._finalized = False

    def __enter__(self) -> DatasetPackageWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
        shutil.rmtree(self.build_root, ignore_errors=True)

    def append(self, day: CompiledDay) -> None:
        if self._finalized:
            raise RuntimeError("cannot append after finalization")
        self.trade_dates.append(day.trade_date)
        self.quality.append(day.quality.to_dict())
        for session in day.sessions:
            self.anchor_count += session.anchors.height
            self._append_session(session)

    @property
    def row_count(self) -> int:
        return self.arrays["features"].rows

    def finalize_and_publish(
        self,
        metadata: DatasetPackageMetadata,
        plan: WalkForwardPlan,
    ) -> Path:
        logger.info(
            "dataset_build.finalize_arrays rows=%d anchors=%d", self.row_count, self.anchor_count
        )
        self._finalize_data()
        anchors_path = self.work_root / "anchors.parquet"
        write_fold_indexes(anchors_path, self.work_root / "folds", plan)
        anchors_path.unlink()
        pl.DataFrame(self.quality).sort("trade_date").write_parquet(
            self.work_root / "quality.parquet"
        )
        (self.work_root / "dataset.json").write_text(
            json.dumps(metadata.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (self.work_root / "_SUCCESS").touch()

        package_root = self.build_root / metadata.dataset_id
        os.replace(self.work_root, package_root)
        logger.info("dataset_build.validate_start dataset_id=%s", metadata.dataset_id)
        validate_dataset_package(package_root)
        logger.info("dataset_build.validate_complete dataset_id=%s", metadata.dataset_id)
        destination = self.output_root / metadata.dataset_id
        if destination.exists():
            validate_dataset_package(destination)
            return destination
        try:
            os.replace(package_root, destination)
        except FileExistsError:
            validate_dataset_package(destination)
        return destination

    def close(self) -> None:
        if self.row_writer is not None:
            self.row_writer.close()
            self.row_writer = None
        if self.anchor_writer is not None:
            self.anchor_writer.close()
            self.anchor_writer = None
        for appender in self.arrays.values():
            appender.close()

    def _append_session(self, session: CompiledSession) -> None:
        self.arrays["features"].append(session.features)
        self.arrays["targets"].append(session.targets)
        self.arrays["validity"].append(session.validity)
        self.arrays["market"].append(session.market)
        self.row_writer = _write_chunk(
            self.work_root / "rows.parquet", session.rows, self.row_writer
        )
        if not session.anchors.is_empty():
            self.anchor_writer = _write_chunk(
                self.work_root / "anchors.parquet", session.anchors, self.anchor_writer
            )

    def _finalize_data(self) -> None:
        if self._finalized:
            raise RuntimeError("writer is already finalized")
        has_anchors = self.anchor_writer is not None
        self.close()
        row_count = self.arrays["features"].rows
        if row_count == 0 or not has_anchors:
            raise ValueError("processed data is empty or contains no valid anchors")
        if any(appender.rows != row_count for appender in self.arrays.values()):
            raise ValueError("array row counts do not match")
        for name, appender in self.arrays.items():
            appender.finalize(self.work_root / f"{name}.npy")
        self._finalized = True

def _write_chunk(
    path: Path,
    frame: pl.DataFrame,
    writer: pq.ParquetWriter | None,
) -> pq.ParquetWriter:
    table = frame.to_arrow()
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema)
    writer.write_table(table)
    return writer
