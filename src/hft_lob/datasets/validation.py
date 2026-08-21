"""预构建数据包的完整性校验。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from hft_lob.datasets.package import (
    SUCCESS_MARKER,
    DatasetPackage,
    DatasetPackageMetadata,
    validate_fold_index,
)

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
        "targets.npy": (row_count, 1),
        "validity.npy": (row_count, 2),
        "market.npy": (row_count, 4),
    }
    arrays = {"features.npy": features, "targets.npy": targets, "validity.npy": validity, "market.npy": market}
    for name, array in arrays.items():
        if array.shape != expected_shapes[name]:
            raise ValueError(f"invalid {name} shape: {array.shape}")
    if features.dtype.name != metadata.feature_dtype or targets.dtype.name != metadata.target_dtype:
        raise ValueError("array dtype does not match dataset metadata")
    if validity.dtype != np.bool_ or market.dtype != np.float32:
        raise ValueError("validity.npy must be bool and market.npy must be float32")

    rows = pl.read_parquet(root / "rows.parquet")
    _validate_rows(rows, row_count)
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
    if rows.height != row_count or rows.get_column("global_index").to_list() != list(range(row_count)):
        raise ValueError("rows.parquet must cover every global row exactly once")


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
        if not validity[start : anchor + 1, 0].all() or not validity[anchor, 1]:
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
