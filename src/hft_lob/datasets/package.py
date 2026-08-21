"""不可变预构建数据包的最小契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from hft_lob.datasets.identity import stable_config_hash

PACKAGE_SCHEMA_VERSION = 2
SUCCESS_MARKER = "_SUCCESS"

ARRAY_FILES = ("features.npy", "targets.npy", "validity.npy", "market.npy")

FOLD_INDEX_COLUMNS = (
    "global_anchor_index",
    "session_start_index",
    "anchor_index",
    "trade_date",
    "session_id",
    "anchor_timestamp",
)

FOLD_INDEX_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "global_anchor_index": pl.Int64,
    "session_start_index": pl.Int64,
    "anchor_index": pl.Int64,
    "trade_date": pl.String,
    "session_id": pl.String,
    "anchor_timestamp": pl.Datetime("us"),
}

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
    """由数据来源、处理语义和 fold 方案生成稳定的数据包 ID。"""
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
    """``dataset.json`` 的固定 schema。"""

    dataset_id: str
    ticker: str
    feature_columns: tuple[str, ...]
    target_column: str
    feature_dtype: str
    target_dtype: str
    snapshot_interval_seconds: int
    history_snapshots: int
    normalization_mode: str
    normalization_window: int
    source_hash: str
    processing_config_hash: str
    fold_plan_hash: str
    schema_version: int = PACKAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        text_fields = (
            "dataset_id",
            "ticker",
            "target_column",
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
        if not self.feature_columns or len(set(self.feature_columns)) != len(
            self.feature_columns
        ):
            raise ValueError("feature_columns must be non-empty and unique")
        if self.snapshot_interval_seconds <= 0:
            raise ValueError("snapshot_interval_seconds must be > 0")
        if self.history_snapshots <= 0:
            raise ValueError("history_snapshots must be > 0")
        if self.normalization_window < 2:
            raise ValueError("normalization_window must be >= 2")
        expected_id = compute_dataset_id(
            ticker=self.ticker,
            source_hash=self.source_hash,
            processing_config_hash=self.processing_config_hash,
            fold_plan_hash=self.fold_plan_hash,
        )
        if self.dataset_id != expected_id:
            raise ValueError("dataset_id does not match package identity fields")

    def to_dict(self) -> dict[str, object]:
        """返回可直接 JSON 编码的表示。"""
        value = asdict(self)
        value["feature_columns"] = list(self.feature_columns)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> DatasetPackageMetadata:
        """从严格、无扩展字段的 JSON 对象恢复 metadata。"""
        expected = set(cls.__dataclass_fields__)
        missing = sorted(expected.difference(value))
        unknown = sorted(set(value).difference(expected))
        if missing or unknown:
            raise ValueError(f"invalid dataset metadata fields: missing={missing}, unknown={unknown}")
        columns = value["feature_columns"]
        if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
            raise ValueError("feature_columns must be a list of strings")
        return cls(**{**value, "feature_columns": tuple(columns)})  # type: ignore[arg-type]


@dataclass(frozen=True)
class DatasetPackage:
    """已完整校验、可由训练各层共享的只读数据包句柄。"""

    root: Path
    metadata: DatasetPackageMetadata

    def __post_init__(self) -> None:
        resolved = self.root.resolve()
        if resolved.name != self.metadata.dataset_id:
            raise ValueError("package root name must equal dataset_id")
        object.__setattr__(self, "root", resolved)


def fold_index_path(package_dir: str | Path, fold_index: int, split: str) -> Path:
    """返回固定的 fold index 路径。"""
    if fold_index <= 0:
        raise ValueError("fold_index must be > 0")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be 'train', 'validation', or 'test'")
    return Path(package_dir) / "folds" / f"fold_{fold_index:03d}" / f"{split}.parquet"


def validate_fold_index(frame: pl.DataFrame) -> None:
    """校验 fold 索引的固定列、类型和样本身份。"""
    if frame.is_empty():
        raise ValueError("fold index must not be empty")
    if tuple(frame.columns) != FOLD_INDEX_COLUMNS:
        raise ValueError(f"invalid fold index columns: {frame.columns}")
    type_errors = {
        name: (frame.schema[name], expected)
        for name, expected in FOLD_INDEX_SCHEMA.items()
        if frame.schema[name] != expected
    }
    if type_errors:
        raise ValueError(f"invalid fold index dtypes: {type_errors}")
    if frame.null_count().select(pl.sum_horizontal(pl.all())).item() != 0:
        raise ValueError("fold index must not contain null values")
    if not frame.filter(
        (pl.col("global_anchor_index") < 0)
        | (pl.col("session_start_index") < 0)
        | (pl.col("anchor_index") < 0)
    ).is_empty():
        raise ValueError("fold indexes must be >= 0")
    duplicate = frame.select(
        pl.col("global_anchor_index").is_duplicated().any()
    ).item()
    if bool(duplicate):
        raise ValueError("fold index contains duplicate samples")
