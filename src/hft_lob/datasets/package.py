"""不可变预构建数据包的最小契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl
import torch

from hft_lob.preprocessing.manifest import stable_config_hash

PACKAGE_SCHEMA_VERSION = 1
SUCCESS_MARKER = "_SUCCESS"

SESSION_KEYS = frozenset(
    {
        "features",
        "targets",
        "row_valid",
        "target_valid",
        "timestamps",
        "mid_price",
        "future_mid",
        "bid1",
        "ask1",
        "trade_date",
        "session_id",
    }
)

FOLD_INDEX_COLUMNS = (
    "session_file",
    "anchor_index",
    "trade_date",
    "session_id",
    "anchor_timestamp",
)

FOLD_INDEX_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "session_file": pl.String,
    "anchor_index": pl.Int64,
    "trade_date": pl.String,
    "session_id": pl.String,
    "anchor_timestamp": pl.Datetime("us"),
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


def fold_index_path(package_dir: str | Path, fold_index: int, split: str) -> Path:
    """返回固定的 fold index 路径。"""
    if fold_index <= 0:
        raise ValueError("fold_index must be > 0")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be 'train', 'validation', or 'test'")
    return Path(package_dir) / "folds" / f"fold_{fold_index:03d}" / f"{split}.parquet"


def validate_fold_index(frame: pl.DataFrame) -> None:
    """校验 fold 索引的固定列、类型和样本身份。"""
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
    if not frame.filter(pl.col("anchor_index") < 0).is_empty():
        raise ValueError("anchor_index must be >= 0")
    duplicate = frame.select(
        pl.struct("session_file", "anchor_index").is_duplicated().any()
    ).item()
    if bool(duplicate):
        raise ValueError("fold index contains duplicate samples")


def validate_session_payload(
    payload: object,
    *,
    feature_count: int,
    feature_dtype: str | None = None,
    target_dtype: str | None = None,
) -> dict[str, object]:
    """校验单个 session ``.pt`` 的张量和逐行字段契约。"""
    if not isinstance(payload, dict):
        raise ValueError("session payload must be a dictionary")
    missing = sorted(SESSION_KEYS.difference(payload))
    unknown = sorted(set(payload).difference(SESSION_KEYS))
    if missing or unknown:
        raise ValueError(f"invalid session fields: missing={missing}, unknown={unknown}")

    features = payload["features"]
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError("session features must be a [R,F] tensor")
    if features.shape[1] != feature_count or not features.dtype.is_floating_point:
        raise ValueError("session features do not match feature count or dtype")
    if feature_dtype is not None and str(features.dtype).removeprefix("torch.") != feature_dtype:
        raise ValueError("session feature dtype does not match dataset metadata")
    row_count = features.shape[0]
    tensor_shapes = {
        "targets": (row_count, 1),
        "row_valid": (row_count,),
        "target_valid": (row_count,),
        "mid_price": (row_count,),
        "future_mid": (row_count,),
        "bid1": (row_count,),
        "ask1": (row_count,),
    }
    for name, shape in tensor_shapes.items():
        value = payload[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"session {name} must have shape {shape}")
    targets = payload["targets"]
    if target_dtype is not None and str(targets.dtype).removeprefix("torch.") != target_dtype:
        raise ValueError("session target dtype does not match dataset metadata")
    if payload["row_valid"].dtype != torch.bool or payload["target_valid"].dtype != torch.bool:
        raise ValueError("session validity tensors must use torch.bool")
    timestamps = payload["timestamps"]
    if not isinstance(timestamps, (list, tuple)) or len(timestamps) != row_count:
        raise ValueError("session timestamps must contain one value per row")
    for name in ("trade_date", "session_id"):
        if not isinstance(payload[name], str) or not payload[name].strip():
            raise ValueError(f"session {name} must be a non-empty string")
    return payload
