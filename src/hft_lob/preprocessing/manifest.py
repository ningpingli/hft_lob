"""数据 manifest（需求文档 §30/§31）：数据集版本追踪；split 以 manifest 表达。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any, cast

import polars as pl

from hft_lob.configs.experiment import FeatureConfig, TargetConfig

#: manifest 列（固定顺序；§31 Data Manifest）。
_MANIFEST_COLUMNS: tuple[str, ...] = (
    "ticker", "trade_date", "session_id", "source_file", "processed_file",
    "raw_hash", "processing_config_hash", "dataset_version",
    "row_count", "valid_row_count", "data_start", "data_end",
    "feature_version", "label_version", "quality_status",
)

_MANIFEST_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "ticker": pl.String,
    "trade_date": pl.String,
    "session_id": pl.String,
    "source_file": pl.String,
    "processed_file": pl.String,
    "raw_hash": pl.String,
    "processing_config_hash": pl.String,
    "dataset_version": pl.String,
    "row_count": pl.Int64,
    "valid_row_count": pl.Int64,
    "data_start": pl.Datetime("us"),
    "data_end": pl.Datetime("us"),
    "feature_version": pl.String,
    "label_version": pl.String,
    "quality_status": pl.String,
}


def raw_file_hash(path: str, *, algorithm: str = "sha256") -> str:
    """流式计算 raw 文件内容哈希，不依赖路径、mtime 或文件名。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(path)
    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"unsupported hash algorithm: {algorithm!r}") from exc
    with source.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_config_hash(config: Mapping[str, Any]) -> str:
    """对 key 排序后的 canonical 配置计算稳定 SHA-256 哈希。"""
    canonical = _canonicalize(config)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataset_version(
    ticker: str,
    raw_hashes: Sequence[str],
    *,
    processing_config_hash: str,
) -> str:
    """生成内容寻址的数据集版本。

    版本由 ticker、排序后的 raw 内容哈希及完整处理配置哈希共同决定；字段映射
    属于处理配置，因此 raw 内容、映射或处理语义变化都会产生新版本。
    """
    if not ticker.strip():
        raise ValueError("ticker must not be empty")
    if not processing_config_hash.strip():
        raise ValueError("processing_config_hash must not be empty")
    hashes = sorted(raw_hashes)
    if not hashes or any(not value.strip() for value in hashes):
        raise ValueError("raw_hashes must contain at least one non-empty hash")
    return stable_config_hash(
        {
            "ticker": ticker,
            "raw_hashes": hashes,
            "processing_config_hash": processing_config_hash,
        }
    )


def feature_version(config: FeatureConfig) -> str:
    """由启用状态、特征名称及顺序生成稳定版本，不以特征数量代替版本。"""
    return stable_config_hash(
        {
            "use_derived": config.use_derived,
            "derived_features": list(config.derived_features) if config.use_derived else [],
        }
    )


def label_version(config: TargetConfig) -> str:
    """标签版本：``<type>_<h>s_tol<tol>``。"""
    return f"{config.type}_{config.horizon_seconds}s_tol{config.tolerance_seconds}"


def build_manifest(*, ticker: str, records: list[dict[str, object]]) -> pl.DataFrame:
    """由逐日记录构建 manifest DataFrame（列序固定，空列表时列仍齐全）。

    Args:
        ticker: 股票代码。
        records: 逐日记录字典列表（键 = manifest 列）。

    Returns:
        manifest DataFrame。
    """
    if not ticker.strip():
        raise ValueError("ticker must not be empty")
    if not records:
        return pl.DataFrame(schema=_MANIFEST_SCHEMA)

    normalized: list[dict[str, object]] = []
    expected_record_keys = set(_MANIFEST_COLUMNS).difference({"ticker"})
    for index, record in enumerate(records):
        unknown = sorted(set(record).difference(_MANIFEST_COLUMNS))
        if unknown:
            raise ValueError(f"manifest record {index} has unknown columns: {unknown}")
        missing = sorted(expected_record_keys.difference(record))
        if missing:
            raise ValueError(f"manifest record {index} missing columns: {missing}")
        record_ticker = record.get("ticker", ticker)
        if record_ticker != ticker:
            raise ValueError(
                f"manifest record {index} ticker {record_ticker!r} != {ticker!r}"
            )
        normalized.append({"ticker": ticker, **record})

    try:
        manifest = pl.DataFrame(normalized, schema=_MANIFEST_SCHEMA, strict=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"manifest record type mismatch: {exc}") from exc
    manifest = manifest.select(_MANIFEST_COLUMNS).sort("trade_date", "session_id")
    _validate_manifest(manifest)
    return manifest


def write_manifest(manifest: pl.DataFrame, path: str) -> None:
    """落盘 manifest（parquet）。

    Args:
        manifest: manifest DataFrame。
        path: 输出路径。
    """
    _validate_manifest(manifest)
    destination = Path(path)
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
        manifest.select(_MANIFEST_COLUMNS).write_parquet(temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def read_manifest(path: str) -> pl.DataFrame:
    """读取 manifest（parquet）。

    Args:
        path: manifest 路径。

    Returns:
        manifest DataFrame。
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(path)
    manifest = pl.read_parquet(source)
    _validate_manifest(manifest)
    return manifest.select(_MANIFEST_COLUMNS)


def _canonicalize(value: Any) -> Any:
    """把配置递归转换为可稳定 JSON 编码的值。"""
    if is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(asdict(value))
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"config mapping keys must be strings, got {type(key).__name__}")
            normalized[key] = _canonicalize(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        return value.isoformat(timespec="microseconds")
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported config value type: {type(value).__name__}")


def _validate_manifest(manifest: pl.DataFrame) -> None:
    """验证固定 schema、唯一 session 记录以及基础计数/时间约束。"""
    missing = sorted(set(_MANIFEST_COLUMNS).difference(manifest.columns))
    unknown = sorted(set(manifest.columns).difference(_MANIFEST_COLUMNS))
    if missing or unknown:
        raise ValueError(f"invalid manifest columns: missing={missing}, unknown={unknown}")
    type_errors = {
        name: (manifest.schema[name], expected)
        for name, expected in _MANIFEST_SCHEMA.items()
        if manifest.schema[name] != expected
    }
    if type_errors:
        raise ValueError(f"invalid manifest dtypes: {type_errors}")
    if manifest.null_count().select(pl.sum_horizontal(pl.all())).item() != 0:
        raise ValueError("manifest must not contain null values")

    invalid_counts = manifest.filter(
        (pl.col("row_count") < 0)
        | (pl.col("valid_row_count") < 0)
        | (pl.col("valid_row_count") > pl.col("row_count"))
    )
    if not invalid_counts.is_empty():
        raise ValueError("manifest counts require 0 <= valid_row_count <= row_count")
    if not manifest.filter(pl.col("data_start") > pl.col("data_end")).is_empty():
        raise ValueError("manifest requires data_start <= data_end")
    duplicates = manifest.select(
        pl.struct("ticker", "trade_date", "session_id").is_duplicated().any()
    ).item()
    if cast(bool, duplicates):
        raise ValueError("manifest contains duplicate ticker/trade_date/session_id records")
