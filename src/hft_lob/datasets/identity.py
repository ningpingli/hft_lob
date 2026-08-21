"""数据包内容寻址标识。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any


def raw_file_hash(path: str | Path) -> str:
    """计算原始文件 SHA-256。"""
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_config_hash(config: Mapping[str, Any]) -> str:
    """计算与 mapping 顺序无关的稳定 SHA-256。"""
    encoded = json.dumps(
        _canonicalize(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def dataset_version(*, ticker: str, raw_hashes: list[str], processing_config_hash: str) -> str:
    """由原始内容和处理语义生成阶段一版本。"""
    if not ticker.strip() or not processing_config_hash.strip():
        raise ValueError("ticker and processing_config_hash must not be empty")
    if not raw_hashes or any(not value.strip() for value in raw_hashes):
        raise ValueError("raw_hashes must contain at least one non-empty hash")
    return stable_config_hash(
        {
            "ticker": ticker,
            "raw_hashes": sorted(raw_hashes),
            "processing_config_hash": processing_config_hash,
        }
    )


def _canonicalize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(asdict(value))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("config mapping keys must be strings")
        return {key: _canonicalize(item) for key, item in value.items()}
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
