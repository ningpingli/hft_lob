"""Prediction artifact：按配置 labels 保存完整的多标签预测矩阵。"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from hft_lob.systems.contracts import SampleMeta

_ARTIFACT_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "model_name": pl.String,
    "model_version": pl.String,
    "dataset_version": pl.String,
    "fold_index": pl.Int64,
    "split": pl.String,
    "ticker": pl.String,
    "trade_date": pl.String,
    "session_id": pl.String,
    "anchor_timestamp": pl.Datetime("us"),
    "mid_t": pl.Float64,
    "bid1": pl.Float64,
    "ask1": pl.Float64,
    "spread": pl.Float64,
}


@dataclass(frozen=True)
class PredictionArtifact:
    """模型和 baseline 共用的多标签内存预测产物。"""

    predictions: np.ndarray
    targets: np.ndarray
    labels: tuple[int, ...]
    metadata: tuple[SampleMeta, ...]
    model_name: str
    model_version: str
    dataset_version: str
    fold_index: int
    split: str

    def __post_init__(self) -> None:
        predictions = _as_matrix(self.predictions, field="predictions")
        targets = _as_matrix(self.targets, field="targets")
        labels = tuple(int(label) for label in self.labels)
        metadata = tuple(self.metadata)
        if predictions.size == 0:
            raise ValueError("prediction artifact must not be empty")
        if (
            targets.shape != predictions.shape
            or predictions.shape[1] != len(labels)
            or predictions.shape[0] != len(metadata)
        ):
            raise ValueError("predictions, targets, labels and metadata must align")
        if not labels or len(set(labels)) != len(labels) or any(label <= 0 for label in labels):
            raise ValueError("labels must be non-empty, unique, and positive")
        if not np.isfinite(predictions).all():
            raise ValueError("predictions must contain only finite values")
        if not np.isfinite(targets).all():
            raise ValueError("targets must contain only finite values")

        sample_keys: set[tuple[str, str, str, str]] = set()
        tickers: set[str] = set()
        for index, meta in enumerate(metadata):
            if not meta.ticker or not meta.trade_date or not meta.session_id:
                raise ValueError(f"metadata[{index}] identity fields must not be empty")
            timestamp = _parse_anchor_timestamp(meta.anchor_timestamp)
            if timestamp.date().isoformat() != meta.trade_date:
                raise ValueError(f"metadata[{index}] trade_date does not match anchor_timestamp")
            numeric = (meta.mid_t, meta.bid1, meta.ask1, meta.spread)
            if not np.isfinite(np.asarray(numeric, dtype=np.float64)).all():
                raise ValueError(f"metadata[{index}] contains non-finite numeric values")
            key = (meta.ticker, meta.trade_date, meta.session_id, meta.anchor_timestamp)
            if key in sample_keys:
                raise ValueError(f"duplicate prediction sample metadata: {key}")
            sample_keys.add(key)
            tickers.add(meta.ticker)
        if len(tickers) != 1:
            raise ValueError("prediction artifact must contain exactly one ticker")

        predictions.setflags(write=False)
        targets.setflags(write=False)
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "metadata", metadata)


def save_prediction_artifact(*, artifact: PredictionArtifact, path: str) -> str:
    """保存带标签列的预测 parquet。"""
    destination = Path(path)
    if destination.suffix.lower() != ".parquet":
        raise ValueError("prediction artifact path must end with .parquet")

    records: list[dict[str, object]] = []
    for index, meta in enumerate(artifact.metadata):
        record: dict[str, object] = {
            "model_name": artifact.model_name,
            "model_version": artifact.model_version,
            "dataset_version": artifact.dataset_version,
            "fold_index": artifact.fold_index,
            "split": artifact.split,
            "ticker": meta.ticker,
            "trade_date": meta.trade_date,
            "session_id": meta.session_id,
            "anchor_timestamp": _parse_anchor_timestamp(meta.anchor_timestamp),
            "mid_t": meta.mid_t,
            "bid1": meta.bid1,
            "ask1": meta.ask1,
            "spread": meta.spread,
        }
        for position, label in enumerate(artifact.labels):
            suffix = f"{label}s"
            record[f"target_{suffix}"] = float(artifact.targets[index, position])
            record[f"prediction_{suffix}"] = float(artifact.predictions[index, position])
        records.append(record)
    schema = dict(_ARTIFACT_SCHEMA)
    for label in artifact.labels:
        suffix = f"{label}s"
        schema.update({f"target_{suffix}": pl.Float64, f"prediction_{suffix}": pl.Float64})
    frame = pl.DataFrame(records, schema=schema, strict=True)
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
    return str(destination.resolve())


def load_prediction_artifact(path: str) -> PredictionArtifact:
    """读取并校验多标签预测 parquet。"""
    source = Path(path)
    if source.suffix.lower() != ".parquet":
        raise ValueError("prediction artifact path must end with .parquet")
    if not source.is_file():
        raise FileNotFoundError(source)
    full_frame = pl.read_parquet(source)
    missing = sorted(set(_ARTIFACT_SCHEMA).difference(full_frame.columns))
    if missing:
        raise ValueError(f"prediction artifact is missing columns: {missing}")
    if full_frame.height == 0:
        raise ValueError("prediction artifact must not be empty")

    labels = _labels_from_columns(full_frame.columns)
    for label in labels:
        suffix = f"{label}s"
        for prefix in ("target", "prediction"):
            if f"{prefix}_{suffix}" not in full_frame.columns:
                raise ValueError(f"prediction artifact is missing column: {prefix}_{suffix}")
    identity = {
        field: _single_artifact_field(full_frame, field)
        for field in ("model_name", "model_version", "dataset_version", "fold_index", "split")
    }
    frame = full_frame.select(list(_ARTIFACT_SCHEMA))
    metadata = tuple(
        SampleMeta(
            ticker=str(row["ticker"]),
            trade_date=str(row["trade_date"]),
            session_id=str(row["session_id"]),
            anchor_timestamp=_format_anchor_timestamp(row["anchor_timestamp"]),
            mid_t=float(row["mid_t"]),
            bid1=float(row["bid1"]),
            ask1=float(row["ask1"]),
            spread=float(row["spread"]),
        )
        for row in frame.to_dicts()
    )
    targets = np.column_stack(
        [np.asarray(full_frame[f"target_{label}s"].to_numpy(), dtype=np.float64) for label in labels]
    )
    predictions = np.column_stack(
        [np.asarray(full_frame[f"prediction_{label}s"].to_numpy(), dtype=np.float64) for label in labels]
    )
    return PredictionArtifact(
        predictions=predictions,
        targets=targets,
        labels=labels,
        metadata=metadata,
        model_name=str(identity["model_name"]),
        model_version=str(identity["model_version"]),
        dataset_version=str(identity["dataset_version"]),
        fold_index=cast(int, identity["fold_index"]),
        split=str(identity["split"]),
    )


def _labels_from_columns(columns: list[str]) -> tuple[int, ...]:
    labels: list[int] = []
    for name in columns:
        if name.startswith("target_") and name.endswith("s"):
            text = name.removeprefix("target_").removesuffix("s")
            if text.isdigit():
                labels.append(int(text))
    result = tuple(labels)
    if not result:
        raise ValueError("prediction artifact contains no target label columns")
    if len(set(result)) != len(result):
        raise ValueError("prediction artifact contains duplicate target labels")
    return result


def _single_artifact_field(frame: pl.DataFrame, field: str) -> object:
    values = frame.get_column(field).unique().to_list()
    if len(values) != 1 or values[0] is None:
        raise ValueError(f"prediction artifact field must be constant: {field}")
    return values[0]


def _format_anchor_timestamp(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        _parse_anchor_timestamp(value)
        return value
    raise ValueError(f"invalid anchor_timestamp: {value!r}")


def _as_matrix(values: np.ndarray, *, field: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{field} must have shape [N, L], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must contain only finite values")
    return np.ascontiguousarray(array.copy())


def _parse_anchor_timestamp(value: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid anchor_timestamp: {value!r}") from exc
    if timestamp.tzinfo is not None:
        raise ValueError("anchor_timestamp must be timezone-naive")
    return timestamp
