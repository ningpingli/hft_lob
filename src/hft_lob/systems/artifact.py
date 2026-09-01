"""Prediction artifact：保存完整 label 向量、有效性与样本上下文。"""

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
    "future_mid": pl.Float64,
    "bid1": pl.Float64,
    "ask1": pl.Float64,
    "spread": pl.Float64,
}


@dataclass(frozen=True)
class PredictionArtifact:
    """模型和 baseline 共用的完整多任务预测产物。"""

    predictions: np.ndarray
    targets: np.ndarray
    target_valid: np.ndarray
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
        target_valid = _as_validity(self.target_valid)
        labels = tuple(self.labels)
        metadata = tuple(self.metadata)
        if predictions.shape != targets.shape or predictions.shape != target_valid.shape:
            raise ValueError("predictions, targets and target_valid must have the same shape")
        if predictions.shape[0] == 0:
            raise ValueError("prediction artifact must not be empty")
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("labels must be non-empty and unique")
        if any(not isinstance(label, int) or isinstance(label, bool) or label <= 0 for label in labels):
            raise ValueError("labels must contain positive integers")
        if predictions.shape[1] != len(labels):
            raise ValueError("labels must match the prediction width")
        if len(metadata) != predictions.shape[0]:
            raise ValueError("predictions, targets and metadata must have the same sample count")
        if not np.isfinite(predictions).all():
            raise ValueError("predictions must contain only finite values")
        if np.any(target_valid) and not np.isfinite(targets[target_valid]).all():
            raise ValueError("valid targets must contain only finite values")

        sample_keys: set[tuple[str, str, str, str]] = set()
        tickers: set[str] = set()
        for index, meta in enumerate(metadata):
            if not meta.ticker or not meta.trade_date or not meta.session_id:
                raise ValueError(f"metadata[{index}] identity fields must not be empty")
            timestamp = _parse_anchor_timestamp(meta.anchor_timestamp)
            if timestamp.date().isoformat() != meta.trade_date:
                raise ValueError(f"metadata[{index}] trade_date does not match anchor_timestamp")
            numeric = (meta.mid_t, meta.future_mid, meta.bid1, meta.ask1, meta.spread)
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
        target_valid.setflags(write=False)
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "target_valid", target_valid)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "metadata", metadata)


def save_prediction_artifact(*, artifact: PredictionArtifact, path: str) -> str:
    """保存带 label、预测、target、validity 列的 prediction parquet。"""
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
            "future_mid": meta.future_mid,
            "bid1": meta.bid1,
            "ask1": meta.ask1,
            "spread": meta.spread,
        }
        for position, label in enumerate(artifact.labels):
            record[f"target_{label}s"] = float(artifact.targets[index, position])
            record[f"prediction_{label}s"] = float(artifact.predictions[index, position])
            record[f"target_valid_{label}s"] = bool(artifact.target_valid[index, position])
        records.append(record)

    schema = dict(_ARTIFACT_SCHEMA)
    for label in artifact.labels:
        schema[f"target_{label}s"] = pl.Float64
        schema[f"prediction_{label}s"] = pl.Float64
        schema[f"target_valid_{label}s"] = pl.Boolean
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
    """加载并校验矩阵 prediction artifact。"""
    source = Path(path)
    if source.suffix.lower() != ".parquet":
        raise ValueError("prediction artifact path must end with .parquet")
    if not source.is_file():
        raise FileNotFoundError(source)

    full_frame = pl.read_parquet(source)
    missing = sorted(set(_ARTIFACT_SCHEMA).difference(full_frame.columns))
    if missing:
        raise ValueError(f"prediction artifact is missing columns: {missing}")
    label_names = [
        name
        for name in full_frame.columns
        if name.startswith("target_")
        and name.endswith("s")
        and not name.startswith("target_valid_")
    ]
    if not label_names:
        raise ValueError("prediction artifact contains no target label columns")
    labels = tuple(int(name.removeprefix("target_").removesuffix("s")) for name in label_names)
    prediction_names = [f"prediction_{label}s" for label in labels]
    validity_names = [f"target_valid_{label}s" for label in labels]
    missing_dynamic = sorted(
        set(prediction_names + validity_names).difference(full_frame.columns)
    )
    if missing_dynamic:
        raise ValueError(f"prediction artifact is missing columns: {missing_dynamic}")
    frame = full_frame.select(list(_ARTIFACT_SCHEMA))
    if frame.height == 0:
        raise ValueError("prediction artifact must not be empty")

    identity = {
        field: _single_artifact_field(frame, field)
        for field in ("model_name", "model_version", "dataset_version", "fold_index", "split")
    }
    metadata = tuple(
        SampleMeta(
            ticker=str(row["ticker"]),
            trade_date=str(row["trade_date"]),
            session_id=str(row["session_id"]),
            anchor_timestamp=_format_anchor_timestamp(row["anchor_timestamp"]),
            mid_t=float(row["mid_t"]),
            future_mid=float(row["future_mid"]),
            bid1=float(row["bid1"]),
            ask1=float(row["ask1"]),
            spread=float(row["spread"]),
        )
        for row in frame.to_dicts()
    )
    targets = np.column_stack([full_frame[name].to_numpy() for name in label_names])
    predictions = np.column_stack([full_frame[name].to_numpy() for name in prediction_names])
    target_valid = np.column_stack([full_frame[name].to_numpy() for name in validity_names])
    return PredictionArtifact(
        predictions=predictions,
        targets=targets,
        target_valid=target_valid,
        labels=labels,
        metadata=metadata,
        model_name=str(identity["model_name"]),
        model_version=str(identity["model_version"]),
        dataset_version=str(identity["dataset_version"]),
        fold_index=cast(int, identity["fold_index"]),
        split=str(identity["split"]),
    )


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
    return np.ascontiguousarray(array.copy())


def _as_validity(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.dtype != np.bool_:
        raise ValueError(f"target_valid must have bool shape [N, L], got {array.shape} {array.dtype}")
    return np.ascontiguousarray(array.copy())


def _parse_anchor_timestamp(value: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid anchor_timestamp: {value!r}") from exc
    if timestamp.tzinfo is not None:
        raise ValueError("anchor_timestamp must be timezone-naive")
    return timestamp
