"""Prediction artifact（需求文档 §28）：parquet 保存完整样本上下文。

禁止只保存 ``[targets, predictions]``——必须保留 ticker / trade_date /
session_id / anchor_timestamp / mid_t / future_mid / bid1 / ask1 / spread /
split / model_version / dataset_version，否则无法定位异常预测。
"""

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
    "target": pl.Float64,
    "prediction": pl.Float64,
    "bid1": pl.Float64,
    "ask1": pl.Float64,
    "spread": pl.Float64,
}


@dataclass(frozen=True)
class PredictionArtifact:
    """模型和 baseline 共用的内存预测产物。"""

    predictions: np.ndarray
    targets: np.ndarray
    metadata: tuple[SampleMeta, ...]
    model_name: str
    model_version: str
    dataset_version: str
    fold_index: int
    split: str

    def __post_init__(self) -> None:
        predictions = _as_vector(self.predictions, field="predictions")
        targets = _as_vector(self.targets, field="targets")
        metadata = tuple(self.metadata)
        if predictions.size == 0:
            raise ValueError("prediction artifact must not be empty")
        if predictions.size != targets.size or predictions.size != len(metadata):
            raise ValueError(
                "predictions, targets and metadata must have the same sample count"
            )
        for field, value in (
            ("model_name", self.model_name),
            ("model_version", self.model_version),
            ("dataset_version", self.dataset_version),
            ("split", self.split),
        ):
            if not value.strip():
                raise ValueError(f"{field} must not be empty")
        if self.fold_index <= 0:
            raise ValueError("fold_index must be > 0")

        sample_keys: set[tuple[str, str, str, str]] = set()
        tickers: set[str] = set()
        for index, meta in enumerate(metadata):
            if not meta.ticker or not meta.trade_date or not meta.session_id:
                raise ValueError(f"metadata[{index}] identity fields must not be empty")
            timestamp = _parse_anchor_timestamp(meta.anchor_timestamp)
            if timestamp.date().isoformat() != meta.trade_date:
                raise ValueError(
                    f"metadata[{index}] trade_date does not match anchor_timestamp"
                )
            numeric = (
                meta.mid_t,
                meta.future_mid,
                meta.bid1,
                meta.ask1,
                meta.spread,
            )
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
        object.__setattr__(self, "metadata", metadata)


def save_prediction_artifact(
    *,
    artifact: PredictionArtifact,
    path: str,
) -> str:
    """保存预测结果 parquet（§28 字段清单）。

    Args:
        artifact: 完整、强类型、已绑定 model/dataset/fold/split 的预测产物。
        path: 输出 parquet 路径。

    Returns:
        输出路径。
    """
    destination = Path(path)
    if destination.suffix.lower() != ".parquet":
        raise ValueError("prediction artifact path must end with .parquet")

    records: list[dict[str, object]] = []
    for prediction, target, meta in zip(
        artifact.predictions,
        artifact.targets,
        artifact.metadata,
        strict=True,
    ):
        records.append(
            {
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
                "target": float(target),
                "prediction": float(prediction),
                "bid1": meta.bid1,
                "ask1": meta.ask1,
                "spread": meta.spread,
            }
        )
    frame = pl.DataFrame(records, schema=_ARTIFACT_SCHEMA, strict=True)
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
    """Load and validate a previously saved prediction artifact."""
    source = Path(path)
    if source.suffix.lower() != ".parquet":
        raise ValueError("prediction artifact path must end with .parquet")
    if not source.is_file():
        raise FileNotFoundError(source)

    frame = pl.read_parquet(source)
    missing = sorted(set(_ARTIFACT_SCHEMA).difference(frame.columns))
    if missing:
        raise ValueError(f"prediction artifact is missing columns: {missing}")
    frame = frame.select(list(_ARTIFACT_SCHEMA))
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
    return PredictionArtifact(
        predictions=np.asarray(frame["prediction"].to_numpy(), dtype=np.float64),
        targets=np.asarray(frame["target"].to_numpy(), dtype=np.float64),
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


def _as_vector(values: np.ndarray, *, field: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"{field} must have shape [N] or [N, 1], got {array.shape}")
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
