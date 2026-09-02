"""按配置 labels 生成多标签收益列。"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from hft_lob.configs.experiment import TargetConfig
from hft_lob.data_pipeline.clean import SessionSegment

_LABEL_TYPE_SHORT: dict[str, str] = {"log_mid_return": "log", "simple_mid_return": "simple"}


def label_columns(config: TargetConfig) -> tuple[str, ...]:
    """Return target column names in the configured label order."""
    return tuple(target_column(config, label) for label in config.labels)


def target_column(config: TargetConfig, label: int) -> str:
    """Return the selected target column name for one label."""
    try:
        short = _LABEL_TYPE_SHORT[config.type]
    except KeyError as exc:
        raise ValueError(f"unsupported target type: {config.type!r}") from exc
    if label not in config.labels:
        raise ValueError(f"label is not configured: {label}")
    return f"Target_{label}s_{short}"


class LabelTransformer:
    """Append one selected target and validity mask for every configured label."""

    def __init__(self, config: TargetConfig) -> None:
        if config.type not in _LABEL_TYPE_SHORT:
            raise ValueError(f"unsupported target type: {config.type!r}")
        self.config = config

    def transform(self, segment: SessionSegment) -> SessionSegment:
        """Add selected target and per-label validity columns."""
        frame = segment.frame
        required = {"trade_date", "session_id", "timestamp", "mid_price", "book_valid"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"label input missing columns: {missing}")
        self._validate_segment(segment)

        output_columns = _output_columns(self.config)
        existing_outputs = [name for name in output_columns if name in frame.columns]
        if existing_outputs:
            frame = frame.drop(existing_outputs)
        if frame.is_empty():
            return SessionSegment(
                trade_date=segment.trade_date,
                session_id=segment.session_id,
                frame=_empty_output_frame(frame, self.config),
            )

        current_valid = (
            pl.col("book_valid").fill_null(False)
            & pl.col("mid_price").is_not_null()
            & pl.col("mid_price").is_finite()
            & (pl.col("mid_price") > 0)
        )
        result = frame
        target_expression = (
            lambda future_mid: (future_mid / pl.col("mid_price")).log()
            if self.config.type == "log_mid_return"
            else future_mid / pl.col("mid_price") - 1.0
        )
        for label in self.config.labels:
            suffix = f"{label}s"
            future_timestamp = f"_future_timestamp_{suffix}"
            future_mid = f"_future_mid_{suffix}"
            future_book_valid = f"_future_book_valid_{suffix}"
            target_timestamp = f"_target_timestamp_{suffix}"
            candidates = frame.select(
                pl.col("timestamp").alias(future_timestamp),
                pl.col("mid_price").alias(future_mid),
                pl.col("book_valid").alias(future_book_valid),
            ).filter(
                pl.col(future_book_valid).fill_null(False)
                & pl.col(future_mid).is_not_null()
                & pl.col(future_mid).is_finite()
                & (pl.col(future_mid) > 0)
            ).sort(future_timestamp)
            result = result.with_columns(
                (pl.col("timestamp") + pl.lit(timedelta(seconds=label))).alias(target_timestamp)
            ).join_asof(
                candidates,
                left_on=target_timestamp,
                right_on=future_timestamp,
                strategy="nearest",
                tolerance=f"{self.config.tolerance_seconds}s",
            )
            target_name = target_column(self.config, label)
            future_valid = (
                pl.col(future_book_valid).fill_null(False)
                & pl.col(future_mid).is_not_null()
                & pl.col(future_mid).is_finite()
                & (pl.col(future_mid) > 0)
                & pl.col(future_timestamp).is_not_null()
            )
            result = result.with_columns(
                pl.when(current_valid & future_valid)
                .then(target_expression(pl.col(future_mid)))
                .otherwise(None)
                .cast(pl.Float64)
                .alias(target_name),
            ).drop([target_timestamp, future_timestamp, future_mid, future_book_valid])
        return SessionSegment(
            trade_date=segment.trade_date,
            session_id=segment.session_id,
            frame=result,
        )

    @staticmethod
    def _validate_segment(segment: SessionSegment) -> None:
        frame = segment.frame
        if frame.is_empty():
            return
        trade_dates = frame.get_column("trade_date").unique().to_list()
        session_ids = frame.get_column("session_id").unique().to_list()
        if len(trade_dates) != 1 or str(trade_dates[0]) != segment.trade_date:
            raise ValueError("frame trade_date must contain exactly the SessionSegment trade_date")
        if len(session_ids) != 1 or str(session_ids[0]) != segment.session_id:
            raise ValueError("frame session_id must contain exactly the SessionSegment session_id")
        if not frame.get_column("timestamp").is_sorted():
            raise ValueError("timestamps must be sorted within a SessionSegment")
        if frame.get_column("timestamp").dtype != pl.Datetime("us"):
            raise ValueError("timestamp must use a Polars Datetime dtype")


def _output_columns(config: TargetConfig) -> list[str]:
    return [target_column(config, label) for label in config.labels]


def _empty_output_frame(frame: pl.DataFrame, config: TargetConfig) -> pl.DataFrame:
    return frame.with_columns(
        [
            pl.lit(None).cast(pl.Float64).alias(target_column(config, label))
            for label in config.labels
        ]
    )
