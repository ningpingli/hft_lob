"""多 horizon 收益标签生成。"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from hft_lob.configs.experiment import TargetConfig
from hft_lob.preprocessing.clean import SessionSegment

_LABEL_TYPE_SHORT: dict[str, str] = {"log_mid_return": "log", "simple_mid_return": "simple"}


def label_column(config: TargetConfig) -> str:
    """Return the primary training target column name."""
    return horizon_label_column(config, config.primary_horizon_seconds)


def horizon_label_column(config: TargetConfig, horizon_seconds: int) -> str:
    """Return the configured target column name for one horizon."""
    try:
        short = _LABEL_TYPE_SHORT[config.type]
    except KeyError as exc:
        raise ValueError(f"unsupported target type: {config.type!r}") from exc
    if horizon_seconds not in config.horizons_seconds:
        raise ValueError(f"horizon_seconds is not configured: {horizon_seconds}")
    return f"Target_{horizon_seconds}s_{short}"


class LabelTransformer:
    """Append one target and validity set for every configured horizon."""

    def __init__(self, config: TargetConfig) -> None:
        if config.type not in _LABEL_TYPE_SHORT:
            raise ValueError(f"unsupported target type: {config.type!r}")
        self.config = config

    def transform(self, segment: SessionSegment) -> SessionSegment:
        """Add future prices, returns, and per-horizon validity columns."""
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
        for horizon in self.config.horizons_seconds:
            suffix = f"{horizon}s"
            future_timestamp = f"future_timestamp_{suffix}"
            future_mid = f"future_mid_{suffix}"
            future_book_valid = f"future_book_valid_{suffix}"
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
                (
                    pl.col("timestamp") + pl.lit(timedelta(seconds=horizon))
                ).alias(target_timestamp)
            ).join_asof(
                candidates,
                left_on=target_timestamp,
                right_on=future_timestamp,
                strategy="nearest",
                tolerance=f"{self.config.tolerance_seconds}s",
            )
            valid_name = f"target_valid_{suffix}"
            log_name = f"Target_{suffix}_log"
            simple_name = f"Target_{suffix}_simple"
            future_valid = (
                pl.col(future_book_valid).fill_null(False)
                & pl.col(future_mid).is_not_null()
                & pl.col(future_mid).is_finite()
                & (pl.col(future_mid) > 0)
                & pl.col(future_timestamp).is_not_null()
            )
            result = result.with_columns((current_valid & future_valid).alias(valid_name))
            result = result.with_columns(
                pl.when(pl.col(valid_name))
                .then((pl.col(future_mid) / pl.col("mid_price")).log())
                .otherwise(None)
                .cast(pl.Float64)
                .alias(log_name),
                pl.when(pl.col(valid_name))
                .then(pl.col(future_mid) / pl.col("mid_price") - 1.0)
                .otherwise(None)
                .cast(pl.Float64)
                .alias(simple_name),
            ).drop(target_timestamp, future_book_valid)

        primary_suffix = f"{self.config.primary_horizon_seconds}s"
        result = result.with_columns(
            pl.col(f"future_timestamp_{primary_suffix}").alias("future_timestamp"),
            pl.col(f"future_mid_{primary_suffix}").alias("future_mid"),
            pl.col(f"target_valid_{primary_suffix}").alias("target_valid"),
        )
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
    columns: list[str] = []
    for horizon in config.horizons_seconds:
        suffix = f"{horizon}s"
        columns.extend(
            [
                f"future_timestamp_{suffix}",
                f"future_mid_{suffix}",
                f"Target_{suffix}_log",
                f"Target_{suffix}_simple",
                f"target_valid_{suffix}",
                f"future_book_valid_{suffix}",
                f"_target_timestamp_{suffix}",
            ]
        )
    columns.extend(("future_timestamp", "future_mid", "target_valid"))
    return columns


def _empty_output_frame(frame: pl.DataFrame, config: TargetConfig) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for horizon in config.horizons_seconds:
        suffix = f"{horizon}s"
        expressions.extend(
            [
                pl.lit(None).cast(pl.Datetime("us")).alias(f"future_timestamp_{suffix}"),
                pl.lit(None).cast(pl.Float64).alias(f"future_mid_{suffix}"),
                pl.lit(None).cast(pl.Float64).alias(f"Target_{suffix}_log"),
                pl.lit(None).cast(pl.Float64).alias(f"Target_{suffix}_simple"),
                pl.lit(False).alias(f"target_valid_{suffix}"),
            ]
        )
    expressions.extend(
        [
            pl.lit(None).cast(pl.Datetime("us")).alias("future_timestamp"),
            pl.lit(None).cast(pl.Float64).alias("future_mid"),
            pl.lit(False).alias("target_valid"),
        ]
    )
    return frame.with_columns(expressions)
