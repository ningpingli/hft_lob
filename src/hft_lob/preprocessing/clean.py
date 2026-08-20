"""数据清洗（需求文档 §4/§5/§6）：schema 校验、session 分割、秒去重、有界 ffill、mid。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from hft_lob.configs.experiment import RAW_FEATURE_COLUMNS, SessionConfig
from hft_lob.preprocessing.quality import QualityReport, run_quality_checks

_LOB_COLUMNS: tuple[str, ...] = RAW_FEATURE_COLUMNS[:20]
_VOLUME_COLUMNS: tuple[str, ...] = tuple(
    name for name in _LOB_COLUMNS if name.startswith(("ASKs", "BIDs"))
)
_PRICE_COLUMNS: tuple[str, ...] = tuple(
    name for name in _LOB_COLUMNS if name.startswith(("ASKp", "BIDp"))
)


@dataclass(frozen=True)
class SessionSegment:
    """单交易日、单连续竞价 session 的物理数据单元。

    frame 中所有行必须具有相同 trade_date/session_id 且时间有序；AM/PM
    不允许共存，从接口边界阻止 shift/join/rolling 跨午休。
    """

    trade_date: str
    session_id: str
    frame: pl.DataFrame


@dataclass(frozen=True)
class CleanDayResult:
    """单日清洗结果：独立 session 集合与日级质量报告。"""

    sessions: tuple[SessionSegment, ...]
    quality_report: QualityReport


class DataCleaner:
    """单日原始快照 → 独立 SessionSegment（§4/§5/§6）。

    每个 segment 的输出列：``trade_date / session_id / timestamp /
    seconds`` + 20 盘口 + 3 标量 + ``mid_price / staleness_seconds / is_ffilled /
    book_valid``。

    行为契约：
    - §3 会话分割：按 SessionConfig 划分 AM/PM（半开区间），非连续竞价时段剔除；
    - §4 秒去重：重复 timestamp 保留同秒最后一条；
    - §5 有界 ffill：整条盘口缺失时，gap ≤ max_ffill_gap_seconds 才前向填充
      （价格 + 数量整体，时间不填充），超限行标记 ``book_valid=False``；
    - §6 mid：双侧有效取均值，单边取存活侧价格，交叉/双侧无效 → NaN 且
      ``book_valid=False``；
    - AM/PM 返回两个独立 segment，不在同一个 DataFrame 中拼接。
    """

    def __init__(
        self,
        sessions: SessionConfig,
        snapshot_interval_seconds: int,
        max_ffill_gap_seconds: int,
        *,
        column_mapping: Mapping[str, str],
    ) -> None:
        """初始化清洗器。

        Args:
            sessions: 交易时段配置（§3）。
            snapshot_interval_seconds: 目标快照周期，用于 session 内补齐时间网格。
            max_ffill_gap_seconds: 缺失策略 gap 上限（§5）。
            column_mapping: 配置文件中的 ``原始列名 -> canonical 列名`` 映射。
        """
        if snapshot_interval_seconds <= 0:
            raise ValueError("snapshot_interval_seconds must be > 0")
        if max_ffill_gap_seconds < 0:
            raise ValueError("max_ffill_gap_seconds must be >= 0")
        if sessions.allow_cross_session:
            raise ValueError("MVP requires sessions.allow_cross_session=False")

        self.sessions = sessions
        self.snapshot_interval_seconds = snapshot_interval_seconds
        self.max_ffill_gap_seconds = max_ffill_gap_seconds
        self.column_mapping = dict(column_mapping)
        self._session_bounds = {
            "AM": (_parse_clock(sessions.morning[0]), _parse_clock(sessions.morning[1])),
            "PM": (_parse_clock(sessions.afternoon[0]), _parse_clock(sessions.afternoon[1])),
        }

    def clean_day(self, path: str, *, ticker: str) -> CleanDayResult:
        """清洗单个交易日原始 parquet 文件。

        Args:
            path: 原始 parquet 文件路径（文件名主名 = 交易日）。
            ticker: 股票代码（原始缺 ``ticker`` 列时填充）。

        Returns:
            独立 AM/PM session 与日级质量报告。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 映射后仍缺少必需列，或多个原始列映射到同一 canonical 列。
        """
        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(path)

        frame = _read_raw_frame(source_path)
        frame = self._apply_column_mapping(frame)
        required = {*RAW_FEATURE_COLUMNS, "timestamp"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"missing required columns after column_mapping: {missing}")

        fallback_date = _resolve_trade_date(frame, source_path)
        timestamps = [
            _parse_timestamp(value, fallback_date) for value in frame.get_column("timestamp")
        ]
        if any(value is None for value in timestamps):
            bad_count = sum(value is None for value in timestamps)
            raise ValueError(f"timestamp contains {bad_count} unparseable value(s)")

        normalized_timestamps = [value for value in timestamps if value is not None]
        dates = {value.date() for value in normalized_timestamps}
        if len(dates) != 1:
            raise ValueError(f"one raw file must contain exactly one trade_date, got {dates}")
        trade_date = next(iter(dates)) if dates else fallback_date

        frame = frame.with_columns(
            pl.Series("timestamp", normalized_timestamps, dtype=pl.Datetime("us")),
            pl.lit(trade_date.isoformat()).alias("trade_date"),
            pl.lit(ticker).alias("ticker"),
        )
        frame = frame.with_columns(
            pl.col(name).cast(pl.Float64, strict=False).alias(name)
            for name in RAW_FEATURE_COLUMNS
        )

        segments: list[SessionSegment] = []
        duplicate_count = 0
        for session_id, (start, end) in self._session_bounds.items():
            session_frame = frame.filter(
                (pl.col("timestamp").dt.time() >= pl.lit(start))
                & (pl.col("timestamp").dt.time() < pl.lit(end))
            )
            if session_frame.is_empty():
                continue
            if not session_frame.get_column("timestamp").is_sorted():
                raise ValueError(
                    f"timestamps must be non-decreasing within {trade_date} {session_id}"
                )

            duplicate_count += session_frame.height - session_frame.select(
                pl.col("timestamp").n_unique()
            ).item()
            session_frame = session_frame.unique(
                subset=["timestamp"], keep="last", maintain_order=True
            ).sort("timestamp")
            session_frame = self._complete_session_grid(
                session_frame, trade_date=trade_date, session_id=session_id
            )
            segments.append(
                SessionSegment(
                    trade_date=trade_date.isoformat(),
                    session_id=session_id,
                    frame=session_frame,
                )
            )

        if segments:
            quality_frame = pl.concat([segment.frame for segment in segments], how="vertical")
            report = run_quality_checks(quality_frame, duplicate_count=duplicate_count)
        else:
            report = QualityReport(
                trade_date=trade_date.isoformat(),
                row_count=0,
                missing_ratio=0.0,
                duplicate_count=duplicate_count,
                crossed_book_count=0,
                one_side_missing_count=0,
                max_gap=0.0,
                p95_gap=0.0,
                stale_snapshot_ratio=0.0,
                invalid_level_order_count=0,
            )
        return CleanDayResult(sessions=tuple(segments), quality_report=report)

    def _apply_column_mapping(self, frame: pl.DataFrame) -> pl.DataFrame:
        """应用配置映射，并拒绝多来源映射到同一 canonical 列。"""
        rename: dict[str, str] = {}
        target_sources: dict[str, str] = {}
        source_columns = set(frame.columns)
        for source, target in self.column_mapping.items():
            if source not in source_columns:
                continue
            previous = target_sources.get(target)
            if previous is not None and previous != source:
                raise ValueError(
                    f"column_mapping is ambiguous: {previous!r} and {source!r} -> {target!r}"
                )
            if source != target and target in source_columns:
                raise ValueError(
                    f"column_mapping collision: both {source!r} and canonical {target!r} exist"
                )
            target_sources[target] = source
            if source != target:
                rename[source] = target
        return frame.rename(rename)

    def _complete_session_grid(
        self,
        frame: pl.DataFrame,
        *,
        trade_date: date,
        session_id: str,
    ) -> pl.DataFrame:
        """补齐 session 内时间网格并完成有界 ffill、mid 与 book_valid。"""
        ticker = str(frame.get_column("ticker").item(0))
        first_timestamp = frame.get_column("timestamp").item(0)
        last_timestamp = frame.get_column("timestamp").item(-1)
        grid = pl.datetime_range(
            first_timestamp,
            last_timestamp,
            interval=f"{self.snapshot_interval_seconds}s",
            time_unit="us",
            eager=True,
        )
        timestamps = pl.concat([grid, frame.get_column("timestamp")]).unique().sort()
        timeline = pl.DataFrame({"timestamp": timestamps})

        frame = frame.with_columns(pl.lit(True).alias("_observed"))
        frame = frame.with_columns(
            pl.col("timestamp")
            .diff()
            .dt.total_seconds()
            .fill_null(0.0)
            .alias("snapshot_gap_seconds")
        )
        exact = timeline.join(frame, on="timestamp", how="left")

        source_rows = frame.filter(
            pl.any_horizontal(pl.col(name).is_not_null() for name in _LOB_COLUMNS)
        )
        previous = source_rows.select(
            pl.col("timestamp").alias("_source_timestamp"),
            *(pl.col(name).alias(f"_previous_{name}") for name in RAW_FEATURE_COLUMNS),
        )
        exact = exact.join_asof(
            previous,
            left_on="timestamp",
            right_on="_source_timestamp",
            strategy="backward",
        ).with_columns(
            (pl.col("timestamp") - pl.col("_source_timestamp"))
            .dt.total_seconds()
            .cast(pl.Float64)
            .alias("staleness_seconds")
        )

        whole_book_missing = pl.all_horizontal(
            pl.col(name).is_null() for name in _LOB_COLUMNS
        )
        can_ffill = (
            whole_book_missing
            & pl.col("_source_timestamp").is_not_null()
            & (pl.col("staleness_seconds") <= self.max_ffill_gap_seconds)
        )
        exact = exact.with_columns(can_ffill.alias("is_ffilled"))
        exact = exact.with_columns(
            pl.when(pl.col("is_ffilled"))
            .then(pl.col(f"_previous_{name}"))
            .otherwise(pl.col(name))
            .alias(name)
            for name in RAW_FEATURE_COLUMNS
        )

        bid1_valid = pl.col("BIDp1").is_not_null() & (pl.col("BIDp1") > 0)
        ask1_valid = pl.col("ASKp1").is_not_null() & (pl.col("ASKp1") > 0)
        crossed = bid1_valid & ask1_valid & (pl.col("BIDp1") > pl.col("ASKp1"))
        invalid_order = _invalid_level_order_expr()
        negative_volume = pl.any_horizontal(
            pl.col(name).is_not_null() & (pl.col(name) < 0) for name in _VOLUME_COLUMNS
        )
        negative_price = pl.any_horizontal(
            pl.col(name).is_not_null() & (pl.col(name) < 0) for name in _PRICE_COLUMNS
        )
        has_side = bid1_valid | ask1_valid

        exact = exact.with_columns(
            pl.when(bid1_valid & ask1_valid & ~crossed)
            .then((pl.col("BIDp1") + pl.col("ASKp1")) / 2.0)
            .when(bid1_valid & ~ask1_valid)
            .then(pl.col("BIDp1"))
            .when(ask1_valid & ~bid1_valid)
            .then(pl.col("ASKp1"))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("mid_price"),
            (has_side & ~crossed & ~invalid_order & ~negative_volume & ~negative_price)
            .alias("book_valid"),
            pl.lit(trade_date.isoformat()).alias("trade_date"),
            pl.lit(ticker).alias("ticker"),
            pl.lit(session_id).alias("session_id"),
            pl.col("timestamp").dt.time().alias("seconds"),
        )

        helper_columns = [
            "_observed",
            "_source_timestamp",
            *(f"_previous_{name}" for name in RAW_FEATURE_COLUMNS),
        ]
        return exact.drop(helper_columns).select(
            "trade_date",
            "session_id",
            "timestamp",
            "seconds",
            "ticker",
            *RAW_FEATURE_COLUMNS,
            "mid_price",
            "snapshot_gap_seconds",
            "staleness_seconds",
            "is_ffilled",
            "book_valid",
        )


def _parse_clock(value: str) -> time:
    try:
        return time.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid session clock: {value!r}") from exc


def _read_raw_frame(path: Path) -> pl.DataFrame:
    """读取原始 Parquet，并将 datetime 索引规范化为内部时间列。

    原始行情文件可由 pandas 以 ``DatetimeIndex`` 写出。时间在这种文件中是
    行索引而不是业务字段；读取边界负责恢复它，后续 Polars 运算则使用统一的
    ``timestamp: Datetime`` 列（Polars 不提供独立索引概念）。
    """
    pandas_frame = pd.read_parquet(path)
    if isinstance(pandas_frame.index, pd.DatetimeIndex):
        if "timestamp" in pandas_frame.columns:
            raise ValueError("timestamp exists both as a DatetimeIndex and a column")
        pandas_frame.index.name = "timestamp"
        pandas_frame = pandas_frame.reset_index()
    return pl.from_pandas(pandas_frame, include_index=False)


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for pattern in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            continue
    return None


def _resolve_trade_date(frame: pl.DataFrame, path: Path) -> date:
    if "trade_date" in frame.columns:
        parsed = {
            value
            for raw in frame.get_column("trade_date").drop_nulls().to_list()
            if (value := _parse_date(raw)) is not None
        }
        if len(parsed) > 1:
            raise ValueError(f"one raw file contains multiple trade_date values: {parsed}")
        if parsed:
            return next(iter(parsed))
    parsed_stem = _parse_date(path.stem)
    if parsed_stem is None:
        raise ValueError("trade_date is missing and cannot be inferred from file name")
    return parsed_stem


def _parse_timestamp(value: Any, fallback_date: date) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, time):
        return datetime.combine(fallback_date, value.replace(tzinfo=None))
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed_datetime = datetime.fromisoformat(normalized)
        if parsed_datetime.tzinfo is not None:
            parsed_datetime = parsed_datetime.replace(tzinfo=None)
        return parsed_datetime
    except ValueError:
        pass
    try:
        return datetime.combine(fallback_date, time.fromisoformat(text))
    except ValueError:
        return None


def _invalid_level_order_expr() -> pl.Expr:
    comparisons: list[pl.Expr] = []
    for level in range(1, 5):
        bid_near = pl.col(f"BIDp{level}")
        bid_far = pl.col(f"BIDp{level + 1}")
        ask_near = pl.col(f"ASKp{level}")
        ask_far = pl.col(f"ASKp{level + 1}")
        comparisons.extend(
            [
                bid_near.is_not_null()
                & bid_far.is_not_null()
                & (bid_near > 0)
                & (bid_far > 0)
                & (bid_near < bid_far),
                ask_near.is_not_null()
                & ask_far.is_not_null()
                & (ask_near > 0)
                & (ask_far > 0)
                & (ask_near > ask_far),
            ]
        )
    return pl.any_horizontal(comparisons)
