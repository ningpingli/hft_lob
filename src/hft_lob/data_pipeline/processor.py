"""原始行情清洗、特征、标签、标准化与质量检查。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import pandas as pd
import polars as pl

from hft_lob.configs.experiment import (
    RAW_FEATURE_COLUMNS,
    FeatureConfig,
    SessionConfig,
    TargetConfig,
)

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

_SUPPORTED_DERIVED_FEATURES: tuple[str, ...] = (
    "spread",
    "relative_spread",
    "mid_price",
    "microprice",
    "l1_imbalance",
    "l5_imbalance",
    "bid_depth",
    "ask_depth",
    "depth_imbalance",
    "price_slope",
    "volume_slope",
)

class FeatureTransformer:
    """在清洗后的 DataFrame 上追加派生特征（§11），原始 23 列保持不变。

    派生特征（``use_derived=True`` 时追加）：spread / relative_spread /
    mid_price / microprice / l1_imbalance / l5_imbalance / bid_depth / ask_depth
    / depth_imbalance / price_slope / volume_slope（§11 定义；除零/NaN 分母
    → NaN，不产生 inf）。
    """

    def __init__(self, config: FeatureConfig) -> None:
        """初始化特征转换器。

        Args:
            config: 特征配置（是否启用派生特征及其清单）。
        """
        derived = tuple(config.derived_features)
        duplicates = sorted({name for name in derived if derived.count(name) > 1})
        if duplicates:
            raise ValueError(f"derived_features contains duplicates: {duplicates}")
        unsupported = sorted(set(derived).difference(_SUPPORTED_DERIVED_FEATURES))
        if unsupported:
            raise ValueError(f"unsupported derived features: {unsupported}")

        self.config = config
        self._derived_features = derived if config.use_derived else ()

    def feature_columns(self) -> list[str]:
        """模型输入特征列：23 原始；开启派生特征后追加（§10/§11）。"""
        return [*RAW_FEATURE_COLUMNS, *self._derived_features]

    def transform(self, segment: SessionSegment) -> SessionSegment:
        """在单个连续 session 内追加派生特征和 ``feature_valid``。

        Args:
            segment: 清洗后的单 session 数据；不得同时包含 AM/PM。

        Returns:
            追加派生特征列及 ``feature_valid`` 后的新 segment。

        Raises:
            ValueError: frame 中出现多个 trade_date/session_id，或元数据与
                SessionSegment 不一致。
        """
        frame = segment.frame
        required = {
            "trade_date",
            "session_id",
            "mid_price",
            "book_valid",
            *RAW_FEATURE_COLUMNS,
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"feature input missing columns: {missing}")

        self._validate_segment(segment)
        if self._derived_features:
            expressions = _derived_expressions()
            frame = frame.with_columns(
                expressions[name].alias(name) for name in self._derived_features
            )

        feature_columns = self.feature_columns()
        feature_valid = pl.col("book_valid").fill_null(False)
        for name in feature_columns:
            feature_valid &= pl.col(name).is_not_null() & pl.col(name).is_finite()
        frame = frame.with_columns(feature_valid.alias("feature_valid"))

        return SessionSegment(
            trade_date=segment.trade_date,
            session_id=segment.session_id,
            frame=frame,
        )

    @staticmethod
    def _validate_segment(segment: SessionSegment) -> None:
        """验证物理 segment 边界，禁止混入其他日期或 session。"""
        frame = segment.frame
        if frame.is_empty():
            return

        trade_dates = frame.get_column("trade_date").unique().to_list()
        session_ids = frame.get_column("session_id").unique().to_list()
        if len(trade_dates) != 1 or str(trade_dates[0]) != segment.trade_date:
            raise ValueError(
                "frame trade_date must contain exactly the SessionSegment trade_date"
            )
        if len(session_ids) != 1 or str(session_ids[0]) != segment.session_id:
            raise ValueError(
                "frame session_id must contain exactly the SessionSegment session_id"
            )

def _derived_expressions() -> dict[str, pl.Expr]:
    """构造全部派生特征表达式；调用方按配置选择，避免无关列落盘。

    ``price_slope`` 是五档两侧平均每档价差扩张：
    ``((ASKp5-ASKp1) + (BIDp1-BIDp5)) / 8``。
    ``volume_slope`` 是每档买卖合计深度对档位 1..5 的 OLS 斜率；其
    ``sum((level-3)^2)`` 固定为 10。
    """
    bid_depth = pl.sum_horizontal(pl.col(f"BIDs{level}") for level in range(1, 6))
    ask_depth = pl.sum_horizontal(pl.col(f"ASKs{level}") for level in range(1, 6))
    l1_total = pl.col("BIDs1") + pl.col("ASKs1")
    total_depth = bid_depth + ask_depth
    spread = pl.col("ASKp1") - pl.col("BIDp1")
    mid_price = (pl.col("ASKp1") + pl.col("BIDp1")) / 2.0
    microprice_numerator = (
        pl.col("ASKp1") * pl.col("BIDs1")
        + pl.col("BIDp1") * pl.col("ASKs1")
    )
    volume_slope = pl.sum_horizontal(
        pl.lit(float(level - 3))
        * (pl.col(f"BIDs{level}") + pl.col(f"ASKs{level}"))
        for level in range(1, 6)
    ) / 10.0

    return {
        "spread": _finite_or_null(spread),
        "relative_spread": _safe_ratio(spread, mid_price),
        # clean.py 已冻结单边盘口的 mid 语义，此处只规范非有限值，不能重算覆盖。
        "mid_price": _finite_or_null(pl.col("mid_price")),
        "microprice": _safe_ratio(microprice_numerator, l1_total),
        "l1_imbalance": _safe_ratio(
            pl.col("BIDs1") - pl.col("ASKs1"), l1_total
        ),
        "l5_imbalance": _safe_ratio(bid_depth - ask_depth, total_depth),
        "bid_depth": _finite_or_null(bid_depth),
        "ask_depth": _finite_or_null(ask_depth),
        "depth_imbalance": _safe_ratio(bid_depth - ask_depth, total_depth),
        "price_slope": _finite_or_null(
            (
                (pl.col("ASKp5") - pl.col("ASKp1"))
                + (pl.col("BIDp1") - pl.col("BIDp5"))
            )
            / 8.0
        ),
        "volume_slope": _finite_or_null(volume_slope),
    }

def _safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """仅在分子分母有限且分母非零时计算比值。"""
    valid = (
        numerator.is_not_null()
        & numerator.is_finite()
        & denominator.is_not_null()
        & denominator.is_finite()
        & (denominator != 0)
    )
    return pl.when(valid).then(numerator / denominator).otherwise(None).cast(pl.Float64)

def _finite_or_null(expression: pl.Expr) -> pl.Expr:
    """将 NaN/inf 归一为 null，避免非有限值进入模型。"""
    return (
        pl.when(expression.is_not_null() & expression.is_finite())
        .then(expression)
        .otherwise(None)
        .cast(pl.Float64)
    )

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
    """按配置标签生成收益目标列；缺失目标由样本编译阶段过滤。"""

    def __init__(self, config: TargetConfig) -> None:
        if config.type not in _LABEL_TYPE_SHORT:
            raise ValueError(f"unsupported target type: {config.type!r}")
        self.config = config

    def transform(self, segment: SessionSegment) -> SessionSegment:
        """追加配置的目标列，不发布标签有效性列。"""
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
        def target_expression(future_mid: pl.Expr) -> pl.Expr:
            if self.config.type == "log_mid_return":
                return (future_mid / pl.col("mid_price")).log()
            return future_mid / pl.col("mid_price") - 1.0
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

@runtime_checkable
class FrameStandardizer(Protocol):
    """Dataset 文件加载阶段消费的标准化协议。"""

    @property
    def output_feature_cols(self) -> list[str]:
        """标准化后供模型读取的列名，顺序与原始特征严格一致。"""
        ...

    def transform_frame(self, frame: pl.DataFrame) -> pl.DataFrame:
        """按时间顺序追加标准化列，不得读取当前行或未来行的统计信息。"""
        ...

    def state_dict(self) -> dict[str, object]:
        """返回可纳入实验 artifact 的纯 Python 配置状态。"""
        ...

@dataclass(frozen=True)
class CausalRollingStandardizer:
    """使用当前行之前固定窗口的统计量进行逐特征 Z-score 标准化。

    每个输入 frame 必须是单一 ``trade_date/session_id`` 且 timestamp 有序。
    对位置 ``t``，均值和总体标准差只来自 ``[t-normalize_window, t)``；历史
    不足、历史窗口含无效行或当前行无效时，输出 null 并令
    ``normalization_valid=False``。标准化列使用 ``normalized__`` 前缀，原始盘口
    和研究元数据保持不变。
    """

    feature_cols: tuple[str, ...]
    normalize_window: int

    def __post_init__(self) -> None:
        columns = tuple(self.feature_cols)
        object.__setattr__(self, "feature_cols", columns)
        if not columns:
            raise ValueError("feature_cols must not be empty")
        if len(set(columns)) != len(columns):
            raise ValueError("feature_cols must be unique")
        if self.normalize_window < 2:
            raise ValueError("normalize_window must be >= 2")

    @property
    def output_feature_cols(self) -> list[str]:
        """返回标准化列名，保持配置中的特征顺序。"""
        return [f"normalized__{name}" for name in self.feature_cols]

    def transform_frame(self, frame: pl.DataFrame) -> pl.DataFrame:
        """追加严格 shift(1) 的滚动标准化特征和有效性标记。"""
        required = {
            "trade_date",
            "session_id",
            "timestamp",
            "feature_valid",
            *self.feature_cols,
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"standardization input missing columns: {missing}")
        non_numeric = [
            name for name in self.feature_cols if not frame.schema[name].is_numeric()
        ]
        if non_numeric:
            raise ValueError(f"standardization features must be numeric: {non_numeric}")
        self._validate_frame(frame)

        current_row_valid = pl.col("feature_valid").fill_null(False)
        expressions: list[pl.Expr] = []
        output_columns = self.output_feature_cols
        for source_name, output_name in zip(
            self.feature_cols, output_columns, strict=True
        ):
            current = pl.col(source_name).cast(pl.Float64)
            finite_current = current.is_not_null() & current.is_finite()
            history_source = (
                pl.when(current_row_valid & finite_current).then(current).otherwise(None)
            )
            history_mean = history_source.rolling_mean(
                window_size=self.normalize_window,
                min_samples=self.normalize_window,
            ).shift(1)
            history_std = history_source.rolling_std(
                window_size=self.normalize_window,
                min_samples=self.normalize_window,
                ddof=0,
            ).shift(1)
            safe_std = pl.when(history_std > 0).then(history_std).otherwise(1.0)
            valid = (
                current_row_valid
                & finite_current
                & history_mean.is_not_null()
                & history_mean.is_finite()
                & history_std.is_not_null()
                & history_std.is_finite()
            )
            expressions.append(
                pl.when(valid)
                .then((current - history_mean) / safe_std)
                .otherwise(None)
                .cast(pl.Float64)
                .alias(output_name)
            )

        result = frame.drop(
            [name for name in output_columns if name in frame.columns],
            strict=False,
        ).with_columns(expressions)
        normalization_valid = pl.all_horizontal(
            pl.col(name).is_not_null() & pl.col(name).is_finite()
            for name in output_columns
        )
        return result.with_columns(normalization_valid.alias("normalization_valid"))

    def state_dict(self) -> dict[str, object]:
        """序列化标准化语义；该算法没有依赖未来数据的拟合状态。"""
        return {
            "version": 1,
            "type": "causal_rolling",
            "feature_cols": list(self.feature_cols),
            "normalize_window": self.normalize_window,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> CausalRollingStandardizer:
        """从 artifact 状态恢复并校验算法版本。"""
        expected_keys = {"version", "type", "feature_cols", "normalize_window"}
        missing = sorted(expected_keys.difference(state))
        unknown = sorted(set(state).difference(expected_keys))
        if missing or unknown:
            raise ValueError(f"invalid standardizer state: missing={missing}, unknown={unknown}")
        if state["version"] != 1 or state["type"] != "causal_rolling":
            raise ValueError("unsupported standardizer state version or type")
        feature_cols = state["feature_cols"]
        normalize_window = state["normalize_window"]
        if not isinstance(feature_cols, list) or not all(
            isinstance(name, str) for name in feature_cols
        ):
            raise ValueError("feature_cols must be a list of strings")
        if not isinstance(normalize_window, int) or isinstance(normalize_window, bool):
            raise ValueError("normalize_window must be an integer")
        return cls(tuple(feature_cols), normalize_window)

    @staticmethod
    def _validate_frame(frame: pl.DataFrame) -> None:
        if frame.is_empty():
            return
        trade_dates = frame.get_column("trade_date").unique().to_list()
        session_ids = frame.get_column("session_id").unique().to_list()
        if len(trade_dates) != 1:
            raise ValueError("standardization frame must contain exactly one trade_date")
        if len(session_ids) != 1:
            raise ValueError("standardization frame must contain exactly one session_id")
        if not frame.get_column("timestamp").is_sorted():
            raise ValueError("standardization timestamps must be sorted")


@dataclass(frozen=True)
class QualityReport:
    """单个交易日的质量报告（§4 数据质量报告字段）。"""

    trade_date: str
    row_count: int
    missing_ratio: float
    duplicate_count: int
    crossed_book_count: int
    one_side_missing_count: int
    max_gap: float
    p95_gap: float
    stale_snapshot_ratio: float
    invalid_level_order_count: int

    def to_dict(self) -> dict[str, object]:
        """转为字典（供 manifest 落盘）。"""
        return asdict(self)

def run_quality_checks(df: pl.DataFrame, *, duplicate_count: int) -> QualityReport:
    """对单日清洗后的 DataFrame 计算质量指标（§4）。

    输入列约定：``trade_date / session_id / timestamp``、20 盘口列、
    ``mid_price / staleness_seconds / is_ffilled``。

    指标：row_count / missing_ratio（整条盘口缺失占比）/ duplicate_count /
    crossed_book_count（bid1 > ask1 且双侧有效）/ one_side_missing_count /
    max_gap 与 p95_gap（会话内相邻快照秒差）/ stale_snapshot_ratio（被 ffill
    的行占比）/ invalid_level_order_count（档位单调性违反行数）。

    Args:
        df: 已清洗的单日数据。
        duplicate_count: 去重阶段剔除的重复 timestamp 数。

    Returns:
        质量报告。
    """
    required = {
        "trade_date",
        "session_id",
        "timestamp",
        "is_ffilled",
        "snapshot_gap_seconds",
        *_LOB_COLUMNS,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"quality input missing columns: {missing}")
    if df.is_empty():
        raise ValueError("quality input must not be empty")

    trade_dates = df.get_column("trade_date").drop_nulls().unique().to_list()
    if len(trade_dates) != 1:
        raise ValueError(f"quality input must contain one trade_date, got {trade_dates}")

    bid1_valid = pl.col("BIDp1").is_not_null() & (pl.col("BIDp1") > 0)
    ask1_valid = pl.col("ASKp1").is_not_null() & (pl.col("ASKp1") > 0)
    whole_book_missing = pl.all_horizontal(
        pl.col(name).is_null() for name in _LOB_COLUMNS
    )
    invalid_order = _invalid_level_order_expr()

    aggregate = df.select(
        whole_book_missing.mean().alias("missing_ratio"),
        (bid1_valid & ask1_valid & (pl.col("BIDp1") > pl.col("ASKp1")))
        .sum()
        .alias("crossed_book_count"),
        (bid1_valid ^ ask1_valid).sum().alias("one_side_missing_count"),
        pl.col("is_ffilled").cast(pl.Float64).mean().alias("stale_snapshot_ratio"),
        invalid_order.sum().alias("invalid_level_order_count"),
    ).row(0, named=True)

    gaps = (
        df.get_column("snapshot_gap_seconds")
        .drop_nulls()
        .filter(df.get_column("snapshot_gap_seconds").drop_nulls() > 0)
    )
    max_value = cast(float | None, gaps.max()) if len(gaps) else None
    p95_value = (
        cast(float | None, gaps.quantile(0.95, interpolation="linear"))
        if len(gaps)
        else None
    )
    max_gap = float(max_value) if max_value is not None else 0.0
    p95_gap = float(p95_value) if p95_value is not None else 0.0

    return QualityReport(
        trade_date=str(trade_dates[0]),
        row_count=df.height,
        missing_ratio=float(aggregate["missing_ratio"] or 0.0),
        duplicate_count=duplicate_count,
        crossed_book_count=int(aggregate["crossed_book_count"] or 0),
        one_side_missing_count=int(aggregate["one_side_missing_count"] or 0),
        max_gap=max_gap,
        p95_gap=p95_gap,
        stale_snapshot_ratio=float(aggregate["stale_snapshot_ratio"] or 0.0),
        invalid_level_order_count=int(aggregate["invalid_level_order_count"] or 0),
    )
