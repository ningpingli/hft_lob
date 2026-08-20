"""数据质量检查（需求文档 §4）：逐交易日输出质量指标。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import cast

import polars as pl

from hft_lob.configs.experiment import RAW_FEATURE_COLUMNS

_LOB_COLUMNS: tuple[str, ...] = RAW_FEATURE_COLUMNS[:20]


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
