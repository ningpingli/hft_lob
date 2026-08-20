"""数据质量检查（需求文档 §4）：逐交易日输出质量指标。"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


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
        raise NotImplementedError("QualityReport.to_dict not implemented")


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
    raise NotImplementedError("run_quality_checks not implemented")
