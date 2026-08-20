"""数据清洗（需求文档 §4/§5/§6）：schema 校验、session 分割、秒去重、有界 ffill、mid。"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from hft_lob.configs.experiment import SessionConfig
from hft_lob.preprocessing.quality import QualityReport


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

    def __init__(self, sessions: SessionConfig, max_ffill_gap_seconds: int) -> None:
        """初始化清洗器。

        Args:
            sessions: 交易时段配置（§3）。
            max_ffill_gap_seconds: 缺失策略 gap 上限（§5）。
        """
        raise NotImplementedError("DataCleaner.__init__ not implemented")

    def clean_day(self, path: str, *, ticker: str) -> CleanDayResult:
        """清洗单个交易日原始 parquet 文件。

        Args:
            path: 原始 parquet 文件路径（文件名主名 = 交易日）。
            ticker: 股票代码（原始缺 ``ticker`` 列时填充）。

        Returns:
            独立 AM/PM session 与日级质量报告。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 缺少必需列（20 盘口 + timestamp）。
        """
        raise NotImplementedError("DataCleaner.clean_day not implemented")
