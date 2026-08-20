"""数据清洗（需求文档 §4/§5/§6）：schema 校验、session 分割、秒去重、有界 ffill、mid。"""

from __future__ import annotations

import polars as pl

from hft_lob.configs.experiment import SessionConfig
from hft_lob.preprocessing.quality import QualityReport


class DataCleaner:
    """单日原始快照 → 规范化 DataFrame（§4 清洗 + §5 缺失策略 + §6 mid）。

    输出列约定（processed 中间格式）：``trade_date / session_id / timestamp /
    seconds`` + 20 盘口 + 3 标量 + ``mid_price / staleness_seconds / is_ffilled /
    valid``。

    行为契约：
    - §3 会话分割：按 SessionConfig 划分 AM/PM（半开区间），非连续竞价时段剔除；
    - §4 秒去重：重复 timestamp 保留同秒最后一条；
    - §5 有界 ffill：整条盘口缺失时，gap ≤ max_ffill_gap_seconds 才前向填充
      （价格 + 数量整体，时间不填充），超限行标记 ``valid=False``；
    - §6 mid：双侧有效取均值，单边取存活侧价格，交叉/双侧无效 → NaN 且
      ``valid=False``。
    """

    def __init__(self, sessions: SessionConfig, max_ffill_gap_seconds: int) -> None:
        """初始化清洗器。

        Args:
            sessions: 交易时段配置（§3）。
            max_ffill_gap_seconds: 缺失策略 gap 上限（§5）。
        """
        raise NotImplementedError("DataCleaner.__init__ not implemented")

    def clean_day(self, path: str, *, ticker: str) -> tuple[pl.DataFrame, QualityReport]:
        """清洗单个交易日原始 parquet 文件。

        Args:
            path: 原始 parquet 文件路径（文件名主名 = 交易日）。
            ticker: 股票代码（原始缺 ``ticker`` 列时填充）。

        Returns:
            (清洗后的单日 DataFrame, 质量报告)。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 缺少必需列（20 盘口 + timestamp）。
        """
        raise NotImplementedError("DataCleaner.clean_day not implemented")
