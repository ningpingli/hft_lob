"""单日 LOB 快照数据清洗：raw → cleaned pl.DataFrame。"""

from __future__ import annotations

import polars as pl

from hft_lob.data_processing.fields import FieldsConfig


class DataCleaner:
    """按 FieldsConfig 清洗每日原始 parquet/CSV（步骤 1-4）：

    列重映射（column_map）+ 时间字段解析 + 缺失盘口前向填充 + 集合竞价剔除
    + 跨日列结构一致性校验。
    """

    def __init__(self, fields: FieldsConfig) -> None:
        """初始化清洗器。

        Args:
            fields: 字段配置（column_map 与时间维度），驱动列重映射与时间解析。
        """
        raise NotImplementedError("DataCleaner.__init__ not implemented")

    def clean_day(self, raw_path: str) -> pl.DataFrame:
        """清洗单个交易日原始文件，返回规范化后的单日 DataFrame。

        Args:
            raw_path: 当日原始 parquet/CSV 文件路径。

        Returns:
            清洗后的单日 DataFrame。
        """
        raise NotImplementedError("DataCleaner.clean_day not implemented")

    def clean_all(self, ticker: str, input_dir: str) -> list[pl.DataFrame]:
        """清洗 ``input_dir`` 下某只股票的全部交易日文件。

        Args:
            ticker: 股票代码。
            input_dir: 存放该股票每日原始文件的目录。

        Returns:
            按日期排序的单日清洗后 DataFrame 列表。
        """
        raise NotImplementedError("DataCleaner.clean_all not implemented")

    @staticmethod
    def validate_schema_consistency(daily_frames: list[pl.DataFrame]) -> int:
        """校验跨日列结构一致性，返回结构一致的帧数。

        Args:
            daily_frames: 多日清洗后的 DataFrame 列表。

        Returns:
            列结构一致的 DataFrame 数量。
        """
        raise NotImplementedError(
            "DataCleaner.validate_schema_consistency not implemented"
        )
