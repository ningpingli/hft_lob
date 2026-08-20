"""特征与标签转换：cleaned DataFrame → 标准化特征 + 双族标签。"""

from __future__ import annotations

from typing import ClassVar

import polars as pl

from hft_lob.data_processing.fields import FieldsConfig


class FeatureTransformer:
    """列重映射 + 中间价 + 滚动 z-score 归一化（步骤 5+6）。

    持有跨日 z-score 状态：``run_pipeline`` 逐日累积调用
    ``transform_features``，滚动窗口状态在日间延续（决策 A1）。
    """

    def __init__(self, fields: FieldsConfig, normalization_window: int) -> None:
        """初始化特征转换器。

        Args:
            fields: 字段配置（column_map 与时间维度）。
            normalization_window: z-score 归一化的滚动窗口（前 N 天为热身期）。
        """
        raise NotImplementedError("FeatureTransformer.__init__ not implemented")

    def transform_features(
        self, df: pl.DataFrame, *, ticker: str, scaling: bool
    ) -> pl.DataFrame:
        """对单日清洗后数据做列重映射、中间价与（可选）滚动 z-score 归一化。

        Args:
            df: 单日清洗后的 DataFrame。
            ticker: 股票代码。
            scaling: 为 True 时应用滚动 z-score 归一化。

        Returns:
            标准化特征 DataFrame。
        """
        raise NotImplementedError("FeatureTransformer.transform_features not implemented")

    def reset_zscore_state(self) -> None:
        """重置跨日滚动 z-score 状态（新股票/新数据集开始前调用）。"""
        raise NotImplementedError("FeatureTransformer.reset_zscore_state not implemented")


class LabelTransformer:
    """双族标签计算（步骤 7）：simple_return / log_return 前向收益。"""

    #: 支持的标签族（默认族在前：simple_return 为默认，log_return 作为对照族）。
    LABEL_TYPES: ClassVar[tuple[str, ...]] = ("simple_return", "log_return")

    #: 标签族到列名短名的映射（``Target_<h>s_<family_short>`` 的推导来源）。
    FAMILY_SHORT: ClassVar[dict[str, str]] = {
        "simple_return": "simple",
        "log_return": "log",
    }

    def __init__(self, label_columns: dict[str, list[str]]) -> None:
        """初始化标签转换器。

        Args:
            label_columns: 必填的 ``{标签族: [列名]}`` 写死标签列名表。
        """
        raise NotImplementedError("LabelTransformer.__init__ not implemented")

    def transform_labels(
        self, df: pl.DataFrame, horizons_sec: list[int]
    ) -> pl.DataFrame:
        """按视界列表为 DataFrame 追加双族前向收益标签列。

        Args:
            df: 已含时间与中间价列的特征 DataFrame。
            horizons_sec: 标签视界（秒）。

        Returns:
            追加双族标签列后的 DataFrame。
        """
        raise NotImplementedError("LabelTransformer.transform_labels not implemented")


def label_columns_for(label_type: str, horizons_sec: list[int]) -> list[str]:
    """推导某标签族在给定视界下的标签列名列表。

    Args:
        label_type: 标签族名（simple_return / log_return）。
        horizons_sec: 标签视界（秒）。

    Returns:
        形如 ``Target_<h>s_<family_short>`` 的列名列表。
    """
    raise NotImplementedError("label_columns_for not implemented")


def forward_return(
    seconds: pl.Series,
    mid: pl.Series,
    horizon_sec: int,
    label_type: str = "simple_return",
) -> pl.Series:
    """计算给定视界的前向收益序列。

    Args:
        seconds: 自午夜起的秒数序列（用于判定同一交易日的边界）。
        mid: 一档中间价序列。
        horizon_sec: 前向视界（秒）。
        label_type: 标签族（simple_return 用价格比，log_return 用对数收益）。

    Returns:
        与输入等长的前向收益序列。
    """
    raise NotImplementedError("forward_return not implemented")
