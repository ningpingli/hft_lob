"""特征工程（需求文档 §10/§11）：保留原始 23 维 + 可选派生特征。"""

from __future__ import annotations

import polars as pl

from hft_lob.configs.experiment import FeatureConfig


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
        raise NotImplementedError("FeatureTransformer.__init__ not implemented")

    def feature_columns(self) -> list[str]:
        """模型输入特征列：23 原始；开启派生特征后追加（§10/§11）。"""
        raise NotImplementedError("FeatureTransformer.feature_columns not implemented")

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """追加派生特征列（若启用）。

        Args:
            df: 清洗后的单日 DataFrame（须含 20 盘口 + last/volume/amount +
                mid_price）。

        Returns:
            追加派生特征列后的 DataFrame。
        """
        raise NotImplementedError("FeatureTransformer.transform not implemented")
