"""特征工程（需求文档 §10/§11）：保留原始 23 维 + 可选派生特征。"""

from __future__ import annotations

from hft_lob.configs.experiment import FeatureConfig
from hft_lob.preprocessing.clean import SessionSegment


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
        raise NotImplementedError("FeatureTransformer.transform not implemented")
