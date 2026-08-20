"""特征工程（需求文档 §10/§11）：保留原始 23 维 + 可选派生特征。"""

from __future__ import annotations

import polars as pl

from hft_lob.configs.experiment import RAW_FEATURE_COLUMNS, FeatureConfig
from hft_lob.preprocessing.clean import SessionSegment

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
