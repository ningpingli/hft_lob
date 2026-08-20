"""标签生成（需求文档 §7）：60 秒中间价对数/简单收益，future 匹配带容差、session 内。"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from hft_lob.configs.experiment import TargetConfig
from hft_lob.preprocessing.clean import SessionSegment

#: 标签类型 → 列名短名（§7.1；``Target_<h>s_<short>`` 的推导来源）。
_LABEL_TYPE_SHORT: dict[str, str] = {"log_mid_return": "log", "simple_mid_return": "simple"}


def label_column(config: TargetConfig) -> str:
    """主标签列名（§7.1：一个实验唯一 primary target）。"""
    try:
        short = _LABEL_TYPE_SHORT[config.type]
    except KeyError as exc:
        raise ValueError(f"unsupported target type: {config.type!r}") from exc
    return f"Target_{config.horizon_seconds}s_{short}"


class LabelTransformer:
    """为清洗后的单日 DataFrame 追加未来中间价与标签列。

    行为契约（§2/§3/§7）：
    - 锚点 t = 当前快照时间；``y_t = return(mid_t, mid_future)``；
    - future 快照取 ``[t + h - tol, t + h + tol]`` 内最近一条（§7.2 容差匹配，
      禁止无上限 ``first timestamp >= t + h``）；
    - 标签只在 same trade_date AND same session_id 内构造：跨 session（如
      11:29:30 + 60s）与跨日自然得到 invalid（§3）；
    - 输出列：``future_mid``、``Target_<h>s_log``、``Target_<h>s_simple``
      （§7.1 主标签 + 对照）。
    """

    def __init__(self, config: TargetConfig) -> None:
        """初始化标签转换器。

        Args:
            config: 标签配置（类型 / 视界 / 容差）。
        """
        if config.type not in _LABEL_TYPE_SHORT:
            raise ValueError(f"unsupported target type: {config.type!r}")
        if config.horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be > 0")
        if config.tolerance_seconds < 0:
            raise ValueError("tolerance_seconds must be >= 0")
        if config.tolerance_seconds >= config.horizon_seconds:
            raise ValueError("tolerance_seconds must be smaller than horizon_seconds")
        self.config = config

    def transform(self, segment: SessionSegment) -> SessionSegment:
        """在单个连续 session 内追加未来中间价、标签与 ``target_valid``。

        Args:
            segment: 含 ``seconds / mid_price`` 的单 session 数据。

        Returns:
            追加 ``future_mid``、双标签列和 ``target_valid`` 后的新 segment。

        Raises:
            ValueError: frame 中出现多个 trade_date/session_id，或元数据与
                SessionSegment 不一致。
        """
        frame = segment.frame
        required = {"trade_date", "session_id", "timestamp", "mid_price", "book_valid"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"label input missing columns: {missing}")
        self._validate_segment(segment)

        log_column = f"Target_{self.config.horizon_seconds}s_log"
        simple_column = f"Target_{self.config.horizon_seconds}s_simple"
        output_columns = (
            "future_timestamp",
            "future_mid",
            log_column,
            simple_column,
            "target_valid",
        )
        existing_outputs = [name for name in output_columns if name in frame.columns]
        if existing_outputs:
            frame = frame.drop(existing_outputs)

        if frame.is_empty():
            return SessionSegment(
                trade_date=segment.trade_date,
                session_id=segment.session_id,
                frame=frame.with_columns(
                    pl.lit(None).cast(pl.Datetime("us")).alias("future_timestamp"),
                    pl.lit(None).cast(pl.Float64).alias("future_mid"),
                    pl.lit(None).cast(pl.Float64).alias(log_column),
                    pl.lit(None).cast(pl.Float64).alias(simple_column),
                    pl.lit(False).alias("target_valid"),
                ),
            )

        candidates = frame.select(
            pl.col("timestamp").alias("future_timestamp"),
            pl.col("mid_price").alias("future_mid"),
            pl.col("book_valid").alias("future_book_valid"),
        ).filter(
            pl.col("future_book_valid").fill_null(False)
            & pl.col("future_mid").is_not_null()
            & pl.col("future_mid").is_finite()
            & (pl.col("future_mid") > 0)
        ).sort("future_timestamp")
        result = (
            frame.with_columns(
                (
                    pl.col("timestamp")
                    + pl.lit(timedelta(seconds=self.config.horizon_seconds))
                ).alias("_target_timestamp")
            )
            .join_asof(
                candidates,
                left_on="_target_timestamp",
                right_on="future_timestamp",
                strategy="nearest",
                tolerance=f"{self.config.tolerance_seconds}s",
            )
        )

        current_valid = (
            pl.col("book_valid").fill_null(False)
            & pl.col("mid_price").is_not_null()
            & pl.col("mid_price").is_finite()
            & (pl.col("mid_price") > 0)
        )
        future_valid = (
            pl.col("future_book_valid").fill_null(False)
            & pl.col("future_mid").is_not_null()
            & pl.col("future_mid").is_finite()
            & (pl.col("future_mid") > 0)
            & pl.col("future_timestamp").is_not_null()
        )
        target_valid = current_valid & future_valid
        result = result.with_columns(target_valid.alias("target_valid"))
        result = result.with_columns(
            pl.when(pl.col("target_valid"))
            .then((pl.col("future_mid") / pl.col("mid_price")).log())
            .otherwise(None)
            .cast(pl.Float64)
            .alias(log_column),
            pl.when(pl.col("target_valid"))
            .then(pl.col("future_mid") / pl.col("mid_price") - 1.0)
            .otherwise(None)
            .cast(pl.Float64)
            .alias(simple_column),
        ).drop("_target_timestamp", "future_book_valid")

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
            raise ValueError(
                "frame trade_date must contain exactly the SessionSegment trade_date"
            )
        if len(session_ids) != 1 or str(session_ids[0]) != segment.session_id:
            raise ValueError(
                "frame session_id must contain exactly the SessionSegment session_id"
            )
        if not frame.get_column("timestamp").is_sorted():
            raise ValueError("timestamps must be sorted within a SessionSegment")
        if not isinstance(frame.schema["timestamp"], pl.Datetime):
            raise ValueError("timestamp must use a Polars Datetime dtype")
