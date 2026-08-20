"""严格因果的滚动窗口特征标准化。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class FrameStandardizer(Protocol):
    """Dataset 文件加载阶段消费的标准化协议。"""

    @property
    def output_feature_cols(self) -> list[str]:
        """标准化后供模型读取的列名，顺序与原始特征严格一致。"""
        ...

    def transform_frame(self, frame: pl.DataFrame) -> pl.DataFrame:
        """按时间顺序追加标准化列，不得读取当前行或未来行的统计信息。"""
        ...

    def state_dict(self) -> dict[str, object]:
        """返回可纳入实验 artifact 的纯 Python 配置状态。"""
        ...


@dataclass(frozen=True)
class CausalRollingStandardizer:
    """使用当前行之前固定窗口的统计量进行逐特征 Z-score 标准化。

    每个输入 frame 必须是单一 ``trade_date/session_id`` 且 timestamp 有序。
    对位置 ``t``，均值和总体标准差只来自 ``[t-normalize_window, t)``；历史
    不足、历史窗口含无效行或当前行无效时，输出 null 并令
    ``normalization_valid=False``。标准化列使用 ``normalized__`` 前缀，原始盘口
    和研究元数据保持不变。
    """

    feature_cols: tuple[str, ...]
    normalize_window: int

    def __post_init__(self) -> None:
        columns = tuple(self.feature_cols)
        object.__setattr__(self, "feature_cols", columns)
        if not columns:
            raise ValueError("feature_cols must not be empty")
        if len(set(columns)) != len(columns):
            raise ValueError("feature_cols must be unique")
        if self.normalize_window < 2:
            raise ValueError("normalize_window must be >= 2")

    @property
    def output_feature_cols(self) -> list[str]:
        """返回标准化列名，保持配置中的特征顺序。"""
        return [f"normalized__{name}" for name in self.feature_cols]

    def transform_frame(self, frame: pl.DataFrame) -> pl.DataFrame:
        """追加严格 shift(1) 的滚动标准化特征和有效性标记。"""
        required = {
            "trade_date",
            "session_id",
            "timestamp",
            "feature_valid",
            *self.feature_cols,
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"standardization input missing columns: {missing}")
        non_numeric = [
            name for name in self.feature_cols if not frame.schema[name].is_numeric()
        ]
        if non_numeric:
            raise ValueError(f"standardization features must be numeric: {non_numeric}")
        self._validate_frame(frame)

        current_row_valid = pl.col("feature_valid").fill_null(False)
        expressions: list[pl.Expr] = []
        output_columns = self.output_feature_cols
        for source_name, output_name in zip(
            self.feature_cols, output_columns, strict=True
        ):
            current = pl.col(source_name).cast(pl.Float64)
            finite_current = current.is_not_null() & current.is_finite()
            history_source = (
                pl.when(current_row_valid & finite_current).then(current).otherwise(None)
            )
            history_mean = history_source.rolling_mean(
                window_size=self.normalize_window,
                min_samples=self.normalize_window,
            ).shift(1)
            history_std = history_source.rolling_std(
                window_size=self.normalize_window,
                min_samples=self.normalize_window,
                ddof=0,
            ).shift(1)
            safe_std = pl.when(history_std > 0).then(history_std).otherwise(1.0)
            valid = (
                current_row_valid
                & finite_current
                & history_mean.is_not_null()
                & history_mean.is_finite()
                & history_std.is_not_null()
                & history_std.is_finite()
            )
            expressions.append(
                pl.when(valid)
                .then((current - history_mean) / safe_std)
                .otherwise(None)
                .cast(pl.Float64)
                .alias(output_name)
            )

        result = frame.drop(
            [name for name in output_columns if name in frame.columns],
            strict=False,
        ).with_columns(expressions)
        normalization_valid = pl.all_horizontal(
            pl.col(name).is_not_null() & pl.col(name).is_finite()
            for name in output_columns
        )
        return result.with_columns(normalization_valid.alias("normalization_valid"))

    def state_dict(self) -> dict[str, object]:
        """序列化标准化语义；该算法没有依赖未来数据的拟合状态。"""
        return {
            "version": 1,
            "type": "causal_rolling",
            "feature_cols": list(self.feature_cols),
            "normalize_window": self.normalize_window,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> CausalRollingStandardizer:
        """从 artifact 状态恢复并校验算法版本。"""
        expected_keys = {"version", "type", "feature_cols", "normalize_window"}
        missing = sorted(expected_keys.difference(state))
        unknown = sorted(set(state).difference(expected_keys))
        if missing or unknown:
            raise ValueError(f"invalid standardizer state: missing={missing}, unknown={unknown}")
        if state["version"] != 1 or state["type"] != "causal_rolling":
            raise ValueError("unsupported standardizer state version or type")
        feature_cols = state["feature_cols"]
        normalize_window = state["normalize_window"]
        if not isinstance(feature_cols, list) or not all(
            isinstance(name, str) for name in feature_cols
        ):
            raise ValueError("feature_cols must be a list of strings")
        if not isinstance(normalize_window, int) or isinstance(normalize_window, bool):
            raise ValueError("normalize_window must be an integer")
        return cls(tuple(feature_cols), normalize_window)

    @staticmethod
    def _validate_frame(frame: pl.DataFrame) -> None:
        if frame.is_empty():
            return
        trade_dates = frame.get_column("trade_date").unique().to_list()
        session_ids = frame.get_column("session_id").unique().to_list()
        if len(trade_dates) != 1:
            raise ValueError("standardization frame must contain exactly one trade_date")
        if len(session_ids) != 1:
            raise ValueError("standardization frame must contain exactly one session_id")
        if not frame.get_column("timestamp").is_sorted():
            raise ValueError("standardization timestamps must be sorted")
