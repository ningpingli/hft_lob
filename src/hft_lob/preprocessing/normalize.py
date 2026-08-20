"""归一化（需求文档 §12）：train-only / causal，状态可序列化、防泄漏。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass
class TrainOnlyNormalizer:
    """train-only 归一化：``fit(train) -> transform(all)``。

    契约（§12）：
    - 归一化参数（逐特征列 mean/std）只来自训练段；
    - 修改 test 数据不影响 train 参数（泄漏测试 B 天然通过）；
    - ``state_dict`` / ``from_state_dict`` 支持序列化（线上 inference，§33）；
    - 零/常量方差 std 兜底 1.0。
    """

    feature_cols: list[str]
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, frames: list[pl.DataFrame]) -> TrainOnlyNormalizer:
        """从训练段各日 DataFrame 累积特征列 mean/std。

        Args:
            frames: 训练段各日 DataFrame 列表。

        Returns:
            self。
        """
        raise NotImplementedError("TrainOnlyNormalizer.fit not implemented")

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """对特征列应用归一化（原地替换列值）。

        Args:
            df: 待变换 DataFrame。

        Returns:
            归一化后的 DataFrame。

        Raises:
            AssertionError: 未先调用 fit。
        """
        raise NotImplementedError("TrainOnlyNormalizer.transform not implemented")

    def state_dict(self) -> dict[str, tuple[float, float]]:
        """可序列化状态：``{列名: (mean, std)}``。"""
        raise NotImplementedError("TrainOnlyNormalizer.state_dict not implemented")

    @classmethod
    def from_state_dict(cls, state: dict[str, tuple[float, float]]) -> TrainOnlyNormalizer:
        """从序列化状态恢复（线上 inference 用，§33）。"""
        raise NotImplementedError("TrainOnlyNormalizer.from_state_dict not implemented")


@dataclass
class CausalRollingNormalizer:
    """causal 滚动归一化（§12）：变换 t 的参数只来自 t 之前的信息。

    MVP 粒度为按交易日滚动：当天 (T) 的变换参数只来自此前 ``window_days`` 个
    交易日（不含当日）；逐日调用 ``transform_day(fit_day=...)`` 推进状态。
    state 可序列化。
    """

    feature_cols: list[str]
    window_days: int = 1
    _means: list[np.ndarray] = field(default_factory=list)
    _mean2s: list[np.ndarray] = field(default_factory=list)

    def transform_day(self, df: pl.DataFrame, *, fit_day: bool) -> pl.DataFrame:
        """变换当日数据；``fit_day=True`` 表示当日并入滚动状态（训练模式）。

        热身期（窗口未满）不变换（返回原帧）；调用方应据此过滤样本。

        Args:
            df: 当日 DataFrame。
            fit_day: 是否将当日并入滚动状态。

        Returns:
            变换后的 DataFrame。
        """
        raise NotImplementedError("CausalRollingNormalizer.transform_day not implemented")

    def state_dict(self) -> dict[str, object]:
        """可序列化状态。"""
        raise NotImplementedError("CausalRollingNormalizer.state_dict not implemented")

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> CausalRollingNormalizer:
        """从序列化状态恢复。"""
        raise NotImplementedError("CausalRollingNormalizer.from_state_dict not implemented")
