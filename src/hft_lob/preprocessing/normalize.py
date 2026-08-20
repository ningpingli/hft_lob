"""MVP train-only 归一化：状态可序列化、防泄漏。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class TensorNormalizer(Protocol):
    """Dataset 可消费的唯一归一化协议。"""

    def transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        """按最后一维特征应用已拟合状态，不允许在此更新统计量。"""
        ...

    def state_dict(self) -> dict[str, object]:
        """返回可序列化、可纳入 checkpoint/artifact 的状态。"""
        ...


@dataclass
class TrainOnlyNormalizer:
    """train-only 归一化：``fit(train)`` 后由 Dataset 应用同一状态。

    契约（§12）：
    - 归一化参数（逐特征列 mean/std）只来自训练段；
    - 修改 test 数据不影响 train 参数（泄漏测试 B 天然通过）；
    - ``state_dict`` / ``from_state_dict`` 支持序列化（线上 inference，§33）；
    - 零/常量方差 std 兜底 1.0。
    """

    feature_cols: list[str]
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        """使用训练段统计量归一化窗口 Tensor。

        Args:
            values: 最后一维与 ``feature_cols`` 对齐的输入 Tensor。

        Returns:
            归一化后的新 Tensor，不修改输入和统计状态。

        Raises:
            AssertionError: 未先调用 fit。
        """
        raise NotImplementedError("TrainOnlyNormalizer.transform_tensor not implemented")

    def state_dict(self) -> dict[str, object]:
        """可序列化状态：``{列名: (mean, std)}``。"""
        raise NotImplementedError("TrainOnlyNormalizer.state_dict not implemented")

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> TrainOnlyNormalizer:
        """从序列化状态恢复（线上 inference 用，§33）。"""
        raise NotImplementedError("TrainOnlyNormalizer.from_state_dict not implemented")


def fit_train_only_normalizer(
    training_files: Sequence[str],
    *,
    feature_cols: Sequence[str],
) -> TrainOnlyNormalizer:
    """唯一拟合入口：从 training parquet 流式统计，不经 Dataset、不读 val/test。"""
    raise NotImplementedError("fit_train_only_normalizer not implemented")
