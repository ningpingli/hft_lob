"""MVP baseline 注册与工厂（需求文档 §17）。"""

from __future__ import annotations

from collections.abc import Sequence

from hft_lob.baselines.base import BaselineModel
from hft_lob.baselines.models import (
    ImbalanceBaseline,
    RidgeBaseline,
    ZeroBaseline,
)
from hft_lob.baselines.runner import BaselineRunner
from hft_lob.configs.experiment import ExperimentConfig

BASELINE_NAMES: tuple[str, ...] = ("zero", "imbalance", "ridge")


def build_baseline(
    name: str,
    config: ExperimentConfig,
    *,
    feature_columns: Sequence[str],
) -> BaselineModel:
    """按实验配置构建 baseline。

    Args:
        name: ``zero`` / ``imbalance`` / ``ridge``。
        config: 特征数、窗口长度及 baseline 参数的唯一来源。
        feature_columns: PreparedDataset 产出的唯一特征 schema。

    Returns:
        满足 ``BaselineModel`` 协议的模型。

    Raises:
        ValueError: baseline 名称未注册，或 imbalance 所需列不存在。
    """
    raise NotImplementedError("build_baseline not implemented")


def volume_feature_indices(
    feature_columns: Sequence[str],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """按实际 FeatureSchema 解析买量、卖量下标。"""
    raise NotImplementedError("volume_feature_indices not implemented")


__all__ = [
    "BASELINE_NAMES",
    "BaselineRunner",
    "BaselineModel",
    "ImbalanceBaseline",
    "RidgeBaseline",
    "ZeroBaseline",
    "build_baseline",
    "volume_feature_indices",
]
