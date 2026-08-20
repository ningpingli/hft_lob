"""MVP baseline 注册与工厂（需求文档 §17）。"""

from __future__ import annotations

from hft_lob.baselines.base import BaselineModel
from hft_lob.baselines.models import (
    ImbalanceBaseline,
    MLPBaseline,
    RidgeBaseline,
    ZeroBaseline,
)
from hft_lob.configs.experiment import ExperimentConfig

BASELINE_NAMES: tuple[str, ...] = ("zero", "imbalance", "ridge", "mlp")


def build_baseline(name: str, config: ExperimentConfig) -> BaselineModel:
    """按实验配置构建 baseline。

    Args:
        name: ``zero`` / ``imbalance`` / ``ridge`` / ``mlp``。
        config: 特征数、窗口长度及 baseline 参数的唯一来源。

    Returns:
        满足 ``BaselineModel`` 协议的模型。

    Raises:
        ValueError: baseline 名称未注册，或 imbalance 所需列不存在。
    """
    raise NotImplementedError("build_baseline not implemented")


def volume_feature_indices() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """返回 canonical raw23 中买量、卖量特征下标，供 ImbalanceBaseline 使用。"""
    raise NotImplementedError("volume_feature_indices not implemented")


__all__ = [
    "BASELINE_NAMES",
    "BaselineModel",
    "ImbalanceBaseline",
    "MLPBaseline",
    "RidgeBaseline",
    "ZeroBaseline",
    "build_baseline",
    "volume_feature_indices",
]
