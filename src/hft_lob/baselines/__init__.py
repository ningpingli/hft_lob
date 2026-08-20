"""MVP baseline 注册与工厂（需求文档 §17）。"""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch

from hft_lob.baselines.base import BaselineModel
from hft_lob.baselines.models import (
    ImbalanceBaseline,
    MLPBaseline,
    RidgeBaseline,
    ZeroBaseline,
)
from hft_lob.baselines.runner import BaselineRunner
from hft_lob.configs.experiment import ExperimentConfig

BASELINE_NAMES: tuple[str, ...] = ("zero", "imbalance", "ridge", "mlp")


def build_baseline(
    name: str,
    config: ExperimentConfig,
    *,
    feature_columns: Sequence[str],
) -> BaselineModel:
    """按实验配置构建 baseline。

    Args:
        name: ``zero`` / ``imbalance`` / ``ridge`` / ``mlp``。
        config: 特征数、窗口长度及 baseline 参数的唯一来源。
        feature_columns: PreparedDataset 产出的唯一特征 schema。

    Returns:
        满足 ``BaselineModel`` 协议的模型。

    Raises:
        ValueError: baseline 名称未注册，或 imbalance 所需列不存在。
    """
    columns = tuple(feature_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("feature_columns must be non-empty and unique")
    if name not in BASELINE_NAMES:
        raise ValueError(f"unsupported baseline {name!r}; expected one of {BASELINE_NAMES}")
    num_features = len(columns)
    history = config.window.history_snapshots
    if name == "zero":
        return ZeroBaseline()
    if name == "imbalance":
        bid_indices, ask_indices = volume_feature_indices(columns)
        return ImbalanceBaseline(
            bid_volume_indices=bid_indices,
            ask_volume_indices=ask_indices,
        )
    if name == "ridge":
        return RidgeBaseline(
            num_features=num_features,
            history_snapshots=history,
            alpha=config.baselines.ridge_alpha,
        )
    with torch.random.fork_rng():
        torch.manual_seed(config.seed)
        return MLPBaseline(
            num_features=num_features,
            history_snapshots=history,
            hidden_dim=config.baselines.mlp_hidden_dim,
            dropout=config.baselines.mlp_dropout,
            epochs=config.training.epochs,
            learning_rate=config.training.learning_rate,
        )


def volume_feature_indices(
    feature_columns: Sequence[str],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """按实际 FeatureSchema 解析买量、卖量下标。"""
    columns = tuple(feature_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("feature_columns must be non-empty and unique")
    pattern = re.compile(r"^(BID|ASK)s([1-9][0-9]*)$")
    by_side: dict[str, dict[int, int]] = {"BID": {}, "ASK": {}}
    for index, name in enumerate(columns):
        match = pattern.fullmatch(name)
        if match is not None:
            by_side[match.group(1)][int(match.group(2))] = index
    levels = sorted(set(by_side["BID"]) | set(by_side["ASK"]))
    if not levels:
        raise ValueError("feature schema contains no BID/ASK volume columns")
    incomplete = [
        level for level in levels
        if level not in by_side["BID"] or level not in by_side["ASK"]
    ]
    if incomplete:
        raise ValueError(f"bid/ask volume columns must be paired by level: {incomplete}")
    return (
        tuple(by_side["BID"][level] for level in levels),
        tuple(by_side["ASK"][level] for level in levels),
    )


__all__ = [
    "BASELINE_NAMES",
    "BaselineModel",
    "BaselineRunner",
    "ImbalanceBaseline",
    "MLPBaseline",
    "RidgeBaseline",
    "ZeroBaseline",
    "build_baseline",
    "volume_feature_indices",
]
