"""Ridge baseline 注册与工厂（需求文档 §17）。"""

from __future__ import annotations

from collections.abc import Sequence

from hft_lob.baselines.base import BaselineModel
from hft_lob.baselines.models import RidgeBaseline
from hft_lob.baselines.runner import BaselineRunner
from hft_lob.configs.experiment import BaselineConfig

BASELINE_NAMES: tuple[str, ...] = ("ridge",)


def build_baseline(
    name: str,
    config: BaselineConfig,
    *,
    feature_columns: Sequence[str],
    history_snapshots: int,
    target_count: int = 1,
) -> BaselineModel:
    """构建唯一支持的 Ridge baseline。

    Args:
        name: 必须为 ``ridge``。
        config: Ridge 正则化参数的唯一来源。
        feature_columns: PreparedDataset 产出的唯一特征 schema。

    Returns:
        满足 ``BaselineModel`` 协议的 Ridge 模型。

    """
    columns = tuple(feature_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("feature_columns must be non-empty and unique")
    if name not in BASELINE_NAMES:
        raise ValueError(f"unsupported baseline {name!r}; expected one of {BASELINE_NAMES}")
    if history_snapshots <= 0:
        raise ValueError("history_snapshots must be > 0")
    return RidgeBaseline(
        num_features=len(columns),
        history_snapshots=history_snapshots,
        alpha=config.ridge_alpha,
        target_count=target_count,
    )


__all__ = [
    "BASELINE_NAMES",
    "BaselineModel",
    "BaselineRunner",
    "RidgeBaseline",
    "build_baseline",
]
