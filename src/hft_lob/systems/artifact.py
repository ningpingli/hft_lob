"""Prediction artifact（需求文档 §28）：parquet 保存完整样本上下文。

禁止只保存 ``[targets, predictions]``——必须保留 ticker / trade_date /
session_id / anchor_timestamp / mid_t / future_mid / bid1 / ask1 / spread /
split / model_version / dataset_version，否则无法定位异常预测。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hft_lob.datasets.lob_dataset import SampleMeta


@dataclass(frozen=True)
class PredictionArtifact:
    """模型和 baseline 共用的内存预测产物。"""

    predictions: np.ndarray
    targets: np.ndarray
    metadata: tuple[SampleMeta, ...]
    model_name: str
    model_version: str
    dataset_version: str
    fold_index: int
    split: str


def git_commit() -> str:
    """当前 git 短提交（§29 可复现；非 git 仓库返回 'unknown'）。"""
    raise NotImplementedError("git_commit not implemented")


def save_prediction_artifact(
    *,
    artifact: PredictionArtifact,
    path: str,
) -> str:
    """保存预测结果 parquet（§28 字段清单）。

    Args:
        artifact: 完整、强类型、已绑定 model/dataset/fold/split 的预测产物。
        path: 输出 parquet 路径。

    Returns:
        输出路径。
    """
    raise NotImplementedError("save_prediction_artifact not implemented")
