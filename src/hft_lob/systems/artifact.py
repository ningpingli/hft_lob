"""Prediction artifact（需求文档 §28）：parquet 保存完整样本上下文。

禁止只保存 ``[targets, predictions]``——必须保留 ticker / trade_date /
session_id / anchor_timestamp / mid_t / future_mid / bid1 / ask1 / spread /
split / model_version / dataset_version，否则无法定位异常预测。
"""

from __future__ import annotations

import numpy as np


def git_commit() -> str:
    """当前 git 短提交（§29 可复现；非 git 仓库返回 'unknown'）。"""
    raise NotImplementedError("git_commit not implemented")


def save_prediction_artifact(
    *,
    preds: np.ndarray,
    targets: np.ndarray,
    meta: dict[str, list[object]],
    path: str,
    model_version: str,
    dataset_version: str,
    split: str = "test",
) -> str:
    """保存预测结果 parquet（§28 字段清单）。

    Args:
        preds: 预测值（与 meta 等长）。
        targets: 已实现收益。
        meta: 每样本元数据（来自 LOBWindowDataset 的 meta dict-of-lists）。
        path: 输出 parquet 路径。
        model_version: 模型版本（§29）。
        dataset_version: 数据集版本（§29/§31）。
        split: 所属切分段（test）。

    Returns:
        输出路径。
    """
    raise NotImplementedError("save_prediction_artifact not implemented")
