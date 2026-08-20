"""预处理总流程（需求文档 §40 流水线前段）：raw → 清洗 → 特征 → 标签 → manifest → split。"""

from __future__ import annotations

from dataclasses import dataclass

from hft_lob.configs.experiment import ExperimentConfig
from hft_lob.preprocessing.split import WalkForwardPlan


@dataclass(frozen=True)
class PreparedDataset:
    """数据准备阶段对训练侧的唯一交付对象。"""

    dataset_version: str
    feature_columns: tuple[str, ...]
    feature_version: str
    label_version: str
    manifest_path: str
    walk_forward_plan: WalkForwardPlan


def prepare_dataset(config: ExperimentConfig) -> PreparedDataset:
    """执行 raw→独立 session parquet→manifest，返回唯一训练交付对象。

    processed 文件名包含 trade_date/session_id；manifest 每行对应一个 session，
    所有 fold 仍按完整 trade_date 切分。
    """
    raise NotImplementedError("prepare_dataset not implemented")
