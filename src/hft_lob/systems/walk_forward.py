"""Walk-forward 统一执行闭环。"""

from __future__ import annotations

from dataclasses import dataclass

from hft_lob.configs.experiment import ExperimentConfig
from hft_lob.preprocessing.pipeline import PreparedDataset
from hft_lob.systems.metrics import EvaluationReport


@dataclass(frozen=True)
class FoldResult:
    """一个 candidate 在一个 fold 上的完整可追踪结果。"""

    fold_index: int
    candidate_name: str
    dataset_version: str
    normalizer_state_path: str
    checkpoint_path: str | None
    predictions_path: str
    evaluation: EvaluationReport


@dataclass(frozen=True)
class WalkForwardReport:
    """全部模型/baseline、全部 fold 的结果集合。"""

    dataset_version: str
    fold_results: tuple[FoldResult, ...]
    summary: dict[str, dict[str, float]]


def run_walk_forward(
    dataset: PreparedDataset,
    config: ExperimentConfig,
) -> WalkForwardReport:
    """执行统一闭环。

    每个 fold 独立解析文件、仅用训练段拟合 normalizer；主模型及配置中的所有
    Zero/Imbalance/Ridge/MLP 均属于 baseline，由 baseline runner 统一适配；主模型
    经 LOBLightningModule 执行。所有 candidate 生成同一 PredictionArtifact parquet，
    再由 build_evaluation_report 评估。禁止跨 fold 复用 normalizer/checkpoint。
    checkpoint 和 early stopping 必须使用 ``config.training.monitor_metric`` 与
    ``config.training.monitor_mode``，不允许 runner 自行定义另一套指标名。
    """
    raise NotImplementedError("run_walk_forward not implemented")
