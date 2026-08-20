"""Walk-forward 统一执行闭环。"""

from __future__ import annotations

from dataclasses import dataclass

from hft_lob.configs.experiment import ExperimentConfig, WalkForwardConfig
from hft_lob.preprocessing.pipeline import PreparedDataset
from hft_lob.preprocessing.split import Fold, WalkForwardPlan
from hft_lob.systems.metrics import EvaluationReport


@dataclass(frozen=True)
class FoldResult:
    """一个 candidate 在一个 fold 上的完整可追踪结果。"""

    fold_index: int
    candidate_name: str
    dataset_version: str
    standardizer_state_path: str
    checkpoint_path: str | None
    predictions_path: str
    evaluation: EvaluationReport


@dataclass(frozen=True)
class WalkForwardReport:
    """全部模型/baseline、全部 fold 的结果集合。"""

    dataset_version: str
    fold_results: tuple[FoldResult, ...]
    summary: dict[str, dict[str, float]]


def select_walk_forward_folds(
    plan: WalkForwardPlan,
    config: WalkForwardConfig,
) -> tuple[Fold, ...]:
    """从固定计划中选择本次要执行的连续周期，不改变任何 fold 内容。

    ``start_fold`` 直接匹配计划中的一基 fold 编号；``num_folds`` 必须能完整
    满足，避免用户要求训练3折却静默只执行剩余2折。
    """
    if not config.enabled:
        return ()
    folds = plan.folds
    if not folds:
        raise ValueError("walk-forward plan contains no folds")

    start_index = next(
        (index for index, fold in enumerate(folds) if fold.index == config.start_fold),
        len(folds),
    )
    if start_index == len(folds):
        raise ValueError(f"walk_forward.start_fold {config.start_fold} is not in the plan")

    if config.num_folds is None:
        return folds[start_index:]
    end_index = start_index + config.num_folds
    if end_index > len(folds):
        available = len(folds) - start_index
        raise ValueError(
            f"walk_forward.num_folds requests {config.num_folds}, only {available} available"
        )
    return folds[start_index:end_index]


def run_walk_forward(
    dataset: PreparedDataset,
    config: ExperimentConfig,
) -> WalkForwardReport:
    """执行统一闭环。

    每个 fold 独立解析文件；特征使用仅依赖当前时刻之前窗口的因果标准化；主模型及配置中的所有
    Zero/Imbalance/Ridge/MLP 均属于 baseline，由 baseline runner 统一适配；主模型
    经 LOBLightningModule 执行。所有 candidate 生成同一 PredictionArtifact parquet，
    再由 build_evaluation_report 评估。禁止跨 fold 复用 checkpoint。
    checkpoint 和 early stopping 必须使用 ``config.training.monitor_metric`` 与
    ``config.training.monitor_mode``，不允许 runner 自行定义另一套指标名。
    """
    raise NotImplementedError("run_walk_forward not implemented")
