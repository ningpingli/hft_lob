"""Walk-forward 统一执行闭环。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from hft_lob.configs.experiment import ExperimentConfig, WalkForwardConfig
from hft_lob.preprocessing.pipeline import PreparedDataset
from hft_lob.preprocessing.split import Fold, WalkForwardPlan
from hft_lob.systems.artifact import PredictionArtifact, save_prediction_artifact
from hft_lob.systems.metrics import EvaluationReport, build_evaluation_report


@dataclass(frozen=True)
class CandidateFoldRun:
    """执行器交给编排层的单次预测结果。

    执行器只负责训练/推理以及给出输出位置；artifact 保存、评估和跨 fold
    汇总由本模块统一完成，模型和 baseline 不得各自维护另一套评估路径。
    """

    artifact: PredictionArtifact
    standardizer_state_path: str
    predictions_path: str
    checkpoint_path: str | None = None


class WalkForwardExecutor(Protocol):
    """模型和 baseline 共用的 fold 执行边界。"""

    def run_candidate(
        self,
        *,
        dataset: PreparedDataset,
        config: ExperimentConfig,
        fold: Fold,
        candidate_name: str,
    ) -> CandidateFoldRun:
        """在一个 fold 上拟合并返回 test split 的预测。"""


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
    *,
    executor: WalkForwardExecutor,
) -> WalkForwardReport:
    """执行统一闭环。

    每个 fold 独立解析文件；特征使用仅依赖当前时刻之前窗口的因果标准化；主模型及配置中的所有
    Zero/Imbalance/Ridge/MLP 均属于 baseline，由 baseline runner 统一适配；主模型
    经 LOBLightningModule 执行。所有 candidate 生成同一 PredictionArtifact parquet，
    再由 build_evaluation_report 评估。禁止跨 fold 复用 checkpoint。
    checkpoint 和 early stopping 必须使用 ``config.training.monitor_metric`` 与
    ``config.training.monitor_mode``，不允许 runner 自行定义另一套指标名。
    """
    if dataset.dataset_version != dataset.walk_forward_plan.dataset_version:
        raise ValueError("prepared dataset and walk-forward plan versions do not match")

    folds = select_walk_forward_folds(dataset.walk_forward_plan, config.walk_forward)
    if not folds:
        raise ValueError("walk-forward execution is disabled")
    candidates = _candidate_names(config)

    results: list[FoldResult] = []
    for fold in folds:
        for candidate_name in candidates:
            run = executor.run_candidate(
                dataset=dataset,
                config=config,
                fold=fold,
                candidate_name=candidate_name,
            )
            _validate_candidate_run(
                run,
                dataset_version=dataset.dataset_version,
                fold_index=fold.index,
                candidate_name=candidate_name,
            )
            predictions_path = save_prediction_artifact(
                artifact=run.artifact,
                path=run.predictions_path,
            )
            evaluation = build_evaluation_report(
                run.artifact,
                config.evaluation,
                seed=config.seed + fold.index,
            )
            results.append(
                FoldResult(
                    fold_index=fold.index,
                    candidate_name=candidate_name,
                    dataset_version=dataset.dataset_version,
                    standardizer_state_path=run.standardizer_state_path,
                    checkpoint_path=run.checkpoint_path,
                    predictions_path=predictions_path,
                    evaluation=evaluation,
                )
            )

    return WalkForwardReport(
        dataset_version=dataset.dataset_version,
        fold_results=tuple(results),
        summary=_summarize_results(results, candidates=candidates),
    )


def _candidate_names(config: ExperimentConfig) -> tuple[str, ...]:
    names = (config.model.name, *config.baselines.names)
    invalid = [name for name in names if not name.strip()]
    if invalid:
        raise ValueError("model and baseline names must not be empty")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"candidate names must be unique: {duplicates}")
    return names


def _validate_candidate_run(
    run: CandidateFoldRun,
    *,
    dataset_version: str,
    fold_index: int,
    candidate_name: str,
) -> None:
    artifact = run.artifact
    expected: Mapping[str, object] = {
        "dataset_version": dataset_version,
        "fold_index": fold_index,
        "model_name": candidate_name,
        "split": "test",
    }
    mismatches = {
        field: (expected_value, getattr(artifact, field))
        for field, expected_value in expected.items()
        if getattr(artifact, field) != expected_value
    }
    if mismatches:
        raise ValueError(f"executor returned an artifact with mismatched identity: {mismatches}")
    if not run.standardizer_state_path.strip():
        raise ValueError("standardizer_state_path must not be empty")
    if not run.predictions_path.strip():
        raise ValueError("predictions_path must not be empty")


def _summarize_results(
    results: list[FoldResult],
    *,
    candidates: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """按 candidate 汇总 fold 级 overall 指标，不混入日级重复统计。"""
    summary: dict[str, dict[str, float]] = {}
    for candidate_name in candidates:
        candidate_results = [
            result for result in results if result.candidate_name == candidate_name
        ]
        metrics = tuple(candidate_results[0].evaluation.overall)
        values: dict[str, float] = {
            "fold_count": float(len(candidate_results)),
            "sample_count": float(
                sum(result.evaluation.sample_count for result in candidate_results)
            ),
        }
        for metric in metrics:
            metric_values = np.asarray(
                [result.evaluation.overall[metric] for result in candidate_results],
                dtype=np.float64,
            )
            finite = metric_values[np.isfinite(metric_values)]
            values[f"{metric}_mean"] = (
                float(np.mean(finite)) if finite.size else float("nan")
            )
            values[f"{metric}_std"] = (
                float(np.std(finite)) if finite.size else float("nan")
            )
        summary[candidate_name] = values
    return summary
