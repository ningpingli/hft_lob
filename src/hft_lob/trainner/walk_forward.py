"""Walk-forward 统一执行闭环。"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from hft_lob.configs.experiment import FoldSelectionConfig, ModelRunConfig
from hft_lob.data_pipeline.writer import DatasetPackage
from hft_lob.metrics.metrics import EvaluationReport, build_evaluation_report
from hft_lob.reporting.artifact import PredictionArtifact, save_prediction_artifact
from hft_lob.reporting.reporter import save_evaluation_outputs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateFoldRun:
    """执行器交给编排层的单次模型预测结果。

    执行器只负责模型训练/推理；artifact 保存、评估和跨 fold 汇总由本模块统一完成。
    baseline 由独立 baseline experiment 生成，不进入模型 candidate 列表。
    """

    artifact: PredictionArtifact
    dataset_metadata_path: str
    predictions_path: str
    checkpoint_path: str | None = None


class WalkForwardExecutor(Protocol):
    """模型实验的 fold 执行边界。"""

    def run_candidate(
        self,
        *,
        package: DatasetPackage,
        config: ModelRunConfig,
        fold_index: int,
        candidate_name: str,
    ) -> CandidateFoldRun:
        """在一个 fold 上拟合并返回模型 test split 的预测。"""
        ...


@dataclass(frozen=True)
class FoldResult:
    """一个 candidate 在一个 fold 上的完整可追踪结果。"""

    fold_index: int
    candidate_name: str
    dataset_version: str
    dataset_metadata_path: str
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
    package: DatasetPackage,
    config: ModelRunConfig,
    *,
    executor: WalkForwardExecutor,
) -> WalkForwardReport:
    """执行模型 walk-forward 闭环；跨 fold 的测试评估统一计算。"""
    root = package.root
    metadata = package.metadata
    fold_indices = select_package_folds(root, config.folds)
    candidates = (config.model.name,)
    logger.info(
        "walk_forward.start dataset_id=%s folds=%s candidates=%s",
        metadata.dataset_id,
        fold_indices,
        candidates,
    )

    results: list[FoldResult] = []
    candidate_runs: dict[str, list[CandidateFoldRun]] = {name: [] for name in candidates}
    for fold_index in fold_indices:
        for candidate_name in candidates:
            candidate_started = time.perf_counter()
            logger.info(
                "walk_forward.candidate_start fold=%d candidate=%s", fold_index, candidate_name
            )
            run = executor.run_candidate(
                package=package,
                config=config,
                fold_index=fold_index,
                candidate_name=candidate_name,
            )
            _validate_candidate_run(
                run,
                dataset_version=metadata.dataset_id,
                fold_index=fold_index,
                candidate_name=candidate_name,
            )
            predictions_path = save_prediction_artifact(
                artifact=run.artifact,
                path=run.predictions_path,
            )
            candidate_runs[candidate_name].append(run)
            fold_evaluation = build_evaluation_report(run.artifact, config.evaluation)
            results.append(
                FoldResult(
                    fold_index=fold_index,
                    candidate_name=candidate_name,
                    dataset_version=metadata.dataset_id,
                    dataset_metadata_path=run.dataset_metadata_path,
                    checkpoint_path=run.checkpoint_path,
                    predictions_path=predictions_path,
                    evaluation=fold_evaluation,
                )
            )
            logger.info(
                "walk_forward.candidate_complete fold=%d candidate=%s samples=%d elapsed_seconds=%.3f",
                fold_index,
                candidate_name,
                fold_evaluation.sample_count,
                time.perf_counter() - candidate_started,
            )

    aggregate_evaluations: dict[str, EvaluationReport] = {}
    for candidate_name in candidates:
        runs = candidate_runs[candidate_name]
        aggregate_artifact = _concatenate_artifacts(
            [run.artifact for run in runs],
            candidate_name=candidate_name,
        )
        aggregate_evaluation = build_evaluation_report(
            aggregate_artifact,
            config.evaluation,
        )
        aggregate_evaluations[candidate_name] = aggregate_evaluation
        output_dir = _aggregate_output_dir(runs[0].predictions_path, candidate_name)
        evaluation_outputs = save_evaluation_outputs(aggregate_evaluation, output_dir)
        logger.info(
            "walk_forward.evaluation_outputs folds=%s candidate=%s paths=%s",
            fold_indices,
            candidate_name,
            evaluation_outputs,
        )

    return WalkForwardReport(
        dataset_version=metadata.dataset_id,
        fold_results=tuple(results),
        summary=_summarize_results(
            results,
            candidates=candidates,
            aggregate_evaluations=aggregate_evaluations,
        ),
    )


def _concatenate_artifacts(
    artifacts: Sequence[PredictionArtifact],
    *,
    candidate_name: str,
) -> PredictionArtifact:
    if not artifacts:
        raise ValueError("artifacts must not be empty")
    first = artifacts[0]
    for artifact in artifacts:
        if (
            artifact.model_name != candidate_name
            or artifact.labels != first.labels
            or artifact.dataset_version != first.dataset_version
            or artifact.split != "test"
        ):
            raise ValueError("walk-forward artifacts must share candidate identity")
    return PredictionArtifact(
        predictions=np.concatenate([artifact.predictions for artifact in artifacts], axis=0),
        targets=np.concatenate([artifact.targets for artifact in artifacts], axis=0),
        labels=first.labels,
        metadata=tuple(meta for artifact in artifacts for meta in artifact.metadata),
        model_name=first.model_name,
        model_version=first.model_version,
        dataset_version=first.dataset_version,
        fold_index=first.fold_index,
        split=first.split,
    )


def _aggregate_output_dir(predictions_path: str, candidate_name: str) -> Path:
    path = Path(predictions_path).resolve()
    try:
        return path.parents[2] / candidate_name
    except IndexError as exc:
        raise ValueError("predictions_path must include fold and candidate directories") from exc


def select_package_folds(root: Path, config: FoldSelectionConfig) -> tuple[int, ...]:
    available = tuple(
        sorted(int(path.name.removeprefix("fold_")) for path in (root / "folds").glob("fold_*"))
    )
    if config.start_fold not in available:
        raise ValueError(f"walk_forward.start_fold {config.start_fold} is not in the package")
    start = available.index(config.start_fold)
    selected = (
        available[start:]
        if config.num_folds is None
        else available[start : start + config.num_folds]
    )
    if config.num_folds is not None and len(selected) != config.num_folds:
        raise ValueError(
            f"walk_forward.num_folds requests {config.num_folds}, only {len(selected)} available"
        )
    return selected


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
    if not run.dataset_metadata_path.strip():
        raise ValueError("dataset_metadata_path must not be empty")
    if not run.predictions_path.strip():
        raise ValueError("predictions_path must not be empty")


def _summarize_results(
    results: list[FoldResult],
    *,
    candidates: tuple[str, ...],
    aggregate_evaluations: Mapping[str, EvaluationReport],
) -> dict[str, dict[str, float]]:
    """汇总 fold 标量指标，并使用连续测试窗口的整体日级 IC。"""
    summary: dict[str, dict[str, float]] = {}
    for candidate_name in candidates:
        candidate_results = [
            result for result in results if result.candidate_name == candidate_name
        ]
        if not candidate_results:
            raise ValueError(f"candidate {candidate_name!r} has no fold results")
        aggregate = aggregate_evaluations[candidate_name]
        metrics = tuple(candidate_results[0].evaluation.overall)
        values: dict[str, float] = {
            "fold_count": float(len(candidate_results)),
            "sample_count": float(
                sum(result.evaluation.sample_count for result in candidate_results)
            ),
            "mean_daily_ic_mean": aggregate.mean_daily_ic,
        }
        mean_daily_ics = np.asarray(
            [result.evaluation.mean_daily_ic for result in candidate_results],
            dtype=np.float64,
        )
        finite_mean_daily_ics = mean_daily_ics[np.isfinite(mean_daily_ics)]
        values["mean_daily_ic_std"] = (
            float(np.std(finite_mean_daily_ics)) if finite_mean_daily_ics.size else float("nan")
        )
        for metric in metrics:
            metric_values = np.asarray(
                [result.evaluation.overall[metric] for result in candidate_results],
                dtype=np.float64,
            )
            finite = metric_values[np.isfinite(metric_values)]
            values[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            values[f"{metric}_std"] = float(np.std(finite)) if finite.size else float("nan")
        summary[candidate_name] = values
    return summary
