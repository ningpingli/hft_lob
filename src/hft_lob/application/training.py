"""训练应用服务：集中组装配置、数据包、实验环境与 walk-forward。"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingRequest:
    """可由 CLI、测试或其他 Python 调用方复用的训练请求。"""

    config_path: str
    dataset_dir: str
    experiment_id: str | None = None
    seed: int | None = None
    gpu_id: int | None = None


@dataclass(frozen=True)
class TrainingResult:
    """训练应用服务返回给适配层的最小结果。"""

    experiment_id: str
    dataset_version: str
    fold_result_count: int


def run_training_application(request: TrainingRequest) -> TrainingResult:
    """执行一次完整训练应用流程。"""
    started = time.perf_counter()
    logger.info("training.start dataset_dir=%s config=%s", request.dataset_dir, request.config_path)
    _configure_gpu(request.gpu_id)

    # GPU 可见设备设置完成后再导入 Torch/Lightning 依赖链。
    from hft_lob.configs import load_model_config
    from hft_lob.datasets.dataset_validator import load_dataset_package
    from hft_lob.systems.executor import DefaultWalkForwardExecutor
    from hft_lob.systems.walk_forward import run_walk_forward
    from hft_lob.utils.checkpoint_utils import backup_experiment_config
    from hft_lob.utils.experiment_manager import (
        resolve_experiment_id,
        resolve_log_dir,
        write_experiment_log,
    )
    from hft_lob.utils.seed import set_seed

    provisional_id = request.experiment_id or "pending"
    config = load_model_config(request.config_path, experiment_id=provisional_id)
    package = load_dataset_package(request.dataset_dir)
    logger.info(
        "training.dataset_loaded dataset_id=%s ticker=%s validation=skipped",
        package.metadata.dataset_id,
        package.metadata.ticker,
    )
    experiment_id = resolve_experiment_id(
        model_name=config.model.name,
        ticker=package.metadata.ticker,
        override_id=request.experiment_id,
    )
    config = replace(
        config,
        experiment_id=experiment_id,
        seed=config.seed if request.seed is None else request.seed,
    )
    set_seed(config.seed)
    log_dir = Path(resolve_log_dir(experiment_id))
    log_dir.mkdir(parents=True, exist_ok=True)
    backup_experiment_config(str(log_dir), asdict(config))
    logger.info(
        "training.experiment_ready experiment_id=%s model=%s folds_start=%d folds_count=%s seed=%d",
        experiment_id,
        config.model.name,
        config.folds.start_fold,
        config.folds.num_folds,
        config.seed,
    )

    report = run_walk_forward(
        package,
        config,
        executor=DefaultWalkForwardExecutor(str(log_dir / "walk_forward")),
    )
    write_experiment_log(
        experiment_id,
        "walk_forward",
        {
            "dataset_version": report.dataset_version,
            "result_count": len(report.fold_results),
            "summary": report.summary,
        },
    )
    logger.info(
        "training.complete experiment_id=%s dataset_id=%s results=%d elapsed_seconds=%.3f",
        experiment_id,
        report.dataset_version,
        len(report.fold_results),
        time.perf_counter() - started,
    )
    return TrainingResult(
        experiment_id=experiment_id,
        dataset_version=report.dataset_version,
        fold_result_count=len(report.fold_results),
    )


def _configure_gpu(gpu_id: int | None) -> None:
    if gpu_id is None:
        return
    if isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0:
        raise ValueError("gpu_id must be >= 0")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
