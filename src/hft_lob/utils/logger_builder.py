"""日志器构建：TensorBoard 本地实验记录。"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from lightning.pytorch.loggers import Logger


def build_logger(
    experiment_id: str,
    log_dir: str,
    hyperparams: dict[str, Any] | None = None,
) -> Logger | None:
    """构建 TensorBoard 日志器，后端不可用时返回 None。

    Args:
        experiment_id: 实验 ID（作为 run name / 本地目录名）。
        log_dir: 本地日志保存路径。
        hyperparams: 需记录的扁平超参（best-effort）。

    Returns:
        TensorBoardLogger 实例，或后端初始化失败时的 None。
    """
    for field, value in (
        ("experiment_id", experiment_id),
        ("log_dir", log_dir),
    ):
        if not value.strip():
            raise ValueError(f"{field} must not be empty")

    Path(log_dir).expanduser().mkdir(parents=True, exist_ok=True)
    try:
        logger = _build_tensorboard_logger(
            experiment_id=experiment_id,
            log_dir=log_dir,
        )
    except Exception as exc:
        _warn_backend_failure("TensorBoard", exc)
        return None
    _log_hyperparams_best_effort(logger, hyperparams)
    return logger


def _build_tensorboard_logger(*, experiment_id: str, log_dir: str) -> Logger:
    """延迟加载 TensorBoard 兜底后端。"""
    from lightning.pytorch.loggers import TensorBoardLogger

    return TensorBoardLogger(
        save_dir=str(Path(log_dir).expanduser()),
        name=experiment_id,
        version="",
    )


def _log_hyperparams_best_effort(logger: Logger, hyperparams: dict[str, Any] | None) -> None:
    if hyperparams is None:
        return
    try:
        logger.log_hyperparams(hyperparams)
    except Exception as exc:
        _warn_backend_failure("hyperparameter logging", exc)


def _warn_backend_failure(backend: str, error: Exception) -> None:
    warnings.warn(
        f"{backend} unavailable; continuing with fallback: {type(error).__name__}: {error}",
        RuntimeWarning,
        stacklevel=3,
    )
