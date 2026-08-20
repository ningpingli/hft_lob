"""日志器构建：wandb 尽力而为降级（online → offline → TensorBoard → None）。"""

from __future__ import annotations

from typing import Any

from lightning.pytorch.loggers import Logger

#: wandb 项目名（§29 实验跟踪；失败不影响本地 metrics）。
_WANDB_PROJECT = "hft_lob"


def build_logger(
    experiment_id: str,
    log_dir: str,
    project_name: str = _WANDB_PROJECT,
    hyperparams: dict[str, Any] | None = None,
    offline_mode: bool = False,
) -> Logger | None:
    """构建日志器：优先 wandb（online→offline），失败降级 TensorBoard，再失败 None。

    Args:
        experiment_id: 实验 ID（作为 run name / 本地目录名）。
        log_dir: 本地日志保存路径。
        project_name: wandb 项目名。
        hyperparams: 需记录的扁平超参（best-effort）。
        offline_mode: 强制 wandb offline 模式。

    Returns:
        日志器实例或 None（全部失败时）。
    """
    raise NotImplementedError("build_logger not implemented")
