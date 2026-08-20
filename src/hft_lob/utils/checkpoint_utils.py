# utils/checkpoint_utils.py
from __future__ import annotations

from typing import Any, Optional


def resolve_ckpt_path(
    log_dir: str,
    filename: str = "best_val_model.ckpt",
    fallback_to_latest: bool = True,
) -> Optional[str]:
    """
    解析最佳检查点或指定检查点的完整路径。

    Args:
        log_dir: 日志目录。
        filename: 检查点文件名（默认 "best_val_model.ckpt"）。
        fallback_to_latest: 若指定文件不存在，是否返回最新的 .ckpt 文件。

    Returns:
        检查点完整路径，若不存在且 fallback_to_latest=False 则返回 None。
    """
    ...


def backup_experiment_config(
    log_dir: str,
    config: dict[str, Any],
    filename: str = "config_used.yaml",
) -> str:
    """
    将运行使用的完整配置备份到实验目录中。

    Args:
        log_dir: 日志目录。
        config: 完整配置字典。
        filename: 备份文件名。

    Returns:
        备份文件的完整路径。

    Raises:
        OSError: 目录不存在且创建失败时。
    """
    ...


def list_checkpoints(log_dir: str) -> list[dict[str, Any]]:
    """
    列出实验目录下所有检查点文件及其元信息。

    Args:
        log_dir: 日志目录。

    Returns:
        [
            {
                "path": str,
                "filename": str,
                "epoch": Optional[int],
                "step": Optional[int],
                "size_mb": float,
                "modified_time": str,
            },
            ...
        ]
    """
    ...


def get_best_checkpoint_metadata(
    log_dir: str,
    monitor_metric: str = "val_ic",
    mode: str = "max",
) -> Optional[dict[str, Any]]:
    """
    从检查点元信息中获取最佳检查点的元数据。

    Args:
        log_dir: 日志目录。
        monitor_metric: 监控的指标名称（如 "val_ic"）。
        mode: "max" 或 "min"。

    Returns:
        {
            "path": str,
            "metric_value": float,
            "epoch": int,
            "step": int,
        }
        若未找到则返回 None。
    """
    ...