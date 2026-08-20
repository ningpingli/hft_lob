"""检查点工具：解析最佳检查点路径与实验配置备份。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hft_lob.utils._yaml_io import atomic_dump_yaml


def resolve_ckpt_path(
    log_dir: str,
    filename: str = "best_val_model.ckpt",
    fallback_to_latest: bool = True,
) -> str | None:
    """解析最佳检查点或指定检查点的完整路径。

    Args:
        log_dir: 日志目录。
        filename: 检查点文件名（默认 best_val_model.ckpt）。
        fallback_to_latest: 指定文件不存在时是否返回最新 .ckpt。

    Returns:
        检查点完整路径；不存在且 ``fallback_to_latest=False`` 时返回 None。
    """
    directory = Path(log_dir).expanduser()
    requested = directory / filename
    if requested.is_file():
        return str(requested)
    if not fallback_to_latest or not directory.is_dir():
        return None

    checkpoints = [path for path in directory.glob("*.ckpt") if path.is_file()]
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda path: (path.stat().st_mtime_ns, path.name))
    return str(latest)


def backup_experiment_config(
    log_dir: str, config: dict[str, Any], filename: str = "config_used.yaml"
) -> str:
    """将运行使用的完整配置备份到实验目录（§29 可复现）。

    Args:
        log_dir: 日志目录。
        config: 完整配置字典。
        filename: 备份文件名。

    Returns:
        备份文件的完整路径。
    """
    directory = Path(log_dir).expanduser()
    destination = directory / filename
    atomic_dump_yaml(destination, config)
    return str(destination)
