"""实验配置备份。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hft_lob.utils._yaml_io import atomic_dump_yaml


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
