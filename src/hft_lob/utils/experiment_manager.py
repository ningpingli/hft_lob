"""实验管理（需求文档 §29）：实验 ID 生成/恢复、结果目录、data.yaml 记录。

合并原 ``loggers/`` 包职责（generate_id / find_save_path / logger），消除与
``utils`` 的重复：实验标识、结果目录与阶段结果 YAML 落盘统一收口在此。
"""

from __future__ import annotations

import os
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from hft_lob.utils._yaml_io import atomic_dump_yaml

#: 输出根目录（cwd 相对）。
_RESULTS_ROOT = "output"

def generate_experiment_id(model_name: str, ticker: str) -> str:
    """生成新实验 ID：``<ticker>_<model>_<YYYY-MM-DD_HH_MM_SS>_<7位随机>``，
    并创建结果目录（§29）。

    Args:
        model_name: 模型名。
        ticker: 股票代码。

    Returns:
        唯一实验 ID。
    """
    _validate_path_component(model_name, field="model_name")
    _validate_path_component(ticker, field="ticker")
    random_part = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(7))
    init_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    experiment_id = f"{ticker}_{model_name}_{init_time}_{random_part}"
    os.makedirs(os.path.join(_RESULTS_ROOT, experiment_id), exist_ok=True)
    return experiment_id


def resolve_log_dir(experiment_id: str) -> str:
    """实验结果目录：``output/<experiment_id>``。"""
    _validate_path_component(experiment_id, field="experiment_id")
    return os.path.join(_RESULTS_ROOT, experiment_id)


def resolve_experiment_id(
    *,
    model_name: str,
    ticker: str,
    override_id: str | None = None,
) -> str:
    """解析实验 ID：显式 ID 优先，否则生成新 ID。

    Args:
        model_name: 模型名。
        ticker: 股票代码。
        override_id: 命令行显式指定的实验 ID。

    Returns:
        实验 ID。

    """
    if override_id is not None:
        _validate_path_component(override_id, field="override_id")
        return override_id
    return generate_experiment_id(model_name, ticker)


def write_experiment_log(experiment_id: str, header: str, contents: dict[str, Any]) -> None:
    """将阶段结果记录到 ``output/<experiment_id>/data.yaml``。

    文件已存在时按 header 键合并，否则创建新文件（§29 结果可追踪）。

    Args:
        experiment_id: 实验 ID。
        header: 记录标题（如 dataset_info）。
        contents: 记录内容字典。
    """
    _validate_path_component(experiment_id, field="experiment_id")
    if not isinstance(header, str) or not header.strip():
        raise ValueError("header must not be empty")
    destination = Path(resolve_log_dir(experiment_id)) / "data.yaml"

    data: dict[str, Any] = {}
    if destination.exists():
        if not destination.is_file():
            raise ValueError(f"experiment log path is not a file: {destination}")
        try:
            loaded = yaml.safe_load(destination.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid experiment log YAML: {destination}") from exc
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ValueError("experiment log root must be a mapping")
            data = loaded

    data[header] = contents
    atomic_dump_yaml(destination, data)


def _validate_path_component(value: str, *, field: str) -> None:
    """验证用于结果目录或 YAML 顶层键的单个非空名称。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must not be empty")
    if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"{field} must be a single path component")
