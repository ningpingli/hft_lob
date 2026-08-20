"""实验管理（需求文档 §29）：实验 ID 生成/恢复、结果目录、data.yaml 记录。

合并原 ``loggers/`` 包职责（generate_id / find_save_path / logger），消除与
``utils`` 的重复：实验标识、结果目录与阶段结果 YAML 落盘统一收口在此。
"""

from __future__ import annotations

import os
import random
import re
import string
from datetime import datetime
from typing import Any

#: 结果根目录（cwd 相对）。
_RESULTS_ROOT = os.path.join("loggers", "results")

#: 实验目录名模式：``loggers/results/<experiment_id>/...``。
_EXP_ID_IN_PATH = re.compile(r"loggers[\\/]results[\\/]([^\\/]+)[\\/]")


def generate_experiment_id(model_name: str, ticker: str) -> str:
    """生成新实验 ID：``<ticker>_<model>_<YYYY-MM-DD_HH_MM_SS>_<7位随机>``，
    并创建结果目录（§29）。

    Args:
        model_name: 模型名。
        ticker: 股票代码。

    Returns:
        唯一实验 ID。
    """
    random_part = "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(7)
    )
    init_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    experiment_id = f"{ticker}_{model_name}_{init_time}_{random_part}"
    os.makedirs(os.path.join(_RESULTS_ROOT, experiment_id), exist_ok=True)
    return experiment_id


def resolve_log_dir(experiment_id: str) -> str:
    """实验结果目录：``loggers/results/<experiment_id>``。"""
    return os.path.join(_RESULTS_ROOT, experiment_id)


def resolve_experiment_id(
    *,
    model_name: str,
    ticker: str,
    override_id: str | None = None,
    resume_ckpt: str | None = None,
) -> str:
    """解析实验 ID：``override_id`` > 检查点路径中提取 > 新生成。

    Args:
        model_name: 模型名。
        ticker: 股票代码。
        override_id: 命令行显式指定的实验 ID。
        resume_ckpt: 恢复训练的检查点路径（用于提取原实验 ID）。

    Returns:
        实验 ID。

    Raises:
        ValueError: ``resume_ckpt`` 存在但无法提取有效实验 ID。
    """
    raise NotImplementedError("resolve_experiment_id not implemented")


def extract_exp_id_from_ckpt(ckpt_path: str) -> str | None:
    """从检查点路径中提取实验 ID（路径须含 ``loggers/results/<id>/`` 段）。"""
    raise NotImplementedError("extract_exp_id_from_ckpt not implemented")


def write_experiment_log(experiment_id: str, header: str, contents: dict[str, Any]) -> None:
    """将阶段结果记录到 ``loggers/results/<experiment_id>/data.yaml``。

    文件已存在时按 header 键合并，否则创建新文件（§29 结果可追踪）。

    Args:
        experiment_id: 实验 ID。
        header: 记录标题（如 dataset_info）。
        contents: 记录内容字典。
    """
    raise NotImplementedError("write_experiment_log not implemented")
