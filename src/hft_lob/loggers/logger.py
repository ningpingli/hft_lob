"""实验日志：实验 ID 生成、结果目录路径与 YAML 落盘。"""

from __future__ import annotations

from typing import Any


def generate_id(name: str, target_stock: str) -> str:
    """基于模型名、目标股票与当前时间戳生成唯一实验 ID，并创建结果目录。

    Args:
        name: 实验中使用的深度学习模型名称。
        target_stock: 目标股票标识。

    Returns:
        唯一实验标识。
    """
    raise NotImplementedError("generate_id not implemented")


def find_save_path(model_id: str) -> str:
    """查找与给定 model_id 关联的结果保存目录路径。

    Args:
        model_id: 模型标识。

    Returns:
        结果保存目录路径。
    """
    raise NotImplementedError("find_save_path not implemented")


def logger(experiment_id: str, header: str, contents: dict[str, Any]) -> None:
    """将实验结果记录到 ``loggers/results/<experiment_id>/data.yaml``。

    文件已存在时向其中追加新数据（按 header 键合并），否则创建新文件。

    Args:
        experiment_id: 实验（模型）标识。
        header: 所记录数据的标题。
        contents: 要记录的数据字典。
    """
    raise NotImplementedError("logger not implemented")
