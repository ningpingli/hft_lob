# utils/config_loader.py
from __future__ import annotations

from typing import Any, Optional


def load_experiment_config(
    config_path: str,
    validate: bool = True,
) -> dict[str, Any]:
    """
    加载并校验实验配置文件。

    Args:
        config_path: YAML 配置文件路径（如 configs/experiment.yaml）。
        validate: 是否执行配置校验（默认 True）。

    Returns:
        {
            "general": dict[str, Any],   # 通用配置（模型名、股票列表等）
            "data": dict[str, Any],      # 数据配置（threshold, window_size 等）
            "loader": dict[str, Any],    # 加载器配置（batch_size, num_workers 等）
            "training": dict[str, Any],  # 训练配置（epochs, patience, loss 等）
        }

    Raises:
        FileNotFoundError: 配置文件不存在。
        ValueError: 配置缺少必要字段或字段类型错误。
        yaml.YAMLError: YAML 解析失败。
    """
    ...


def load_model_config(
    model_name: str,
    config_dir: str = "configs/models",
) -> dict[str, Any]:
    """
    加载模型专属配置文件。

    Args:
        model_name: 模型名称（如 "deeplob", "transformer"）。
        config_dir: 模型配置根目录。

    Returns:
        {
            "data_features": {
                "num_features": int,
                "levels": int,
                "history_length": int,
            },
            "model_params": dict[str, Any],  # 模型特定超参数
        }

    Raises:
        FileNotFoundError: configs/models/{model_name}.yaml 不存在。
        ValueError: 配置文件缺少 data_features 或 model_params 字段。
    """
    ...


def load_config_from_cli(
    config_path: str,
    model_override: Optional[str] = None,
    exp_id_override: Optional[str] = None,
) -> tuple[dict[str, Any], str, Optional[str]]:
    """
    从命令行参数加载配置（封装备用解析逻辑）。

    Args:
        config_path: 配置文件路径。
        model_override: 命令行指定的模型名（覆盖配置文件）。
        exp_id_override: 命令行指定的实验 ID（覆盖自动生成）。

    Returns:
        (config_dict, model_name, experiment_id)
    """
    ...