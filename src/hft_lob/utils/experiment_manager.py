# utils/experiment_manager.py
from __future__ import annotations

from typing import Optional, Any


def resolve_experiment_id(
    config: dict[str, Any],
    resume_ckpt: Optional[str] = None,
    override_id: Optional[str] = None,
) -> str:
    """
    解析并生成实验唯一标识符。

    优先级：override_id > resume_ckpt 中提取的 ID > 基于时间戳和模型名生成的新 ID。

    Args:
        config: 完整实验配置字典（至少包含 model 字段）。
        resume_ckpt: 恢复训练的检查点路径（用于提取原实验 ID）。
        override_id: 命令行指定的实验 ID（--exp-id）。

    Returns:
        实验唯一标识符字符串（如 "deeplob_20250101_120000"）。

    Raises:
        ValueError: resume_ckpt 存在但无法提取有效 ID 时。
    """
    ...


def extract_exp_id_from_ckpt(ckpt_path: str) -> str:
    """
    从检查点路径中提取实验 ID。

    支持的路径格式：
    - ./logs/exp_001/checkpoints/best.ckpt → "exp_001"
    - ./logs/20250101_120000/checkpoints/last.ckpt → "20250101_120000"
    - /absolute/path/logs/my_exp/checkpoints/model.ckpt → "my_exp"

    Args:
        ckpt_path: 检查点文件路径。

    Returns:
        提取出的实验 ID。

    Raises:
        ValueError: 路径格式不符合预期，无法提取 ID。
    """
    ...


def resolve_log_dir(
    experiment_id: str,
    base_log_dir: str = "./logs",
    suffix: Optional[str] = None,
) -> str:
    """
    解析日志/检查点保存目录路径。

    Args:
        experiment_id: 实验 ID。
        base_log_dir: 日志根目录（默认 "./logs"）。
        suffix: 可选的子目录后缀（如 "resume" 用于恢复训练）。

    Returns:
        完整的日志目录路径（如 "./logs/exp_001" 或 "./logs/exp_001/resume"）。
    """
    ...


def generate_experiment_id(
    model_name: str,
    timestamp_format: str = "%Y%m%d_%H%M%S",
) -> str:
    """
    基于模型名和时间戳生成新实验 ID。

    Args:
        model_name: 模型名称。
        timestamp_format: 时间戳格式。

    Returns:
        格式为 "{model_name}_{timestamp}" 的字符串。
    """
    ...