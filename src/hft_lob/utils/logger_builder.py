# utils/logger_builder.py
from __future__ import annotations

from typing import Any, Optional

from lightning.pytorch.loggers import Logger


def build_logger(
    experiment_id: str,
    log_dir: str,
    project_name: str = "Limit_Order_Book",
    hyperparams: Optional[dict[str, Any]] = None,
    offline_mode: bool = False,
    resume_run_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> Optional[Logger]:
    """
    构建日志器（支持 WandB 降级到 TensorBoard）。

    策略：
    1. 若 resume_run_id 存在，恢复已有的 WandB run。
    2. 否则创建新 run，若 offline_mode=True 则以 offline 模式启动。
    3. WandB 初始化失败时，降级为 TensorBoardLogger。
    4. TensorBoardLogger 失败时返回 None。

    Args:
        experiment_id: 实验 ID（作为 run name）。
        log_dir: 本地日志保存路径。
        project_name: WandB 项目名称。
        hyperparams: 需要记录的超参数字典（扁平化）。
        offline_mode: 是否强制以 offline 模式运行 WandB。
        resume_run_id: 恢复 WandB run 的 ID（用于续跑）。
        tags: WandB 标签列表（如 ["baseline", "attention"]）。

    Returns:
        日志器实例（WandbLogger / TensorBoardLogger / None）。
    """
    ...


def build_tensorboard_logger(
    experiment_id: str,
    log_dir: str,
) -> Logger:
    """
    构建 TensorBoard 日志器（兜底方案）。

    Args:
        experiment_id: 实验 ID。
        log_dir: 日志保存路径。

    Returns:
        TensorBoardLogger 实例。
    """
    ...


def flatten_config(
    config: dict[str, Any],
    parent_key: str = "",
    sep: str = "/",
    max_depth: int = 5,
) -> dict[str, Any]:
    """
    将嵌套配置字典扁平化，供 WandB log_hyperparams 使用。

    Args:
        config: 嵌套配置字典。
        parent_key: 父级键名前缀（递归使用）。
        sep: 层级分隔符（默认 "/"）。
        max_depth: 最大递归深度（防止循环引用）。

    Returns:
        扁平化后的字典。
        例：{"data/threshold": 0.01, "training/epochs": 100}

    Raises:
        TypeError: 当 config 包含不可序列化对象时（如 nn.Module）。
        RecursionError: 嵌套深度超过 max_depth。
    """
    ...


def is_serializable(value: Any) -> bool:
    """
    检查值是否可被 WandB 序列化记录。

    Args:
        value: 待检查的值。

    Returns:
        True 表示可序列化，False 表示不可序列化。
    """
    ...