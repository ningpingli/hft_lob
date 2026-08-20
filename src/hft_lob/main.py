"""hft_lob 流水线 CLI 入口（需求文档 §40/§41 MVP 阶段编排）。
"""

from __future__ import annotations

import argparse

VALID_STAGES: tuple[str, ...] = (
    "data_processing", "dataset_preparation", "training", "evaluation",
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的参数对象（--config / --experiment-id / --resume-ckpt /
        --stages / --seed）。
    """
    raise NotImplementedError("parse_args not implemented")


def main() -> None:
    """
    训练流程主入口。

    完整流程：
    1. 加载配置（load_experiment_config）
    2. 解析实验 ID（resolve_experiment_id）
    3. 解析日志目录（resolve_log_dir）
    4. 备份配置（backup_experiment_config）
    5. 构建日志器（build_logger）
    6. 构建数据模块（build_datamodule）
    7. 构建模型（build_model）
    8. 构建回调（build_checkpoint_callback, build_early_stopping_callback）
    9. 构建 Trainer（build_trainer）
    10. 执行训练（run_training）
    11. 执行测试（run_test）或 预测（run_predict）

    Args:
        config_path: 实验配置文件路径。
        model_name: 若指定，覆盖配置文件中的 model 字段。
        resume_ckpt: 恢复训练的检查点路径。
        skip_test: 是否跳过测试阶段。
        offline_log: 是否强制使用 offline 日志模式。
        override_exp_id: 覆盖自动生成的实验 ID。
    """
    raise NotImplementedError("train.main not implemented")

if __name__ == "__main__":
    main()

