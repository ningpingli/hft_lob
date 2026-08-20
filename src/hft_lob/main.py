"""hft_lob 流水线 CLI 入口（需求文档 §40/§41 MVP 阶段编排）。
"""

from __future__ import annotations

import argparse

VALID_STAGES: tuple[str, ...] = (
    "prepare-data", "walk-forward", "evaluate", "predict-offline",
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
    1. 加载配置（load_config）
    2. 解析实验 ID（resolve_experiment_id）
    3. 解析日志目录（resolve_log_dir）
    4. 备份配置（backup_experiment_config）
    5. 构建日志器（build_logger）
    6. prepare_dataset 生成固定版本和 WalkForwardPlan
    7. run_walk_forward 对每个 fold 独立拟合 normalizer 并运行模型/baseline
    8. 所有 candidate 输出 PredictionArtifact
    9. build_evaluation_report 输出 EvaluationReport/FoldResult
    """
    raise NotImplementedError("train.main not implemented")

if __name__ == "__main__":
    main()

