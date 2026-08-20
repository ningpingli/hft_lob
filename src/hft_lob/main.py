"""hft_lob 流水线 CLI 入口（需求文档 §40/§41 MVP 阶段编排）。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from hft_lob.configs import load_config
from hft_lob.preprocessing import prepare_dataset
from hft_lob.utils.checkpoint_utils import backup_experiment_config
from hft_lob.utils.experiment_manager import (
    resolve_experiment_id,
    resolve_log_dir,
    write_experiment_log,
)
from hft_lob.utils.seed import set_seed

VALID_STAGES: tuple[str, ...] = (
    "prepare-data", "walk-forward", "evaluate", "predict-offline",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的参数对象（--config / --experiment-id / --resume-ckpt /
        --stages / --seed）。
    """
    parser = argparse.ArgumentParser(description="LOB walk-forward experiment pipeline")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--experiment-id")
    parser.add_argument("--resume-ckpt")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=VALID_STAGES,
        default=["prepare-data"],
    )
    parser.add_argument("--seed", type=int)
    return parser.parse_args(argv)


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
    7. run_walk_forward 使用严格因果滚动标准化并运行模型/baseline
    8. 所有 candidate 输出 PredictionArtifact
    9. build_evaluation_report 输出 EvaluationReport/FoldResult
    """
    args = parse_args()
    provisional_id = args.experiment_id or "pending"
    config = load_config(args.config, experiment_id=provisional_id)
    experiment_id = resolve_experiment_id(
        model_name=config.model.name,
        ticker=config.ticker,
        override_id=args.experiment_id,
        resume_ckpt=args.resume_ckpt,
    )
    config = replace(
        config,
        experiment_id=experiment_id,
        seed=config.seed if args.seed is None else args.seed,
    )
    set_seed(config.seed)
    log_dir = Path(resolve_log_dir(experiment_id))
    log_dir.mkdir(parents=True, exist_ok=True)
    backup_experiment_config(str(log_dir), asdict(config))

    prepared = None
    for stage in args.stages:
        if stage == "prepare-data":
            prepared = prepare_dataset(config)
            fold_count = len(prepared.walk_forward_plan.folds)
            write_experiment_log(
                experiment_id,
                "dataset_info",
                {
                    "dataset_version": prepared.dataset_version,
                    "feature_version": prepared.feature_version,
                    "label_version": prepared.label_version,
                    "feature_count": len(prepared.feature_columns),
                    "manifest_path": prepared.manifest_path,
                    "quality_report_path": prepared.quality_report_path,
                    "fold_count": fold_count,
                },
            )
            print(f"dataset_version={prepared.dataset_version}")
            print(f"manifest_path={prepared.manifest_path}")
            print(f"quality_report_path={prepared.quality_report_path}")
            print(f"fold_count={fold_count}")
            continue
        raise NotImplementedError(
            f"stage {stage!r} is not executable yet; complete the walk-forward executor first"
        )

if __name__ == "__main__":
    main()

