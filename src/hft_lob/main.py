"""hft_lob 流水线 CLI 入口（需求文档 §40/§41 MVP 阶段编排）。
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from hft_lob.configs import load_config
from hft_lob.systems.executor import DefaultWalkForwardExecutor
from hft_lob.systems.walk_forward import run_walk_forward
from hft_lob.utils.checkpoint_utils import backup_experiment_config
from hft_lob.utils.experiment_manager import (
    resolve_experiment_id,
    resolve_log_dir,
    write_experiment_log,
)

VALID_STAGES: tuple[str, ...] = ("walk-forward",)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的参数对象（--config / --experiment-id / --resume-ckpt /
        --stages / --seed）。
    """
    parser = argparse.ArgumentParser(description="LOB walk-forward experiment pipeline")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--experiment-id")
    parser.add_argument("--resume-ckpt")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=VALID_STAGES,
        default=["walk-forward"],
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--gpu-id",
        type=int,
        help="绑定的物理 GPU 编号；设置后进程内部使用 cuda:0",
    )
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
    if args.gpu_id is not None:
        if args.gpu_id < 0:
            raise ValueError("gpu_id must be >= 0")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # 必须在 CUDA_VISIBLE_DEVICES 设置之后导入 torch 相关 seed 模块。
    from hft_lob.utils.seed import set_seed

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

    for stage in args.stages:
        if stage == "walk-forward":
            report = run_walk_forward(
                args.dataset_dir,
                config,
                executor=DefaultWalkForwardExecutor(str(log_dir / "walk_forward")),
            )
            write_experiment_log(
                experiment_id,
                "walk_forward",
                {
                    "dataset_version": report.dataset_version,
                    "result_count": len(report.fold_results),
                    "summary": report.summary,
                },
            )
            print(f"walk_forward_results={len(report.fold_results)}")
            continue
        raise NotImplementedError(
            f"stage {stage!r} is not executable yet"
        )

if __name__ == "__main__":
    main()

