"""hft_lob 训练 CLI 适配层。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from hft_lob.application import TrainingRequest, run_training_application


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOB walk-forward experiment pipeline")
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--experiment-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu-id", type=int, help="绑定的物理 GPU 编号")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """解析 CLI 参数并委托给训练应用服务。"""
    args = parse_args(argv)
    result = run_training_application(
        TrainingRequest(
            config_path=args.config,
            dataset_dir=args.dataset_dir,
            experiment_id=args.experiment_id,
            seed=args.seed,
            gpu_id=args.gpu_id,
        )
    )
    print(f"walk_forward_results={result.fold_result_count}")


if __name__ == "__main__":
    main()
