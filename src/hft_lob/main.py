"""hft_lob 统一 CLI 适配层。"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence

from hft_lob.application import (
    DatasetBuildRequest,
    TrainingRequest,
    build_dataset,
    inspect_dataset,
    run_training_application,
    verify_dataset,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOB dataset and training pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="build or validate immutable datasets")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    build = data_commands.add_parser("build")
    build.add_argument("--config", required=True)
    build.add_argument("--output-root", required=True)
    for name in ("verify", "inspect"):
        command = data_commands.add_parser(name)
        command.add_argument("--dataset-dir", required=True)

    train = commands.add_parser("train", help="train models from an immutable dataset")
    train.add_argument("--config", default="configs/model.yaml")
    train.add_argument("--dataset-dir", required=True)
    train.add_argument("--experiment-id")
    train.add_argument("--seed", type=int)
    train.add_argument("--gpu-id", type=int, help="绑定的物理 GPU 编号")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """解析命令并委托给相应应用用例。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args(argv)
    if args.command == "data":
        _run_data_command(args)
        return

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


def _run_data_command(args: argparse.Namespace) -> None:
    if args.data_command == "build":
        package = build_dataset(DatasetBuildRequest(args.config, args.output_root))
        print(package)
        return
    if args.data_command == "verify":
        metadata = verify_dataset(args.dataset_dir)
        print(f"valid dataset package: {metadata.dataset_id}")
        return
    metadata = inspect_dataset(args.dataset_dir)
    print(json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
