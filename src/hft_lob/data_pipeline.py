"""独立数据工程命令：构建、校验和查看不可变训练数据包。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from hft_lob.configs import load_data_config
from hft_lob.datasets.builder import build_dataset_package
from hft_lob.datasets.dataset_validator import validate_dataset_package


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and validate LOB dataset packages")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--config", required=True)
    build.add_argument("--output-root", required=True)
    for name in ("verify", "inspect"):
        command = commands.add_parser(name)
        command.add_argument("--dataset-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "build":
        config = load_data_config(args.config)
        package = build_dataset_package(config, args.output_root)
        print(package)
        return
    metadata = validate_dataset_package(args.dataset_dir)
    if args.command == "verify":
        print(f"valid dataset package: {metadata.dataset_id}")
    else:
        print(json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
