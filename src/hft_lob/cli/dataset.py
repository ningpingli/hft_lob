"""数据工程应用用例。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hft_lob.configs import load_data_config
from hft_lob.data_pipeline.pipeline import build_dataset_package
from hft_lob.data_pipeline.writer import DatasetPackageMetadata, validate_dataset_package


@dataclass(frozen=True)
class DatasetBuildRequest:
    """构建不可变训练数据包所需的输入。"""

    config_path: str
    output_root: str


def build_dataset(request: DatasetBuildRequest) -> Path:
    """从配置构建并发布一个不可变训练数据包。"""
    return build_dataset_package(load_data_config(request.config_path), request.output_root)


def verify_dataset(dataset_dir: str | Path) -> DatasetPackageMetadata:
    """校验已发布的数据包。"""
    return validate_dataset_package(dataset_dir)


def inspect_dataset(dataset_dir: str | Path) -> DatasetPackageMetadata:
    """读取元数据，同时保证数据包有效。"""
    return validate_dataset_package(dataset_dir)
