"""数据 manifest（需求文档 §30/§31）：数据集版本追踪；split 以 manifest 表达。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from hft_lob.configs.experiment import FeatureConfig, TargetConfig

#: manifest 列（固定顺序；§31 Data Manifest）。
_MANIFEST_COLUMNS: tuple[str, ...] = (
    "ticker", "trade_date", "session_id", "source_file", "processed_file",
    "raw_hash", "processing_config_hash", "dataset_version",
    "row_count", "valid_row_count", "data_start", "data_end",
    "feature_version", "label_version", "quality_status",
)


def raw_file_hash(path: str, *, algorithm: str = "sha256") -> str:
    """流式计算 raw 文件内容哈希，不依赖路径、mtime 或文件名。"""
    raise NotImplementedError("raw_file_hash not implemented")


def stable_config_hash(config: Mapping[str, Any]) -> str:
    """对 key 排序后的 canonical 配置计算稳定 SHA-256 哈希。"""
    raise NotImplementedError("stable_config_hash not implemented")


def dataset_version(
    ticker: str,
    raw_hashes: Sequence[str],
    *,
    processing_config_hash: str,
) -> str:
    """生成内容寻址的数据集版本。

    版本由 ticker、排序后的 raw 内容哈希及完整处理配置哈希共同决定；字段映射
    属于处理配置，因此 raw 内容、映射或处理语义变化都会产生新版本。
    """
    raise NotImplementedError("dataset_version not implemented")


def feature_version(config: FeatureConfig) -> str:
    """由启用状态、特征名称及顺序生成稳定版本，不以特征数量代替版本。"""
    raise NotImplementedError("feature_version not implemented")


def label_version(config: TargetConfig) -> str:
    """标签版本：``<type>_<h>s_tol<tol>``。"""
    return f"{config.type}_{config.horizon_seconds}s_tol{config.tolerance_seconds}"


def build_manifest(*, ticker: str, records: list[dict[str, object]]) -> pl.DataFrame:
    """由逐日记录构建 manifest DataFrame（列序固定，空列表时列仍齐全）。

    Args:
        ticker: 股票代码。
        records: 逐日记录字典列表（键 = manifest 列）。

    Returns:
        manifest DataFrame。
    """
    raise NotImplementedError("build_manifest not implemented")


def write_manifest(manifest: pl.DataFrame, path: str) -> None:
    """落盘 manifest（parquet）。

    Args:
        manifest: manifest DataFrame。
        path: 输出路径。
    """
    raise NotImplementedError("write_manifest not implemented")


def read_manifest(path: str) -> pl.DataFrame:
    """读取 manifest（parquet）。

    Args:
        path: manifest 路径。

    Returns:
        manifest DataFrame。
    """
    raise NotImplementedError("read_manifest not implemented")
