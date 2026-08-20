"""数据 manifest（需求文档 §30/§31）：数据集版本追踪；split 以 manifest 表达。"""

from __future__ import annotations

import polars as pl

from hft_lob.configs.experiment import FeatureConfig, TargetConfig

#: manifest 列（固定顺序；§31 Data Manifest）。
_MANIFEST_COLUMNS: tuple[str, ...] = (
    "ticker", "trade_date", "session_id", "source_file", "processed_file",
    "row_count", "valid_row_count", "data_start", "data_end",
    "feature_version", "label_version", "quality_status",
)


def dataset_version(ticker: str, dates: list[str]) -> str:
    """数据集版本标识：``<ticker>_<首日>_<末日>``。"""
    if not dates:
        return f"{ticker}_empty"
    return f"{ticker}_{min(dates)}_{max(dates)}"


def feature_version(config: FeatureConfig) -> str:
    """特征版本：``raw23`` 或 ``raw23+<派生数>``。"""
    if not config.use_derived:
        return "raw23"
    return f"raw23+{len(config.derived_features)}"


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
