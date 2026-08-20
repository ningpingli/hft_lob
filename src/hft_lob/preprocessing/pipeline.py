"""预处理总流程（需求文档 §40 流水线前段）：raw → 清洗 → 特征 → 标签 → manifest → split。"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from hft_lob.configs.experiment import ExperimentConfig
from hft_lob.preprocessing.quality import QualityReport
from hft_lob.preprocessing.split import ChronologicalSplit


@dataclass(frozen=True)
class PipelineResult:
    """预处理结果汇总。"""

    processed_files: list[str]
    quality_reports: dict[str, QualityReport]
    manifest: pl.DataFrame
    split: ChronologicalSplit
    manifest_dir: str
    dataset_version: str


def run_pipeline(config: ExperimentConfig) -> PipelineResult:
    """执行完整预处理：逐交易日 raw parquet → 清洗 + 特征 + 标签 → processed
    parquet，生成数据 manifest，并按 chronological split 落盘三段 split manifest。

    原始数据只读（immutable，§30）：split 通过 manifest 表达，不移动/删除任何
    原始文件。

    Args:
        config: 实验配置根。

    Returns:
        预处理结果汇总。

    Raises:
        FileNotFoundError: 原始目录下无 parquet 文件。
    """
    raise NotImplementedError("run_pipeline not implemented")
