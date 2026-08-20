"""预处理总流程：raw parquet → ETL + 切分 → 落盘 CSV（步骤 8，串联 1-7 +9）。"""

from __future__ import annotations

from dataclasses import dataclass

from hft_lob.configs.experiment import ExperimentConfig
from hft_lob.preprocessing.split import SplitResult


@dataclass(frozen=True)
class PipelineResult:
    """预处理总流程的结果汇总。"""

    processed_tickers: list[str]
    skipped_tickers: list[tuple[str, str]]  # (ticker, reason)
    split: SplitResult
    error_log_path: str


def run_pipeline(
    config: ExperimentConfig,
    *,
    input_dir: str,
    output_dir: str,
    logs_dir: str,
) -> PipelineResult:
    """执行完整预处理总流程（步骤 8：串联清洗/转换/标签/落盘 + 切分 1-7 +9）。

    逐 ticker 调用 DataCleaner / FeatureTransformer / LabelTransformer 完成
    ETL 并落盘 CSV，再以 split_into_stages 切分三段，返回结果汇总。

    Args:
        config: 实验配置根（含 general/data/loader 段超参数）。
        input_dir: 原始 parquet 数据根目录。
        output_dir: 处理后 CSV 落盘目录。
        logs_dir: 处理日志（跳过原因/错误）目录。

    Returns:
        处理结果汇总（成功 ticker、跳过 ticker 及原因、切分结果、错误日志路径）。
    """
    raise NotImplementedError("run_pipeline not implemented")
