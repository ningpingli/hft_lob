"""数据切分：processed CSV → train/validation/test 三段目录（步骤 9）。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitResult:
    """切分结果：三段文件路径列表与生效的 max_days。"""

    training_files: list[str]
    validation_files: list[str]
    test_files: list[str]
    max_days: int | None


def split_into_stages(
    *,
    dataset_root: str,
    training_stocks: list[str],
    target_stocks: list[str],
    training_ratio: float,
    validation_ratio: float,
    include_target_stock_in_training: bool,
    max_days: int | None = None,
) -> SplitResult:
    """按比例把处理后的每日 CSV 划分为 train/validation/test 三段（步骤 9）。

    只计算并返回三段文件路径，不移动/删除文件；目标股票按比例切三份
    （``include_target_stock_in_training`` 决定是否排除出训练段），
    其余训练股票整体归入训练段。

    Args:
        dataset_root: 处理后数据根目录。
        training_stocks: 用于训练的股票列表。
        target_stocks: 用于验证与测试的目标股票列表。
        training_ratio: 训练数据比例。
        validation_ratio: 验证数据比例。
        include_target_stock_in_training: 目标股票是否包含进训练集。
        max_days: 仅使用最近 max_days 个交易日划分；None 表示全部。

    Returns:
        三段文件路径列表的切分结果。
    """
    raise NotImplementedError("split_into_stages not implemented")
