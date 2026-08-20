"""chronological split（需求文档 §15/§16）：按完整交易日切分；walk-forward 支持。"""

from __future__ import annotations

from dataclasses import dataclass

from hft_lob.configs.experiment import SplitConfig


@dataclass(frozen=True)
class ChronologicalSplit:
    """chronological 切分结果（§15：max(train) < min(val) < min(test)）。"""

    train_dates: list[str]
    validation_dates: list[str]
    test_dates: list[str]

    def dates_for(self, stage: str) -> list[str]:
        """按阶段名（training / validation / test）取日期列表。"""
        return {
            "training": self.train_dates,
            "validation": self.validation_dates,
            "test": self.test_dates,
        }[stage]


@dataclass(frozen=True)
class Fold:
    """walk-forward 的一个折（§16）。"""

    index: int
    train_dates: list[str]
    validation_dates: list[str]
    test_dates: list[str]


@dataclass(frozen=True)
class WalkForwardPlan:
    """绑定数据版本的完整 walk-forward 执行计划。"""

    dataset_version: str
    folds: tuple[Fold, ...]


def chronological_split(dates: list[str], config: SplitConfig) -> ChronologicalSplit:
    """按完整交易日 chronological 切分（§15）。

    优先使用显式日期范围（``train_dates / validation_dates / test_dates``，
    %Y-%m-%d，含两端），否则按 ``train_ratio / validation_ratio`` 切分
    （test 为余数）。``dates`` 为升序 %Y-%m-%d 列表。

    Args:
        dates: 全部交易日（升序）。
        config: 切分配置。

    Returns:
        三段日期切分结果。

    Raises:
        ValueError: 三段日期有重叠或未覆盖全部日期。
    """
    raise NotImplementedError("chronological_split not implemented")


def walk_forward_folds(dates: list[str], config: SplitConfig) -> list[Fold]:
    """生成 walk-forward 折（§16）：训练扩张、以自然月滚动 val/test。

    每个折：val = 上一个月，test = 当月，train = 更早的全部交易日。

    Args:
        dates: 升序 %Y-%m-%d 列表。
        config: 切分配置。

    Returns:
        折列表（index 从 1 开始）。
    """
    raise NotImplementedError("walk_forward_folds not implemented")
