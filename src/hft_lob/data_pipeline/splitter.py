"""时间切分与 walk-forward fold 计划。"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from hft_lob.configs.experiment import DataBuildConfig, SplitConfig, WalkForwardConfig

FOLD_INDEX_COLUMNS = (
    "global_anchor_index",
    "session_start_index",
    "anchor_index",
    "trade_date",
    "session_id",
    "anchor_timestamp",
)

FOLD_INDEX_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "global_anchor_index": pl.Int64,
    "session_start_index": pl.Int64,
    "anchor_index": pl.Int64,
    "trade_date": pl.String,
    "session_id": pl.String,
    "anchor_timestamp": pl.Datetime("us"),
}


@dataclass(frozen=True)
class ChronologicalSplit:
    """chronological 切分结果（§15：max(train) < min(val) < min(test)）。"""

    train_dates: list[str]
    validation_dates: list[str]
    test_dates: list[str]

    def dates_for(self, stage: str) -> list[str]:
        """按阶段名（training / validation / test）取日期列表。"""
        stages = {
            "training": self.train_dates,
            "validation": self.validation_dates,
            "test": self.test_dates,
        }
        try:
            return list(stages[stage])
        except KeyError as exc:
            raise ValueError(f"unsupported split stage: {stage!r}") from exc

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
    _validate_dates(dates)
    selected = list(dates)
    explicit_ranges = (config.train_dates, config.validation_dates, config.test_dates)
    has_explicit = [value is not None for value in explicit_ranges]
    if any(has_explicit) and not all(has_explicit):
        raise ValueError(
            "train_dates, validation_dates and test_dates must be configured together"
        )

    if all(has_explicit):
        train_range = _validate_range(config.train_dates, field="split.train_dates")
        validation_range = _validate_range(
            config.validation_dates, field="split.validation_dates"
        )
        test_range = _validate_range(config.test_dates, field="split.test_dates")
        train = _dates_in_range(selected, train_range)
        validation = _dates_in_range(selected, validation_range)
        test = _dates_in_range(selected, test_range)
        assigned = [*train, *validation, *test]
        if sorted(assigned) != selected or len(set(assigned)) != len(selected):
            raise ValueError(
                "explicit split ranges must cover every selected date exactly once"
            )
    else:
        train_end = int(len(selected) * config.train_ratio)
        validation_end = train_end + int(len(selected) * config.validation_ratio)
        train = selected[:train_end]
        validation = selected[train_end:validation_end]
        test = selected[validation_end:]

    _validate_partition(train, validation, test)
    return ChronologicalSplit(train, validation, test)

def walk_forward_folds(dates: list[str], config: WalkForwardConfig) -> list[Fold]:
    """按交易日数量生成固定训练窗口的 walk-forward 折（§16）。

    每个折依次包含固定长度 train/validation/test，下一折整体向前移动
    ``step_days``。训练窗口不会扩张，只保留最近 ``train_window_days``。

    Args:
        dates: 升序 %Y-%m-%d 列表。
        config: 切分配置。

    Returns:
        折列表（index 从 1 开始）。
    """
    _validate_dates(dates)
    selected = list(dates)
    required_days = (
        config.train_window_days
        + config.validation_window_days
        + config.test_window_days
    )
    if len(selected) < required_days:
        raise ValueError(
            f"walk-forward requires at least {required_days} trade dates, got {len(selected)}"
        )

    folds: list[Fold] = []
    fold_start = 0
    while fold_start + required_days <= len(selected):
        train_end = fold_start + config.train_window_days
        validation_end = train_end + config.validation_window_days
        test_end = validation_end + config.test_window_days
        train = selected[fold_start:train_end]
        validation = selected[train_end:validation_end]
        test = selected[validation_end:test_end]
        _validate_partition(train, validation, test)
        folds.append(Fold(len(folds) + 1, train, validation, test))
        fold_start += config.step_days
    return folds

def _validate_dates(dates: list[str]) -> list[date]:
    if not dates:
        raise ValueError("dates must not be empty")
    parsed = [_parse_iso_date(value, field="dates") for value in dates]
    if parsed != sorted(parsed):
        raise ValueError("dates must be sorted ascending")
    if len(set(parsed)) != len(parsed):
        raise ValueError("dates must contain unique trade dates")
    return parsed

def _parse_iso_date(value: str, *, field: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must use YYYY-MM-DD: {value!r}") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{field} must use YYYY-MM-DD: {value!r}")
    return parsed

def _validate_range(
    value: tuple[str, str] | None,
    *,
    field: str,
) -> tuple[date, date]:
    if value is None:
        raise ValueError(f"{field} is required")
    start = _parse_iso_date(value[0], field=field)
    end = _parse_iso_date(value[1], field=field)
    if start > end:
        raise ValueError(f"{field} start must be <= end")
    return start, end

def _dates_in_range(values: list[str], bounds: tuple[date, date]) -> list[str]:
    start, end = bounds
    return [value for value in values if start <= date.fromisoformat(value) <= end]

def _validate_partition(
    train: list[str],
    validation: list[str],
    test: list[str],
) -> None:
    if not train or not validation or not test:
        raise ValueError("training, validation and test must each contain at least one date")
    if not max(train) < min(validation) < min(test):
        raise ValueError("split must satisfy max(train) < min(validation) < min(test)")

def build_fold_plan(
    trade_dates: tuple[str, ...],
    config: DataBuildConfig,
    dataset_version: str,
) -> WalkForwardPlan:
    """从完整交易日集合生成固定 fold 计划。"""
    if not trade_dates:
        raise ValueError("trade_dates must not be empty")
    if config.walk_forward.enabled:
        folds = tuple(walk_forward_folds(list(trade_dates), config.walk_forward))
    else:
        split = chronological_split(list(trade_dates), config.split)
        folds = (Fold(1, split.train_dates, split.validation_dates, split.test_dates),)
    return WalkForwardPlan(dataset_version=dataset_version, folds=folds)

def write_fold_indexes(
    anchors_path: Path,
    folds_root: Path,
    plan: WalkForwardPlan,
) -> None:
    """把日期计划物化为只含 sample index 的 fold parquet。"""
    anchors = pl.scan_parquet(anchors_path)
    for fold in plan.folds:
        for split, dates in (
            ("train", fold.train_dates),
            ("validation", fold.validation_dates),
            ("test", fold.test_dates),
        ):
            path = folds_root / f"fold_{fold.index:03d}" / f"{split}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            anchors.filter(pl.col("trade_date").is_in(dates)).select(
                FOLD_INDEX_COLUMNS
            ).sink_parquet(path)
    del anchors
    gc.collect()
