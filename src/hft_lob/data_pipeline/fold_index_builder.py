"""数据包 fold 计划与索引物化。"""

from __future__ import annotations

import gc
from pathlib import Path

import polars as pl

from hft_lob.configs.experiment import DataBuildConfig
from hft_lob.data_pipeline.dataset_validator import FOLD_INDEX_COLUMNS
from hft_lob.data_pipeline.split import (
    Fold,
    WalkForwardPlan,
    chronological_split,
    walk_forward_folds,
)


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
    gc.collect()  # Windows 上及时释放 parquet 扫描句柄，允许随后原子重命名目录。
