"""阶段一入口：编排 raw source、样本编译和不可变数据包发布。"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path

from hft_lob.configs.experiment import DataBuildConfig
from hft_lob.datasets.dataset_validator import (
    DatasetPackageMetadata,
    compute_dataset_id,
    stable_config_hash,
)
from hft_lob.datasets.fold_index_builder import build_fold_plan
from hft_lob.datasets.package_writer import DatasetPackageWriter
from hft_lob.datasets.sample_compiler import SampleCompiler, SourceSet, discover_sources
from hft_lob.preprocessing.split import WalkForwardPlan

logger = logging.getLogger(__name__)


def build_dataset_package(config: DataBuildConfig, output_root: str | Path) -> Path:
    """从 immutable raw 构建并原子发布一个内容寻址数据包。"""
    started = time.perf_counter()
    logger.info("dataset_build.start ticker=%s output_root=%s", config.ticker, output_root)
    logger.info("dataset_build.discover_sources_start raw_dir=%s", config.data.raw_dir)
    sources = discover_sources(config)
    logger.info(
        "dataset_build.sources_ready files=%d source_version=%s",
        len(sources.files),
        sources.version,
    )
    compiler = SampleCompiler(config)
    with DatasetPackageWriter(
        output_root,
        len(compiler.feature_columns),
        target_count=config.target.target_count,
    ) as writer:
        for index, day in enumerate(compiler.compile(sources.files), start=1):
            writer.append(day)
            if index == 1 or index % 50 == 0 or index == len(sources.files):
                logger.info(
                    "dataset_build.progress files=%d/%d trade_date=%s rows=%d anchors=%d",
                    index,
                    len(sources.files),
                    day.trade_date,
                    writer.row_count,
                    writer.anchor_count,
                )
        plan = build_fold_plan(tuple(writer.trade_dates), config, sources.version)
        logger.info("dataset_build.fold_plan_ready folds=%d", len(plan.folds))
        metadata = _metadata(config, sources, plan, compiler.feature_columns)
        package = writer.finalize_and_publish(metadata, plan)
        logger.info(
            "dataset_build.complete dataset_id=%s path=%s rows=%d anchors=%d folds=%d elapsed_seconds=%.3f",
            metadata.dataset_id,
            package,
            writer.row_count,
            writer.anchor_count,
            len(plan.folds),
            time.perf_counter() - started,
        )
        return package


def _metadata(
    config: DataBuildConfig,
    sources: SourceSet,
    plan: WalkForwardPlan,
    feature_columns: tuple[str, ...],
) -> DatasetPackageMetadata:
    source_hash = stable_config_hash({"raw_hashes": sorted(sources.raw_hashes)})
    fold_plan_hash = stable_config_hash({"folds": [asdict(fold) for fold in plan.folds]})
    return DatasetPackageMetadata(
        dataset_id=compute_dataset_id(
            ticker=config.ticker,
            source_hash=source_hash,
            processing_config_hash=sources.processing_hash,
            fold_plan_hash=fold_plan_hash,
        ),
        ticker=config.ticker,
        feature_columns=feature_columns,
        target_columns=config.target_columns,
        feature_dtype="float32",
        target_dtype="float32",
        snapshot_interval_seconds=config.data.snapshot_interval_seconds,
        history_snapshots=config.window.history_snapshots,
        normalization_mode=config.normalization.mode,
        normalization_window=config.normalization.normalize_window,
        source_hash=source_hash,
        processing_config_hash=sources.processing_hash,
        fold_plan_hash=fold_plan_hash,
        labels=tuple(config.target.label),
    )
