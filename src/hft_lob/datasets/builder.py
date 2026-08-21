"""阶段一入口：编排 raw source、样本编译和不可变数据包发布。"""

from __future__ import annotations

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


def build_dataset_package(config: DataBuildConfig, output_root: str | Path) -> Path:
    """从 immutable raw 构建并原子发布一个内容寻址数据包。"""
    sources = discover_sources(config)
    compiler = SampleCompiler(config)
    with DatasetPackageWriter(output_root, len(compiler.feature_columns)) as writer:
        for day in compiler.compile(sources.files):
            writer.append(day)
        plan = build_fold_plan(tuple(writer.trade_dates), config, sources.version)
        metadata = _metadata(config, sources, plan, compiler.feature_columns)
        return writer.finalize_and_publish(metadata, plan)


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
        target_column=config.target_column,
        feature_dtype="float32",
        target_dtype="float32",
        snapshot_interval_seconds=config.data.snapshot_interval_seconds,
        history_snapshots=config.window.history_snapshots,
        normalization_mode=config.normalization.mode,
        normalization_window=config.normalization.normalize_window,
        source_hash=source_hash,
        processing_config_hash=sources.processing_hash,
        fold_plan_hash=fold_plan_hash,
    )
