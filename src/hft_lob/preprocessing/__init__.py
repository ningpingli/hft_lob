"""preprocessing 包：LOB 数据清洗、特征、标签、质量、manifest、切分与归一化。"""

from hft_lob.preprocessing.clean import CleanDayResult, DataCleaner, SessionSegment
from hft_lob.preprocessing.features import FeatureTransformer
from hft_lob.preprocessing.labels import LabelTransformer, label_column
from hft_lob.preprocessing.manifest import (
    build_manifest,
    dataset_version,
    feature_version,
    label_version,
    raw_file_hash,
    read_manifest,
    stable_config_hash,
    write_manifest,
)
from hft_lob.preprocessing.normalize import CausalRollingNormalizer, TrainOnlyNormalizer
from hft_lob.preprocessing.pipeline import PipelineResult, run_pipeline
from hft_lob.preprocessing.quality import QualityReport, run_quality_checks
from hft_lob.preprocessing.split import (
    ChronologicalSplit,
    Fold,
    chronological_split,
    walk_forward_folds,
)

__all__ = [
    "CausalRollingNormalizer",
    "CleanDayResult",
    "ChronologicalSplit",
    "DataCleaner",
    "FeatureTransformer",
    "Fold",
    "LabelTransformer",
    "PipelineResult",
    "QualityReport",
    "SessionSegment",
    "TrainOnlyNormalizer",
    "build_manifest",
    "chronological_split",
    "dataset_version",
    "feature_version",
    "label_column",
    "label_version",
    "read_manifest",
    "raw_file_hash",
    "run_pipeline",
    "run_quality_checks",
    "stable_config_hash",
    "walk_forward_folds",
    "write_manifest",
]
