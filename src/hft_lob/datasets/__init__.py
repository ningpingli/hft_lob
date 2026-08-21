"""LOB 运行时 Dataset 与不可变预构建数据包契约。"""

from hft_lob.datasets.builder import build_dataset_package
from hft_lob.datasets.lob_dataset import LOBBatch, LOBWindowDataset, SampleMeta
from hft_lob.datasets.package import DatasetPackageMetadata, compute_dataset_id
from hft_lob.datasets.prebuilt_dataset import PrebuiltLOBDataset
from hft_lob.datasets.validation import validate_dataset_package

__all__ = [
    "DatasetPackageMetadata",
    "LOBBatch",
    "LOBWindowDataset",
    "PrebuiltLOBDataset",
    "SampleMeta",
    "build_dataset_package",
    "compute_dataset_id",
    "validate_dataset_package",
]
