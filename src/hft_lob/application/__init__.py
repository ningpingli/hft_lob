"""应用用例：连接外部入口与数据、训练领域模块。"""

from hft_lob.application.baseline import (
    BaselineRunRequest,
    BaselineRunResult,
    run_baseline_application,
)
from hft_lob.application.data_build import (
    DatasetBuildRequest,
    build_dataset,
    inspect_dataset,
    verify_dataset,
)
from hft_lob.application.testing import (
    StandaloneTestRequest,
    StandaloneTestResult,
    run_standalone_test,
)
from hft_lob.application.training import (
    TrainingRequest,
    TrainingResult,
    run_training_application,
)

__all__ = [
    "BaselineRunRequest",
    "BaselineRunResult",
    "DatasetBuildRequest",
    "StandaloneTestRequest",
    "StandaloneTestResult",
    "TrainingRequest",
    "TrainingResult",
    "build_dataset",
    "inspect_dataset",
    "run_baseline_application",
    "run_standalone_test",
    "run_training_application",
    "verify_dataset",
]
