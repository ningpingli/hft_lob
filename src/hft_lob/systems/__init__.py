"""systems 包：训练算法层（LightningModule / 损失 / 指标 / artifact）。"""

from hft_lob.systems.artifact import PredictionArtifact, save_prediction_artifact
from hft_lob.systems.executor import DefaultWalkForwardExecutor
from hft_lob.systems.lob_data_module import LOBDataModule
from hft_lob.systems.lob_module import LOBLightningModule
from hft_lob.systems.losses import build_loss
from hft_lob.systems.metrics import EvaluationReport, build_evaluation_report
from hft_lob.systems.walk_forward import (
    CandidateFoldRun,
    FoldResult,
    WalkForwardExecutor,
    WalkForwardReport,
    run_walk_forward,
)

__all__ = [
    "CandidateFoldRun",
    "DefaultWalkForwardExecutor",
    "EvaluationReport",
    "FoldResult",
    "LOBDataModule",
    "LOBLightningModule",
    "PredictionArtifact",
    "WalkForwardReport",
    "WalkForwardExecutor",
    "build_evaluation_report",
    "build_loss",
    "run_walk_forward",
    "save_prediction_artifact",
]
