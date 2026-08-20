"""systems 包：训练算法层（LightningModule / 损失 / 指标 / artifact）。"""

from hft_lob.systems.artifact import PredictionArtifact, git_commit, save_prediction_artifact
from hft_lob.systems.lob_data_module import LOBDataModule, StageFiles
from hft_lob.systems.lob_module import LOBLightningModule
from hft_lob.systems.losses import LOSS_NAMES, build_loss
from hft_lob.systems.metrics import (
    ConfidenceInterval,
    DailyMetricRecord,
    EvaluationReport,
    METRIC_NAMES,
    PredictionBinRecord,
    block_bootstrap_confidence_interval,
    build_evaluation_report,
    daily_metric_records,
    direction_accuracy,
    directional_precision_recall,
    evaluate,
    evaluate_by_day,
    icir,
    mae,
    positive_ic_day_ratio,
    prediction_quantile_bins,
    rank_ic,
    rmse,
    ts_ic,
)
from hft_lob.systems.walk_forward import FoldResult, WalkForwardReport, run_walk_forward

__all__ = [
    "ConfidenceInterval",
    "DailyMetricRecord",
    "EvaluationReport",
    "LOSS_NAMES",
    "LOBLightningModule",
    "METRIC_NAMES",
    "PredictionBinRecord",
    "PredictionArtifact",
    "FoldResult",
    "WalkForwardReport",
    "block_bootstrap_confidence_interval",
    "build_evaluation_report",
    "build_loss",
    "direction_accuracy",
    "directional_precision_recall",
    "daily_metric_records",
    "evaluate",
    "evaluate_by_day",
    "git_commit",
    "icir",
    "mae",
    "positive_ic_day_ratio",
    "prediction_quantile_bins",
    "rank_ic",
    "rmse",
    "run_walk_forward",
    "save_prediction_artifact",
    "ts_ic",
    "LOBDataModule",
    "StageFiles",
]
