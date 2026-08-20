"""评估指标（需求文档 §21）：TS-IC / RankIC / MAE / RMSE / Direction + 稳定性汇总。

对单只股票时序预测，Pearson 相关明确命名为 TS-IC（时间序列 IC），而非横截面 IC。
所有指标为 numpy 纯函数，便于单测与线上评估复用。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from hft_lob.configs.experiment import EvaluationConfig
from hft_lob.systems.artifact import PredictionArtifact

#: 与配置 EvaluationConfig.metrics 对齐的指标名。
METRIC_NAMES: tuple[str, ...] = (
    "mae", "rmse", "ts_ic", "rank_ic", "direction_accuracy",
    "up_precision", "up_recall", "down_precision", "down_recall",
)


@dataclass(frozen=True)
class ConfidenceInterval:
    """单个指标的置信区间。"""

    estimate: float
    lower: float
    upper: float
    confidence_level: float
    method: str = "moving_block_bootstrap"


@dataclass(frozen=True)
class DailyMetricRecord:
    """单个交易日的完整指标，保留样本数用于审计。"""

    trade_date: str
    sample_count: int
    metrics: dict[str, float]


@dataclass(frozen=True)
class PredictionBinRecord:
    """一个预测分位桶的边界、样本量及平均预测/实现收益。"""

    bin_index: int
    lower_quantile: float
    upper_quantile: float
    sample_count: int
    mean_prediction: float
    mean_realized_return: float


@dataclass(frozen=True)
class EvaluationReport:
    """需求 §14/§21/§23 的结构化评估结果。"""

    sample_count: int
    overall: dict[str, float]
    daily: tuple[DailyMetricRecord, ...]
    daily_summary: dict[str, float]
    confidence_intervals: dict[str, ConfidenceInterval]
    prediction_bins: tuple[PredictionBinRecord, ...]


def mae(preds: np.ndarray, targets: np.ndarray) -> float:
    """平均绝对误差。"""
    raise NotImplementedError("mae not implemented")


def rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    """均方根误差。"""
    raise NotImplementedError("rmse not implemented")


def ts_ic(preds: np.ndarray, targets: np.ndarray) -> float:
    """TS-IC：预测与已实现收益的 Pearson 相关（§21；退化输入 → NaN）。"""
    raise NotImplementedError("ts_ic not implemented")


def rank_ic(preds: np.ndarray, targets: np.ndarray) -> float:
    """RankIC：Spearman 秩相关（§21；退化输入 → NaN）。"""
    raise NotImplementedError("rank_ic not implemented")


def direction_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    """方向准确率：sign(pred) == sign(target) 占比（排除目标为 0 的样本）。"""
    raise NotImplementedError("direction_accuracy not implemented")


def directional_precision_recall(
    preds: np.ndarray,
    targets: np.ndarray,
    *,
    direction: str,
) -> tuple[float, float]:
    """计算上涨或下跌方向的 Precision / Recall。

    Args:
        preds: 预测收益。
        targets: 实现收益。
        direction: ``"up"`` 或 ``"down"``；零收益不属于任一方向。

    Returns:
        ``(precision, recall)``；无正预测或无真实正例时相应值为 NaN。

    Raises:
        ValueError: direction 不是 up/down。
    """
    raise NotImplementedError("directional_precision_recall not implemented")


def icir(daily_ics: np.ndarray) -> float:
    """ICIR = mean(daily_IC) / std(daily_IC)（§21 稳定性）。"""
    raise NotImplementedError("icir not implemented")


def positive_ic_day_ratio(daily_ics: np.ndarray) -> float:
    """Positive IC Day Ratio：daily IC > 0 的天占比。"""
    raise NotImplementedError("positive_ic_day_ratio not implemented")


def evaluate(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """按 METRIC_NAMES 计算全部指标（不含日级稳定性）。

    Returns:
        ``METRIC_NAMES`` 中的全部基础与方向分类指标。
    """
    raise NotImplementedError("evaluate not implemented")


def prediction_quantile_bins(
    preds: np.ndarray,
    targets: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[PredictionBinRecord, ...]:
    """按预测值分位数分桶并统计每桶实现收益（§23）。

    重复分位点必须使用确定性策略处理；返回记录按预测值从低到高排列。

    Raises:
        ValueError: n_bins < 2、输入长度不同或有效样本不足。
    """
    raise NotImplementedError("prediction_quantile_bins not implemented")


def block_bootstrap_confidence_interval(
    preds: np.ndarray,
    targets: np.ndarray,
    trade_dates: np.ndarray,
    session_ids: np.ndarray,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float],
    confidence_level: float = 0.95,
    n_resamples: int = 1_000,
    block_size: int = 20,
    seed: int = 42,
) -> ConfidenceInterval:
    """使用 moving block bootstrap 估计指标置信区间（§14）。

    连续块只在同一 trade_date/session_id 内抽样，保留局部序列相关性且禁止
    跨日、跨午休或退化为 IID 行抽样。
    """
    raise NotImplementedError("block_bootstrap_confidence_interval not implemented")


def build_evaluation_report(
    artifact: PredictionArtifact,
    config: EvaluationConfig,
    *,
    seed: int,
) -> EvaluationReport:
    """从统一 PredictionArtifact 构建唯一对外评估报告。"""
    raise NotImplementedError("build_evaluation_report not implemented")
