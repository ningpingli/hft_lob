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
    "mae",
    "rmse",
    "ts_ic",
    "rank_ic",
    "direction_accuracy",
    "up_precision",
    "up_recall",
    "down_precision",
    "down_recall",
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
class DailyICRecord:
    """单个交易日的 TS-IC。"""

    trade_date: str
    sample_count: int
    ic: float


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
    daily_ic: tuple[DailyICRecord, ...]
    mean_daily_ic: float
    daily_summary: dict[str, float]
    confidence_intervals: dict[str, ConfidenceInterval]
    prediction_bins: tuple[PredictionBinRecord, ...]


def mae(preds: np.ndarray, targets: np.ndarray) -> float:
    """平均绝对误差。"""
    prediction, target = _valid_pairs(preds, targets)
    return float(np.mean(np.abs(prediction - target))) if prediction.size else float("nan")


def rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    """均方根误差。"""
    prediction, target = _valid_pairs(preds, targets)
    return (
        float(np.sqrt(np.mean(np.square(prediction - target)))) if prediction.size else float("nan")
    )


def ts_ic(preds: np.ndarray, targets: np.ndarray) -> float:
    """TS-IC：预测与已实现收益的 Pearson 相关（§21；退化输入 → NaN）。"""
    prediction, target = _valid_pairs(preds, targets)
    if prediction.size < 2 or _is_constant(prediction) or _is_constant(target):
        return float("nan")
    return float(np.corrcoef(prediction, target)[0, 1])


def rank_ic(preds: np.ndarray, targets: np.ndarray) -> float:
    """RankIC：Spearman 秩相关（§21；退化输入 → NaN）。"""
    prediction, target = _valid_pairs(preds, targets)
    if prediction.size < 2 or _is_constant(prediction) or _is_constant(target):
        return float("nan")
    return ts_ic(_average_ranks(prediction), _average_ranks(target))


def direction_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    """方向准确率：sign(pred) == sign(target) 占比（排除目标为 0 的样本）。"""
    prediction, target = _valid_pairs(preds, targets)
    nonzero_target = target != 0
    if not np.any(nonzero_target):
        return float("nan")
    return float(np.mean(np.sign(prediction[nonzero_target]) == np.sign(target[nonzero_target])))


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
    prediction, target = _valid_pairs(preds, targets)
    if direction == "up":
        predicted_positive = prediction > 0
        actual_positive = target > 0
    elif direction == "down":
        predicted_positive = prediction < 0
        actual_positive = target < 0
    else:
        raise ValueError("direction must be 'up' or 'down'")

    true_positive = np.count_nonzero(predicted_positive & actual_positive)
    predicted_count = np.count_nonzero(predicted_positive)
    actual_count = np.count_nonzero(actual_positive)
    precision = float(true_positive / predicted_count) if predicted_count else float("nan")
    recall = float(true_positive / actual_count) if actual_count else float("nan")
    return precision, recall


def mean_daily_ic(daily_ics: np.ndarray) -> float:
    """Mean Daily IC：有限交易日 TS-IC 的算术平均。"""
    values = _finite_vector(daily_ics)
    return float(np.mean(values)) if values.size else float("nan")

def icir(daily_ics: np.ndarray) -> float:
    """ICIR = mean(daily_IC) / std(daily_IC)（§21 稳定性）。"""
    values = _finite_vector(daily_ics)
    if values.size < 2:
        return float("nan")
    standard_deviation = float(np.std(values))
    if standard_deviation == 0:
        return float("nan")
    return float(np.mean(values) / standard_deviation)


def positive_ic_day_ratio(daily_ics: np.ndarray) -> float:
    """Positive IC Day Ratio：daily IC > 0 的天占比。"""
    values = _finite_vector(daily_ics)
    return float(np.mean(values > 0)) if values.size else float("nan")


def evaluate(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """按 METRIC_NAMES 计算全部指标（不含日级稳定性）。

    Returns:
        ``METRIC_NAMES`` 中的全部基础与方向分类指标。
    """
    up_precision, up_recall = directional_precision_recall(preds, targets, direction="up")
    down_precision, down_recall = directional_precision_recall(preds, targets, direction="down")
    return {
        "mae": mae(preds, targets),
        "rmse": rmse(preds, targets),
        "ts_ic": ts_ic(preds, targets),
        "rank_ic": rank_ic(preds, targets),
        "direction_accuracy": direction_accuracy(preds, targets),
        "up_precision": up_precision,
        "up_recall": up_recall,
        "down_precision": down_precision,
        "down_recall": down_recall,
    }


def daily_ic_records(
    preds: np.ndarray,
    targets: np.ndarray,
    trade_dates: np.ndarray,
) -> tuple[DailyICRecord, ...]:
    """按交易日计算 TS-IC，保持输入中交易日的首次出现顺序。"""
    prediction, target = _paired_vectors(preds, targets)
    dates = _metadata_vector(trade_dates, field="trade_dates", size=prediction.size)
    records: list[DailyICRecord] = []
    for trade_date in dict.fromkeys(dates.tolist()):
        mask = dates == trade_date
        records.append(
            DailyICRecord(
                trade_date=str(trade_date),
                sample_count=int(np.count_nonzero(mask)),
                ic=ts_ic(prediction[mask], target[mask]),
            )
        )
    return tuple(records)


def prediction_quantile_bins(
    preds: np.ndarray,
    targets: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[PredictionBinRecord, ...]:
    """计算时序分组收益：全体时序样本按预测值排序后等量分 bin。

    这不是截面分组收益。所有有效时序样本先按预测值稳定排序，再切分为 ``n_bins`` 个
    等量 bin；每个 bin 的真实收益均值用于绘制时序分组收益曲线。

    Raises:
        ValueError: n_bins < 2、输入长度不同或有效样本不足。
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    prediction, target = _valid_pairs(preds, targets)
    if prediction.size < n_bins:
        raise ValueError("valid sample count must be >= n_bins")

    order = np.argsort(prediction, kind="stable")
    groups = np.array_split(order, n_bins)
    return tuple(
        PredictionBinRecord(
            bin_index=index,
            lower_quantile=(index - 1) / n_bins,
            upper_quantile=index / n_bins,
            sample_count=int(group.size),
            mean_prediction=float(np.mean(prediction[group])),
            mean_realized_return=float(np.mean(target[group])),
        )
        for index, group in enumerate(groups, start=1)
    )


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
    prediction, target = _paired_vectors(preds, targets)
    dates = _metadata_vector(trade_dates, field="trade_dates", size=prediction.size)
    sessions = _metadata_vector(session_ids, field="session_ids", size=prediction.size)
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be > 0")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    groups = _session_group_indices(dates, sessions)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=np.float64)
    for sample_index in range(n_resamples):
        sampled_indices = np.concatenate(
            [_resample_group(group, block_size=block_size, rng=rng) for group in groups]
        )
        estimates[sample_index] = metric(prediction[sampled_indices], target[sampled_indices])

    finite_estimates = estimates[np.isfinite(estimates)]
    estimate = float(metric(prediction, target))
    if finite_estimates.size == 0:
        lower = upper = float("nan")
    else:
        alpha = (1 - confidence_level) / 2
        lower, upper = (float(value) for value in np.quantile(finite_estimates, [alpha, 1 - alpha]))
    return ConfidenceInterval(
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
    )


def build_evaluation_report(
    artifact: PredictionArtifact,
    config: EvaluationConfig,
    *,
    seed: int,
) -> EvaluationReport:
    """从统一 PredictionArtifact 构建唯一对外评估报告。"""
    unknown_metrics = sorted(set(config.metrics).difference(METRIC_NAMES))
    if unknown_metrics:
        raise ValueError(f"unsupported evaluation metrics: {unknown_metrics}")
    if len(set(config.metrics)) != len(config.metrics):
        raise ValueError("evaluation metrics must be unique")

    predictions = artifact.predictions
    targets = artifact.targets
    trade_dates = np.asarray([meta.trade_date for meta in artifact.metadata])
    session_ids = np.asarray([meta.session_id for meta in artifact.metadata])
    overall_all = evaluate(predictions, targets)
    overall = {name: overall_all[name] for name in config.metrics}

    daily_ic = daily_ic_records(predictions, targets, trade_dates)
    daily_ic_values = np.asarray([record.ic for record in daily_ic], dtype=np.float64)
    report_mean_daily_ic = mean_daily_ic(daily_ic_values)
    daily: tuple[DailyMetricRecord, ...] = ()
    daily_summary: dict[str, float] = {"mean_daily_ic": report_mean_daily_ic}
    if config.report_daily:
        records: list[DailyMetricRecord] = []
        for daily_record in daily_ic:
            mask = trade_dates == daily_record.trade_date
            metrics = evaluate(predictions[mask], targets[mask])
            records.append(
                DailyMetricRecord(
                    trade_date=daily_record.trade_date,
                    sample_count=daily_record.sample_count,
                    metrics={name: metrics[name] for name in config.metrics},
                )
            )
        daily = tuple(records)
        for name in config.metrics:
            values = _finite_vector(np.asarray([record.metrics[name] for record in daily]))
            daily_summary[f"{name}_mean"] = float(np.mean(values)) if values.size else float("nan")
            daily_summary[f"{name}_std"] = float(np.std(values)) if values.size else float("nan")
        daily_summary["icir"] = icir(daily_ic_values)
        daily_summary["positive_ic_day_ratio"] = positive_ic_day_ratio(daily_ic_values)

    metric_functions: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "mae": mae,
        "rmse": rmse,
        "ts_ic": ts_ic,
        "rank_ic": rank_ic,
        "direction_accuracy": direction_accuracy,
        "up_precision": lambda p, t: directional_precision_recall(p, t, direction="up")[0],
        "up_recall": lambda p, t: directional_precision_recall(p, t, direction="up")[1],
        "down_precision": lambda p, t: directional_precision_recall(p, t, direction="down")[0],
        "down_recall": lambda p, t: directional_precision_recall(p, t, direction="down")[1],
    }
    confidence_intervals = {
        name: block_bootstrap_confidence_interval(
            predictions,
            targets,
            trade_dates,
            session_ids,
            metric=metric_functions[name],
            confidence_level=config.confidence_level,
            n_resamples=config.bootstrap_samples,
            block_size=config.bootstrap_block_size,
            seed=seed,
        )
        for name in config.metrics
    }
    bins = prediction_quantile_bins(predictions, targets, n_bins=config.prediction_bins)
    return EvaluationReport(
        sample_count=int(predictions.size),
        overall=overall,
        daily=daily,
        daily_ic=daily_ic,
        mean_daily_ic=report_mean_daily_ic,
        daily_summary=daily_summary,
        confidence_intervals=confidence_intervals,
        prediction_bins=bins,
    )


def _paired_vectors(preds: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction = _as_vector(preds, field="preds")
    target = _as_vector(targets, field="targets")
    if prediction.size != target.size:
        raise ValueError("preds and targets must have the same length")
    return prediction, target


def _valid_pairs(preds: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction, target = _paired_vectors(preds, targets)
    valid = np.isfinite(prediction) & np.isfinite(target)
    return prediction[valid], target[valid]


def _as_vector(values: np.ndarray, *, field: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"{field} must have shape [N] or [N, 1], got {array.shape}")
    return array


def _finite_vector(values: np.ndarray) -> np.ndarray:
    vector = _as_vector(values, field="values")
    return vector[np.isfinite(vector)]


def _is_constant(values: np.ndarray) -> bool:
    return bool(np.all(values == values[0]))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def _metadata_vector(values: np.ndarray, *, field: str, size: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size != size:
        raise ValueError(f"{field} must be one-dimensional with length {size}")
    if any(value is None or str(value) == "" for value in array):
        raise ValueError(f"{field} must not contain empty values")
    return array


def _session_group_indices(trade_dates: np.ndarray, session_ids: np.ndarray) -> list[np.ndarray]:
    grouped: dict[tuple[object, object], list[int]] = {}
    for index, key in enumerate(zip(trade_dates, session_ids, strict=True)):
        grouped.setdefault(key, []).append(index)
    return [np.asarray(indices, dtype=np.int64) for indices in grouped.values()]


def _resample_group(
    indices: np.ndarray, *, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    effective_block_size = min(block_size, indices.size)
    max_start = indices.size - effective_block_size
    block_count = int(np.ceil(indices.size / effective_block_size))
    starts = rng.integers(0, max_start + 1, size=block_count)
    sampled = np.concatenate([indices[start : start + effective_block_size] for start in starts])
    return sampled[: indices.size]
