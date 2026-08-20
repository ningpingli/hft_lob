"""评估指标（需求文档 §21）：TS-IC / RankIC / MAE / RMSE / Direction + 稳定性汇总。

对单只股票时序预测，Pearson 相关明确命名为 TS-IC（时间序列 IC），而非横截面 IC。
所有指标为 numpy 纯函数，便于单测与线上评估复用。
"""

from __future__ import annotations

import numpy as np

#: 与配置 EvaluationConfig.metrics 对齐的指标名。
METRIC_NAMES: tuple[str, ...] = (
    "mae", "rmse", "ts_ic", "rank_ic", "direction_accuracy",
)


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


def icir(daily_ics: np.ndarray) -> float:
    """ICIR = mean(daily_IC) / std(daily_IC)（§21 稳定性）。"""
    raise NotImplementedError("icir not implemented")


def positive_ic_day_ratio(daily_ics: np.ndarray) -> float:
    """Positive IC Day Ratio：daily IC > 0 的天占比。"""
    raise NotImplementedError("positive_ic_day_ratio not implemented")


def evaluate(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """按 METRIC_NAMES 计算全部指标（不含日级稳定性）。

    Returns:
        ``{mae, rmse, ts_ic, rank_ic, direction_accuracy}``。
    """
    raise NotImplementedError("evaluate not implemented")


def evaluate_by_day(
    preds: np.ndarray, targets: np.ndarray, trade_dates: np.ndarray
) -> dict[str, float]:
    """日级稳定性汇总（§14/§21）：daily 指标 mean/std + ICIR + Positive Day Ratio。

    3 秒采样 × 60 秒 horizon 的标签高度重叠（§14），日级聚合是处理序列相关的
    最小要求；每日先算 TS-IC，再聚合。

    Returns:
        ``{daily_ic_mean, daily_ic_std, icir, positive_ic_day_ratio}``。
    """
    raise NotImplementedError("evaluate_by_day not implemented")
