"""交易模拟：基于测试集预测执行回测并保存交易历史。"""

from __future__ import annotations

from typing import Any


def backtest(
    experiment_id: str, trading_hyperparameters: dict[str, Any]
) -> None:
    """基于预测结果执行交易模拟，把交易历史落盘为 trading_simulation.pkl。

    读取 ``prediction.pkl``，按 ``mid_side_trading`` 策略逐快照驱动
    ``Trading`` agent 开平仓（日内收盘强制平仓），结果写入
    ``loggers/results/<experiment_id>/trading_simulation.pkl``。

    Args:
        experiment_id: 实验 ID。
        trading_hyperparameters: trading 段超参数（mid_side_trading /
            probability_threshold 等）。
    """
    raise NotImplementedError("backtest not implemented")
