"""交易后分析：评估回测表现并生成分类/盈亏图表。"""

from __future__ import annotations

from typing import Any


def post_trading_analysis(
    experiment_id: str,
    general_hyperparameters: dict[str, Any],
    trading_hyperparameters: dict[str, Any],
    data_hyperparameters: dict[str, Any],
    loader_hyperparameters: dict[str, Any],
) -> None:
    """对回测结果进行交易后分析：分类报告、ROC、混淆矩阵、P&L 与价格图。

    读取 ``prediction.pkl`` 与 ``trading_simulation.pkl``，打印分类指标
    （MCC / AUC-ROC / top-k）并绘制预测分布、ROC 曲线、混淆矩阵、
    P&L 分布/累计曲线与带多空标记的中间价走势图。

    Args:
        experiment_id: 实验 ID。
        general_hyperparameters: 通用超参数（training_stocks / target_stocks）。
        trading_hyperparameters: trading 段超参数（simulation_type 等）。
        data_hyperparameters: data 段超参数（threshold / prediction_horizon）。
        loader_hyperparameters: loader 段超参数（batch_size / num_workers）。
    """
    raise NotImplementedError("post_trading_analysis not implemented")
