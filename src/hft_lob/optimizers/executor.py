"""模型实例化、配置契约校验与训练执行器。"""

from __future__ import annotations

from typing import Any

from torch import nn


def validate_model_data_contract(
    data_features: dict[str, Any], data_hyperparameters: dict[str, Any]
) -> None:
    """校验模型配置 data_features 与实验 data 段契约一致。

    num_features / levels 逐项一致；history_length 与 window_size 是同一滑窗
    长度的两个名字，必须相等。不匹配时抛出含双方实际值的 ValueError。

    Args:
        data_features: 模型配置的 data_features 段（契约方）。
        data_hyperparameters: 实验配置的 data 段。
    """
    raise NotImplementedError("validate_model_data_contract not implemented")


def validate_training_contract(training_hyperparameters: dict[str, Any]) -> None:
    """校验训练段（experiment.yaml training 段）配置契约。

    loss 必须是 ``LOSS_NAMES`` 之一（大小写不敏感）；loss_huber_delta 必须是
    数值；optimizer.betas 必须是长度为 2 的数值序列；optimizer.weight_decay
    必须是数值。键缺失时按 KeyError 自然暴露。

    Args:
        training_hyperparameters: 训练段配置字典。
    """
    raise NotImplementedError("validate_training_contract not implemented")


def build_model(
    model_name: str,
    data_features: dict[str, Any],
    model_params: dict[str, Any],
    *,
    homological_structures: dict[str, Any] | None = None,
) -> nn.Module:
    """按配置实例化模型（configs/models/<model>.yaml 的 data_features + model_params）。

    数据侧契约值（num_features / levels / history_length）由调用方按注入规则
    传入；hlob 的同调结构经 ``homological_structures`` 注入。未注册的模型名
    抛 ValueError。

    Args:
        model_name: 模型名（deeplob / transformer / hlob 等）。
        data_features: 模型配置的数据契约段。
        model_params: 模型专属参数段。
        homological_structures: hlob 模型所需的同调结构（默认 None）。

    Returns:
        实例化后的模型。
    """
    raise NotImplementedError("build_model not implemented")


class Executor:
    """训练执行器：数据集准备、模型构建与训练/评估编排。"""

    def __init__(
        self,
        experiment_id: str,
        general_hyperparameters: dict[str, Any],
        data_hyperparameters: dict[str, Any],
        loader_hyperparameters: dict[str, Any],
        training_hyperparameters: dict[str, Any],
        torch_dataset_preparation: bool = False,
        torch_dataset_preparation_backtest: bool = False,
    ) -> None:
        """初始化执行器：校验训练契约并按模式准备数据集或构建模型与 manager。

        Args:
            experiment_id: 实验 ID。
            general_hyperparameters: 通用超参数。
            data_hyperparameters: data 段超参数。
            loader_hyperparameters: loader 段超参数。
            training_hyperparameters: training 段超参数。
            torch_dataset_preparation: 仅生成 torch 数据集（不构建模型）。
            torch_dataset_preparation_backtest: 仅生成回测用 torch 数据集。
        """
        raise NotImplementedError("Executor.__init__ not implemented")

    def execute_training(self) -> None:
        """执行模型训练（manager.train）。"""
        raise NotImplementedError("Executor.execute_training not implemented")

    def execute_testing(self) -> None:
        """执行模型测试（manager.test）。"""
        raise NotImplementedError("Executor.execute_testing not implemented")

    def logger_clean_up(self) -> None:
        """清理实验目录下的 wandb 日志文件（尽力而为，无产物时无副作用）。"""
        raise NotImplementedError("Executor.logger_clean_up not implemented")
