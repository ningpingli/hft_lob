"""ExperimentConfig：实验超参数 dataclass 组。

整个流水线的接口面都收 ``ExperimentConfig``，避免到处传递
``(experiment_id, general, data, loader, training)`` 4 件套。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hft_lob.data_processing.fields import FieldsConfig


#: 模型 data_features 契约（架构期望的输入形状）：num_features / levels / history_length
#: 分别与 ExperimentConfig.data 三项逐项校验，构建模型时在 executor.build_model 强制。
@dataclass(frozen=True)
class ModelDataFeatures:
    """模型架构期望的输入形状契约。"""

    num_features: int
    levels: int
    history_length: int


#: 模型超参数（按 configs/models/<model>.yaml 中 model_params 段读取，schema
#: 由具体模型类决定，因此这里用 dict 透传）。
ModelParams = dict[str, Any]


@dataclass(frozen=True)
class GeneralHyperparameters:
    """实验通用超参数（来自 ``configs/experiment.yaml`` general 段）。"""

    experiment_id: str
    dataset: str
    model: str
    training_stocks: list[str]
    target_stocks: list[str]
    normalization_window: int
    horizons: list[int]
    label_columns: dict[str, list[str]]
    prediction_label_type: str
    prediction_label: str
    training_ratio: float
    validation_ratio: float
    test_ratio: float
    stages: list[str]
    include_target_stock_in_training: bool
    max_days: int | None = None


@dataclass(frozen=True)
class DataHyperparameters:
    """数据侧超参数（来自 ``configs/experiment.yaml`` data 段）。"""

    num_features: int
    levels: int
    window_size: int
    prediction_horizon: int
    threshold: float
    fields: FieldsConfig


@dataclass(frozen=True)
class LoaderHyperparameters:
    """DataLoader 超参数（来自 ``configs/experiment.yaml`` loader 段）。"""

    batch_size: int
    num_workers: int
    shuffling_seed: int
    balanced_sampling: bool = False


@dataclass(frozen=True)
class OptimizerHyperparameters:
    """训练优化器超参数（嵌套在 ``training.optimizer`` 段）。"""

    betas: tuple[float, float]
    weight_decay: float


@dataclass(frozen=True)
class TrainingHyperparameters:
    """训练超参数（来自 ``configs/experiment.yaml`` training 段）。"""

    epochs: int
    log_interval_epochs: int
    learning_rate: float
    patience: int
    loss: str
    loss_huber_delta: float
    optimizer: OptimizerHyperparameters


@dataclass(frozen=True)
class TradingHyperparameters:
    """回测超参数（来自 ``configs/experiment.yaml`` trading 段）。"""

    initial_cash: float
    trading_fee: float
    mid_side_trading: str
    simulation_type: str
    probability_threshold: float


@dataclass(frozen=True)
class ModelHyperparameters:
    """模型侧超参数（来自 ``configs/models/<model>.yaml``）。"""

    name: str
    data_features: ModelDataFeatures
    model_params: ModelParams = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    """整个实验的配置根对象。

    Attributes:
        general: 实验通用超参数（含 experiment_id 与运行身份）。
        data: 数据侧超参数。
        loader: DataLoader 超参数。
        training: 训练超参数（含优化器嵌套）。
        trading: 回测超参数。
        model: 模型侧超参数（含契约 data_features 与可变 model_params）。
    """

    general: GeneralHyperparameters
    data: DataHyperparameters
    loader: LoaderHyperparameters
    training: TrainingHyperparameters
    trading: TradingHyperparameters
    model: ModelHyperparameters

    @property
    def experiment_id(self) -> str:
        """便捷访问实验 ID（lobx 调用方最常读）。"""
        return self.general.experiment_id

    @property
    def dataset(self) -> str:
        """便捷访问数据集名。"""
        return self.general.dataset

    @property
    def model_name(self) -> str:
        """便捷访问模型名（架构分发键）。"""
        return self.general.model