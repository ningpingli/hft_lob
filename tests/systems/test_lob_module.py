from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from hft_lob.configs.experiment import (
    BaselineConfig,
    CleaningConfig,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    FeatureConfig,
    LoaderConfig,
    ModelConfig,
    NormalizationConfig,
    SessionConfig,
    SplitConfig,
    TargetConfig,
    TaskConfig,
    TrainingConfig,
    WalkForwardConfig,
    WindowConfig,
)
from hft_lob.datasets.lob_dataset import LOBBatch, SampleMeta
from hft_lob.systems.lob_module import LOBLightningModule


class MeanModel(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.output = nn.Linear(features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(x.mean(dim=1))


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="lob-module-test",
        task=TaskConfig(ticker="TEST"),
        data=DataConfig(),
        cleaning=CleaningConfig(),
        target=TargetConfig(),
        sessions=SessionConfig(),
        window=WindowConfig(history_snapshots=3),
        features=FeatureConfig(),
        normalization=NormalizationConfig(),
        loader=LoaderConfig(),
        model=ModelConfig(name="cnn1"),
        baselines=BaselineConfig(),
        training=TrainingConfig(epochs=1),
        evaluation=EvaluationConfig(
            metrics=("mae", "ts_ic", "direction_accuracy"),
            prediction_bins=2,
            bootstrap_samples=2,
        ),
        split=SplitConfig(),
        walk_forward=WalkForwardConfig(),
    )


def _batch() -> LOBBatch:
    features = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]],
            [[-1.0, 2.0], [0.0, 1.0], [1.0, 0.0]],
        ]
    )
    targets = torch.tensor([[0.2], [-0.1]])
    metadata = tuple(
        SampleMeta(
            ticker="TEST",
            trade_date="2026-01-05",
            session_id="AM",
            anchor_timestamp=f"2026-01-05T09:30:0{index}",
            mid_t=10.0,
            future_mid=10.1,
            bid1=9.9,
            ask1=10.1,
            spread=0.2,
        )
        for index in range(2)
    )
    return LOBBatch(features, targets, metadata)


def _module() -> LOBLightningModule:
    module = LOBLightningModule(
        MeanModel(2),
        _config(),
        dataset_version="dataset-v1",
        model_version="model-v1",
        fold_index=1,
    )
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    return module


def test_training_and_optimizer_contract() -> None:
    module = _module()

    loss = module.training_step(_batch(), 0)

    assert loss.ndim == 0
    optimizer = module.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)


def test_validation_logs_epoch_metrics_and_clears_buffers() -> None:
    module = _module()
    logged: dict[str, Any] = {}
    module.log = lambda name, value, **kwargs: logged.__setitem__(name, value)  # type: ignore[method-assign]

    module.validation_step(_batch(), 0)
    module.on_validation_epoch_end()

    assert {"val/loss", "val/mae", "val/ts_ic", "val/direction_accuracy"} <= logged.keys()
    assert not module._validation_predictions


def test_predict_and_test_use_complete_artifact_contract() -> None:
    module = _module()
    batch = _batch()

    prediction = module.predict_step(batch, 0)
    module.test_step(batch, 0)
    module.on_test_epoch_end()

    assert prediction.predictions.shape == (2,)
    assert prediction.dataset_version == "dataset-v1"
    assert prediction.fold_index == 1
    assert module.test_artifact is not None
    assert module.test_artifact.metadata == batch.metadata


def test_rejects_noncanonical_model_output() -> None:
    module = LOBLightningModule(nn.Identity(), _config())

    with pytest.raises(ValueError, match="model output"):
        module(torch.randn(2, 3, 2))
