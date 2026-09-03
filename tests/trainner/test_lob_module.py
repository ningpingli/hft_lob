from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from hft_lob.configs.experiment import (
    EvaluationConfig,
    LoaderConfig,
    ModelConfig,
    ModelRunConfig,
    TrainingConfig,
)
from hft_lob.data_types import LOBBatch, SampleMeta
from hft_lob.trainner.lob_module import LOBLightningModule


class MeanModel(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.output = nn.Linear(features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(x.mean(dim=1))


def _config() -> ModelRunConfig:
    return ModelRunConfig(
        experiment_id="lob-module-test",
        loader=LoaderConfig(),
        model=ModelConfig(name="cnn1"),
        training=TrainingConfig(epochs=1),
        evaluation=EvaluationConfig(prediction_bins=2),
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


def test_validation_logs_only_fast_metrics_and_clears_accumulators() -> None:
    module = _module()
    batch = _batch()
    expected = module(batch.features).detach()
    logged: dict[str, Any] = {}
    module.log = lambda name, value, **kwargs: logged.__setitem__(name, value)  # type: ignore[method-assign]

    module.on_validation_epoch_start()
    module.validation_step(batch, 0)
    module.on_validation_epoch_end()

    assert set(logged) == {"val/loss", "val/mse", "val/mae"}
    assert logged["val/mse"].item() == pytest.approx(
        torch.mean((expected - batch.targets).square()).item()
    )
    assert logged["val/mae"].item() == pytest.approx(
        torch.mean((expected - batch.targets).abs()).item()
    )
    assert module._validation_element_count == 0
    assert module._validation_mse_sum is None
    assert module._validation_mae_sum is None

def test_test_uses_complete_artifact_contract() -> None:
    module = _module()
    batch = _batch()

    module.on_test_epoch_start()
    module.test_step(batch, 0)
    module.on_test_epoch_end()

    assert module.test_artifact.predictions.shape == (2, 1)
    assert module.test_artifact.dataset_version == "dataset-v1"
    assert module.test_artifact.fold_index == 1


def test_rejects_noncanonical_model_output() -> None:
    module = LOBLightningModule(nn.Identity(), _config())

    with pytest.raises(ValueError, match="model output"):
        module(torch.randn(2, 3, 2))
