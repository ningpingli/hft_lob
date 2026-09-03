from __future__ import annotations

import pytest
import torch

from hft_lob.baselines import BASELINE_NAMES, build_baseline
from hft_lob.baselines.models import RidgeBaseline
from hft_lob.baselines.runner import BaselineRunner
from hft_lob.configs.experiment import BaselineConfig
from hft_lob.data_types import LOBBatch, SampleMeta


def _meta() -> tuple[SampleMeta, ...]:
    return tuple(
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


def test_ridge_fits_and_serializes_parameters() -> None:
    torch.manual_seed(2)
    x = torch.randn(20, 2, 3)
    expected_weight = torch.arange(6, dtype=torch.float32).reshape(6, 1) / 10
    y = x.reshape(20, 6) @ expected_weight + 0.25
    y = y.repeat(1, 2)
    model = RidgeBaseline(num_features=3, history_snapshots=2, alpha=0.0, target_count=2).fit(x, y)
    torch.testing.assert_close(model(x), y, atol=1e-5, rtol=1e-5)
    assert {"weight", "intercept", "fitted"}.issubset(model.state_dict())


def test_runner_builds_prediction_artifact() -> None:
    features = torch.randn(2, 3, 2)
    targets = torch.tensor([[0.1], [-0.2]])
    batches = (LOBBatch(features, targets, _meta()),)
    runner = BaselineRunner(
        "ridge",
        RidgeBaseline(num_features=2, history_snapshots=3),
        "v1",
        "dataset-v1",
        1,
    )
    runner.fit(lambda: iter(batches))
    artifact = runner.predict(batches, split="test")
    assert artifact.model_name == "ridge"
    assert artifact.predictions.shape == targets.shape
    assert artifact.targets.tolist() == targets.tolist()


def test_ridge_rejects_empty_training_batches() -> None:
    model = RidgeBaseline(num_features=2, history_snapshots=3)
    with pytest.raises(ValueError, match="must not be empty"):
        model.fit_batches(lambda: iter(()))

def test_only_ridge_is_registered() -> None:
    assert BASELINE_NAMES == ("ridge",)
    with pytest.raises(ValueError, match="unsupported baseline"):
        build_baseline(
            "zero",
            BaselineConfig(),
            feature_columns=("feature",),
            history_snapshots=1,
        )
