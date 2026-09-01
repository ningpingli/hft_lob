from __future__ import annotations

import pytest
import torch

from hft_lob.baselines import build_baseline, volume_feature_indices
from hft_lob.baselines.models import (
    ImbalanceBaseline,
    RidgeBaseline,
    ZeroBaseline,
)
from hft_lob.baselines.runner import BaselineRunner
from hft_lob.configs.experiment import BaselineConfig
from hft_lob.systems.contracts import LOBBatch, SampleMeta


def test_zero_baseline_preserves_batch_contract() -> None:
    x = torch.randn(3, 4, 2)
    y = torch.randn(3, 1)
    model = ZeroBaseline().fit(x, y)

    assert torch.equal(model(x), torch.zeros(3, 1))


def test_imbalance_fits_anchor_linear_mapping() -> None:
    x = torch.zeros(4, 2, 2)
    x[:, -1, 0] = torch.tensor([3.0, 2.0, 1.0, 4.0])
    x[:, -1, 1] = torch.tensor([1.0, 2.0, 3.0, 0.0])
    imbalance = (x[:, -1, 0] - x[:, -1, 1]) / x[:, -1, :].sum(dim=1)
    y = (2 * imbalance + 0.5).unsqueeze(1)

    model = ImbalanceBaseline(bid_volume_indices=(0,), ask_volume_indices=(1,))
    model.fit(x, y)

    torch.testing.assert_close(model(x), y)


def test_ridge_fits_and_serializes_parameters() -> None:
    torch.manual_seed(2)
    x = torch.randn(20, 2, 3)
    expected_weight = torch.arange(6, dtype=torch.float32).reshape(6, 1) / 10
    y = x.reshape(20, 6) @ expected_weight + 0.25
    model = RidgeBaseline(num_features=3, history_snapshots=2, alpha=0.0).fit(x, y)

    torch.testing.assert_close(model(x), y, atol=1e-5, rtol=1e-5)
    assert {"weight", "intercept", "fitted"}.issubset(model.state_dict())


def test_streaming_statistics_match_single_batch_fit() -> None:
    torch.manual_seed(8)
    x = torch.randn(12, 2, 3)
    y = x.reshape(12, 6).sum(dim=1, keepdim=True)

    def batches():  # type: ignore[no-untyped-def]
        return iter(((x[:5], y[:5]), (x[5:], y[5:])))

    expected_ridge = RidgeBaseline(num_features=3, history_snapshots=2, alpha=0.5).fit(x, y)
    streamed_ridge = RidgeBaseline(num_features=3, history_snapshots=2, alpha=0.5)
    streamed_ridge.fit_batches(batches)
    torch.testing.assert_close(streamed_ridge(x), expected_ridge(x), atol=1e-5, rtol=1e-5)

    expected_imbalance = ImbalanceBaseline(bid_volume_indices=(0,), ask_volume_indices=(1,)).fit(
        x, y
    )
    streamed_imbalance = ImbalanceBaseline(bid_volume_indices=(0,), ask_volume_indices=(1,))
    streamed_imbalance.fit_batches(batches)
    torch.testing.assert_close(streamed_imbalance(x), expected_imbalance(x), atol=1e-5, rtol=1e-5)


def test_volume_indices_follow_level_order() -> None:
    columns = ("ASKp1", "ASKs2", "BIDs1", "ASKs1", "BIDs2", "last")

    bid, ask = volume_feature_indices(columns)

    assert bid == (2, 4)
    assert ask == (3, 1)


def test_runner_builds_prediction_artifact() -> None:
    features = torch.randn(2, 3, 2)
    targets = torch.tensor([[0.1], [-0.2]])
    target_valid = torch.ones_like(targets, dtype=torch.bool)
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
    batches = (LOBBatch(features, targets, target_valid, metadata),)
    runner = BaselineRunner("zero", ZeroBaseline(), "v1", "dataset-v1", 1)

    runner.fit(lambda: iter(batches))
    artifact = runner.predict(batches, split="test")

    assert artifact.model_name == "zero"
    assert artifact.predictions.tolist() == [[0.0], [0.0]]
    assert artifact.targets.tolist() == targets.tolist()


def test_all_baselines_reject_empty_training_batches() -> None:
    models = (
        ZeroBaseline(),
        ImbalanceBaseline(bid_volume_indices=(0,), ask_volume_indices=(1,)),
        RidgeBaseline(num_features=2, history_snapshots=3),
    )
    for model in models:
        with pytest.raises(ValueError, match="must not be empty"):
            model.fit_batches(lambda: iter(()))


def test_factory_rejects_removed_mlp_baseline() -> None:
    with pytest.raises(ValueError, match="unsupported baseline"):
        build_baseline(
            "mlp",
            BaselineConfig(),
            feature_columns=("BIDs1", "ASKs1"),
            history_snapshots=3,
        )
