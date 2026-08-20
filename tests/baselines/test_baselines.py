from __future__ import annotations

import torch

from hft_lob.baselines import volume_feature_indices
from hft_lob.baselines.models import (
    ImbalanceBaseline,
    MLPBaseline,
    RidgeBaseline,
    ZeroBaseline,
)
from hft_lob.baselines.runner import BaselineRunner
from hft_lob.datasets.lob_dataset import LOBBatch, SampleMeta


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


def test_mlp_fit_switches_to_deterministic_evaluation() -> None:
    torch.manual_seed(3)
    x = torch.randn(8, 2, 3)
    y = x.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
    model = MLPBaseline(
        num_features=3,
        history_snapshots=2,
        hidden_dim=4,
        dropout=0.5,
        epochs=2,
    ).fit(x, y)

    assert model(x).shape == (8, 1)
    torch.testing.assert_close(model(x), model(x))
    assert not model.training


def test_volume_indices_follow_level_order() -> None:
    columns = ("ASKp1", "ASKs2", "BIDs1", "ASKs1", "BIDs2", "last")

    bid, ask = volume_feature_indices(columns)

    assert bid == (2, 4)
    assert ask == (3, 1)


def test_runner_builds_prediction_artifact() -> None:
    features = torch.randn(2, 3, 2)
    targets = torch.tensor([[0.1], [-0.2]])
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
    batches = (LOBBatch(features, targets, metadata),)
    runner = BaselineRunner("zero", ZeroBaseline(), "v1", "dataset-v1", 1)

    runner.fit(batches)
    artifact = runner.predict(batches, split="test")

    assert artifact.model_name == "zero"
    assert artifact.predictions.tolist() == [0.0, 0.0]
    assert artifact.targets.tolist() == targets[:, 0].tolist()
