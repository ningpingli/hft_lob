from __future__ import annotations

import pytest
import torch
from torch import nn

from hft_lob.systems.losses import build_loss


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [("mse", nn.MSELoss), ("mae", nn.L1Loss), ("huber", nn.HuberLoss)],
)
def test_build_loss_returns_expected_module(
    name: str, expected_type: type[nn.Module]
) -> None:
    assert isinstance(build_loss(name), expected_type)


def test_build_loss_normalizes_name() -> None:
    assert isinstance(build_loss("  HuBeR  "), nn.HuberLoss)


def test_build_loss_uses_huber_delta() -> None:
    loss = build_loss("huber", huber_delta=2.0)
    prediction = torch.tensor([3.0])
    target = torch.tensor([0.0])

    assert loss(prediction, target).item() == pytest.approx(4.0)


@pytest.mark.parametrize("name", ["", "cross_entropy", "msee"])
def test_build_loss_rejects_unknown_name(name: str) -> None:
    with pytest.raises(ValueError, match="unsupported loss name"):
        build_loss(name)


@pytest.mark.parametrize("delta", [0.0, -1.0])
def test_build_loss_rejects_non_positive_huber_delta(delta: float) -> None:
    with pytest.raises(ValueError, match="huber_delta must be > 0"):
        build_loss("huber", huber_delta=delta)


