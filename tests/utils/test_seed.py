from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

from hft_lob.utils.seed import set_seed


def test_set_seed_replays_python_numpy_and_torch_streams() -> None:
    set_seed(42)
    first = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )

    set_seed(42)
    second = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_set_seed_configures_hash_cuda_and_cudnn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda_seeds: list[int] = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", cuda_seeds.append)

    set_seed(123)

    assert os.environ["PYTHONHASHSEED"] == "123"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert cuda_seeds
    assert all(seed == 123 for seed in cuda_seeds)


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5, "42", True])
def test_set_seed_rejects_invalid_values(seed: object) -> None:
    with pytest.raises(ValueError, match="seed must be an integer"):
        set_seed(seed)  # type: ignore[arg-type]
