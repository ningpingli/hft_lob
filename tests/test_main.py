from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hft_lob.main import main, parse_args


def test_parse_training_cli_requires_dataset() -> None:
    args = parse_args(
        [
            "--config",
            "custom.yaml",
            "--dataset-dir",
            "data/prebuilt/dataset-id",
            "--experiment-id",
            "train-1",
            "--seed",
            "7",
            "--gpu-id",
            "2",
        ]
    )

    assert args.dataset_dir == "data/prebuilt/dataset-id"
    assert args.stages == ["walk-forward"]
    assert args.seed == 7
    assert args.gpu_id == 2


def test_parse_rejects_unknown_stage_or_missing_dataset() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--dataset-dir", "dataset", "--stages", "prepare-data"])
    with pytest.raises(SystemExit):
        parse_args([])


def test_main_passes_dataset_directly_to_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Path(__file__).parents[1] / "configs" / "model.yaml"
    dataset_dir = tmp_path / "immutable-dataset"
    seen: dict[str, object] = {}

    def fake_run(dataset, loaded_config, *, executor):  # type: ignore[no-untyped-def]
        seen["dataset"] = dataset
        return SimpleNamespace(dataset_version="dataset-id", fold_results=(), summary={})

    monkeypatch.chdir(tmp_path)
    main_module = importlib.import_module("hft_lob.main")
    monkeypatch.setattr(main_module, "run_walk_forward", fake_run)
    monkeypatch.setattr(
        main_module,
        "open_dataset_package",
        lambda path: SimpleNamespace(metadata=SimpleNamespace(ticker="TEST")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hft_lob",
            "--config",
            str(config),
            "--dataset-dir",
            str(dataset_dir),
            "--experiment-id",
            "cli-training",
        ],
    )

    main()

    assert seen["dataset"].metadata.ticker == "TEST"  # type: ignore[union-attr]
