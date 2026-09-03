from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from hft_lob.application.train import TrainingResult
from hft_lob.cli.main import main, parse_args


def test_parse_training_cli_requires_dataset() -> None:
    args = parse_args(
        [
            "train",
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
    assert args.seed == 7
    assert args.gpu_id == 2


def test_parse_baseline_cli_supports_replace_default() -> None:
    args = parse_args(
        [
            "baseline",
            "run",
            "--config",
            "configs/baselines.yaml",
            "--dataset-dir",
            "dataset",
            "--experiment-id",
            "baseline-1",
            "--replace-default",
        ]
    )

    assert args.command == "baseline"
    assert args.baseline_command == "run"
    assert args.replace_default is True


def test_parse_standalone_test_cli_requires_model_and_dataset() -> None:
    args = parse_args(
        [
            "test",
            "--test-data-dir",
            "datasets/test-set",
            "--model-name",
            "cnn1",
            "--model-dir",
            "models/cnn1",
            "--output-dir",
            "results/cnn1",
        ]
    )

    assert args.command == "test"
    assert args.test_data_dir == "datasets/test-set"
    assert args.model_name == "cnn1"
    assert args.model_dir == "models/cnn1"
    assert args.output_dir == "results/cnn1"


def test_parse_rejects_unknown_stage_or_missing_dataset() -> None:
    with pytest.raises(SystemExit):
        parse_args(["train", "--dataset-dir", "dataset", "--stages", "prepare-data"])
    with pytest.raises(SystemExit):
        parse_args([])


def test_main_passes_dataset_directly_to_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Path(__file__).parents[1] / "configs" / "train.yaml"
    dataset_dir = tmp_path / "immutable-dataset"
    seen: dict[str, object] = {}

    def fake_run(request):  # type: ignore[no-untyped-def]
        seen["request"] = request
        return TrainingResult("cli-training", "dataset-id", 1)

    main_module = importlib.import_module("hft_lob.cli.main")
    monkeypatch.setattr(main_module, "run_training_application", fake_run)
    main(
        [
            "train",
            "--config",
            str(config),
            "--dataset-dir",
            str(dataset_dir),
            "--experiment-id",
            "cli-training",
        ]
    )
    request = seen["request"]
    assert request.dataset_dir == str(dataset_dir)  # type: ignore[union-attr]
    assert request.experiment_id == "cli-training"  # type: ignore[union-attr]
