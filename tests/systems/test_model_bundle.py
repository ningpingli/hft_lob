from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from hft_lob.configs.experiment import (
    EvaluationConfig,
    FoldSelectionConfig,
    LoaderConfig,
    ModelConfig,
    ModelRunConfig,
    TrainingConfig,
)
from hft_lob.datasets.dataset_validator import (
    DatasetPackageMetadata,
    compute_dataset_id,
)
from hft_lob.systems.model_bundle import (
    MODEL_CONFIG_FILENAME,
    MODEL_METADATA_FILENAME,
    load_model_bundle,
    save_model_bundle,
    validate_model_data_contract,
)


def _config() -> ModelRunConfig:
    return ModelRunConfig(
        experiment_id="training-run",
        loader=LoaderConfig(batch_size=8),
        model=ModelConfig(name="cnn1"),
        training=TrainingConfig(epochs=1),
        evaluation=EvaluationConfig(prediction_bins=2),
        folds=FoldSelectionConfig(start_fold=1, num_folds=1),
        seed=7,
    )


def _metadata(*, source_hash: str = "source") -> DatasetPackageMetadata:
    processing_hash = "processing"
    fold_hash = "folds"
    return DatasetPackageMetadata(
        dataset_id=compute_dataset_id(
            ticker="TEST",
            source_hash=source_hash,
            processing_config_hash=processing_hash,
            fold_plan_hash=fold_hash,
        ),
        ticker="TEST",
        feature_columns=("BIDp1", "BIDq1", "ASKp1", "ASKq1"),
        target_columns=("Target_60s_log",),
        feature_dtype="float32",
        target_dtype="float32",
        snapshot_interval_seconds=3,
        history_snapshots=8,
        normalization_mode="rolling_zscore",
        normalization_window=2,
        source_hash=source_hash,
        processing_config_hash=processing_hash,
        fold_plan_hash=fold_hash,
        labels=(60,),
    )


def _write_checkpoint(
    path: Path,
    *,
    model_version: str,
    metadata: DatasetPackageMetadata | None = None,
    fold_index: int = 1,
) -> None:
    dataset = metadata or _metadata()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hyper_parameters": {
                "dataset_version": dataset.dataset_id,
                "model_version": model_version,
                "fold_index": fold_index,
            },
            "state_dict": {"model.weight": torch.ones(1)},
        },
        path,
    )


def test_model_bundle_round_trips_training_checkpoint_and_sidecars(tmp_path: Path) -> None:
    model_dir = tmp_path / "fold_001" / "cnn1"
    checkpoint = model_dir / "checkpoints" / "best_val_model.ckpt"
    _write_checkpoint(checkpoint, model_version="training-run-fold1-cnn1")

    saved = save_model_bundle(
        model_dir,
        config=_config(),
        dataset_metadata=_metadata(),
        checkpoint_path=checkpoint,
        model_version="training-run-fold1-cnn1",
        fold_index=1,
    )
    loaded = load_model_bundle(model_dir)

    assert saved == loaded
    assert loaded.checkpoint_path == checkpoint.resolve()
    assert loaded.metadata.checkpoint_file == "checkpoints/best_val_model.ckpt"
    assert loaded.metadata.model_name == "cnn1"
    assert loaded.config.model.name == "cnn1"
    assert loaded.config.loader.batch_size == 8
    assert (model_dir / MODEL_CONFIG_FILENAME).is_file()
    assert (model_dir / MODEL_METADATA_FILENAME).is_file()


def test_model_bundle_rejects_checkpoint_identity_mismatch_on_load(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    checkpoint = model_dir / "checkpoints" / "best.ckpt"
    _write_checkpoint(checkpoint, model_version="checkpoint-v2")
    save_model_bundle(
        model_dir,
        config=_config(),
        dataset_metadata=_metadata(),
        checkpoint_path=checkpoint,
        model_version="metadata-v1",
        fold_index=1,
    )

    with pytest.raises(ValueError, match="checkpoint identity"):
        load_model_bundle(model_dir)


def test_model_bundle_accepts_new_dataset_identity_with_same_contract(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    checkpoint = model_dir / "checkpoints" / "best.ckpt"
    _write_checkpoint(checkpoint, model_version="model-v1")
    bundle = save_model_bundle(
        model_dir,
        config=_config(),
        dataset_metadata=_metadata(),
        checkpoint_path=checkpoint,
        model_version="model-v1",
        fold_index=1,
    )

    validate_model_data_contract(
        bundle.metadata.data_contract,
        _metadata(source_hash="independent-test-source"),
    )


def test_model_bundle_rejects_incompatible_test_contract(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    checkpoint = model_dir / "checkpoints" / "best.ckpt"
    _write_checkpoint(checkpoint, model_version="model-v1")
    bundle = save_model_bundle(
        model_dir,
        config=_config(),
        dataset_metadata=_metadata(),
        checkpoint_path=checkpoint,
        model_version="model-v1",
        fold_index=1,
    )

    with pytest.raises(ValueError, match="history_snapshots"):
        validate_model_data_contract(
            bundle.metadata.data_contract,
            replace(_metadata(), history_snapshots=9),
        )


def test_model_bundle_rejects_checkpoint_outside_model_directory(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    checkpoint = tmp_path / "outside.ckpt"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="contained in model_dir"):
        save_model_bundle(
            model_dir,
            config=_config(),
            dataset_metadata=_metadata(),
            checkpoint_path=checkpoint,
            model_version="model-v1",
            fold_index=1,
        )
