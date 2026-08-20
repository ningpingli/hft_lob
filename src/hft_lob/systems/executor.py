"""模型与 baseline 共用的默认 walk-forward fold 执行器。"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from lightning.pytorch.callbacks import ModelCheckpoint

from hft_lob.baselines import BASELINE_NAMES, BaselineRunner, build_baseline
from hft_lob.configs.experiment import ExperimentConfig
from hft_lob.datasets.lob_dataset import LOBBatch
from hft_lob.models import build_model
from hft_lob.preprocessing.normalize import CausalRollingStandardizer
from hft_lob.preprocessing.pipeline import PreparedDataset
from hft_lob.preprocessing.split import Fold
from hft_lob.systems.artifact import PredictionArtifact
from hft_lob.systems.lob_data_module import LOBDataModule, resolve_stage_files
from hft_lob.systems.lob_module import LOBLightningModule
from hft_lob.systems.walk_forward import CandidateFoldRun
from hft_lob.train import (
    build_checkpoint_callback,
    build_early_stopping_callback,
    build_trainer,
    run_predict,
    run_training,
)


@dataclass(frozen=True)
class DefaultWalkForwardExecutor:
    """在独立 fold 目录中执行训练、预测并返回统一结果。"""

    output_root: str
    accelerator: str = "auto"
    devices: int | str = 1

    def run_candidate(
        self,
        *,
        dataset: PreparedDataset,
        config: ExperimentConfig,
        fold: Fold,
        candidate_name: str,
    ) -> CandidateFoldRun:
        output_dir = (
            Path(self.output_root)
            / f"fold_{fold.index:03d}"
            / candidate_name
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        standardizer = CausalRollingStandardizer(
            dataset.feature_columns,
            config.normalization.normalize_window,
        )
        state_path = _write_json_atomic(
            output_dir / "standardizer.json",
            standardizer.state_dict(),
        )
        stage_files = resolve_stage_files(dataset, fold_index=fold.index)
        datamodule = LOBDataModule(
            config,
            stage_files=stage_files,
            standardizer=standardizer,
        )
        predictions_path = str((output_dir / "predictions.parquet").resolve())
        model_version = f"{config.experiment_id}-fold{fold.index}-{candidate_name}"

        if candidate_name in BASELINE_NAMES:
            artifact = self._run_baseline(
                candidate_name=candidate_name,
                dataset=dataset,
                config=config,
                datamodule=datamodule,
                model_version=model_version,
                fold_index=fold.index,
            )
            return CandidateFoldRun(
                artifact=artifact,
                standardizer_state_path=state_path,
                predictions_path=predictions_path,
            )

        if candidate_name != config.model.name:
            raise ValueError(f"candidate {candidate_name!r} is neither model nor baseline")
        checkpoint = cast(
            ModelCheckpoint,
            build_checkpoint_callback(
                str(output_dir / "checkpoints"),
                monitor=config.training.monitor_metric,
                mode=config.training.monitor_mode,
            ),
        )
        early_stopping = build_early_stopping_callback(
            monitor=config.training.monitor_metric,
            mode=config.training.monitor_mode,
            patience=config.training.patience,
        )
        trainer = build_trainer(
            str(output_dir),
            config.training.epochs,
            config.training.patience,
            callbacks=[checkpoint, early_stopping],
            accelerator=self.accelerator,
            devices=self.devices,
        )
        lightning_module = LOBLightningModule(
            build_model(config, feature_columns=dataset.feature_columns),
            config,
            dataset_version=dataset.dataset_version,
            model_version=model_version,
            fold_index=fold.index,
        )
        run_training(trainer, lightning_module, datamodule)
        checkpoint_path = checkpoint.best_model_path
        if not checkpoint_path or not Path(checkpoint_path).is_file():
            raise RuntimeError(
                f"training completed without a best checkpoint for {candidate_name} fold {fold.index}"
            )
        artifact = run_predict(
            trainer,
            lightning_module,
            datamodule,
            checkpoint_path,
            split="test",
        )
        return CandidateFoldRun(
            artifact=artifact,
            standardizer_state_path=state_path,
            checkpoint_path=str(Path(checkpoint_path).resolve()),
            predictions_path=predictions_path,
        )

    @staticmethod
    def _run_baseline(
        *,
        candidate_name: str,
        dataset: PreparedDataset,
        config: ExperimentConfig,
        datamodule: LOBDataModule,
        model_version: str,
        fold_index: int,
    ) -> PredictionArtifact:
        datamodule.setup("fit")
        training_batches = tuple(
            cast(LOBBatch, batch) for batch in datamodule.train_dataloader()
        )
        datamodule.teardown("fit")
        runner = BaselineRunner(
            name=candidate_name,
            model=build_baseline(
                candidate_name,
                config,
                feature_columns=dataset.feature_columns,
            ),
            model_version=model_version,
            dataset_version=dataset.dataset_version,
            fold_index=fold_index,
        )
        runner.fit(training_batches)
        datamodule.setup("predict")
        test_batches = tuple(
            cast(LOBBatch, batch) for batch in datamodule.predict_dataloader()
        )
        datamodule.teardown("predict")
        return runner.predict(test_batches, split="test")


def _write_json_atomic(path: Path, value: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(value, temporary, ensure_ascii=False, sort_keys=True, indent=2)
            temporary.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return str(path.resolve())
