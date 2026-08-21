"""模型与 baseline 共用的默认 walk-forward fold 执行器。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from lightning.pytorch.callbacks import ModelCheckpoint

from hft_lob.baselines import BASELINE_NAMES, BaselineRunner, build_baseline
from hft_lob.configs.experiment import ExperimentConfig
from hft_lob.datasets.lob_dataset import LOBBatch
from hft_lob.datasets.package import DatasetPackageMetadata
from hft_lob.models import build_model
from hft_lob.systems.artifact import PredictionArtifact
from hft_lob.systems.lob_data_module import LOBDataModule
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
        dataset_dir: str,
        metadata: DatasetPackageMetadata,
        config: ExperimentConfig,
        fold_index: int,
        candidate_name: str,
    ) -> CandidateFoldRun:
        output_dir = (
            Path(self.output_root)
            / f"fold_{fold_index:03d}"
            / candidate_name
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        datamodule = LOBDataModule(
            dataset_dir,
            fold_index=fold_index,
            loader=config.loader,
            seed=config.seed,
        )
        predictions_path = str((output_dir / "predictions.parquet").resolve())
        model_version = f"{config.experiment_id}-fold{fold_index}-{candidate_name}"
        state_path = str((Path(dataset_dir) / "dataset.json").resolve())

        if candidate_name in BASELINE_NAMES:
            artifact = self._run_baseline(
                candidate_name=candidate_name,
                metadata=metadata,
                config=config,
                datamodule=datamodule,
                model_version=model_version,
                fold_index=fold_index,
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
            build_model(config, feature_columns=metadata.feature_columns),
            config,
            dataset_version=metadata.dataset_id,
            model_version=model_version,
            fold_index=fold_index,
        )
        run_training(trainer, lightning_module, datamodule)
        checkpoint_path = checkpoint.best_model_path
        if not checkpoint_path or not Path(checkpoint_path).is_file():
            raise RuntimeError(
                f"training completed without a best checkpoint for {candidate_name} fold {fold_index}"
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
        metadata: DatasetPackageMetadata,
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
                feature_columns=metadata.feature_columns,
            ),
            model_version=model_version,
            dataset_version=metadata.dataset_id,
            fold_index=fold_index,
        )
        runner.fit(training_batches)
        datamodule.setup("predict")
        test_batches = tuple(
            cast(LOBBatch, batch) for batch in datamodule.predict_dataloader()
        )
        datamodule.teardown("predict")
        return runner.predict(test_batches, split="test")
