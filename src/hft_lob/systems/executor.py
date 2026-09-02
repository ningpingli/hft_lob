"""模型与 baseline 共用的默认 walk-forward fold 执行器。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import Logger

from hft_lob.baselines import BaselineRunner, build_baseline
from hft_lob.configs.experiment import BaselineConfig, BaselineRunConfig, ModelRunConfig
from hft_lob.datasets.datamodule import LOBDataModule
from hft_lob.datasets.dataset_validator import DatasetPackage, DatasetPackageMetadata
from hft_lob.models.lob_model import build_model
from hft_lob.modules.lob_module import LOBLightningModule
from hft_lob.reporting.artifact import PredictionArtifact
from hft_lob.systems.contracts import LOBBatch
from hft_lob.systems.model_bundle import save_model_bundle
from hft_lob.systems.walk_forward import CandidateFoldRun


@dataclass(frozen=True)
class DefaultWalkForwardExecutor:
    """在独立 fold 目录中执行训练、预测并返回统一结果。"""

    output_root: str
    accelerator: str = "auto"
    devices: int | str = 1

    def run_candidate(
        self,
        *,
        package: DatasetPackage,
        config: ModelRunConfig,
        fold_index: int,
        candidate_name: str,
    ) -> CandidateFoldRun:
        metadata = package.metadata
        output_dir = Path(self.output_root) / f"fold_{fold_index:03d}" / candidate_name
        output_dir.mkdir(parents=True, exist_ok=True)
        datamodule = LOBDataModule(
            package,
            fold_index=fold_index,
            loader=config.loader,
            seed=config.seed,
        )
        predictions_path = str((output_dir / "predictions.parquet").resolve())
        model_version = f"{config.experiment_id}-fold{fold_index}-{candidate_name}"
        state_path = str((package.root / "dataset.json").resolve())

        if candidate_name != config.model.name:
            raise ValueError(f"candidate {candidate_name!r} is not the configured model")

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
            build_model(
                config,
                feature_columns=metadata.feature_columns,
                history_snapshots=metadata.history_snapshots,
            ),
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
        save_model_bundle(
            output_dir,
            config=config,
            dataset_metadata=metadata,
            checkpoint_path=checkpoint_path,
            model_version=model_version,
            fold_index=fold_index,
        )
        artifact = run_test(
            trainer,
            lightning_module,
            datamodule,
            checkpoint_path,
        )
        return CandidateFoldRun(
            artifact=artifact,
            dataset_metadata_path=state_path,
            checkpoint_path=str(Path(checkpoint_path).resolve()),
            predictions_path=predictions_path,
        )

    def run_baseline_candidate(
        self,
        *,
        package: DatasetPackage,
        config: BaselineRunConfig,
        fold_index: int,
        candidate_name: str,
    ) -> CandidateFoldRun:
        metadata = package.metadata
        output_dir = Path(self.output_root) / f"fold_{fold_index:03d}" / candidate_name
        output_dir.mkdir(parents=True, exist_ok=True)
        datamodule = LOBDataModule(
            package,
            fold_index=fold_index,
            loader=config.loader,
            seed=config.seed,
        )
        artifact = self._run_baseline(
            candidate_name=candidate_name,
            metadata=metadata,
            config=config.baselines,
            datamodule=datamodule,
            model_version=f"{config.experiment_id}-fold{fold_index}-{candidate_name}",
            fold_index=fold_index,
        )
        return CandidateFoldRun(
            artifact=artifact,
            dataset_metadata_path=str((package.root / "dataset.json").resolve()),
            predictions_path=str((output_dir / "predictions.parquet").resolve()),
        )

    @staticmethod
    def _run_baseline(
        *,
        candidate_name: str,
        metadata: DatasetPackageMetadata,
        config: BaselineConfig,
        datamodule: LOBDataModule,
        model_version: str,
        fold_index: int,
    ) -> PredictionArtifact:
        datamodule.setup("fit")
        runner = BaselineRunner(
            name=candidate_name,
            model=build_baseline(
                candidate_name,
                config,
                feature_columns=metadata.feature_columns,
                history_snapshots=metadata.history_snapshots,
            ),
            model_version=model_version,
            dataset_version=metadata.dataset_id,
            fold_index=fold_index,
        )
        runner.fit(lambda: (cast(LOBBatch, batch) for batch in datamodule.train_dataloader()))
        datamodule.teardown("fit")
        datamodule.setup("test")
        artifact = runner.predict(
            (cast(LOBBatch, batch) for batch in datamodule.test_dataloader()),
            split="test",
        )
        datamodule.teardown("test")
        return artifact


def build_checkpoint_callback(
    log_dir: str,
    *,
    monitor: str,
    mode: str,
    save_top_k: int = 1,
    filename: str = "best_val_model",
) -> Callback:
    """构建单 fold 最佳模型检查点回调。"""
    _validate_monitor_mode(mode)
    if not log_dir.strip():
        raise ValueError("log_dir must not be empty")
    if save_top_k < -1:
        raise ValueError("save_top_k must be >= -1")
    if not filename.strip():
        raise ValueError("filename must not be empty")
    Path(log_dir).expanduser().mkdir(parents=True, exist_ok=True)
    return ModelCheckpoint(
        dirpath=str(Path(log_dir).expanduser()),
        filename=filename,
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_weights_only=False,
        auto_insert_metric_name=False,
    )


def build_early_stopping_callback(
    *,
    monitor: str,
    mode: str,
    patience: int = 20,
    min_delta: float = 0.001,
    check_finite: bool = False,
) -> Callback:
    """构建单 fold 早停回调。"""
    _validate_monitor_mode(mode)
    if not monitor.strip():
        raise ValueError("monitor must not be empty")
    if patience < 0:
        raise ValueError("patience must be >= 0")
    if min_delta < 0:
        raise ValueError("min_delta must be >= 0")
    return EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        min_delta=min_delta,
        check_finite=check_finite,
    )


def build_trainer(
    log_dir: str,
    epochs: int,
    patience: int,
    callbacks: list[Callback] | None = None,
    logger: Logger | None = None,
    accelerator: str = "auto",
    devices: int | list[int] | str = 1,
    precision: str = "32-true",
    gradient_clip_val: float | None = None,
    **kwargs: Any,
) -> L.Trainer:
    """构建 executor 使用的 Lightning Trainer。"""
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if patience < 0:
        raise ValueError("patience must be >= 0")
    if gradient_clip_val is not None and gradient_clip_val < 0:
        raise ValueError("gradient_clip_val must be >= 0")
    configured_callbacks = list(callbacks or [])
    if not any(isinstance(callback, ModelCheckpoint) for callback in configured_callbacks):
        configured_callbacks.append(
            build_checkpoint_callback(log_dir, monitor="val/mse", mode="min")
        )
    if not any(isinstance(callback, EarlyStopping) for callback in configured_callbacks):
        configured_callbacks.append(
            build_early_stopping_callback(monitor="val/mse", mode="min", patience=patience)
        )
    trainer_kwargs: dict[str, Any] = {
        "default_root_dir": str(Path(log_dir).expanduser()),
        "max_epochs": epochs,
        "callbacks": configured_callbacks,
        "logger": logger if logger is not None else False,
        "accelerator": accelerator,
        "devices": devices,
        "precision": precision,
        "deterministic": True,
        **kwargs,
    }
    if gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = gradient_clip_val
    return L.Trainer(**trainer_kwargs)




def run_training(
    trainer: L.Trainer,
    lightning_module: L.LightningModule,
    datamodule: LOBDataModule,
    ckpt_path: str | None = None,
) -> None:
    """执行单 fold 模型训练。"""
    if ckpt_path is not None and not ckpt_path.strip():
        raise ValueError("ckpt_path must be None or a non-empty path")
    trainer.fit(model=lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)


def run_test(
    trainer: L.Trainer,
    lightning_module: L.LightningModule,
    datamodule: LOBDataModule,
    ckpt_path: str,
) -> PredictionArtifact:
    """恢复最佳检查点并返回 test split 的预测产物。"""
    if not ckpt_path.strip():
        raise ValueError("ckpt_path must not be empty")
    trainer.test(model=lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)
    artifact = getattr(lightning_module, "test_artifact", None)
    if not isinstance(artifact, PredictionArtifact):
        raise RuntimeError("test completed without a PredictionArtifact")
    if artifact.split != "test":
        raise RuntimeError("test artifact split must be 'test'")
    return artifact


def _validate_monitor_mode(mode: str) -> None:
    if mode not in {"min", "max"}:
        raise ValueError("mode must be 'min' or 'max'")
