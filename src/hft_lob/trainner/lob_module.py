"""LOBLightningModule（需求文档 §18/§20/§21/§28）：回归训练循环 + 评估 + artifact。"""

from __future__ import annotations

import lightning.pytorch as L
import numpy as np
import torch
from torch import nn

from hft_lob.configs.experiment import ModelRunConfig
from hft_lob.data_types import LOBBatch, SampleMeta
from hft_lob.metrics.metrics import evaluate
from hft_lob.reporting.artifact import PredictionArtifact
from hft_lob.trainner.losses import build_loss


class LOBLightningModule(L.LightningModule):
    """多标签回归训练循环；模型和 batch 均使用 ``[B,L]``。"""

    def __init__(
        self,
        model: nn.Module,
        config: ModelRunConfig,
        *,
        dataset_version: str | None = None,
        model_version: str = "unknown",
        fold_index: int | None = None,
        target_count: int = 1,
        labels: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if target_count <= 0:
            raise ValueError("target_count must be positive")
        if labels is not None and len(labels) != target_count:
            raise ValueError("labels length must equal target_count")
        self.save_hyperparameters(
            {
                "dataset_version": dataset_version,
                "model_version": model_version,
                "fold_index": fold_index,
                "target_count": target_count,
                "labels": labels,
            }
        )
        self.model = model
        self.config = config
        self.dataset_version = dataset_version
        self.model_version = model_version
        self.fold_index = fold_index
        self.target_count = target_count
        self.labels = (
            tuple(labels)
            if labels is not None
            else ((1,) if target_count == 1 else tuple(range(1, target_count + 1)))
        )
        self.loss_fn = build_loss(
            config.training.loss,
            huber_delta=config.training.loss_huber_delta,
        )
        self._validation_predictions: list[torch.Tensor] = []
        self._validation_targets: list[torch.Tensor] = []
        self._test_predictions: list[torch.Tensor] = []
        self._test_targets: list[torch.Tensor] = []
        self._test_metadata: list[SampleMeta] = []
        self.test_artifact: PredictionArtifact | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：委托内部模型并校验 ``[B,L]`` 输出。"""
        if x.ndim != 3:
            raise ValueError(f"model input must have shape [B,T,F], got {tuple(x.shape)}")
        predictions = self.model(x)
        if not isinstance(predictions, torch.Tensor):
            raise TypeError("model.forward must return a torch.Tensor")
        if predictions.shape != (x.shape[0], self.target_count):
            raise ValueError(
                f"model output must have shape [B,{self.target_count}], "
                f"got {tuple(predictions.shape)}"
            )
        return predictions


    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(predictions, targets)
        if not isinstance(loss, torch.Tensor):
            raise TypeError("loss function must return a torch.Tensor")
        return loss
    def transfer_batch_to_device(
        self,
        batch: LOBBatch,
        device: torch.device,
        dataloader_idx: int,
    ) -> LOBBatch:
        return LOBBatch(
            features=batch.features.to(device, non_blocking=True),
            targets=batch.targets.to(device, non_blocking=True),
            metadata=batch.metadata,
        )
    def training_step(self, batch: LOBBatch, batch_idx: int) -> torch.Tensor:
        predictions, targets = self._shared_step(batch)
        loss = self._compute_loss(predictions, targets)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=targets.shape[0],
        )
        return loss

    def validation_step(self, batch: LOBBatch, batch_idx: int) -> torch.Tensor:
        predictions, targets = self._shared_step(batch)
        loss = self._compute_loss(predictions, targets)
        self._validation_predictions.append(predictions.detach().cpu())
        self._validation_targets.append(targets.detach().cpu())
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=targets.shape[0],
            sync_dist=True,
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self._validation_predictions:
            return
        predictions = torch.cat(self._validation_predictions).numpy()
        targets = torch.cat(self._validation_targets).numpy()
        metric_values = [
            evaluate(predictions[:, position], targets[:, position])
            for position in range(self.target_count)
        ]
        for name in ("mse", "mae"):
            value = float(np.mean([metrics[name] for metrics in metric_values]))
            self.log(
                f"val/{name}",
                torch.tensor(value, dtype=torch.float32, device=self.device),
                on_step=False,
                on_epoch=True,
                prog_bar=name == "mse",
                sync_dist=True,
            )
        self._validation_predictions.clear()
        self._validation_targets.clear()

    def test_step(self, batch: LOBBatch, batch_idx: int) -> torch.Tensor:
        predictions, targets = self._shared_step(batch)
        loss = self._compute_loss(predictions, targets)
        self._test_predictions.append(predictions.detach().cpu())
        self._test_targets.append(targets.detach().cpu())
        self._test_metadata.extend(batch.metadata)
        self.log("test/loss", loss, batch_size=targets.shape[0], sync_dist=True)
        return loss

    def on_test_epoch_start(self) -> None:
        self._test_predictions.clear()
        self._test_targets.clear()
        self._test_metadata.clear()
        self.test_artifact = None

    def on_test_epoch_end(self) -> None:
        if not self._test_predictions:
            raise RuntimeError("test completed without any batches")
        self.test_artifact = self._make_artifact(
            predictions=torch.cat(self._test_predictions),
            targets=torch.cat(self._test_targets),
            metadata=tuple(self._test_metadata),
            split="test",
        )
        self._test_predictions.clear()
        self._test_targets.clear()
        self._test_metadata.clear()
    def configure_optimizers(self) -> torch.optim.Optimizer:
        training = self.config.training
        if training.learning_rate <= 0 or training.weight_decay < 0:
            raise ValueError("learning_rate must be > 0 and weight_decay must be >= 0")
        return torch.optim.AdamW(
            self.parameters(),
            lr=training.learning_rate,
            betas=training.betas,
            weight_decay=training.weight_decay,
        )

    def _shared_step(self, batch: LOBBatch) -> tuple[torch.Tensor, torch.Tensor]:
        if batch.features.ndim != 3:
            raise ValueError("batch features must have shape [B,T,F]")
        targets = batch.targets
        expected_shape = (batch.features.shape[0], self.target_count)
        if targets.shape != expected_shape:
            raise ValueError(f"batch targets must have shape {expected_shape}")
        if len(batch.metadata) != batch.features.shape[0]:
            raise ValueError("batch metadata count must match batch size")
        if not torch.isfinite(batch.features).all() or not torch.isfinite(targets).all():
            raise ValueError("batch features and targets must be finite")
        return self(batch.features), targets

    def _make_artifact(
        self,
        *,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: tuple[SampleMeta, ...],
        split: str,
    ) -> PredictionArtifact:
        if self.dataset_version is None or not self.dataset_version.strip():
            raise RuntimeError("dataset_version is required to generate prediction artifacts")
        if self.fold_index is None or self.fold_index <= 0:
            raise RuntimeError("positive fold_index is required to generate prediction artifacts")
        return PredictionArtifact(
            predictions=np.asarray(predictions.detach().cpu(), dtype=np.float64),
            targets=np.asarray(targets.detach().cpu(), dtype=np.float64),
            labels=self.labels,
            metadata=metadata,
            model_name=self.config.model.name,
            model_version=self.model_version,
            dataset_version=self.dataset_version,
            fold_index=self.fold_index,
            split=split,
        )
