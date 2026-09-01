"""LOBLightningModule：完整 label 向量训练、评估与 prediction artifact。"""

from __future__ import annotations

import lightning.pytorch as L
import numpy as np
import torch
from torch import nn

from hft_lob.configs.experiment import ModelRunConfig
from hft_lob.systems.artifact import PredictionArtifact
from hft_lob.systems.contracts import LOBBatch, SampleMeta
from hft_lob.systems.losses import build_loss, masked_loss
from hft_lob.systems.metrics import evaluate


class LOBLightningModule(L.LightningModule):
    """统一多任务回归训练：模型输出和 target 均为 ``[B, L]``。"""

    def __init__(
        self,
        model: nn.Module,
        config: ModelRunConfig,
        *,
        target_count: int | None = None,
        labels: tuple[int, ...] | None = None,
        dataset_version: str | None = None,
        model_version: str = "unknown",
        fold_index: int | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(
            {
                "dataset_version": dataset_version,
                "model_version": model_version,
                "fold_index": fold_index,
            }
        )
        self.output_dim = config.model.output_dim if target_count is None else target_count
        self.labels = tuple(range(1, self.output_dim + 1)) if labels is None else tuple(labels)
        if (
            not self.labels
            or len(set(self.labels)) != len(self.labels)
            or any(label <= 0 for label in self.labels)
        ):
            raise ValueError("labels must be non-empty, unique, and positive")
        if len(self.labels) != self.output_dim:
            raise ValueError("labels must match target_count")
        if self.output_dim <= 0:
            raise ValueError("target_count must be > 0")
        self.dataset_version = dataset_version
        self.model_version = model_version
        self.fold_index = fold_index
        self.loss_fn = build_loss(
            config.training.loss,
            huber_delta=config.training.loss_huber_delta,
        )
        self._validation_predictions: list[torch.Tensor] = []
        self._validation_targets: list[torch.Tensor] = []
        self._validation_validity: list[torch.Tensor] = []
        self._test_predictions: list[torch.Tensor] = []
        self._test_targets: list[torch.Tensor] = []
        self._test_validity: list[torch.Tensor] = []
        self._test_metadata: list[SampleMeta] = []
        self.test_artifact: PredictionArtifact | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"model input must have shape [B,T,F], got {tuple(x.shape)}")
        predictions = self.model(x)
        if not isinstance(predictions, torch.Tensor):
            raise TypeError("model.forward must return a torch.Tensor")
        if predictions.shape != (x.shape[0], self.output_dim):
            raise ValueError(
                f"model output must have shape [B,{self.output_dim}], got {tuple(predictions.shape)}"
            )
        return predictions

    def transfer_batch_to_device(
        self,
        batch: LOBBatch,
        device: torch.device,
        dataloader_idx: int,
    ) -> LOBBatch:
        return LOBBatch(
            features=batch.features.to(device, non_blocking=True),
            targets=batch.targets.to(device, non_blocking=True),
            target_valid=batch.target_valid.to(device, non_blocking=True),
            metadata=batch.metadata,
        )

    def training_step(self, batch: LOBBatch, batch_idx: int) -> torch.Tensor:
        predictions, targets, target_valid = self._shared_step(batch)
        loss = masked_loss(self.loss_fn, predictions, targets, target_valid)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=targets.shape[0])
        return loss

    def validation_step(self, batch: LOBBatch, batch_idx: int) -> torch.Tensor:
        predictions, targets, target_valid = self._shared_step(batch)
        loss = masked_loss(self.loss_fn, predictions, targets, target_valid)
        self._validation_predictions.append(predictions.detach().cpu())
        self._validation_targets.append(targets.detach().cpu())
        self._validation_validity.append(target_valid.detach().cpu())
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=targets.shape[0], sync_dist=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self._validation_predictions:
            return
        predictions = torch.cat(self._validation_predictions)
        targets = torch.cat(self._validation_targets)
        target_valid = torch.cat(self._validation_validity)
        metrics = _evaluate_valid(predictions, targets, target_valid)
        for name in self.config.evaluation.metrics:
            if name not in metrics:
                raise ValueError(f"unsupported validation metric: {name!r}")
            self.log(
                f"val/{name}",
                torch.tensor(metrics[name], dtype=torch.float32, device=self.device),
                on_step=False,
                on_epoch=True,
                prog_bar=name == "ts_ic",
                sync_dist=True,
            )
        self._validation_predictions.clear()
        self._validation_targets.clear()
        self._validation_validity.clear()

    def test_step(self, batch: LOBBatch, batch_idx: int) -> torch.Tensor:
        predictions, targets, target_valid = self._shared_step(batch)
        loss = masked_loss(self.loss_fn, predictions, targets, target_valid)
        self._test_predictions.append(predictions.detach().cpu())
        self._test_targets.append(targets.detach().cpu())
        self._test_validity.append(target_valid.detach().cpu())
        self._test_metadata.extend(batch.metadata)
        self.log("test/loss", loss, batch_size=targets.shape[0], sync_dist=True)
        return loss

    def on_test_epoch_start(self) -> None:
        self._test_predictions.clear()
        self._test_targets.clear()
        self._test_validity.clear()
        self._test_metadata.clear()
        self.test_artifact = None

    def on_test_epoch_end(self) -> None:
        if not self._test_predictions:
            raise RuntimeError("test completed without any batches")
        self.test_artifact = self._make_artifact(
            predictions=torch.cat(self._test_predictions),
            targets=torch.cat(self._test_targets),
            target_valid=torch.cat(self._test_validity),
            metadata=tuple(self._test_metadata),
            split="test",
        )
        self._test_predictions.clear()
        self._test_targets.clear()
        self._test_validity.clear()
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

    def _shared_step(
        self, batch: LOBBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch.features.ndim != 3:
            raise ValueError("batch features must have shape [B,T,F]")
        targets = batch.targets
        target_valid = batch.target_valid
        expected = (batch.features.shape[0], self.output_dim)
        if targets.shape != expected or target_valid.shape != expected:
            raise ValueError(f"batch targets and target_valid must have shape {expected}")
        if target_valid.dtype is not torch.bool:
            raise TypeError("batch target_valid must have dtype torch.bool")
        if len(batch.metadata) != batch.features.shape[0]:
            raise ValueError("batch metadata count must match batch size")
        if not torch.isfinite(batch.features).all():
            raise ValueError("batch features must be finite")
        if target_valid.any() and not torch.isfinite(targets[target_valid]).all():
            raise ValueError("valid batch targets must be finite")
        return self(batch.features), targets, target_valid

    def _make_artifact(
        self,
        *,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_valid: torch.Tensor,
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
            target_valid=np.asarray(target_valid.detach().cpu(), dtype=np.bool_),
            labels=self.labels,
            metadata=metadata,
            model_name=self.config.model.name,
            model_version=self.model_version,
            dataset_version=self.dataset_version,
            fold_index=self.fold_index,
            split=split,
        )


def _evaluate_valid(
    predictions: torch.Tensor, targets: torch.Tensor, target_valid: torch.Tensor
) -> dict[str, float]:
    valid = target_valid
    if not valid.any():
        return evaluate(np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64))
    return evaluate(
        np.asarray(predictions[valid].detach().cpu(), dtype=np.float64),
        np.asarray(targets[valid].detach().cpu(), dtype=np.float64),
    )
