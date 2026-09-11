"""LOBLightningModule：验证阶段轻量聚合，测试阶段生成完整预测 artifact。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import lightning.pytorch as L
import numpy as np
import torch
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch import nn

from hft_lob.configs.experiment import ModelRunConfig
from hft_lob.data_types import LOBBatch, SampleMeta
from hft_lob.metrics.metrics import VALIDATION_METRIC_NAMES, daily_ic_records, mean_daily_ic
from hft_lob.models.TABL.bin_tabl import enforce_weight_constraints
from hft_lob.reporting.artifact import PredictionArtifact
from hft_lob.trainner.losses import build_loss


@dataclass(frozen=True)
class _LearningRateSchedule:
    """Step-based linear warmup followed by cosine decay."""

    total_steps: int
    warmup_steps: int
    min_lr_ratio: float
    start_lr_ratio: float = 0.1

    def multiplier(self, step: int) -> float:
        if self.total_steps <= 1:
            return self.min_lr_ratio
        if self.warmup_steps > 0 and step < self.warmup_steps:
            progress = step / self.warmup_steps
            return self.start_lr_ratio + (1.0 - self.start_lr_ratio) * progress
        decay_steps = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, (step - self.warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * float(cosine)


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
        self._validation_mse_sum: torch.Tensor | None = None
        self._validation_mae_sum: torch.Tensor | None = None
        self._validation_element_count = 0
        self._validation_predictions: list[torch.Tensor] = []
        self._validation_targets: list[torch.Tensor] = []
        self._validation_trade_dates: list[str] = []
        self._validation_series_ids: list[str] = []
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

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer | LightningOptimizer,
        optimizer_closure: Callable[[], Any] | None = None,
    ) -> None:
        """每个优化器步之后施加 TABL 权重范数约束。

        约束是权重的性质、与输入无关（旧实现放在 ``BiN_BTABL``/``BiN_CTABL`` 的 forward 内，
        每次前向 5-7 次 ``matrix_norm`` 检查且每次触发 device→CPU 同步）。放在更新之后：
        权重在两次前向之间不变，前向看到的取值与旧实现一致，同时前向热路径不再有同步
        （binctabl 单样本延迟 −28%，见 issue #16）。
        """
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        enforce_weight_constraints(self.model)

    def on_validation_epoch_start(self) -> None:
        """开始验证，初始化误差与日级 IC 的聚合器。"""
        self._reset_validation_metrics()

    def validation_step(self, batch: LOBBatch, batch_idx: int) -> torch.Tensor:
        predictions, targets = self._shared_step(batch)
        loss = self._compute_loss(predictions, targets)
        errors = predictions.detach() - targets.detach()
        self._validation_mse_sum = (
            errors.square().sum()
            if self._validation_mse_sum is None
            else self._validation_mse_sum + errors.square().sum()
        )
        self._validation_mae_sum = (
            errors.abs().sum()
            if self._validation_mae_sum is None
            else self._validation_mae_sum + errors.abs().sum()
        )
        self._validation_element_count += errors.numel()
        self._validation_predictions.append(predictions.detach().cpu())
        self._validation_targets.append(targets.detach().cpu())
        self._validation_trade_dates.extend(meta.trade_date for meta in batch.metadata)
        self._validation_series_ids.extend(meta.ticker for meta in batch.metadata)
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
        """记录验证误差与 mean daily IC，不生成测试报告或曲线数据。"""
        if self._validation_element_count == 0:
            return
        if (
            self._validation_mse_sum is None
            or self._validation_mae_sum is None
            or not self._validation_predictions
            or not self._validation_targets
        ):
            raise RuntimeError("validation metric accumulators are incomplete")

        predictions = torch.cat(self._validation_predictions).numpy()
        targets = torch.cat(self._validation_targets).numpy()
        trade_dates = np.asarray(self._validation_trade_dates, dtype=object)
        series_ids = np.asarray(self._validation_series_ids, dtype=object)
        daily_ic = daily_ic_records(
            predictions.T.reshape(-1),
            targets.T.reshape(-1),
            np.tile(trade_dates, self.target_count),
            np.tile(series_ids, self.target_count),
        )
        daily_values = np.asarray([record.ic for record in daily_ic], dtype=np.float64)
        metrics = {
            "mse": self._validation_mse_sum / self._validation_element_count,
            "mae": self._validation_mae_sum / self._validation_element_count,
            "mean_daily_ic": self._validation_mse_sum.new_tensor(mean_daily_ic(daily_values)),
        }
        for name in VALIDATION_METRIC_NAMES:
            self.log(
                f"val/{name}",
                metrics[name],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        self.log(
            "val/mean_daily_ic",
            metrics["mean_daily_ic"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self._reset_validation_metrics()

    def _reset_validation_metrics(self) -> None:
        self._validation_mse_sum = None
        self._validation_mae_sum = None
        self._validation_element_count = 0
        self._validation_predictions.clear()
        self._validation_targets.clear()
        self._validation_trade_dates.clear()
        self._validation_series_ids.clear()

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
    def configure_optimizers(self) -> Any:
        training = self.config.training
        if (
            training.learning_rate <= 0
            or training.min_learning_rate <= 0
            or training.min_learning_rate > training.learning_rate
            or training.weight_decay < 0
            or not 0 <= training.warmup_ratio < 1
        ):
            raise ValueError("invalid optimizer stability configuration")
        if training.scheduler != "cosine":
            raise ValueError("training.scheduler must be 'cosine'")

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=training.learning_rate,
            betas=training.betas,
            weight_decay=training.weight_decay,
        )
        try:
            total_steps = int(self.trainer.estimated_stepping_batches)
        except (RuntimeError, TypeError, ValueError, OverflowError):
            total_steps = training.epochs
        total_steps = max(1, total_steps)
        warmup_steps = min(
            max(0, total_steps - 1),
            int(round(total_steps * training.warmup_ratio)),
        )
        schedule = _LearningRateSchedule(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=training.min_learning_rate / training.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=schedule.multiplier,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

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
