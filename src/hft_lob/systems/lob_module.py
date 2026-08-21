"""LOBLightningModule（需求文档 §18/§20/§21/§28）：回归训练循环 + 评估 + artifact。"""

from __future__ import annotations

from typing import cast

import lightning.pytorch as L
import numpy as np
import torch
from torch import nn

from hft_lob.configs.experiment import ModelRunConfig
from hft_lob.datasets.contracts import LOBBatch, SampleMeta
from hft_lob.systems.artifact import PredictionArtifact
from hft_lob.systems.losses import build_loss
from hft_lob.systems.metrics import evaluate


class LOBLightningModule(L.LightningModule):
    """回归任务的 LightningModule：模型包装 + 损失 + 指标 + prediction artifact。

    契约：
    - 模型统一 ``forward(x) -> [B, 1]``（§18），最后一层 Linear(hidden, 1)，
      无 softmax/sigmoid；
    - 损失：Huber 默认（§20，``systems.losses.build_loss``）；
    - 评估：TS-IC / RankIC / MAE / RMSE / Direction（§21，
      ``systems.metrics``）+ 日级稳定性（§14 序列相关处理）+ prediction
      artifact parquet（§28，``systems.artifact``）。
    """

    def __init__(
        self,
        model: nn.Module,
        config: ModelRunConfig,
        *,
        dataset_version: str | None = None,
        model_version: str = "unknown",
        fold_index: int | None = None,
    ) -> None:
        """初始化 LightningModule。

        Args:
            model: 待训练的模型（forward(x) -> [B, 1]）。
            config: 实验配置根（training/evaluation 段）。
            dataset_version: 数据集版本标识（§31）。
            model_version: 模型版本标识（§29）。
            fold_index: 当前 walk-forward fold 编号；生成 artifact 时必须提供。
        """
        super().__init__()
        # 不把 ModelRunConfig dataclass pickle 进 checkpoint；PyTorch 2.6 的
        # weights_only 安全加载只接受张量和基础类型。配置已由实验目录单独备份。
        self.save_hyperparameters(
            {
                "dataset_version": dataset_version,
                "model_version": model_version,
                "fold_index": fold_index,
            }
        )
        self.model = model
        self.config = config
        self.dataset_version = dataset_version
        self.model_version = model_version
        self.fold_index = fold_index
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
        """前向：委托内部模型（§18 契约）。"""
        if x.ndim != 3:
            raise ValueError(f"model input must have shape [B,T,F], got {tuple(x.shape)}")
        predictions = self.model(x)
        if not isinstance(predictions, torch.Tensor):
            raise TypeError("model.forward must return a torch.Tensor")
        if predictions.shape != (x.shape[0], 1):
            raise ValueError(
                f"model output must have shape [B,1], got {tuple(predictions.shape)}"
            )
        return predictions

    def transfer_batch_to_device(
        self,
        batch: LOBBatch,
        device: torch.device,
        dataloader_idx: int,
    ) -> LOBBatch:
        """迁移 frozen LOBBatch 中的张量，metadata 保持在 CPU/Python 侧。"""
        return LOBBatch(
            features=batch.features.to(device, non_blocking=True),
            targets=batch.targets.to(device, non_blocking=True),
            metadata=batch.metadata,
        )

    def training_step(self, batch: LOBBatch, batch_idx: int) -> torch.Tensor:
        """训练步：回归损失（§20）。"""
        predictions, targets = self._shared_step(batch)
        loss = cast(torch.Tensor, self.loss_fn(predictions, targets))
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
        """验证步：累积 preds/targets（epoch 端统一计算指标）。"""
        predictions, targets = self._shared_step(batch)
        loss = cast(torch.Tensor, self.loss_fn(predictions, targets))
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
        """验证期结束：整 epoch 计算指标，并以 ``val/<metric>`` 记录。

        TS-IC 的稳定 key 为 ``val/ts_ic``，供 checkpoint 与 early stopping 使用。
        """
        if not self._validation_predictions:
            return
        predictions = torch.cat(self._validation_predictions)[:, 0].numpy()
        targets = torch.cat(self._validation_targets)[:, 0].numpy()
        metrics = evaluate(predictions, targets)
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

    def test_step(self, batch: LOBBatch, batch_idx: int) -> torch.Tensor:
        """测试步：记录 loss，并累积生成统一 artifact 所需的数据。"""
        predictions, targets = self._shared_step(batch)
        loss = cast(torch.Tensor, self.loss_fn(predictions, targets))
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
        """AdamW（参数来自 training 段；无 scheduler）。"""
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
        if targets.ndim != 2 or targets.shape != (batch.features.shape[0], 1):
            raise ValueError("batch targets must have shape [B,1]")
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
            predictions=np.asarray(predictions[:, 0].detach().cpu(), dtype=np.float64),
            targets=np.asarray(targets[:, 0].detach().cpu(), dtype=np.float64),
            metadata=metadata,
            model_name=self.config.model.name,
            model_version=self.model_version,
            dataset_version=self.dataset_version,
            fold_index=self.fold_index,
            split=split,
        )
