"""非梯度 baseline 的流式训练与预测适配器。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import torch

from hft_lob.baselines.base import BaselineModel
from hft_lob.datasets.contracts import LOBBatch, SampleMeta
from hft_lob.systems.artifact import PredictionArtifact


@dataclass
class BaselineRunner:
    name: str
    model: BaselineModel
    model_version: str
    dataset_version: str
    fold_index: int

    def fit(self, batches: Callable[[], Iterable[LOBBatch]]) -> None:
        """流式拟合；模型可为多个 epoch 重建 DataLoader 迭代器。"""
        self.model.fit_batches(
            lambda: ((batch.features, batch.targets) for batch in batches())
        )

    def predict(self, batches: Iterable[LOBBatch], *, split: str) -> PredictionArtifact:
        """逐 batch 推理，仅累积最终预测、目标和必要 metadata。"""
        if not split.strip():
            raise ValueError("split must not be empty")
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        metadata: list[SampleMeta] = []
        with torch.no_grad():
            for batch in batches:
                _validate_batch(batch)
                output = self.model.forward(batch.features)
                if output.shape != (batch.features.shape[0], 1):
                    raise ValueError(
                        f"baseline prediction must have shape [B,1], got {tuple(output.shape)}"
                    )
                predictions.append(output[:, 0].detach().cpu().numpy())
                targets.append(batch.targets[:, 0].detach().cpu().numpy())
                metadata.extend(batch.metadata)
        if not predictions:
            raise ValueError("prediction batches must not be empty")
        return PredictionArtifact(
            predictions=np.concatenate(predictions),
            targets=np.concatenate(targets),
            metadata=tuple(metadata),
            model_name=self.name,
            model_version=self.model_version,
            dataset_version=self.dataset_version,
            fold_index=self.fold_index,
            split=split,
        )


def _validate_batch(batch: LOBBatch) -> None:
    if batch.features.ndim != 3 or batch.targets.ndim != 2 or batch.targets.shape[1] != 1:
        raise ValueError("batches must follow [B,T,F] features and [B,1] targets")
    if batch.features.shape[0] != batch.targets.shape[0] or batch.features.shape[0] != len(batch.metadata):
        raise ValueError("batch features, targets and metadata counts must match")
    if not torch.isfinite(batch.features).all() or not torch.isfinite(batch.targets).all():
        raise ValueError("batch features and targets must be finite")
