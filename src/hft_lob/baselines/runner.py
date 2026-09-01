"""非梯度 baseline 的流式训练与矩阵预测适配器。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import torch

from hft_lob.baselines.base import BaselineModel
from hft_lob.systems.artifact import PredictionArtifact
from hft_lob.systems.contracts import LOBBatch, SampleMeta


@dataclass
class BaselineRunner:
    name: str
    model: BaselineModel
    model_version: str
    dataset_version: str
    fold_index: int
    labels: tuple[int, ...] = (60,)

    def fit(self, batches: Callable[[], Iterable[LOBBatch]]) -> None:
        self.model.fit_batches(lambda: ((batch.features, batch.targets[:, :1]) for batch in batches()))

    def predict(self, batches: Iterable[LOBBatch], *, split: str) -> PredictionArtifact:
        """逐 batch 推理，累积完整 prediction、target、validity 矩阵。"""
        if not split.strip():
            raise ValueError("split must not be empty")
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        target_valid: list[np.ndarray] = []
        metadata: list[SampleMeta] = []
        with torch.no_grad():
            for batch in batches:
                _validate_batch(batch)
                output = self.model.forward(batch.features)
                if output.shape == (batch.features.shape[0], 1) and batch.targets.shape[1] > 1:
                    output = output.expand(-1, batch.targets.shape[1])
                if output.shape != batch.targets.shape:
                    raise ValueError(
                        f"baseline prediction must have shape {tuple(batch.targets.shape)}, got {tuple(output.shape)}"
                    )
                predictions.append(output.detach().cpu().numpy())
                targets.append(batch.targets.detach().cpu().numpy())
                target_valid.append(batch.target_valid.detach().cpu().numpy())
                metadata.extend(batch.metadata)
        if not predictions:
            raise ValueError("prediction batches must not be empty")
        return PredictionArtifact(
            predictions=np.concatenate(predictions),
            targets=np.concatenate(targets),
            target_valid=np.concatenate(target_valid),
            labels=self.labels,
            metadata=tuple(metadata),
            model_name=self.name,
            model_version=self.model_version,
            dataset_version=self.dataset_version,
            fold_index=self.fold_index,
            split=split,
        )


def _validate_batch(batch: LOBBatch) -> None:
    if (
        batch.features.ndim != 3
        or batch.targets.ndim != 2
        or batch.targets.shape[1] == 0
        or batch.target_valid.shape != batch.targets.shape
        or batch.target_valid.dtype is not torch.bool
    ):
        raise ValueError("batches must follow [B,T,F] features and [B,L] targets/validity")
    if batch.features.shape[0] != batch.targets.shape[0] or batch.features.shape[0] != len(
        batch.metadata
    ):
        raise ValueError("batch features, targets and metadata counts must match")
    if not torch.isfinite(batch.features).all():
        raise ValueError("batch features must be finite")
    if batch.target_valid.any() and not torch.isfinite(batch.targets[batch.target_valid]).all():
        raise ValueError("valid batch targets must be finite")
