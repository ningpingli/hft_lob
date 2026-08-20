"""非梯度 baseline 的统一运行适配器。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hft_lob.baselines.base import BaselineModel
from hft_lob.datasets.lob_dataset import LOBBatch, SampleMeta
from hft_lob.systems.artifact import PredictionArtifact


@dataclass
class BaselineRunner:
    """让 Zero/Imbalance/Ridge 与神经模型共享 PredictionArtifact 出口。"""

    name: str
    model: BaselineModel
    model_version: str
    dataset_version: str
    fold_index: int

    def fit(self, batches: tuple[LOBBatch, ...]) -> None:
        """仅使用当前 fold 的 training batches 拟合。"""
        features, targets, _ = _merge_batches(batches)
        self.model.fit(features, targets)

    def predict(self, batches: tuple[LOBBatch, ...], *, split: str) -> PredictionArtifact:
        """生成与神经模型相同的预测产物。"""
        if not split.strip():
            raise ValueError("split must not be empty")
        features, targets, metadata = _merge_batches(batches)
        with torch.no_grad():
            predictions = self.model.forward(features)
        if predictions.shape != (features.shape[0], 1):
            raise ValueError(
                f"baseline prediction must have shape [B,1], got {tuple(predictions.shape)}"
            )
        return PredictionArtifact(
            predictions=predictions[:, 0].detach().cpu().numpy(),
            targets=targets[:, 0].detach().cpu().numpy(),
            metadata=metadata,
            model_name=self.name,
            model_version=self.model_version,
            dataset_version=self.dataset_version,
            fold_index=self.fold_index,
            split=split,
        )


def _merge_batches(
    batches: tuple[LOBBatch, ...],
) -> tuple[torch.Tensor, torch.Tensor, tuple[SampleMeta, ...]]:
    if not batches:
        raise ValueError("batches must not be empty")
    features = torch.cat([batch.features for batch in batches], dim=0).detach().cpu()
    targets = torch.cat([batch.targets for batch in batches], dim=0).detach().cpu()
    metadata = tuple(meta for batch in batches for meta in batch.metadata)
    if features.ndim != 3 or targets.ndim != 2 or targets.shape[1] != 1:
        raise ValueError("batches must follow [B,T,F] features and [B,1] targets")
    if features.shape[0] != targets.shape[0] or features.shape[0] != len(metadata):
        raise ValueError("batch features, targets and metadata counts must match")
    if not torch.isfinite(features).all() or not torch.isfinite(targets).all():
        raise ValueError("batch features and targets must be finite")
    return features, targets, metadata
