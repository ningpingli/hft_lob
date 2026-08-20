"""非梯度 baseline 的统一运行适配器。"""

from __future__ import annotations

from dataclasses import dataclass

from hft_lob.baselines.base import BaselineModel
from hft_lob.datasets.lob_dataset import LOBBatch
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
        raise NotImplementedError("BaselineRunner.fit not implemented")

    def predict(self, batches: tuple[LOBBatch, ...], *, split: str) -> PredictionArtifact:
        """生成与神经模型相同的预测产物。"""
        raise NotImplementedError("BaselineRunner.predict not implemented")
