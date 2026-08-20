"""LOBLightningModule（需求文档 §18/§20/§21/§28）：回归训练循环 + 评估 + artifact。"""

from __future__ import annotations

import lightning.pytorch as L
import torch
from torch import nn

from hft_lob.configs.experiment import ExperimentConfig
from hft_lob.datasets.lob_dataset import LOBBatch
from hft_lob.systems.artifact import PredictionArtifact


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
        config: ExperimentConfig,
        *,
        artifact_dir: str | None = None,
        dataset_version: str | None = None,
        model_version: str = "unknown",
    ) -> None:
        """初始化 LightningModule。

        Args:
            model: 待训练的模型（forward(x) -> [B, 1]）。
            config: 实验配置根（training/evaluation 段）。
            artifact_dir: 预测产物落盘目录（None 不落盘）。
            dataset_version: 数据集版本标识（§31）。
            model_version: 模型版本标识（§29）。
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：委托内部模型（§18 契约）。"""
        raise NotImplementedError("LOBLightningModule.forward not implemented")

    def training_step(self, batch: LOBBatch, batch_idx: int) -> torch.Tensor:
        """训练步：回归损失（§20）。"""
        raise NotImplementedError("LOBLightningModule.training_step not implemented")

    def validation_step(self, batch: LOBBatch, batch_idx: int) -> torch.Tensor:
        """验证步：累积 preds/targets（epoch 端统一计算指标）。"""
        raise NotImplementedError("LOBLightningModule.validation_step not implemented")

    def test_step(self, batch: LOBBatch, batch_idx: int) -> torch.Tensor:
        """测试步：累积 preds/targets/meta（供 artifact 与日级指标，§28）。"""
        raise NotImplementedError("LOBLightningModule.test_step not implemented")

    def on_validation_epoch_end(self) -> None:
        """验证期结束：整 epoch 计算指标，并以 ``val/<metric>`` 记录。

        TS-IC 的稳定 key 为 ``val/ts_ic``，供 checkpoint 与 early stopping 使用。
        """
        raise NotImplementedError("LOBLightningModule.on_validation_epoch_end not implemented")

    def on_test_epoch_end(self) -> None:
        """测试期结束：只汇集预测记录；评估与落盘由外部统一处理。"""
        raise NotImplementedError("LOBLightningModule.on_test_epoch_end not implemented")

    def predict_step(self, batch: LOBBatch, batch_idx: int) -> PredictionArtifact:
        """生成带完整 metadata 的 batch 级预测产物。"""
        raise NotImplementedError("LOBLightningModule.predict_step not implemented")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """AdamW（参数来自 training 段；无 scheduler）。"""
        raise NotImplementedError("LOBLightningModule.configure_optimizers not implemented")
