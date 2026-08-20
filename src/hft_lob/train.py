# train.py - 编排层接口（精简版）
# 职责：组装零件 → 配置 Trainer → 执行训练/测试/预测
# 注意：具体功能实现（配置加载、日志构建、错误处理）来自 utils 模块

from __future__ import annotations

from typing import Any

import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger

from hft_lob.systems.artifact import PredictionArtifact
from hft_lob.systems.lob_data_module import LOBDataModule
from hft_lob.systems.metrics import EvaluationReport

__all__: list[str] = []

# ============================================================
# 从 utils 导入（消除重复定义）
# ============================================================

# ============================================================
# 1. 回调工厂接口
# ============================================================

def build_checkpoint_callback(
    log_dir: str,
    *,
    monitor: str,
    mode: str,
    save_top_k: int = 1,
    filename: str = "best_val_model",
) -> Callback:
    """
    构建模型检查点回调。

    Args:
        log_dir: 检查点保存目录。
        monitor: 监控的指标名称。
        mode: 指标优化方向（"max" / "min"）。
        save_top_k: 保存最佳模型的数量。
        filename: 检查点文件名前缀。

    Returns:
        ModelCheckpoint 实例。
    """
    raise NotImplementedError("train.build_checkpoint_callback not implemented")


def build_early_stopping_callback(
    *,
    monitor: str,
    mode: str,
    patience: int = 20,
    min_delta: float = 0.001,
    check_finite: bool = False,
) -> Callback:
    """
    构建早停回调。

    Args:
        monitor: 监控的指标名称。
        patience: 容忍轮数。
        min_delta: 最小改善阈值。
        mode: 指标优化方向（"max" / "min"）。
        check_finite: 是否检查指标有限性。

    Returns:
        EarlyStopping 实例。
    """
    raise NotImplementedError("train.build_early_stopping_callback not implemented")


# ============================================================
# 2. 训练器构建接口
# ============================================================

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
    """
    构建 Trainer 实例（所有工程配置在此集中管理）。

    Args:
        log_dir: 日志和检查点保存路径。
        epochs: 最大训练轮数。
        patience: 早停耐心值。
        callbacks: 额外的回调列表。
        logger: 日志器实例。
        accelerator: 硬件加速器。
        devices: 设备数量。
        precision: 训练精度。
        gradient_clip_val: 梯度裁剪阈值。
        **kwargs: 其他 Trainer 参数。

    Returns:
        配置好的 Trainer 实例。
    """
    raise NotImplementedError("train.build_trainer not implemented")


# ============================================================
# 3. 单 fold 执行接口
# ============================================================

def run_training(
    trainer: L.Trainer,
    lightning_module: L.LightningModule,
    datamodule: LOBDataModule,
    ckpt_path: str | None = None,
) -> None:
    """
    执行训练流程。

    Args:
        trainer: 已配置的 Trainer 实例。
        lightning_module: 训练模块。
        datamodule: 数据模块。
        ckpt_path: 恢复训练的检查点路径（可选）。
    """
    raise NotImplementedError("train.run_training not implemented")


def run_test(
    trainer: L.Trainer,
    lightning_module: L.LightningModule,
    datamodule: LOBDataModule,
    ckpt_path: str,
) -> EvaluationReport:
    """
    执行测试流程（加载最佳检查点后评估）。

    Args:
        trainer: 已配置的 Trainer 实例。
        lightning_module: 训练模块（用于提供模型架构）。
        datamodule: 数据模块。
        ckpt_path: 最佳检查点路径。

    Returns:
        统一结构化评估报告。
    """
    raise NotImplementedError("train.run_test not implemented")


def run_predict(
    trainer: L.Trainer,
    lightning_module: L.LightningModule,
    datamodule: LOBDataModule,
    ckpt_path: str,
    split: str = "test",
) -> PredictionArtifact:
    """
    执行预测流程（加载检查点后推理）。

    Args:
        trainer: 已配置的 Trainer 实例。
        lightning_module: 预测模块。
        datamodule: 数据模块。
        ckpt_path: 检查点路径。
        split: 当前预测所属 split。

    Returns:
        含完整 metadata/model/dataset/fold 的统一预测产物。
    """
    raise NotImplementedError("train.run_predict not implemented")
