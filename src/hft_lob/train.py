# train.py - 编排层接口（精简版）
# 职责：组装零件 → 配置 Trainer → 执行训练/测试/预测
# 注意：具体功能实现（配置加载、日志构建、错误处理）来自 utils 模块

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import Logger

from hft_lob.systems.artifact import PredictionArtifact
from hft_lob.systems.lob_data_module import LOBDataModule

__all__ = [
    "build_checkpoint_callback",
    "build_early_stopping_callback",
    "build_trainer",
    "run_test",
    "run_training",
]

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
    _validate_monitor_mode(mode)
    if not log_dir.strip():
        raise ValueError("log_dir must not be empty")
    if save_top_k < -1:
        raise ValueError("save_top_k must be >= -1")
    if not filename.strip():
        raise ValueError("filename must not be empty")
    Path(log_dir).expanduser().mkdir(parents=True, exist_ok=True)
    return ModelCheckpoint(
        dirpath=str(Path(log_dir).expanduser()),
        filename=filename,
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_weights_only=False,
        auto_insert_metric_name=False,
    )


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
    _validate_monitor_mode(mode)
    if not monitor.strip():
        raise ValueError("monitor must not be empty")
    if patience < 0:
        raise ValueError("patience must be >= 0")
    if min_delta < 0:
        raise ValueError("min_delta must be >= 0")
    return EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        min_delta=min_delta,
        check_finite=check_finite,
    )


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
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if patience < 0:
        raise ValueError("patience must be >= 0")
    if gradient_clip_val is not None and gradient_clip_val < 0:
        raise ValueError("gradient_clip_val must be >= 0")

    configured_callbacks = list(callbacks or [])
    if not any(isinstance(callback, ModelCheckpoint) for callback in configured_callbacks):
        configured_callbacks.append(
            build_checkpoint_callback(log_dir, monitor="val/ts_ic", mode="max")
        )
    if not any(isinstance(callback, EarlyStopping) for callback in configured_callbacks):
        configured_callbacks.append(
            build_early_stopping_callback(
                monitor="val/ts_ic", mode="max", patience=patience
            )
        )

    trainer_kwargs: dict[str, Any] = {
        "default_root_dir": str(Path(log_dir).expanduser()),
        "max_epochs": epochs,
        "callbacks": configured_callbacks,
        "logger": logger if logger is not None else False,
        "accelerator": accelerator,
        "devices": devices,
        "precision": precision,
        "deterministic": True,
        **kwargs,
    }
    if gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = gradient_clip_val
    return L.Trainer(**trainer_kwargs)


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
    if ckpt_path is not None and not ckpt_path.strip():
        raise ValueError("ckpt_path must be None or a non-empty path")
    trainer.fit(
        model=lightning_module,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )


def run_test(
    trainer: L.Trainer,
    lightning_module: L.LightningModule,
    datamodule: LOBDataModule,
    ckpt_path: str,
) -> PredictionArtifact:
    """加载最佳检查点，在 test split 上运行并返回统一预测产物。

    Args:
        trainer: 已配置的 Trainer 实例。
        lightning_module: 测试模块。
        datamodule: 数据模块。
        ckpt_path: 检查点路径。

    Returns:
        含完整 metadata/model/dataset/fold 的统一预测产物。
    """
    _require_checkpoint(ckpt_path)
    trainer.test(
        model=lightning_module,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )
    artifact = getattr(lightning_module, "test_artifact", None)
    if not isinstance(artifact, PredictionArtifact):
        raise RuntimeError("test completed without a PredictionArtifact")
    if artifact.split != "test":
        raise RuntimeError("test artifact split must be 'test'")
    return artifact


def _validate_monitor_mode(mode: str) -> None:
    if mode not in {"min", "max"}:
        raise ValueError("mode must be 'min' or 'max'")


def _require_checkpoint(ckpt_path: str) -> None:
    if not ckpt_path.strip():
        raise ValueError("ckpt_path must not be empty")

