"""纯模型层：统一 ``forward(x) -> [B, 1]`` 契约（需求文档 §18）。

MVP 实际可用：``cnn1`` / ``deeplob``（§1.1/§17/§39）。其余 9 个模型（transformer
/ itransformer / lobtransformer / cnn2 / axiallob / dla / binbtabl / binctabl /
hlob）**接口保留、暂不实现**（构造/前向为 ``raise NotImplementedError`` 骨架，
待 §39 Phase 3-4 分阶段实装）。所有模型均为纯 ``torch.nn.Module``，训练职责由
``systems.LOBLightningModule`` 统一封装；``build_model`` 是唯一模型工厂，运行时调用
未实现模型会得到明确的 ``NotImplementedError``。
"""

from __future__ import annotations

from typing import Any

from torch import nn

from hft_lob.configs.experiment import ExperimentConfig
from hft_lob.models.AxialLob.axiallob import AxialLOB, GatedAxialAttention
from hft_lob.models.CNN1.cnn1 import CNN1
from hft_lob.models.CNN2.cnn2 import CNN2
from hft_lob.models.CompleteHCNN.complete_hcnn import Complete_HCNN
from hft_lob.models.DeepLob.deeplob import DeepLOB
from hft_lob.models.DLA.DLA import DLA
from hft_lob.models.iTransformer.itransformer import ITransformer
from hft_lob.models.LobTransformer.lobtransformer import LobTransformer
from hft_lob.models.TABL.bin_nn import BiN
from hft_lob.models.TABL.bin_tabl import BiN_BTABL, BiN_CTABL
from hft_lob.models.TABL.bl_layer import BL_layer
from hft_lob.models.TABL.tabl_layer import TABL_layer
from hft_lob.models.Transformer.transformer import SinusoidalPositionalEmbedding, Transformer


def build_model(
    config: ExperimentConfig,
    *,
    homological_structures: dict[str, Any] | None = None,
) -> nn.Module:
    """按配置实例化模型（§18：统一 forward(x) -> [B, 1]）。

    Args:
        config: 实验配置根（model 段 + window/features 契约）。
        homological_structures: hlob 模型所需的同调结构（仅 ``hlob`` 使用；
            由调用方提供，缺失抛 ValueError）。

    Returns:
        实例化模型。

    Raises:
        ValueError: 未注册的模型名 / hlob 缺同调结构。
        NotImplementedError: 目标模型接口尚未实现（非 MVP 模型骨架）。
    """
    name = config.model.name
    num_features = config.model.num_features or config.feature_count
    history = config.window.history_snapshots
    levels = config.data.levels

    if name == "cnn1":
        return CNN1(num_features=num_features, history_length=history)
    if name == "deeplob":
        return DeepLOB(num_features=num_features, levels=levels)
    # ---- 以下接口保留、暂不实现（§39 Phase 3-4）----
    if name == "transformer":
        return Transformer(num_features=num_features, history_length=history)
    if name == "itransformer":
        return ITransformer(num_features=num_features, history_length=history)
    if name == "lobtransformer":
        return LobTransformer(num_features=num_features, levels=levels)
    if name == "cnn2":
        return CNN2(num_features=num_features, history_length=history)
    if name == "axiallob":
        return AxialLOB(W=num_features, H=history)
    if name == "dla":
        return DLA(num_features=num_features, num_snapshots=history)
    if name == "binbtabl":
        # d2/d3/t3 沿用 lobx 5 档默认（待模型实现阶段配置化）。
        return BiN_BTABL(d2=120, d1=num_features, t1=history, t2=levels, d3=1, t3=1)
    if name == "binctabl":
        return BiN_CTABL(
            d2=120, d1=num_features, t1=history, t2=levels,
            d3=120, t3=5, d4=1, t4=1,
        )
    if name == "hlob":
        if homological_structures is None:
            raise ValueError(
                "build_model('hlob', ...) requires homological_structures "
                "(prepared by the caller; see doc/需求文档.md §1.2 — HLOB 不在 MVP)"
            )
        return Complete_HCNN(homological_structures=homological_structures, num_features=num_features)
    raise ValueError(
        f"Unsupported model {name!r}; registered: cnn1 | deeplob | transformer | "
        "itransformer | lobtransformer | cnn2 | axiallob | dla | binbtabl | "
        "binctabl | hlob"
    )


__all__ = [
    "AxialLOB",
    "BiN",
    "BiN_BTABL",
    "BiN_CTABL",
    "BL_layer",
    "CNN1",
    "CNN2",
    "Complete_HCNN",
    "DLA",
    "DeepLOB",
    "GatedAxialAttention",
    "ITransformer",
    "LobTransformer",
    "SinusoidalPositionalEmbedding",
    "TABL_layer",
    "Transformer",
    "build_model",
]
