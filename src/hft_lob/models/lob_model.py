"""纯模型层：统一 ``forward([B,T,F]) -> [B,1]`` 契约（需求文档 §18）。

``[B,T,F]`` 是与模型无关的数据语义；CNN/DeepLOB 在各自 ``forward`` 内部
增加 channel 维，Transformer 类模型直接消费三维输入。

模型结构与 lobx 对齐：``cnn1`` / ``deeplob`` / ``cnn2`` / ``transformer`` /
``itransformer`` / ``lobtransformer`` / ``axiallob`` / ``dla`` / ``binbtabl`` /
``binctabl`` / ``hlob`` 全部实装（MLP 属 baseline，见 ``hft_lob.baselines``）。
所有模型均为纯 ``torch.nn.Module``，训练职责由 ``hft_lob.modules.LOBLightningModule``
统一封装；``build_model`` 是唯一模型工厂，按训练配置 + 数据集元数据
实例化模型（``hlob`` 额外需要调用方提供同调结构）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import nn

from hft_lob.configs.experiment import ModelRunConfig
from hft_lob.models.AxialLob.axiallob import AxialLOB
from hft_lob.models.CNN1.cnn1 import CNN1
from hft_lob.models.CNN2.cnn2 import CNN2
from hft_lob.models.CompleteHCNN.complete_hcnn import Complete_HCNN
from hft_lob.models.DeepLob.deeplob import DeepLOB
from hft_lob.models.DLA.DLA import DLA
from hft_lob.models.iTransformer.itransformer import ITransformer
from hft_lob.models.LobTransformer.lobtransformer import LobTransformer
from hft_lob.models.TABL.bin_tabl import BiN_BTABL, BiN_CTABL
from hft_lob.models.Transformer.transformer import Transformer


def build_model(
    config: ModelRunConfig,
    *,
    feature_columns: Sequence[str],
    history_snapshots: int,
    homological_structures: dict[str, Any] | None = None,
) -> nn.Module:
    """按配置实例化模型（§18：统一 forward([B,T,F]) -> [B,1]）。

    Args:
        config: 实验配置根（model 段 + window/features 契约）。
        feature_columns: PreparedDataset 产出的唯一特征 schema。
        homological_structures: hlob 模型所需的同调结构（仅 ``hlob`` 使用；
            由调用方提供，缺失抛 ValueError）。

    Returns:
        实例化模型。

    Raises:
        ValueError: 未注册的模型名 / hlob 缺同调结构。
        NotImplementedError: 目标模型接口尚未实现（非 MVP 模型骨架）。
    """
    name = config.model.name
    num_features = len(feature_columns)
    history = history_snapshots
    levels = sum(name.startswith("ASKp") for name in feature_columns) or num_features // 4
    if history <= 0 or levels <= 0:
        raise ValueError("dataset metadata must contain positive history and LOB levels")

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


__all__ = ["CNN1", "DeepLOB", "build_model"]
