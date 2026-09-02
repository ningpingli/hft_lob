"""纯模型层：统一 ``forward([B,T,F]) -> [B,L]`` 契约。

``L`` 是数据包元数据中的 ``labels`` 数量；模型层不再从训练配置读取输出宽度。
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
    target_count: int = 1,
    homological_structures: dict[str, Any] | None = None,
) -> nn.Module:
    """按数据包 labels 数量实例化模型，统一返回 ``[B,L]``。"""
    name = config.model.name
    num_features = len(feature_columns)
    history = history_snapshots
    levels = sum(name.startswith("ASKp") for name in feature_columns) or num_features // 4
    if history <= 0 or levels <= 0 or target_count <= 0:
        raise ValueError("dataset metadata must contain positive dimensions")

    if name == "cnn1":
        return CNN1(num_features=num_features, history_length=history, output_dim=target_count)
    if name == "deeplob":
        return DeepLOB(num_features=num_features, levels=levels, output_dim=target_count)
    if name == "transformer":
        return Transformer(
            num_features=num_features, history_length=history, output_dim=target_count
        )
    if name == "itransformer":
        return ITransformer(
            num_features=num_features, history_length=history, output_dim=target_count
        )
    if name == "lobtransformer":
        return LobTransformer(num_features=num_features, levels=levels, output_dim=target_count)
    if name == "cnn2":
        return CNN2(num_features=num_features, history_length=history, output_dim=target_count)
    if name == "axiallob":
        return AxialLOB(W=num_features, H=history, output_dim=target_count)
    if name == "dla":
        return DLA(num_features=num_features, num_snapshots=history, output_dim=target_count)
    if name == "binbtabl":
        return BiN_BTABL(d2=120, d1=num_features, t1=history, t2=levels, d3=target_count, t3=1)
    if name == "binctabl":
        return BiN_CTABL(
            d2=120, d1=num_features, t1=history, t2=levels,
            d3=120, t3=5, d4=target_count, t4=1,
        )
    if name == "hlob":
        if homological_structures is None:
            raise ValueError(
                "build_model('hlob', ...) requires homological_structures "
                "(prepared by the caller; see doc/需求文档.md §1.2 — HLOB 不在 MVP)"
            )
        return Complete_HCNN(
            homological_structures=homological_structures,
            num_features=num_features,
            output_dim=target_count,
        )
    raise ValueError(
        f"Unsupported model {name!r}; registered: cnn1 | deeplob | transformer | "
        "itransformer | lobtransformer | cnn2 | axiallob | dla | binbtabl | "
        "binctabl | hlob"
    )


__all__ = ["CNN1", "DeepLOB", "build_model"]
