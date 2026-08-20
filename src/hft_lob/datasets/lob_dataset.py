"""滑窗 torch Dataset（需求文档 §2/§13）：锚点语义、session 内构造、样本元数据。

职责单一：本模块只做「显式单-session processed parquet 文件列表 → 锚点滑窗样本」的纯
采样；文件发现/切分（manifest）与 DataLoader 装配分别属于 preprocessing 与
systems（LOBDataModule）层。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from hft_lob.preprocessing.normalize import TensorNormalizer

#: 构造样本所需的处理列（元数据侧）。
_META_COLUMNS: tuple[str, ...] = (
    "trade_date", "session_id", "seconds", "mid_price", "future_mid",
    "ASKp1", "BIDp1",
)


@dataclass(frozen=True)
class SampleMeta:
    """§13 样本元数据（研究 artifact 定位用）。"""

    ticker: str
    trade_date: str
    session_id: str
    anchor_timestamp: str
    mid_t: float
    future_mid: float
    bid1: float
    ask1: float
    spread: float


@dataclass(frozen=True)
class LOBBatch:
    """训练、预测和 artifact 共用的唯一 batch 契约。"""

    features: torch.Tensor  # [B, 1, T, F]
    targets: torch.Tensor  # [B, 1]
    metadata: tuple[SampleMeta, ...]


class LOBWindowDataset(Dataset):
    """对显式 processed parquet 列表构造锚点滑窗样本。

    锚点语义（§2 样本契约，禁止 ``X=data[i:i+N], y=target[i+N]``）：
    - ``X = seg[i - window_size + 1 : i + 1]``（**包含 anchor 快照 t**）；
    - ``y = target[i]``（anchor 行的主标签）；
    - 窗口与标签只在 same trade_date AND same session_id 内构造（§3.1）；
    - 每个 processed 文件只能包含一个 trade_date/session_id；多 session 文件
      立即拒绝，不能依赖调用方记得 group-by；
    - 窗口内所有行必须 ``book_valid AND feature_valid``；只有 anchor 行必须
      ``target_valid``，历史行不因自身未来标签无效而被删除；
    - 不变量（单元测试必须检查）：``max(timestamp(X)) == anchor_timestamp``。
    """

    def __init__(
        self,
        file_paths: Sequence[str],
        *,
        ticker: str,
        window_size: int,
        feature_cols: Sequence[str],
        target_col: str,
        cache_size: int = 4,
        normalizer: TensorNormalizer | None = None,
    ) -> None:
        """初始化数据集：逐文件逐 session 扫描有效样本并建立索引。

        Args:
            file_paths: 单-session processed parquet 文件列表（升序，时间顺序）。
            ticker: 股票代码（写入样本元数据）。
            window_size: 每个窗口的快照数（§2：含 anchor 共 N 帧）。
            feature_cols: 模型输入特征列（23 原始，或 +派生）。
            target_col: 主标签列名（§7.1）。
            cache_size: 文件内存缓存上限（按文件数计，FIFO 逐出）。
            normalizer: 仅从 training split 拟合并冻结的归一化器；Dataset 是
                唯一调用 ``transform_tensor`` 的位置，None 表示不归一化。

        Raises:
            ValueError: 参数非法，或任一文件包含多个 trade_date/session_id。
        """
        raise NotImplementedError("LOBWindowDataset.__init__ not implemented")

    @property
    def n_features(self) -> int:
        """每个快照的特征数。"""
        raise NotImplementedError("LOBWindowDataset.n_features not implemented")

    @property
    def feature_cols(self) -> list[str]:
        """实际使用的特征列名。"""
        raise NotImplementedError("LOBWindowDataset.feature_cols not implemented")

    def __len__(self) -> int:
        """全部有效样本数。"""
        raise NotImplementedError("LOBWindowDataset.__len__ not implemented")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, SampleMeta]:
        """返回 (特征窗口, 回归标签, 样本元数据)。

        Returns:
            - 特征窗口：``(1, window_size, n_features)`` float32，含 anchor 帧；
            - 标签：``(1,)`` float32，collate 后严格为 ``[B, 1]``；
            - 元数据：完整 ``SampleMeta``（§13/§28）。
        """
        raise NotImplementedError("LOBWindowDataset.__getitem__ not implemented")
