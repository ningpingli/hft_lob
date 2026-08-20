"""滑窗 torch Dataset（需求文档 §2/§13）：锚点语义、session 内构造、样本元数据。

职责单一：本模块只做「显式单-session processed parquet 文件列表 → 锚点滑窗样本」的纯
采样；文件发现/切分（manifest）与 DataLoader 装配分别属于 preprocessing 与
systems（LOBDataModule）层。
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import Dataset

from hft_lob.preprocessing.normalize import FrameStandardizer

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
    - 窗口内所有行必须 ``book_valid AND feature_valid``；启用 standardizer 时还
      必须 ``normalization_valid``；只有 anchor 行必须 ``target_valid``，历史行
      不因自身未来标签无效而被删除；
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
        standardizer: FrameStandardizer | None = None,
    ) -> None:
        """初始化数据集：逐文件逐 session 扫描有效样本并建立索引。

        Args:
            file_paths: 单-session processed parquet 文件列表（升序，时间顺序）。
            ticker: 股票代码（写入样本元数据）。
            window_size: 每个窗口的快照数（§2：含 anchor 共 N 帧）。
            feature_cols: 模型输入特征列（23 原始，或 +派生）。
            target_col: 主标签列名（§7.1）。
            cache_size: 文件内存缓存上限（按文件数计，FIFO 逐出）。
            standardizer: 严格因果的滚动标准化器；Dataset 在单-session 文件
                加载后、随机切窗前统一调用，None 表示不标准化。

        Raises:
            ValueError: 参数非法，或任一文件包含多个 trade_date/session_id。
        """
        paths = [Path(path).resolve() for path in file_paths]
        columns = list(feature_cols)
        if not paths:
            raise ValueError("file_paths must not be empty")
        if len(set(paths)) != len(paths):
            raise ValueError("file_paths must be unique")
        if not ticker.strip():
            raise ValueError("ticker must not be empty")
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("feature_cols must be non-empty and unique")
        if not target_col:
            raise ValueError("target_col must not be empty")
        if cache_size <= 0:
            raise ValueError("cache_size must be > 0")
        missing_files = [str(path) for path in paths if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(f"dataset files do not exist: {missing_files}")

        self._file_paths = tuple(paths)
        self._ticker = ticker
        self._window_size = window_size
        self._source_feature_cols = columns
        self._target_col = target_col
        self._cache_size = cache_size
        self._standardizer = standardizer
        self._model_feature_cols = (
            list(standardizer.output_feature_cols)
            if standardizer is not None
            else list(columns)
        )
        if len(self._model_feature_cols) != len(columns):
            raise ValueError("standardizer output feature count must match feature_cols")

        self._cache: OrderedDict[int, pl.DataFrame] = OrderedDict()
        self._sample_index: list[tuple[int, int]] = []
        file_order: list[datetime] = []
        for file_index in range(len(self._file_paths)):
            frame = self._load_file(file_index)
            if frame.is_empty():
                raise ValueError(f"dataset file is empty: {self._file_paths[file_index]}")
            first_timestamp = frame.get_column("timestamp").item(0)
            if not isinstance(first_timestamp, datetime):
                raise ValueError("timestamp values must be datetime")
            file_order.append(first_timestamp)
            self._index_file(file_index, frame)
        if file_order != sorted(file_order):
            raise ValueError("file_paths must be ordered chronologically")

    @property
    def n_features(self) -> int:
        """每个快照的特征数。"""
        return len(self._model_feature_cols)

    @property
    def feature_cols(self) -> list[str]:
        """实际使用的特征列名。"""
        return list(self._model_feature_cols)

    def __len__(self) -> int:
        """全部有效样本数。"""
        return len(self._sample_index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, SampleMeta]:
        """返回 (特征窗口, 回归标签, 样本元数据)。

        Returns:
            - 特征窗口：``(1, window_size, n_features)`` float32，含 anchor 帧；
            - 标签：``(1,)`` float32，collate 后严格为 ``[B, 1]``；
            - 元数据：完整 ``SampleMeta``（§13/§28）。
        """
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        file_index, anchor_index = self._sample_index[index]
        frame = self._load_file(file_index)
        window_start = anchor_index - self._window_size + 1
        window = frame.slice(window_start, self._window_size)
        values = torch.tensor(
            window.select(self._model_feature_cols).to_numpy(),
            dtype=torch.float32,
        ).unsqueeze(0)
        target = torch.tensor(
            [float(frame.get_column(self._target_col).item(anchor_index))],
            dtype=torch.float32,
        )
        timestamp = frame.get_column("timestamp").item(anchor_index)
        if not isinstance(timestamp, datetime):
            raise ValueError("anchor timestamp must be datetime")
        bid1 = float(frame.get_column("BIDp1").item(anchor_index))
        ask1 = float(frame.get_column("ASKp1").item(anchor_index))
        metadata = SampleMeta(
            ticker=self._ticker,
            trade_date=str(frame.get_column("trade_date").item(anchor_index)),
            session_id=str(frame.get_column("session_id").item(anchor_index)),
            anchor_timestamp=timestamp.isoformat(),
            mid_t=float(frame.get_column("mid_price").item(anchor_index)),
            future_mid=float(frame.get_column("future_mid").item(anchor_index)),
            bid1=bid1,
            ask1=ask1,
            spread=ask1 - bid1,
        )
        return values, target, metadata

    def __getstate__(self) -> dict[str, object]:
        """DataLoader spawn worker 只复制索引，不复制主进程文件缓存。"""
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        return state

    def _load_file(self, file_index: int) -> pl.DataFrame:
        cached = self._cache.get(file_index)
        if cached is not None:
            return cached
        frame = pl.read_parquet(self._file_paths[file_index])
        self._validate_frame(frame, self._file_paths[file_index])
        if self._standardizer is not None:
            frame = self._standardizer.transform_frame(frame)
        self._cache[file_index] = frame
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return frame

    def _validate_frame(self, frame: pl.DataFrame, path: Path) -> None:
        required = {
            "ticker",
            "timestamp",
            "book_valid",
            "feature_valid",
            "target_valid",
            self._target_col,
            *self._source_feature_cols,
            *_META_COLUMNS,
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"dataset file {path} missing columns: {missing}")
        if frame.is_empty():
            return
        trade_dates = frame.get_column("trade_date").unique().to_list()
        session_ids = frame.get_column("session_id").unique().to_list()
        tickers = frame.get_column("ticker").unique().to_list()
        if len(trade_dates) != 1 or trade_dates[0] is None:
            raise ValueError(f"dataset file {path} must contain one trade_date")
        if len(session_ids) != 1 or session_ids[0] is None:
            raise ValueError(f"dataset file {path} must contain one session_id")
        if tickers != [self._ticker]:
            raise ValueError(f"dataset file {path} ticker does not match {self._ticker!r}")
        timestamps = frame.get_column("timestamp")
        if not isinstance(frame.schema["timestamp"], pl.Datetime):
            raise ValueError(f"dataset file {path} timestamp must be Datetime")
        if timestamps.null_count() > 0 or not timestamps.is_sorted():
            raise ValueError(f"dataset file {path} timestamps must be non-null and sorted")
        if timestamps.n_unique() != frame.height:
            raise ValueError(f"dataset file {path} timestamps must be unique")
        numeric_columns = [*self._source_feature_cols, self._target_col, "future_mid"]
        non_numeric = [name for name in numeric_columns if not frame.schema[name].is_numeric()]
        if non_numeric:
            raise ValueError(f"dataset file {path} has non-numeric columns: {non_numeric}")

    def _index_file(self, file_index: int, frame: pl.DataFrame) -> None:
        row_valid = (
            pl.col("book_valid").fill_null(False)
            & pl.col("feature_valid").fill_null(False)
            & pl.all_horizontal(
                pl.col(name).is_not_null() & pl.col(name).is_finite()
                for name in self._model_feature_cols
            )
        )
        if self._standardizer is not None:
            row_valid &= pl.col("normalization_valid").fill_null(False)
        anchor_valid = (
            pl.col("target_valid").fill_null(False)
            & pl.col(self._target_col).is_not_null()
            & pl.col(self._target_col).is_finite()
            & pl.col("future_mid").is_not_null()
            & pl.col("future_mid").is_finite()
        )
        validity = frame.select(
            row_valid.alias("row_valid"), anchor_valid.alias("anchor_valid")
        )
        rows = validity.get_column("row_valid").to_list()
        anchors = validity.get_column("anchor_valid").to_list()
        prefix = [0]
        for valid in rows:
            prefix.append(prefix[-1] + int(bool(valid)))
        for anchor_index in range(self._window_size - 1, frame.height):
            window_start = anchor_index - self._window_size + 1
            valid_count = prefix[anchor_index + 1] - prefix[window_start]
            if valid_count == self._window_size and bool(anchors[anchor_index]):
                self._sample_index.append((file_index, anchor_index))
