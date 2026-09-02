"""从连续 memory-mapped tensor 中按 anchor 返回滑动窗口样本。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from hft_lob.datasets.dataset_validator import DatasetPackageMetadata, fold_index_path
from hft_lob.systems.contracts import SampleMeta


class PrebuiltLOBDataset(Dataset):
    """一个 index 对应 fold 索引中的一个 anchor，也对应一个训练 sample。"""

    def __init__(self, package_dir: str | Path, metadata: DatasetPackageMetadata, *, fold_index: int, split: str) -> None:
        self.package_dir = Path(package_dir).resolve()
        self.metadata = metadata
        self.index = pl.read_parquet(fold_index_path(self.package_dir, fold_index, split))
        if self.index.is_empty():
            raise ValueError(f"fold {fold_index} {split} contains no samples")
        self._features: np.ndarray | None = None
        self._targets: np.ndarray | None = None
        self._market: np.ndarray | None = None

    def __len__(self) -> int:
        return self.index.height

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor], SampleMeta]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        self._ensure_arrays()
        row = self.index.row(index, named=True)
        anchor = cast(int, row["global_anchor_index"])
        session_start = cast(int, row["session_start_index"])
        start = anchor - self.metadata.history_snapshots + 1
        if start < session_start:
            raise ValueError("sample window crosses a session boundary")
        features = torch.from_numpy(cast(np.ndarray, self._features)[start : anchor + 1].copy())
        target_row = cast(np.ndarray, self._targets)[anchor]
        target = torch.from_numpy(target_row[:1].copy())
        targets_by_horizon = {
            label: torch.from_numpy(target_row[position : position + 1].copy())
            for position, label in enumerate(self.metadata.labels)
        }
        market = cast(np.ndarray, self._market)[anchor]
        timestamp = cast(datetime, row["anchor_timestamp"])
        sample = SampleMeta(
            ticker=self.metadata.ticker,
            trade_date=cast(str, row["trade_date"]),
            session_id=cast(str, row["session_id"]),
            anchor_timestamp=timestamp.isoformat(),
            mid_t=float(market[0]),
            future_mid=float(market[1]),
            bid1=float(market[2]),
            ask1=float(market[3]),
            spread=float(market[3] - market[2]),
        )
        return features, target, targets_by_horizon, sample

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state.update(_features=None, _targets=None, _market=None)
        return state

    def _ensure_arrays(self) -> None:
        if self._features is None:
            self._features = np.load(self.package_dir / "features.npy", mmap_mode="r")
            self._targets = np.load(self.package_dir / "targets.npy", mmap_mode="r")
            self._market = np.load(self.package_dir / "market.npy", mmap_mode="r")
