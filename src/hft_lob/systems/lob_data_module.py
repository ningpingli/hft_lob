"""只读装配预构建数据包的 Lightning DataModule。"""

from __future__ import annotations

import random
from functools import partial
from typing import cast

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader

from hft_lob.configs.experiment import LoaderConfig
from hft_lob.datasets.dataset_validator import DatasetPackage
from hft_lob.systems.contracts import LOBBatch, SampleMeta
from hft_lob.systems.prebuilt_dataset import PrebuiltLOBDataset


def _seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = (base_seed + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, SampleMeta]],
) -> LOBBatch:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    features, targets, target_valid, metadata = zip(*batch, strict=True)
    target_width = targets[0].shape
    if target_width != target_valid[0].shape or len(target_width) != 1:
        raise ValueError("targets and target_valid must have shape [L]")
    if any(target.shape != target_width for target in targets):
        raise ValueError("all samples in a batch must have the same target width")
    if any(valid.shape != target_width for valid in target_valid):
        raise ValueError("all target_valid vectors must match target width")
    return LOBBatch(
        torch.stack(features),
        torch.stack(targets),
        torch.stack(target_valid),
        tuple(metadata),
    )


class LOBDataModule(pl.LightningDataModule):
    """加载一个不可变数据包的指定 fold；不包含 ETL fallback。"""

    def __init__(self, package: DatasetPackage, *, fold_index: int, loader: LoaderConfig, seed: int) -> None:
        super().__init__()
        self.package = package
        self.dataset_dir = package.root
        self.metadata = package.metadata
        self.fold_index = fold_index
        self.loader = loader
        self.seed = seed
        self.train_dataset: PrebuiltLOBDataset | None = None
        self.val_dataset: PrebuiltLOBDataset | None = None
        self.test_dataset: PrebuiltLOBDataset | None = None
        self.save_hyperparameters({"dataset_id": self.metadata.dataset_id, "fold_index": fold_index})

    @property
    def dataset_version(self) -> str:
        return self.metadata.dataset_id

    def prepare_data(self) -> None:
        return None

    def setup(self, stage: str | None = None) -> None:
        if stage not in {None, "fit", "validate", "test"}:
            raise ValueError(f"unsupported Lightning stage: {stage!r}")
        if stage in (None, "fit"):
            self.train_dataset = self._dataset("train")
            self.val_dataset = self._dataset("validation")
        elif stage == "validate":
            self.val_dataset = self._dataset("validation")
        if stage in (None, "test"):
            self.test_dataset = self._dataset("test")

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self._require("train_dataset"), shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._require("val_dataset"), shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self._require("test_dataset"), shuffle=False)

    def teardown(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = None
            self.val_dataset = None
        elif stage == "validate":
            self.val_dataset = None
        if stage in (None, "test"):
            self.test_dataset = None

    def _dataset(self, split: str) -> PrebuiltLOBDataset:
        return PrebuiltLOBDataset(self.dataset_dir, self.metadata, fold_index=self.fold_index, split=split)

    def _make_loader(self, dataset: PrebuiltLOBDataset, *, shuffle: bool) -> DataLoader:
        if self.loader.batch_size <= 0:
            raise ValueError("loader.batch_size must be > 0")
        if self.loader.num_workers < 0:
            raise ValueError("loader.num_workers must be >= 0")
        if self.loader.persistent_workers and self.loader.num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")
        if self.loader.prefetch_factor is not None and self.loader.num_workers == 0:
            raise ValueError("prefetch_factor requires num_workers > 0")
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=self.loader.batch_size,
            shuffle=shuffle,
            num_workers=self.loader.num_workers,
            pin_memory=self.loader.pin_memory,
            persistent_workers=self.loader.persistent_workers,
            prefetch_factor=self.loader.prefetch_factor,
            collate_fn=_collate,
            worker_init_fn=partial(_seed_worker, base_seed=self.seed),
            generator=generator,
        )

    def _require(self, name: str) -> PrebuiltLOBDataset:
        dataset = cast(PrebuiltLOBDataset | None, getattr(self, name))
        if dataset is None:
            raise RuntimeError(f"{name} is not initialized; call setup() first")
        return dataset
