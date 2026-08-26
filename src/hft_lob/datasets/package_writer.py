"""训练数据包的流式写入与原子发布事务。"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from hft_lob.datasets.dataset_validator import DatasetPackageMetadata, validate_dataset_package
from hft_lob.datasets.fold_index_builder import write_fold_indexes
from hft_lob.datasets.sample_compiler import CompiledDay, CompiledSession
from hft_lob.preprocessing.split import WalkForwardPlan

logger = logging.getLogger(__name__)


class _ArrayAppender:
    """顺序追加二维数组，完成后封装成标准 ``.npy``。"""

    def __init__(self, path: Path, *, dtype: Any, width: int) -> None:
        self.path = path
        self.dtype = np.dtype(dtype)
        self.width = width
        self.rows = 0
        self._file: BinaryIO = path.open("wb")

    def append(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=self.dtype)
        if array.ndim != 2 or array.shape[1] != self.width:
            raise ValueError(f"expected array [N,{self.width}], got {array.shape}")
        array.tofile(self._file)
        self.rows += array.shape[0]

    def finalize(self, destination: Path) -> None:
        self.close()
        output = np.lib.format.open_memmap(
            destination,
            mode="w+",
            dtype=self.dtype,
            shape=(self.rows, self.width),
        )
        source = np.memmap(
            self.path,
            mode="r",
            dtype=self.dtype,
            shape=(self.rows, self.width),
        )
        chunk_rows = max(1, (64 * 1024 * 1024) // (self.dtype.itemsize * self.width))
        for start in range(0, self.rows, chunk_rows):
            output[start : start + chunk_rows] = source[start : start + chunk_rows]
        output.flush()
        del source, output
        self.path.unlink()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


class DatasetPackageWriter:
    """持有一次构建的文件句柄、临时目录和原子发布事务。"""

    def __init__(
        self, output_root: str | Path, feature_count: int, target_count: int = 1
    ) -> None:
        if feature_count <= 0:
            raise ValueError("feature_count must be > 0")
        if target_count <= 0:
            raise ValueError("target_count must be > 0")
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.build_root = self.output_root / f".building-{uuid.uuid4().hex}"
        self.work_root = self.build_root / "package"
        self.work_root.mkdir(parents=True)
        self.arrays = {
            "features": _ArrayAppender(
                self.work_root / ".features.bin", dtype=np.float32, width=feature_count
            ),
            "targets": _ArrayAppender(
                self.work_root / ".targets.bin", dtype=np.float32, width=target_count
            ),
            "validity": _ArrayAppender(
                self.work_root / ".validity.bin", dtype=np.bool_, width=target_count + 1
            ),
            "market": _ArrayAppender(self.work_root / ".market.bin", dtype=np.float32, width=4),
        }
        self.row_writer: pq.ParquetWriter | None = None
        self.anchor_writer: pq.ParquetWriter | None = None
        self.trade_dates: list[str] = []
        self.quality: list[dict[str, object]] = []
        self.anchor_count = 0
        self._finalized = False

    def __enter__(self) -> DatasetPackageWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
        shutil.rmtree(self.build_root, ignore_errors=True)

    def append(self, day: CompiledDay) -> None:
        if self._finalized:
            raise RuntimeError("cannot append after finalization")
        self.trade_dates.append(day.trade_date)
        self.quality.append(day.quality.to_dict())
        for session in day.sessions:
            self.anchor_count += session.anchors.height
            self._append_session(session)

    @property
    def row_count(self) -> int:
        return self.arrays["features"].rows

    def finalize_and_publish(
        self,
        metadata: DatasetPackageMetadata,
        plan: WalkForwardPlan,
    ) -> Path:
        logger.info(
            "dataset_build.finalize_arrays rows=%d anchors=%d", self.row_count, self.anchor_count
        )
        self._finalize_data()
        anchors_path = self.work_root / "anchors.parquet"
        write_fold_indexes(anchors_path, self.work_root / "folds", plan)
        anchors_path.unlink()
        pl.DataFrame(self.quality).sort("trade_date").write_parquet(
            self.work_root / "quality.parquet"
        )
        (self.work_root / "dataset.json").write_text(
            json.dumps(metadata.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (self.work_root / "_SUCCESS").touch()

        package_root = self.build_root / metadata.dataset_id
        os.replace(self.work_root, package_root)
        logger.info("dataset_build.validate_start dataset_id=%s", metadata.dataset_id)
        validate_dataset_package(package_root)
        logger.info("dataset_build.validate_complete dataset_id=%s", metadata.dataset_id)
        destination = self.output_root / metadata.dataset_id
        if destination.exists():
            validate_dataset_package(destination)
            return destination
        try:
            os.replace(package_root, destination)
        except FileExistsError:
            validate_dataset_package(destination)
        return destination

    def close(self) -> None:
        if self.row_writer is not None:
            self.row_writer.close()
            self.row_writer = None
        if self.anchor_writer is not None:
            self.anchor_writer.close()
            self.anchor_writer = None
        for appender in self.arrays.values():
            appender.close()

    def _append_session(self, session: CompiledSession) -> None:
        self.arrays["features"].append(session.features)
        self.arrays["targets"].append(session.targets)
        self.arrays["validity"].append(session.validity)
        self.arrays["market"].append(session.market)
        self.row_writer = _write_chunk(
            self.work_root / "rows.parquet", session.rows, self.row_writer
        )
        if not session.anchors.is_empty():
            self.anchor_writer = _write_chunk(
                self.work_root / "anchors.parquet", session.anchors, self.anchor_writer
            )

    def _finalize_data(self) -> None:
        if self._finalized:
            raise RuntimeError("writer is already finalized")
        has_anchors = self.anchor_writer is not None
        self.close()
        row_count = self.arrays["features"].rows
        if row_count == 0 or not has_anchors:
            raise ValueError("processed data is empty or contains no valid anchors")
        if any(appender.rows != row_count for appender in self.arrays.values()):
            raise ValueError("array row counts do not match")
        for name, appender in self.arrays.items():
            appender.finalize(self.work_root / f"{name}.npy")
        self._finalized = True


def _write_chunk(
    path: Path,
    frame: pl.DataFrame,
    writer: pq.ParquetWriter | None,
) -> pq.ParquetWriter:
    table = frame.to_arrow()
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema)
    writer.write_table(table)
    return writer
