"""预构建数据包的完整性校验。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import polars as pl
import torch

from hft_lob.datasets.package import (
    SUCCESS_MARKER,
    DatasetPackageMetadata,
    validate_fold_index,
    validate_session_payload,
)


def validate_dataset_package(package_dir: str | Path) -> DatasetPackageMetadata:
    """验证已发布数据包并返回其 metadata；不修复、不回退构建。"""
    root = Path(package_dir).resolve()
    if not (root / SUCCESS_MARKER).is_file():
        raise ValueError(f"dataset package is not published: missing {SUCCESS_MARKER}")
    metadata_path = root / "dataset.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    try:
        raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("dataset.json is not valid UTF-8 JSON") from exc
    if not isinstance(raw_metadata, dict):
        raise ValueError("dataset.json root must be an object")
    metadata = DatasetPackageMetadata.from_dict(raw_metadata)
    if root.name != metadata.dataset_id:
        raise ValueError("package directory name must equal dataset_id")

    fold_dirs = sorted(path for path in (root / "folds").glob("fold_*") if path.is_dir())
    if not fold_dirs:
        raise ValueError("dataset package contains no fold indexes")
    referenced_sessions: set[Path] = set()
    maximum_anchor: dict[Path, int] = {}
    for fold_dir in fold_dirs:
        expected = {fold_dir / f"{split}.parquet" for split in ("train", "validation", "test")}
        if not all(path.is_file() for path in expected):
            raise ValueError(f"fold is missing a required split: {fold_dir.name}")
        for index_path in sorted(expected):
            frame = pl.read_parquet(index_path)
            validate_fold_index(frame)
            for relative, anchor_index in frame.select(
                "session_file", "anchor_index"
            ).iter_rows():
                session_path = (root / relative).resolve()
                if not session_path.is_relative_to(root) or not session_path.is_file():
                    raise ValueError(f"fold index references missing session: {relative}")
                referenced_sessions.add(session_path)
                maximum_anchor[session_path] = max(
                    maximum_anchor.get(session_path, -1), anchor_index
                )

    for session_path in referenced_sessions:
        payload = torch.load(session_path, map_location="cpu", weights_only=True)
        validated = validate_session_payload(
            payload,
            feature_count=len(metadata.feature_columns),
            feature_dtype=metadata.feature_dtype,
            target_dtype=metadata.target_dtype,
        )
        features = cast(torch.Tensor, validated["features"])
        if maximum_anchor[session_path] >= features.shape[0]:
            raise ValueError(f"fold index anchor exceeds session rows: {session_path.name}")
    return metadata
