"""YAML 落盘的内部共享实现。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def atomic_dump_yaml(path: Path, contents: dict[str, Any]) -> None:
    """在目标目录中原子写入 YAML，避免留下半写文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            yaml.safe_dump(
                contents,
                temporary,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
