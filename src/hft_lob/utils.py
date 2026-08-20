"""共享工具：YAML 读取。"""

from __future__ import annotations

from typing import Any


def load_yaml(path: str, subsection: str) -> dict[str, Any]:
    """加载 YAML 文件并返回指定子节。

    Args:
        path: YAML 文件路径。
        subsection: 要读取的子节名（general / data / loader / training / trading）。

    Returns:
        包含 YAML 子节内容的字典。
    """
    raise NotImplementedError("load_yaml not implemented")
