"""LOB 快照数据的字段配置：原始列名到标准列名的映射与时间维度。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: 字段配置字典中时间段与字段段的键名。
_TIME_KEY = "time"
_TIME_SOURCE_KEY = "source"
_TIME_UNIT_KEY = "unit"
_INDEX_MARKER = "__index__"

_FIELDS_KEY = "fields"
_COLUMN_MAP_KEY = "column_map"


@dataclass(frozen=True)
class TimeConfig:
    """时间维度的来源：列名，或 ``"__index__"``（时间即 DataFrame 索引）。"""

    source: str
    unit: str = "seconds"

    @property
    def is_index(self) -> bool:
        """时间是否取自 DataFrame 索引（``source == "__index__"``）。"""
        raise NotImplementedError("TimeConfig.is_index not implemented")


@dataclass(frozen=True)
class FieldsConfig:
    """数据集原始列名到标准列名的映射（全部字段统一走该映射）。"""

    column_map: dict[str, str]
    time: TimeConfig = TimeConfig(_INDEX_MARKER)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldsConfig:
        """从配置字典构造字段配置。

        Args:
            data: 含 fields.column_map 与 time 段的配置字典。

        Returns:
            构造好的 FieldsConfig。
        """
        raise NotImplementedError("FieldsConfig.from_dict not implemented")

    @classmethod
    def from_yaml(cls, path: str) -> FieldsConfig:
        """从 YAML 文件构造字段配置。

        Args:
            path: 字段配置 YAML 文件路径。

        Returns:
            构造好的 FieldsConfig。
        """
        raise NotImplementedError("FieldsConfig.from_yaml not implemented")
