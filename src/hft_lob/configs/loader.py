"""YAML 实验配置加载。"""

from __future__ import annotations

from hft_lob.configs.experiment import ExperimentConfig


def load_config(config_path: str, *, experiment_id: str) -> ExperimentConfig:
    """读取、校验 YAML 并组装 ExperimentConfig。

    Raises:
        FileNotFoundError: 配置文件不存在。
        ValueError: YAML 结构、字段或取值不符合配置契约。
    """
    raise NotImplementedError("load_config not implemented")
