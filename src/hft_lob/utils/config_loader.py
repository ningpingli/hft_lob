"""配置加载（需求文档 §42 冻结规格）：configs/experiment.yaml → ExperimentConfig。"""

from __future__ import annotations

from hft_lob.configs.experiment import ExperimentConfig


def load_config(config_path: str, *, experiment_id: str) -> ExperimentConfig:
    """读取 YAML 配置并组装 ExperimentConfig（缺段/缺键用默认值兜底）。

    YAML 结构按 §42 冻结规格：task / data / target / sessions / window /
    features / normalization / loader / model / training / evaluation / split /
    seed（完整模板见 ``configs/experiment.yaml``）。

    Args:
        config_path: configs/experiment.yaml 路径。
        experiment_id: 实验 ID（main 生成/恢复后注入）。

    Returns:
        实验配置根。

    Raises:
        FileNotFoundError: 配置文件不存在。
        ValueError: 配置文件顶层不是 YAML mapping。
    """
    raise NotImplementedError("load_config not implemented")
