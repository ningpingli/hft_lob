"""滑窗 torch Dataset：在处理后的 LOB CSV/pt 上构造 (特征窗口, 回归标签) 样本。"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from hft_lob.configs.experiment import ExperimentConfig

#: 标识盘口特征列的列名前缀。
_FEATURE_PREFIXES: tuple[str, ...] = ("ASKp", "ASKs", "BIDp", "BIDs")


class LOBWindowDataset(Dataset):
    """滑窗读取处理后的 LOB CSV/pt，返回 ``(1, window_size, n_features)`` 特征
    + 连续回归标签（``label_columns[label_index]`` 的未来收益）。

    原 CustomDataset 已删除（无别名）：旧 import 请改为 LOBWindowDataset。
    """

    def __init__(
        self,
        dataset_root: str,
        learning_stage: str,
        window_size: int,
        shuffling_seed: int,
        cache_size: int,
        lighten: bool,
        threshold: float,
        label_columns: list[str],
        label_index: int = 0,
        balanced_dataloader: bool = False,
        backtest: bool = False,
        training_stocks: list[str] | None = None,
        validation_stocks: list[str] | None = None,
        target_stocks: list[str] | None = None,
    ) -> None:
        """初始化数据集：展开 CSV/pt 文件并构建滑窗有效样本索引。

        ``label_columns`` 必须非空；训练阶段按 ``shuffling_seed`` 打乱文件，
        验证/测试阶段按时间顺序读取。

        Args:
            dataset_root: 处理后数据根目录。
            learning_stage: 学习阶段（training / validation / testing）。
            window_size: 每个窗口的时间步数。
            shuffling_seed: 训练集文件随机打乱的种子。
            cache_size: 文件内存缓存上限（按文件数计）。
            lighten: 是否仅使用前 5 档（20 个特征列）。
            threshold: 标签阈值（与 executor 兼容保留）。
            label_columns: 标签列名列表（显式传入，禁止 f-string 推导）。
            label_index: 训练目标标签在 label_columns 中的下标。
            balanced_dataloader: 是否使用平衡采样（回归管线未使用，兼容保留）。
            backtest: 是否为回测数据集模式。
            training_stocks: 训练股票列表。
            validation_stocks: 验证股票列表。
            target_stocks: 测试股票列表。
        """
        raise NotImplementedError("LOBWindowDataset.__init__ not implemented")

    def __len__(self) -> int:
        """所有输入文件的累计有效样本数。"""
        raise NotImplementedError("LOBWindowDataset.__len__ not implemented")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回第 index 个滑窗样本 (特征张量, 回归标签)。

        Args:
            index: 全局样本下标。

        Returns:
            (特征张量, 标签张量)：特征形状为 ``(1, window_size, n_features)``。
        """
        raise NotImplementedError("LOBWindowDataset.__getitem__ not implemented")


def build_dataset_path(config: ExperimentConfig, *, stage: str) -> str:
    """按配置与阶段推导处理后数据集的目录路径。

    Args:
        config: 实验配置根（含 dataset 名等路径要素）。
        stage: 阶段名（training / validation / testing）。

    Returns:
        该阶段对应的数据目录路径。
    """
    raise NotImplementedError("build_dataset_path not implemented")
