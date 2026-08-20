"""ExperimentConfig：单股 LOB 60 秒收益预测系统的实验配置 dataclass 组。

按 ``doc/需求文档.md`` §42 冻结的核心规格组织：task / data / target / sessions /
window / features / normalization / loader / model / training / evaluation / split。
"""

from __future__ import annotations

from dataclasses import dataclass

#: A 股连续竞价时段（半开区间 [start, end)）。
MORNING_SESSION: tuple[str, str] = ("09:30:00", "11:30:00")
AFTERNOON_SESSION: tuple[str, str] = ("13:00:00", "14:57:00")

#: 特征列契约：20 盘口 + 3 标量（§10 第一版保留原始 23 维）。
RAW_FEATURE_COLUMNS: tuple[str, ...] = (
    "ASKp1", "ASKs1", "BIDp1", "BIDs1",
    "ASKp2", "ASKs2", "BIDp2", "BIDs2",
    "ASKp3", "ASKs3", "BIDp3", "BIDs3",
    "ASKp4", "ASKs4", "BIDp4", "BIDs4",
    "ASKp5", "ASKs5", "BIDp5", "BIDs5",
    "last", "volume", "amount",
)

#: 盘口价格列（整条盘口缺失判定 / 质量检查用）。
PRICE_COLUMNS: tuple[str, ...] = tuple(
    c for c in RAW_FEATURE_COLUMNS if c.startswith(("ASKp", "BIDp"))
)


@dataclass(frozen=True)
class TaskConfig:
    """任务定义：单只股票、回归（§0）。"""

    ticker: str
    task_type: str = "regression"


@dataclass(frozen=True)
class DataConfig:
    """数据规格与目录（§30 目录分层：raw immutable，split 用 manifest 表达）。"""

    levels: int = 5
    snapshot_interval_seconds: int = 3
    raw_dir: str = "data/raw"  # 原始 parquet 根目录（只读，immutable）
    processed_dir: str = "data/processed"  # 清洗 + 特征 + 标签落盘根目录
    manifest_dir: str = "data/datasets"  # split manifest 根目录


@dataclass(frozen=True)
class TargetConfig:
    """标签定义（§7）：60 秒中间价对数收益，future 匹配容差 3 秒（§7.2）。"""

    type: str = "log_mid_return"  # log_mid_return | simple_mid_return
    horizon_seconds: int = 60
    tolerance_seconds: int = 3


@dataclass(frozen=True)
class SessionConfig:
    """交易时段（§3）；窗口与标签禁止跨 session。"""

    morning: tuple[str, str] = MORNING_SESSION
    afternoon: tuple[str, str] = AFTERNOON_SESSION
    allow_cross_session: bool = False


@dataclass(frozen=True)
class WindowConfig:
    """输入窗口（§2/§9）：快照数；锚点语义 ``X = data[i-N+1 : i+1]`` 含 t 时刻。"""

    history_snapshots: int = 180
    include_anchor_snapshot: bool = True

    @property
    def history_seconds(self) -> int:
        """窗口覆盖的墙钟秒数（快照数 × 3 秒周期）。"""
        return self.history_snapshots * 3


@dataclass(frozen=True)
class FeatureConfig:
    """特征（§10/§11）：第一版保留原始 23 维；派生特征 P1 默认关闭。"""

    use_derived: bool = False
    derived_features: tuple[str, ...] = (
        "spread", "relative_spread", "mid_price", "microprice",
        "l1_imbalance", "l5_imbalance", "bid_depth", "ask_depth",
        "depth_imbalance", "price_slope", "volume_slope",
    )


@dataclass(frozen=True)
class NormalizationConfig:
    """归一化（§12）：只允许 t 之前（或 train 段）信息；禁止全量统计。"""

    mode: str = "train_only"  # train_only | causal
    max_ffill_gap_seconds: int = 6  # §5 缺失策略：gap 上限，超过则标记该段 invalid


@dataclass(frozen=True)
class LoaderConfig:
    """DataLoader 装配参数。"""

    batch_size: int = 128
    num_workers: int = 0
    seed: int = 42
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None  # None = PyTorch 默认
    cache_size: int = 4  # LOBWindowDataset 文件内存缓存上限（按文件数计）


@dataclass(frozen=True)
class ModelConfig:
    """模型（§18）：统一 ``forward(x) -> [B, 1]``；MVP 仅 CNN1 / DeepLOB。"""

    name: str = "cnn1"
    output_dim: int = 1
    num_features: int | None = None  # None → 由特征工程产出列数决定（契约校验用）


@dataclass(frozen=True)
class BaselineConfig:
    """MVP 基线（§17）：Zero / Imbalance / Ridge / 轻量 MLP。"""

    names: tuple[str, ...] = ("zero", "imbalance", "ridge", "mlp")
    ridge_alpha: float = 1.0
    mlp_hidden_dim: int = 64
    mlp_dropout: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    """训练（§20/§29）：primary loss = huber；全随机种子。"""

    loss: str = "huber"
    loss_huber_delta: float = 1.0
    epochs: int = 50
    patience: int = 10
    learning_rate: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1e-5
    log_interval_epochs: int = 1
    seed: int = 42


@dataclass(frozen=True)
class EvaluationConfig:
    """评估指标（§21）：TS-IC / RankIC / MAE / RMSE / Direction + 稳定性。"""

    metrics: tuple[str, ...] = (
        "mae", "rmse", "ts_ic", "rank_ic", "direction_accuracy",
    )
    report_daily: bool = True  # §14：daily metric mean/std/CI，显式处理序列相关


@dataclass(frozen=True)
class SplitConfig:
    """切分（§15/§16）：chronological（禁止 random row split）；walk-forward 可选。"""

    strategy: str = "chronological"
    walk_forward: bool = True
    train_ratio: float = 0.6
    validation_ratio: float = 0.2
    train_dates: tuple[str, str] | None = None  # 可选显式日期范围（含两端，%Y-%m-%d）
    validation_dates: tuple[str, str] | None = None
    test_dates: tuple[str, str] | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    """整个实验的配置根（§42 冻结规格）。"""

    experiment_id: str
    task: TaskConfig
    data: DataConfig
    target: TargetConfig
    sessions: SessionConfig
    window: WindowConfig
    features: FeatureConfig
    normalization: NormalizationConfig
    loader: LoaderConfig
    model: ModelConfig
    baselines: BaselineConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    split: SplitConfig
    seed: int = 42

    @property
    def ticker(self) -> str:
        """便捷访问：任务股票代码。"""
        return self.task.ticker

    @property
    def model_name(self) -> str:
        """便捷访问：模型名。"""
        return self.model.name

    @property
    def feature_count(self) -> int:
        """模型输入特征数：23 原始；开启派生特征后追加（§10/§11）。"""
        n = len(RAW_FEATURE_COLUMNS)
        if self.features.use_derived:
            n += len(self.features.derived_features)
        return n

    @property
    def target_column(self) -> str:
        """主标签列名（§7.1：唯一 primary target）。"""
        short = {"log_mid_return": "log", "simple_mid_return": "simple"}[self.target.type]
        return f"Target_{self.target.horizon_seconds}s_{short}"
