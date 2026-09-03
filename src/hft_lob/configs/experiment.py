"""数据构建与模型训练相互独立的配置 dataclass 组。

按 ``doc/需求文档.md`` §42 冻结的核心规格组织：task / data / target / sessions /
window / features / normalization / loader / model / training / evaluation / split。
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    """原始数据规格；阶段一只读取 immutable raw。"""

    levels: int = 5
    snapshot_interval_seconds: int = 3
    raw_dir: str = "data/raw"  # 原始 parquet 根目录（只读，immutable）
    column_mapping: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CleaningConfig:
    """清洗参数的唯一来源。"""

    max_ffill_gap_seconds: int = 6


@dataclass(frozen=True)
class TargetConfig:
    """收益标签配置；labels 的顺序是整个数据契约的列顺序。"""

    type: str = "log_mid_return"
    labels: list[int] = field(default_factory=lambda: [60, 120, 300, 600])
    tolerance_seconds: int = 3

    def __post_init__(self) -> None:
        labels = list(self.labels)
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("labels must be non-empty and unique")
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in labels):
            raise ValueError("labels must contain positive integers")
        if self.tolerance_seconds < 0:
            raise ValueError("tolerance_seconds must be >= 0")
        if any(self.tolerance_seconds >= value for value in labels):
            raise ValueError("tolerance_seconds must be smaller than label values")
        object.__setattr__(self, "labels", labels)

    @property
    def target_count(self) -> int:
        return len(self.labels)


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
    """严格因果滚动标准化（§12）：统计量只能来自当前时刻之前。"""

    mode: str = "causal_rolling"
    normalize_window: int = 180

    def __post_init__(self) -> None:
        if self.mode != "causal_rolling":
            raise ValueError("normalization.mode must be 'causal_rolling'")
        if self.normalize_window < 2:
            raise ValueError("normalization.normalize_window must be >= 2")


@dataclass(frozen=True)
class LoaderConfig:
    """DataLoader 装配参数。"""

    batch_size: int = 128
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None  # None = PyTorch 默认


@dataclass(frozen=True)
class ModelConfig:
    """模型结构配置；输出维度由数据包 labels 决定。"""

    name: str = "cnn1"


@dataclass(frozen=True)
class BaselineConfig:
    """唯一支持的 baseline 配置：Ridge。"""

    names: tuple[str, ...] = ("ridge",)
    ridge_alpha: float = 1.0




@dataclass(frozen=True)
class TrainingConfig:
    """训练配置：验证阶段只使用快速的 val/mse 或 val/mae 监控。"""

    loss: str = "huber"
    loss_huber_delta: float = 1.0
    epochs: int = 50
    patience: int = 10
    monitor_metric: str = "val/mse"
    monitor_mode: str = "min"
    learning_rate: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1e-5
    log_interval_epochs: int = 1


@dataclass(frozen=True)
class EvaluationConfig:
    """测试阶段完整评估配置：标量指标、日级 IC 与诊断曲线。"""

    prediction_bins: int = 10

    def __post_init__(self) -> None:
        if (
            isinstance(self.prediction_bins, bool)
            or not isinstance(self.prediction_bins, int)
            or self.prediction_bins < 2
        ):
            raise ValueError("evaluation.prediction_bins must be an integer >= 2")


@dataclass(frozen=True)
class SplitConfig:
    """单次 chronological 切分（§15），禁止 random row split。"""

    strategy: str = "chronological"
    train_ratio: float = 0.6
    validation_ratio: float = 0.2
    train_dates: tuple[str, str] | None = None  # 可选显式日期范围（含两端，%Y-%m-%d）
    validation_dates: tuple[str, str] | None = None
    test_dates: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        if self.strategy != "chronological":
            raise ValueError("split.strategy must be 'chronological'")
        if not 0 < self.train_ratio < 1:
            raise ValueError("split.train_ratio must be between 0 and 1")
        if not 0 < self.validation_ratio < 1:
            raise ValueError("split.validation_ratio must be between 0 and 1")
        if self.train_ratio + self.validation_ratio >= 1:
            raise ValueError("train_ratio + validation_ratio must be < 1")


@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward 方案及执行范围（§16）。

    先基于完整历史生成固定 folds，再从 ``start_fold`` 开始执行 ``num_folds``
    个 fold；执行范围不会截断任何 fold 的训练历史。fold 编号从1开始。
    """

    enabled: bool = True
    train_window_days: int = 60
    validation_window_days: int = 20
    test_window_days: int = 20
    step_days: int = 20
    start_fold: int = 1
    num_folds: int | None = None

    def __post_init__(self) -> None:
        window_fields = {
            "train_window_days": self.train_window_days,
            "validation_window_days": self.validation_window_days,
            "test_window_days": self.test_window_days,
            "step_days": self.step_days,
        }
        invalid = [name for name, value in window_fields.items() if value <= 0]
        if invalid:
            raise ValueError(f"walk_forward day parameters must be > 0: {invalid}")
        if self.start_fold <= 0:
            raise ValueError("walk_forward.start_fold must be > 0")
        if self.num_folds is not None and self.num_folds <= 0:
            raise ValueError("walk_forward.num_folds must be > 0")


@dataclass(frozen=True)
class FoldSelectionConfig:
    """训练阶段只负责选择数据包中已经生成的 folds。"""

    start_fold: int = 1
    num_folds: int | None = None

    def __post_init__(self) -> None:
        if self.start_fold <= 0:
            raise ValueError("folds.start_fold must be > 0")
        if self.num_folds is not None and self.num_folds <= 0:
            raise ValueError("folds.num_folds must be > 0")

@dataclass(frozen=True)
class BaselineRunConfig:
    """共享 Ridge baseline 实验配置：覆盖选定 fold。"""

    experiment_id: str
    loader: LoaderConfig
    baselines: BaselineConfig
    evaluation: EvaluationConfig
    folds: FoldSelectionConfig = field(default_factory=FoldSelectionConfig)
    seed: int = 42

@dataclass(frozen=True)
class DataBuildConfig:
    """阶段一配置：从原始行情构建不可变数据集。"""

    task: TaskConfig
    data: DataConfig
    cleaning: CleaningConfig
    target: TargetConfig
    sessions: SessionConfig
    window: WindowConfig
    features: FeatureConfig
    normalization: NormalizationConfig
    split: SplitConfig
    walk_forward: WalkForwardConfig

    @property
    def ticker(self) -> str:
        return self.task.ticker

    @property
    def target_columns(self) -> tuple[str, ...]:
        short = {"log_mid_return": "log", "simple_mid_return": "simple"}[self.target.type]
        return tuple(f"Target_{label}s_{short}" for label in self.target.labels)


@dataclass(frozen=True)
class ModelRunConfig:
    """阶段二配置：消费数据集元数据并训练候选模型。"""

    experiment_id: str
    loader: LoaderConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    folds: FoldSelectionConfig = field(default_factory=FoldSelectionConfig)
    seed: int = 42
