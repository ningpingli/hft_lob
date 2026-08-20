# hft_lob

面向单只股票限价订单簿（LOB）的深度学习收益预测框架。项目覆盖原始盘口清洗、
严格因果特征标准化、60 秒收益标签、按交易日 walk-forward 切分、Lightning 训练、
预测产物和结构化评估。

```text
raw parquet → PreparedDataset → WalkForwardPlan → LOBDataModule
→ model/baseline → PredictionArtifact → EvaluationReport
```

## 环境安装

项目要求 Python 3.12 及 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync
uv run hft_lob --help
```

## 数据准备教程

### 1. 准备原始文件

默认配置读取 `data/raw/<ticker>/*.parquet`，例如：

```text
data/raw/000001/
├── 20260105.parquet
├── 20260106.parquet
└── ...
```

建议一个文件只包含一个交易日。每行是一条盘口快照，时间必须递增。项目内部
canonical schema 包含：

```text
timestamp
ASKp1, ASKs1, BIDp1, BIDs1
ASKp2, ASKs2, BIDp2, BIDs2
ASKp3, ASKs3, BIDp3, BIDs3
ASKp4, ASKs4, BIDp4, BIDs4
ASKp5, ASKs5, BIDp5, BIDs5
last, volume, amount
```

`timestamp` 必须能解析为 datetime。卖价应随档位递增，买价应随档位递减，
且正常盘口满足 `BIDp1 < ASKp1`。`trade_date` 和 `ticker` 可以由时间及配置补充。

如果数据源使用不同列名，在 `configs/experiment.yaml` 的
`data.column_mapping` 中配置“原始列名 → canonical 列名”：

```yaml
task:
  ticker: "000001"

data:
  raw_dir: data/raw
  processed_dir: data/processed
  manifest_dir: data/datasets
  column_mapping:
    Time: timestamp
    AskPrice1: ASKp1
    AskVolume1: ASKs1
    BidPrice1: BIDp1
    BidVolume1: BIDs1
    Volume: volume
```

五档盘口的其余列也需要完整映射。

### 2. 配置时间语义

```yaml
data:
  snapshot_interval_seconds: 3

target:
  type: log_mid_return
  horizon_seconds: 60
  tolerance_seconds: 3

window:
  history_snapshots: 180

normalization:
  mode: causal_rolling
  normalize_window: 180
```

这表示：

- 盘口按 3 秒快照处理；
- 标签为 `log(mid[t+60s] / mid[t])`；
- 每个输入窗口包含锚点在内的 180 条快照；
- 标准化统计量只使用当前时刻之前的 180 条记录；
- 窗口、标签和前向填充都不会跨越午休或交易日。

启用默认 walk-forward 参数时，第一个 fold 至少需要
`60 + 20 + 20 = 100` 个有效交易日。若执行 `N` 个 fold，通常至少需要：

```text
train_window_days + validation_window_days + test_window_days
+ (N - 1) * step_days
```

### 3. 执行数据准备

```powershell
uv run hft_lob --config configs/experiment.yaml --experiment-id demo_prepare --stages prepare-data --seed 42
```

成功后终端会输出：

```text
dataset_version=...
manifest_path=...
quality_report_path=...
fold_count=...
```

主要产物包括：

```text
data/processed/<ticker>/<dataset_version>/  # 按交易日/session 保存的处理数据
data/datasets/<ticker>/<dataset_version>/   # manifest 与质量报告
loggers/results/demo_prepare/config_used.yaml
loggers/results/demo_prepare/data.yaml
```

`manifest.parquet` 是训练切分的事实来源；不要通过移动或删除原始文件来表达
train/validation/test。

### 4. 检查数据质量

```python
import polars as pl

manifest = pl.read_parquet("<命令输出的 manifest_path>")
quality = pl.read_parquet("<命令输出的 quality_report_path>")

print(manifest.select("trade_date", "session_id", "valid_row_count"))
print(quality)
```

训练前至少确认：

- `valid_row_count` 不为零；
- 交易日数量足以生成配置的 walk-forward folds；
- crossed book、缺失率和无效档位数量符合数据源预期；
- 上午 session 没有被错误拼接。

## 训练教程

目前 CLI 已接通 `prepare-data`；`walk-forward`、`evaluate` 和
`predict-offline` 尚未接入 CLI 编排。下面使用现有 Python API 训练并评估一个 fold。

将以下内容保存为 `train_one_fold.py`：

```python
from dataclasses import asdict
from pathlib import Path

from hft_lob.configs import load_config
from hft_lob.models import build_model
from hft_lob.preprocessing import CausalRollingStandardizer, prepare_dataset
from hft_lob.systems import LOBDataModule, LOBLightningModule
from hft_lob.systems.lob_data_module import resolve_stage_files
from hft_lob.train import (
    build_checkpoint_callback,
    build_early_stopping_callback,
    build_trainer,
    run_test,
    run_training,
)
from hft_lob.utils.checkpoint_utils import backup_experiment_config
from hft_lob.utils.experiment_manager import resolve_log_dir
from hft_lob.utils.logger_builder import build_logger
from hft_lob.utils.seed import set_seed


EXPERIMENT_ID = "cnn1_demo"
FOLD_INDEX = 1

config = load_config("configs/experiment.yaml", experiment_id=EXPERIMENT_ID)
set_seed(config.seed)

# 数据已由 prepare-data 生成时，重复调用会解析出相同的内容寻址版本。
prepared = prepare_dataset(config)
stage_files = resolve_stage_files(prepared, fold_index=FOLD_INDEX)

standardizer = CausalRollingStandardizer(
    feature_cols=prepared.feature_columns,
    normalize_window=config.normalization.normalize_window,
)
datamodule = LOBDataModule(
    config,
    stage_files=stage_files,
    standardizer=standardizer,
)

model = build_model(config, feature_columns=prepared.feature_columns)
lightning_module = LOBLightningModule(
    model,
    config,
    dataset_version=prepared.dataset_version,
    model_version="v1",
    fold_index=FOLD_INDEX,
)

log_dir = Path(resolve_log_dir(EXPERIMENT_ID)) / f"fold_{FOLD_INDEX}"
log_dir.mkdir(parents=True, exist_ok=True)
backup_experiment_config(str(log_dir), asdict(config))

logger = build_logger(EXPERIMENT_ID, str(log_dir), hyperparams=asdict(config))
checkpoint = build_checkpoint_callback(
    str(log_dir),
    monitor=config.training.monitor_metric,
    mode=config.training.monitor_mode,
)
early_stopping = build_early_stopping_callback(
    monitor=config.training.monitor_metric,
    mode=config.training.monitor_mode,
    patience=config.training.patience,
)
trainer = build_trainer(
    str(log_dir),
    epochs=config.training.epochs,
    patience=config.training.patience,
    callbacks=[checkpoint, early_stopping],
    logger=logger,
    accelerator="auto",
    devices=1,
)

run_training(trainer, lightning_module, datamodule)
best_ckpt = checkpoint.best_model_path
if not best_ckpt:
    raise RuntimeError("training finished without a best checkpoint")

report = run_test(
    trainer,
    lightning_module,
    datamodule,
    ckpt_path=best_ckpt,
)
print("best checkpoint:", best_ckpt)
print("sample count:", report.sample_count)
print("overall metrics:", report.overall)
print("daily stability:", report.daily_summary)
```

执行：

```bash
uv run python train_one_fold.py
```

训练日志可通过 TensorBoard 查看：

```bash
uv run tensorboard --logdir loggers/results
```

然后访问终端显示的本地地址，通常是 `http://localhost:6006`。

### GPU 与恢复训练

Lightning 默认使用 `accelerator="auto"`。如需限制物理 GPU，应在 Python 启动前设置：

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
uv run python train_one_fold.py
```

恢复训练时将 checkpoint 传入：

```python
run_training(
    trainer,
    lightning_module,
    datamodule,
    ckpt_path="loggers/results/cnn1_demo/fold_1/best_val_model.ckpt",
)
```

恢复时必须沿用原实验的数据版本、特征 schema、模型配置和标准化配置。

## 包结构

```text
src/hft_lob/
├── configs/                  # 强类型实验配置
├── preprocessing/            # 清洗、特征、标签、manifest、split、normalizer
├── datasets/                 # LOBWindowDataset / LOBBatch
├── models/                   # 纯 torch.nn.Module 模型
├── baselines/                # Zero / Imbalance / Ridge / MLP
├── systems/                  # DataModule、Lightning wrapper、artifact、评估
└── utils/                    # 实验、TensorBoard、检查点、随机种子
```

## 开发验证

```bash
uv run mypy src/hft_lob
uv run ruff check src/hft_lob
uv run pytest tests/ -q
```

更完整的设计约束见 `doc/需求文档.md`。
