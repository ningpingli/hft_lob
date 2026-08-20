# hft_lob

面向单只股票限价订单簿（LOB）的收益预测框架，覆盖盘口清洗、严格因果标准化、
60 秒收益标签、按交易日 walk-forward、Lightning 训练、预测 artifact 和结构化评估。

```text
raw parquet → PreparedDataset → WalkForwardPlan → LOBDataModule
→ model/baseline → PredictionArtifact → EvaluationReport
```

## 环境安装

要求 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync
uv run hft_lob --help
```

## 原始数据和数据集构建

默认从 `data/raw/<ticker>/*.parquet` 读取数据：

```text
data/raw/000001/
├── 20250102.parquet
├── 20250103.parquet
└── ...
```

建议每个文件只包含一个交易日，快照时间递增。内部 canonical schema 为：

```text
timestamp
ASKp1, ASKs1, BIDp1, BIDs1
ASKp2, ASKs2, BIDp2, BIDs2
ASKp3, ASKs3, BIDp3, BIDs3
ASKp4, ASKs4, BIDp4, BIDs4
ASKp5, ASKs5, BIDp5, BIDs5
last, volume, amount
```

原始字段名称不同时，在 `configs/experiment.yaml` 的 `data.column_mapping` 中配置
“原始字段 → canonical 字段”，不需要修改清洗代码。

默认 walk-forward 第一折至少需要 `60 + 20 + 20 = 100` 个有效交易日。执行
`N` 折通常至少需要：

```text
train_window_days + validation_window_days + test_window_days
+ (N - 1) * step_days
```

构建数据集：

```bash
uv run hft_lob \
  --config configs/experiment.yaml \
  --experiment-id prepare-000001 \
  --stages prepare-data
```

PowerShell：

```powershell
uv run hft_lob `
  --config configs/experiment.yaml `
  --experiment-id prepare-000001 `
  --stages prepare-data
```

成功后会打印：

```text
dataset_version=...
manifest_path=...
quality_report_path=...
fold_count=...
```

主要数据产物：

```text
data/processed/<ticker>/<dataset_version>/
data/datasets/<ticker>/<dataset_version>/manifest.parquet
data/datasets/<ticker>/<dataset_version>/quality_reports.parquet
```

`manifest.parquet` 是 split 的事实来源，不要移动 processed 文件来表达
train/validation/test。

## 训练配置

### 模型和 baseline

```yaml
model:
  name: cnn1                 # cnn1 或 deeplob
  output_dim: 1

baselines:
  names: [zero, imbalance, ridge, mlp]
```

一次 walk-forward 会执行 `model.name + baselines.names`。仅调试主模型时可关闭
baseline：

```yaml
baselines:
  names: []
```

### Fold 范围

```yaml
walk_forward:
  enabled: true
  train_window_days: 60
  validation_window_days: 20
  test_window_days: 20
  step_days: 20
  start_fold: 1
  num_folds: 3              # null 表示从 start_fold 执行到最后
```

`start_fold: 2, num_folds: 3` 表示执行固定方案中的第 2、3、4 折，不会截断任何
fold 的训练历史。

首次烟雾测试建议：

```yaml
training:
  epochs: 1

baselines:
  names: []

walk_forward:
  start_fold: 1
  num_folds: 1
```

确认 checkpoint、预测 parquet 和评估结果正常后，再恢复正式参数。

## 单 GPU 训练

`--gpu-id` 指定物理 GPU。绑定后，训练进程内部使用逻辑设备 `cuda:0`：

```bash
uv run hft_lob \
  --config configs/experiment.yaml \
  --experiment-id cnn1-production \
  --gpu-id 0 \
  --stages walk-forward
```

PowerShell：

```powershell
uv run hft_lob `
  --config configs/experiment.yaml `
  --experiment-id cnn1-production `
  --gpu-id 0 `
  --stages walk-forward
```

不传 `--gpu-id` 时，Lightning 使用 `accelerator="auto"` 自动选择设备。

## 从原始数据开始训练

同一任务中顺序执行数据构建和训练：

```bash
uv run hft_lob \
  --config configs/experiment.yaml \
  --experiment-id cnn1-full-run \
  --gpu-id 0 \
  --stages prepare-data walk-forward
```

只指定 `walk-forward` 时，CLI 也会使用相同配置解析内容寻址的数据版本，确保训练
使用的 manifest、特征 schema 和标签版本一致。

## 多 GPU 训练不同模型

为模型准备独立配置：

```text
configs/cnn1.yaml       model.name: cnn1
configs/deeplob.yaml    model.name: deeplob
```

Linux Bash：

```bash
uv run hft_lob --config configs/cnn1.yaml \
  --experiment-id cnn1-gpu0 --gpu-id 0 --stages walk-forward \
  > cnn1-gpu0.log 2>&1 &

uv run hft_lob --config configs/deeplob.yaml \
  --experiment-id deeplob-gpu1 --gpu-id 1 --stages walk-forward \
  > deeplob-gpu1.log 2>&1 &

wait
```

PowerShell：

```powershell
Start-Process uv -WindowStyle Hidden -ArgumentList @(
  "run", "hft_lob", "--config", "configs/cnn1.yaml",
  "--experiment-id", "cnn1-gpu0", "--gpu-id", "0", "--stages", "walk-forward"
)

Start-Process uv -WindowStyle Hidden -ArgumentList @(
  "run", "hft_lob", "--config", "configs/deeplob.yaml",
  "--experiment-id", "deeplob-gpu1", "--gpu-id", "1", "--stages", "walk-forward"
)
```

不同进程必须使用不同的 `experiment-id`，否则产物可能互相覆盖。

## 训练产物

每个 candidate/fold 使用独立目录：

```text
loggers/results/<experiment_id>/
├── config_used.yaml
├── data.yaml
└── walk_forward/
    └── fold_001/
        ├── cnn1/
        │   ├── checkpoints/best_val_model.ckpt
        │   ├── standardizer.json
        │   └── predictions.parquet
        ├── zero/
        │   ├── standardizer.json
        │   └── predictions.parquet
        └── ...
```

`predictions.parquet` 包含 ticker、交易日、session、anchor timestamp、目标值、
预测值、盘口价格、模型版本、数据版本和 fold 编号。`data.yaml` 保存跨 fold 汇总。

checkpoint 和 early stopping 使用统一指标：

```yaml
training:
  monitor_metric: val/ts_ic
  monitor_mode: max
```

## 验证训练链路

下面的测试会真实运行 1 epoch CNN 训练、checkpoint 恢复、test 推理、预测 parquet
和评估报告：

```bash
uv run pytest tests/systems/test_executor.py -q
```

正式训练前建议运行：

```bash
uv run pytest tests/preprocessing/test_pipeline.py -q
uv run pytest tests/systems/test_executor.py -q
uv run ruff check src/hft_lob
uv run mypy src/hft_lob
```

如果 Polars 提示缺少 AVX2/FMA，请安装兼容运行时后再长时间训练。

## 包结构

```text
src/hft_lob/
├── configs/                  # 强类型实验配置
├── preprocessing/            # 清洗、特征、标签、manifest、split、normalizer
├── datasets/                 # LOBWindowDataset / LOBBatch
├── models/                   # 纯 torch.nn.Module 模型
├── baselines/                # Zero / Imbalance / Ridge / MLP
├── systems/                  # DataModule、Lightning、executor、artifact、评估
└── utils/                    # 实验、日志、检查点、随机种子
```

完整设计约束见 `doc/需求文档.md`。
