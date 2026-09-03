# hft_lob

单股限价订单簿（LOB）收益率预测框架。整个流程只有两个阶段：

```text
阶段一：原始行情 → 不可变训练数据集
阶段二：不可变训练数据集 → 模型训练与测试
```

两个阶段通过数据集目录连接。训练不会运行清洗、标签、滑动窗口或数据分割。

## 安装

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync --dev
uv run pre-commit install
uv run hft_lob --help
```

## 阶段一：构建数据集

在 `configs/dataset.yaml` 中设置原始数据位置和构建参数。当前真实数据位于：

```yaml
data:
  raw_dir: C:/Users/GF/hft-project/raw_data
```

不要将 `raw_dir` 直接设置为 `raw_data/688981`。程序会根据：

```yaml
task:
  ticker: "688981"
```

自动定位 `raw_dir/688981/`。

原始数据建议按交易日保存：

```text
<raw_dir>/688981/
├── 20250102.parquet
├── 20250103.parquet
└── ...
```

构建真实数据集：

Bash：

```bash
uv run hft_lob data build \
  --config configs/dataset.yaml \
  --output-root data/datasets/688981
```

PowerShell：

```powershell
uv run hft_lob data build `
  --config configs/dataset.yaml `
  --output-root data/datasets/688981
```

一行写法：

```powershell
uv run hft_lob data build --config configs/dataset.yaml --output-root data/datasets/688981
```

`data build` 成功发布数据集后会自动运行共享 Ridge baseline，并输出 baseline manifest。只需要数据集、不运行 baseline 时可显式跳过：

```powershell
uv run hft_lob data build `
  --config configs/dataset.yaml `
  --output-root data/datasets/688981 `
  --skip-baseline
```

构建结果会输出到 `data/datasets/688981/<dataset_id>/`，同时生成 `loggers/results/baselines/<dataset_id>/default_manifest.yaml`。验证构建结果：

```powershell
uv run hft_lob data verify `
  --dataset-dir data/datasets/688981/<dataset_id>
```

查看元数据：

```powershell
uv run hft_lob data inspect `
  --dataset-dir data/datasets/688981/<dataset_id>
```

命令完成后会打印数据集目录：

```text
data/datasets/688981/<dataset_id>/
├── dataset.json
├── features.npy
├── targets.npy
├── validity.npy
├── market.npy
├── rows.parquet
├── quality.parquet
├── folds/
│   └── fold_001/
│       ├── train.parquet
│       ├── validation.parquet
│       └── test.parquet
└── _SUCCESS
```

baseline 已在数据集发布后自动完成；如果 baseline 配置 hash 未变化，重复执行会复用已有 default manifest，文件位于 `loggers/results/<dataset_id>/baseline/manifest.yaml`。

数据集发布后应保持只读。需要完整检查时运行：

```bash
uv run hft_lob data verify --dataset-dir data/datasets/688981/<dataset_id>
```

查看数据集元数据：

```bash
uv run hft_lob data inspect --dataset-dir data/datasets/688981/<dataset_id>
```

## 阶段二：共享 Ridge baseline（可单独重跑）

通常无需手动执行 baseline。需要使用不同配置或重新发布 default manifest 时，可以单独运行：

```bash
uv run hft_lob baseline run \
  --config configs/baselines.yaml \
  --dataset-dir data/datasets/688981/<dataset_id> \
  --experiment-id baseline-688981
```

如果 baseline 配置 hash 与当前 default 不同，必须显式替换：

```bash
uv run hft_lob baseline run \
  --config configs/baselines.yaml \
  --dataset-dir data/datasets/688981/<dataset_id> \
  --experiment-id baseline-688981-v2 \
  --replace-default
```

## 阶段三：训练模型

模型配置只包含模型、训练、评测和 fold 选择，不再包含 baseline candidate：

```yaml
model:
  name: cnn1
  output_dim: 1

folds:
  start_fold: 1
  num_folds: 1
```

模型启动前会校验数据集 default Ridge baseline manifest、请求 fold 覆盖范围及 Ridge artifact；
缺失或不一致时不会开始模型训练。

每个已注册模型都有可直接加载的默认模板，位于 `configs/models/`：
`cnn1.yaml`、`cnn2.yaml`、`deeplob.yaml`、`transformer.yaml`、`itransformer.yaml`、
`lobtransformer.yaml`、`axiallob.yaml`、`dla.yaml`、`binbtabl.yaml`、`binctabl.yaml`
和 `hlob.yaml`。选择对应文件即可替代手工修改 `configs/train.yaml`，例如：

```bash
uv run hft_lob train \
  --config configs/models/transformer.yaml \
  --dataset-dir data/datasets/688981/<dataset_id> \
  --experiment-id transformer-688981
```

使用 DeepLOB 模型训练：

```bash
uv run hft_lob train \
  --config configs/models/deeplob.yaml \
  --dataset-dir data/datasets/688981/<dataset_id> \
  --experiment-id deeplob-688981
```

启动训练：

```bash
uv run hft_lob train \
  --config configs/train.yaml \
  --dataset-dir data/datasets/688981/<dataset_id> \
  --experiment-id cnn1-688981
```

指定物理 GPU：

```bash
uv run hft_lob train \
  --config configs/train.yaml \
  --dataset-dir data/datasets/688981/<dataset_id> \
  --experiment-id cnn1-688981 \
  --gpu-id 0
```

不传 `--gpu-id` 时由 Lightning 自动选择设备。同一份只读数据集可以被多个训练进程或不同模型同时消费，但每个进程应使用不同的 `experiment-id`。

训练中的验证集只计算快速标量指标 `val/mse` 和 `val/mae`，用于 checkpoint
选择与 early stopping；不会生成预测 artifact、评测报告或曲线。测试集在训练结束后
单独运行完整评估，额外计算日级 IC、正 IC 日占比、预测分桶，并生成下方的
`evaluation.yaml` 与两张诊断曲线。


模型训练结果保存在：

```text
output/<experiment_id>/
├── config_used.yaml
└── walk_forward/
    └── fold_001/
        └── <model>/
            ├── checkpoints/best_val_model.ckpt
            ├── model_config.yaml
            ├── model_metadata.yaml
            ├── predictions.parquet
            ├── evaluation.yaml
            ├── daily_ic_curve.png
            └── time_series_grouped_return_curve.png
```

每个 fold/model 目录都是可移动的自包含模型目录。独立测试只需要测试数据集、
模型名称和该模型目录，不读取训练实验目录：

```bash
uv run hft_lob test \
  --test-data-dir data/datasets/688981/<test_dataset_id> \
  --model-name cnn1 \
  --model-dir output/<experiment_id>/walk_forward/fold_001/cnn1
```

使用刚训练的 DeepLOB checkpoint 进行独立测试：

```bash
uv run hft_lob test \
  --test-data-dir data/datasets/688981/<test_dataset_id> \
  --model-name deeplob \
  --model-dir output/deeplob-688981/walk_forward/fold_001/deeplob \
  --output-dir output/deeplob-688981/standalone_test
```

可通过 `--output-dir` 指定输出位置；默认写入
`output/standalone_test/<model_version>/<test_dataset_id>/`。测试数据集必须
与 `model_metadata.yaml` 记录的特征顺序、窗口、采样间隔、归一化和目标契约一致，
且必须包含 `model_metadata.yaml` 中 `fold_index` 对应的 test split（评测在该 fold
定义的测试窗口上进行，不会用新划分训练或验证）。命令严格加载训练生成的 Lightning
checkpoint，只运行 test，不训练或重新选择 checkpoint。

共享 baseline 结果和权威引用保存在数据集实验空间：

```text
output/<dataset_id>/baseline/
├── manifest.yaml
└── runs/
    └── <baseline_run_id>/
        └── fold_001/
            └── ridge/
```
评测报告中的 `mean_daily_ic` 是各交易日 TS-IC 的有限值算术平均；`daily_ic_curve.png`
绘制按日期排列的逐日 TS-IC，`time_series_grouped_return_curve.png` 绘制时序分组收益曲线。
后者将评测窗口内的全部有效样本按预测值排序后分成 `k` 个等量 bin，绘制各 bin
真实收益均值；它不是按每个时点做横截面排序的普通分组收益曲线。

## 快速检查

提交前运行全部 pre-commit 检查：

```bash
uv run pre-commit run --all-files
```

完整检查：

```bash
uv run pytest -q
uv run ruff check src tests
uv run mypy src/hft_lob
```

核心代码结构：

```text
src/hft_lob/
├── application/     # dataset、baseline、train、独立测试应用用例
├── cli/             # 命令行参数解析、分发与结果输出
├── configs/         # Python 配置契约与 YAML 配置
├── data_types.py    # 跨模块共享的底层 LOBBatch / SampleMeta 类型
├── data_pipeline/   # 数据加载、处理、切分与写入
├── datasets/        # LOB Dataset 与 Lightning DataModule
├── baselines/       # Ridge baseline 与 baseline manifest
├── models/          # PyTorch 模型、模型工厂与 model bundle
├── trainner/        # 训练模块、损失函数、executor 与 walk-forward
├── metrics/         # 评测指标
├── reporting/       # 预测产物与评测报告
├── utils/           # seed、实验目录、配置 hash 等通用工具
└── main.py          # CLI 兼容入口
```
