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

在 `configs/data.yaml` 中设置原始数据位置和构建参数。原始数据建议按交易日保存：

```text
<raw_dir>/688981/
├── 20250102.parquet
├── 20250103.parquet
└── ...
```

构建数据集：

```bash
uv run hft_lob data build \
  --config configs/data.yaml \
  --output-root data/datasets/688981
```

PowerShell：

```powershell
uv run hft_lob data build `
  --config configs/data.yaml `
  --output-root data/datasets/688981
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

数据集发布后应保持只读。需要完整检查时运行：

```bash
uv run hft_lob data verify --dataset-dir data/datasets/688981/<dataset_id>
```

查看数据集元数据：

```bash
uv run hft_lob data inspect --dataset-dir data/datasets/688981/<dataset_id>
```

## 阶段二：共享 baseline 实验

baseline 必须先于模型实验生成。一次 baseline 实验覆盖配置中的全部 baseline 和 fold：

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

模型启动前会校验数据集 default baseline manifest、请求 fold 覆盖范围及所有 baseline artifact；
缺失或不一致时不会开始模型训练。

每个已注册模型都有可直接加载的默认模板，位于 `configs/models/`：
`cnn1.yaml`、`cnn2.yaml`、`deeplob.yaml`、`transformer.yaml`、`itransformer.yaml`、
`lobtransformer.yaml`、`axiallob.yaml`、`dla.yaml`、`binbtabl.yaml`、`binctabl.yaml`
和 `hlob.yaml`。选择对应文件即可替代手工修改 `configs/model.yaml`，例如：

```bash
uv run hft_lob train \
  --config configs/models/transformer.yaml \
  --dataset-dir data/datasets/688981/<dataset_id> \
  --experiment-id transformer-688981
```

启动训练：

```bash
uv run hft_lob train \
  --config configs/model.yaml \
  --dataset-dir data/datasets/688981/<dataset_id> \
  --experiment-id cnn1-688981
```

指定物理 GPU：

```bash
uv run hft_lob train \
  --config configs/model.yaml \
  --dataset-dir data/datasets/688981/<dataset_id> \
  --experiment-id cnn1-688981 \
  --gpu-id 0
```

不传 `--gpu-id` 时由 Lightning 自动选择设备。同一份只读数据集可以被多个训练进程或不同模型同时消费，但每个进程应使用不同的 `experiment-id`。


模型训练结果保存在：

```text
loggers/results/<experiment_id>/
├── config_used.yaml
└── walk_forward/
    └── fold_001/
        └── <model>/
            ├── checkpoints/best_val_model.ckpt
            ├── predictions.parquet
            ├── evaluation.yaml
            ├── daily_ic_curve.png
            └── time_series_grouped_return_curve.png
```

共享 baseline 结果和权威引用保存在数据集实验空间：

```text
loggers/results/<dataset_id>/baseline/
├── manifest.yaml
└── runs/
    └── <baseline_run_id>/
        └── fold_001/
            ├── zero/
            ├── imbalance/
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
├── application/     # 数据构建与训练用例
├── datasets/        # 样本编译、fold 索引、数据包写入与校验
├── preprocessing/   # 清洗、特征、标签、标准化与分割
├── models/          # 神经网络模型
├── baselines/       # Zero、Imbalance、Ridge
├── systems/         # DataModule、训练执行、预测与评估
└── main.py          # 统一 CLI 入口
```
