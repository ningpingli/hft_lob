# hft_lob

LOB（Limit Order Book）深度学习预测 / HLOB 框架的接口骨架仓库。

## 当前阶段

本仓库当前为 **接口抄写阶段**（interface skeleton ported from `lobx`）。

- 所有公开函数、类、模块常量均已按 lobx 一比一抄写签名（参数 / 返回类型）。
- 所有方法体均为 `raise NotImplementedError(...)` 占位，等待后续阶段实装。
- 所有公开接口已添加完整 type annotation，并通过 mypy / ruff 检查。
- docstring 使用 Google 风格（中文）。

## 包结构

```
src/hft_lob/
├── configs/                  # 实验配置（YAML 由后续阶段填入）
├── data_processing/          # 数据处理流水线
├── loaders/                  # torch 数据集与 DataLoader
├── models/                   # 模型库（CNN / Transformer / TABL / AxialLOB）
├── optimizers/               # 训练 executor / lightning module / losses
├── simulator/                # 回测 / 撮合 / 交易 agent
└── loggers/                  # 实验日志
```

## 开发

```bash
# 类型检查
uv run mypy src/hft_lob
# 代码风格
uv run ruff check src/hft_lob
# 导入健全性 + 烟雾测试
uv run pytest tests/ -q
```

## 来源

接口对照自 `ningpingli/lobx` 主分支提交 31d2811。