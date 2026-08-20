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
├── configs/                  # 强类型实验配置
├── preprocessing/            # session 清洗、特征、标签、manifest、split、normalizer
├── datasets/                 # LOBWindowDataset / LOBBatch
├── models/                   # 纯 nn.Module 模型
├── baselines/                # Zero / Imbalance / Ridge 及运行适配器
├── systems/                  # DataModule、Lightning wrapper、artifact、评估、walk-forward
└── utils/                    # 配置加载、实验、日志、检查点、随机种子
```

统一接口闭环：

```text
PreparedDataset → WalkForwardPlan → fold normalizer → LOBBatch
→ model/baseline → PredictionArtifact → EvaluationReport → FoldResult
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
