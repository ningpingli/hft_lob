# hft_lob 接口抄写任务书（dsh 执行）

## 背景（≤5 行）

新仓库 `~/projects/hft_lob` 已初始化（pyproject + mypy + ruff + README），目前只有空目录与根 `__init__.py`。本阶段把所有 lobx 公开接口**一比一**抄到 hft_lob，仅签名 + Google 中文 docstring + `raise NotImplementedError(...)` 占位，**不实装任何业务逻辑**。后续阶段再逐模块填实现。

## 项目根

`/home/ubuntu/projects/hft_lob`（派单 cd 目标）

## 必读（dsh 动手前）

- `/home/ubuntu/projects/hft_lob/pyproject.toml` —— mypy / ruff / 项目元数据（硬约束）
- `/home/ubuntu/projects/hft_lob/.gitignore` —— 忽略规则
- `/home/ubuntu/projects/hft_lob/README.md` —— 当前阶段说明
- lobx 源（**只读**，用于抄签名 + 抄 docstring 摘要，不要抄实现）：
  - `/home/ubuntu/projects/lobx/data_processing/{fields,data_process,data_process_utils,complete_homological_utils}.py`
  - `/home/ubuntu/projects/lobx/loaders/custom_dataset.py`
  - `/home/ubuntu/projects/lobx/loggers/logger.py`
  - `/home/ubuntu/projects/lobx/optimizers/{executor,lightning_batch_gd,losses}.py`
  - `/home/ubuntu/projects/lobx/simulator/{market_sim,post_trading_analysis,trading_agent}.py`
  - `/home/ubuntu/projects/lobx/models/**/*.py`
  - `/home/ubuntu/projects/lobx/utils.py`
  - `/home/ubuntu/projects/lobx/main.py`
- **风格参考（lobx 现有中文 docstring 写法）**：上述各模块开篇的模块 docstring 与函数 docstring 风格，可在保留严谨度的前提下做精简（用户明确要求"不要写得很啰嗦"）

## 目标（验收标准，可机器验证）

- [ ] `src/hft_lob/__init__.py` 存在并暴露 `__version__ = "0.1.0"` 与 `main() -> None`
- [ ] 27 个 Python 源文件全部就位（见下方"改动范围声明"列表）
- [ ] `uv run mypy src/hft_lob` 通过（exit 0）
- [ ] `uv run ruff check src/hft_lob` 通过（exit 0）
- [ ] `python -c "from hft_lob.data_processing.fields import FieldsConfig, TimeConfig; from hft_lob.optimizers.executor import Executor, build_model; from hft_lob.models.DeepLob.deeplob import DeepLOB; from hft_lob.simulator.trading_agent import Trading; print('imports ok')"` 通过
- [ ] 所有函数/方法/类方法的 `raise NotImplementedError` 字符串包含函数名（如 `raise NotImplementedError("process_data not implemented")`），便于主控 grep 验证完整性

## 改动范围声明（file scoping）

**只允许在以下 27 个文件写入新内容**（按 lobx 一比一对应）：

| hft_lob 路径 | lobx 对应路径 | 用途 |
|---|---|---|
| `src/hft_lob/__init__.py` | `src/lobx/__init__.py` | 包入口 + `main()` shim |
| `src/hft_lob/main.py` | `lobx/main.py` | CLI 入口 `parse_args` + `main` |
| `src/hft_lob/utils.py` | `lobx/utils.py` | `load_yaml` / `data_split` / `save_dataset_info` / `load_experiment_config` / `create_hyperparameters_yaml` / `create_tree` |
| `src/hft_lob/configs/__init__.py` | 新建 | configs 包入口（空 `__init__.py`，无公开符号） |
| `src/hft_lob/data_processing/__init__.py` | `lobx/data_processing/__init__.py` | 数据处理包入口（按需 re-export） |
| `src/hft_lob/data_processing/fields.py` | `lobx/data_processing/fields.py` | `TimeConfig` / `FieldsConfig` |
| `src/hft_lob/data_processing/data_process.py` | `lobx/data_processing/data_process.py` | `LABEL_TYPES` / `process_data` + 私有常量 `_FAMILY_SHORT` / `_LEGACY_MINUTES` / `_LEVEL_FAMILIES` / `_A_SHARE_SESSIONS` |
| `src/hft_lob/data_processing/data_process_utils.py` | `lobx/data_processing/data_process_utils.py` | `DataUtils` |
| `src/hft_lob/data_processing/complete_homological_utils.py` | `lobx/data_processing/complete_homological_utils.py` | `compute_pairwise_mi` / `process_file` / `mean_tmfg` / `extract_components` / `execute_pipeline` / `get_complete_homology` |
| `src/hft_lob/loaders/__init__.py` | 新建 | loaders 包入口（空 `__init__.py`） |
| `src/hft_lob/loaders/custom_dataset.py` | `lobx/loaders/custom_dataset.py` | `CustomDataset`（继承 `torch.utils.data.Dataset`） |
| `src/hft_lob/loggers/__init__.py` | `lobx/loggers/__init__.py` | loggers 包入口 |
| `src/hft_lob/loggers/logger.py` | `lobx/loggers/logger.py` | `generate_id` / `find_save_path` / `logger` |
| `src/hft_lob/optimizers/__init__.py` | `lobx/optimizers/__init__.py` | optimizers 包入口 |
| `src/hft_lob/optimizers/executor.py` | `lobx/optimizers/executor.py` | `validate_model_data_contract` / `validate_training_contract` / `build_model` / `Executor` |
| `src/hft_lob/optimizers/lightning_batch_gd.py` | `lobx/optimizers/lightning_batch_gd.py` | `LOBLightningModule` / `BatchGDManager` |
| `src/hft_lob/optimizers/losses.py` | `lobx/optimizers/losses.py` | `LOSS_NAMES` / `build_loss` |
| `src/hft_lob/simulator/__init__.py` | `lobx/simulator/__init__.py` | simulator 包入口 |
| `src/hft_lob/simulator/market_sim.py` | `lobx/simulator/market_sim.py` | `backtest` |
| `src/hft_lob/simulator/post_trading_analysis.py` | `lobx/simulator/post_trading_analysis.py` | `post_trading_analysis` |
| `src/hft_lob/simulator/trading_agent.py` | `lobx/simulator/trading_agent.py` | `Trading` |
| `src/hft_lob/models/AxialLob/axiallob.py` | `lobx/models/AxialLob/axiallob.py` | `GatedAxialAttention` / `AxialLOB` |
| `src/hft_lob/models/CNN1/cnn1.py` | `lobx/models/CNN1/cnn1.py` | `CNN1` |
| `src/hft_lob/models/CNN2/cnn2.py` | `lobx/models/CNN2/cnn2.py` | `CNN2` |
| `src/hft_lob/models/CompleteHCNN/complete_hcnn.py` | `lobx/models/CompleteHCNN/complete_hcnn.py` | `Complete_HCNN` |
| `src/hft_lob/models/DLA/DLA.py` | `lobx/models/DLA/DLA.py` | `DLA` |
| `src/hft_lob/models/DeepLob/deeplob.py` | `lobx/models/DeepLob/deeplob.py` | `DeepLOB` |
| `src/hft_lob/models/LobTransformer/lobtransformer.py` | `lobx/models/LobTransformer/lobtransformer.py` | `LobTransformer` |
| `src/hft_lob/models/TABL/bin_nn.py` | `lobx/models/TABL/bin_nn.py` | `BiN` |
| `src/hft_lob/models/TABL/bin_tabl.py` | `lobx/models/TABL/bin_tabl.py` | `BiN_BTABL` / `BiN_CTABL` |
| `src/hft_lob/models/TABL/bl_layer.py` | `lobx/models/TABL/bl_layer.py` | `BL_layer` |
| `src/hft_lob/models/TABL/tabl_layer.py` | `lobx/models/TABL/tabl_layer.py` | `TABL_layer` |
| `src/hft_lob/models/Transformer/transformer.py` | `lobx/models/Transformer/transformer.py` | `SinusoidalPositionalEmbedding` / `Transformer` |
| `src/hft_lob/models/iTransformer/itransformer.py` | `lobx/models/iTransformer/itransformer.py` | `ITransformer` |

禁止改动：

- `/home/ubuntu/projects/lobx/`（**只读**：lobx 是已部署的生产代码，**禁止任何写操作**）
- `/home/ubuntu/projects/hft_lob/pyproject.toml`（已在主控阶段固化）
- `/home/ubuntu/projects/hft_lob/README.md`
- `/home/ubuntu/projects/hft_lob/.gitignore`
- 任何 YAML / 配置数据真源（**本阶段不复制 `configs/experiment.yaml` 等**，仅创建包目录 `__init__.py`）
- `/home/ubuntu/projects/hft_lob/tests/`（本阶段不写测试，由主控在质量门阶段做烟雾测试）

## 执行纪律（三铁律 + git 唯一写者）

1. **Plan 先行**：动手前先读「必读」+ 目标文件，输出改动清单（27 个文件 × 每文件抄写的类/函数/常量），再开始实现
2. **测试先行（TDD）**：本阶段无单元测试；改完后主控会跑 `mypy + ruff + import smoke`，请在 dsh 自检中**先**用 `python3 -c "import ast; ast.parse(open('src/hft_lob/<file>.py').read())"` 逐文件验证语法，再跑 mypy 自检
3. **小步报告**：每写完 1 个子模块报告一次（不攒到最后）
4. **不 commit**：git 唯一写者 = Hermes 主控（主控验证全绿后统一 commit）；dsh 完成即报告，禁止 `git commit / git add / git push / git reset`

## 模型等级（复杂度路由）

- [x] **常规（Flash-max 档）**：deepseek-v4-flash（无 patch）—— **27 个文件跨模块接口抄写**
  - **降级理由**：本机 `~/.dsh/profiles/headless/` 不存在 `pro-model.patch.yml`，Pro 档无法加载；按 memory 实测"Flash-max 常规阶段 3/3 通过"，任务书已按 Pro 粒度细化（明确每个文件抄写的符号清单、docstring 模板、类型规范），Flash 可逐文件机械照抄，跨文件类型对账风险已消除
  - 派单命令不带 `--patch`
  - 兜底：若 dsh 报告出现类型错漏 > 3 文件，回炉升级路径见「回炉与升级」节

## 视觉场景（派单前勾选）

- [ ] 本阶段需要视觉（看设计稿/截图对照/截图自检）→ **派单必须带** vision-model.patch.yml
- [x] 本阶段纯编码无视觉 → **不带** vision-model.patch.yml

## 实施要点

### 1. 文件骨架格式（统一模板）

每个 `.py` 文件统一遵循以下结构：

```python
"""<一句话中文模块说明>。"""

from __future__ import annotations

# 1. 必要 import（仅类型与抽象基类，不引业务模块）
# 2. 模块级常量（按 lobx 原样）
# 3. 类（继承抽象基类时显式 import torch / nn / Dataset / pl.LightningModule）
# 4. 模块级函数（按 lobx 原顺序）
```

**类骨架示例**：

```python
class FieldsConfig:
    """字段配置：从 dict / YAML 构造。

    Attributes:
        time: 时间字段配置。
        column_map: 原始列名到标准列名的映射。
    """

    def __init__(self, time: TimeConfig, column_map: dict[str, str]) -> None:
        """初始化字段配置。

        Args:
            time: 时间字段配置。
            column_map: 原始列名到标准列名映射。
        """
        raise NotImplementedError("FieldsConfig.__init__ not implemented")
```

**方法体规范**：

```python
def process_data(ticker: str, ...) -> None:
    """处理单个 ticker 的原始数据，产出标准化 / 未标准化 CSV。

    Args:
        ticker: 股票代码。
        ...
    """
    raise NotImplementedError("process_data not implemented")
```

### 2. 类型标注规则（mypy 必须过）

- **强制**：`from __future__ import annotations` 在每个文件首行（PEP 563）
- **强制**：所有函数/方法参数 + 返回类型**显式标注**（不省略）
- **强制**：模型类 `__init__` 参数类型精确照抄 lobx；forward 输入输出使用 `torch.Tensor`
- **强制**：使用 PEP 604 语法 `int | None`（lobx 已是 Python 3.12）
- **复杂外部类型**：torch / lightning / wandb / polars 等库**没有官方 typing**，使用 `Any` 时**仅限**参数声明外部对象边界（如 `data_features: Any`），且必须在该行末尾加 `# noqa: ANN401` 注释或在 module 顶部 `# mypy: disable-error-code="any-expr,no-untyped-def"`
- **避免**：`# type: ignore` 只在确有必要时使用（库边界/未实现的方法签名），**禁止**用 `cast` / `Any` 漏勺
- **`mypy.ini` 允许 `ignore_missing_imports = true`**，所以缺库不会报错；但**业务代码自身类型必须严格**
- **dataclass 替代手写 `__init__`**：仅当 lobx 已用 dataclass 时保持；本阶段保持 lobx 现状（不重构为 dataclass）

### 3. docstring 规范（Google 风格，中文）

每个公开类/函数/方法**必须有** docstring，结构：

```python
"""<一句话概要>。

<详细说明（可选，1-3 句话）>。

Args:
    <name>: <描述，每行一个>。
Returns:
    <描述>。
Raises:
    <描述>。
"""
```

- **不要照搬 lobx 的长 docstring**——用户明确要求"不要写得很啰嗦"。精简到"一句话说明 + 必要参数/返回说明"
- **模块 docstring**：仅一句说明模块用途，不写背景
- **常量 docstring**：模块级常量（如 `LABEL_TYPES`）用单行 `#:` 注释，**不写 docstring**
- **类继承签名**：保留 `class CustomDataset(Dataset)`、`class LOBLightningModule(pl.LightningModule)` 等继承关系

### 4. 模型类特殊处理

模型类继承 `torch.nn.Module`，forward 必须返回 `torch.Tensor`：

```python
class DeepLOB(nn.Module):
    """DeepLOB 双流 CNN 模型。"""

    def __init__(self, num_features: int = ..., levels: int = ...) -> None:
        """初始化 DeepLOB。

        Args:
            num_features: 单层特征维度。
            levels: 盘口档位。
        """
        raise NotImplementedError("DeepLOB.__init__ not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量。
        Returns:
            模型输出。
        """
        raise NotImplementedError("DeepLOB.forward not implemented")
```

**模型参数默认值**：保持 lobx 现状（部分有默认值，部分无）。`__init__` 中 `num_features: int = ...` 是 lobx 的写法——hft_lob 也保留相同写法（`...` 是 Ellipsis，合法默认）。

### 5. import 策略

- `from __future__ import annotations`：**每个文件首行**
- 类型仅引：`from typing import Any` / `from pathlib import Path` 等**仅类型/路径用到**的 import
- 业务模块互引：`from hft_lob.configs import ...` / `from hft_lob.data_processing.fields import FieldsConfig`
- **绝对不引** lobx 任何路径
- `from hft_lob import configs as configs_module` 之类**循环风险**，**优先**懒导入或在 docstring 中标注"运行时再导入"——本阶段保持简单，必要时用 `# noqa: E402` 标注延后 import

### 6. __init__.py 内容

- `src/hft_lob/__init__.py`：`__version__ = "0.1.0"` + `main() -> None` 占位（raise NotImplementedError）
- 子包 `__init__.py`：仅 `"""<包名> 包。"""` 文档字符串，无 re-export（本阶段简化）
- `src/hft_lob/main.py`：参照 lobx/main.py，但**不** `import` lobx 子模块——本阶段只抄签名（`parse_args() -> Any` + `main() -> None`），body 用 `raise NotImplementedError`

### 7. 私有符号处理

- **模块级私有常量**（如 `_TIME_KEY`、`_CANONICAL_SIZE_COLUMNS`、`_LABEL_TYPES`）：**保留**——它们是公开 API 的支持符号，mypy/IDE 推断会用
- **下划线开头方法**：**不抄**——只抄公开接口
- 私有方法实现一律 `raise NotImplementedError`，**不允许 `pass`**

### 8. 严禁

- 禁止 `pass`（用 `raise NotImplementedError`）
- 禁止抄 lobx 实现（哪怕是 1 行算法）
- 禁止写 YAML / 数据文件 / 测试 / .ipynb
- 禁止 commit / push / 改 pyproject / 改 README
- 禁止把 lobx 的注释、字符串常量原样复制——只抄**签名**与**行为契约**（docstring 摘要可借用意译）

## 完成标准（质量门，主控执行）

- [ ] `cd /home/ubuntu/projects/hft_lob && python3 -m mypy src/hft_lob` 全绿（exit 0）
- [ ] `cd /home/ubuntu/projects/hft_lob && python3 -m ruff check src/hft_lob` 全绿（exit 0）
- [ ] 主控 `git diff --stat` 审查：27 个新文件 + pyproject/.gitignore/README（已在主控阶段固化，不计入 dsh 改动）
- [ ] **可执行性铁门**：主控 `python3 -c "from hft_lob.data_processing.fields import FieldsConfig, TimeConfig; from hft_lob.optimizers.executor import Executor, build_model; from hft_lob.models.DeepLob.deeplob import DeepLOB; from hft_lob.simulator.trading_agent import Trading; print('imports ok')"` 可执行
- [ ] **末尾输出铁门**：dsh 报告里每条质量门命令必须贴**命令末尾输出**而非仅写"全绿"

## 报告格式（防止自报失实的最低要求）

dsh 完成后报告必须包含：

1. **改动清单**：27 个文件 × 每个文件抄写的公开符号（类/函数/常量）+ 对应原 lobx 行号
2. **红→绿记录**：每个 mypy / ruff 命令的**末尾输出**粘贴（`Success: no issues found` / `Found N errors` 等具体行）
3. **质量门命令输出**：每条质量门命令的**末尾输出**粘贴
4. **范围外发现**：列**具体文件/函数**而非"我注意到几个可能问题"

## 派单命令

```bash
cd /home/ubuntu/projects/hft_lob && dsh --profile headless "$(cat /home/ubuntu/projects/hft_lob/.hermes/tasks/01-interface-port.md)"
```

## 回炉与升级

- 主控验证失败 → 把失败原因写进新任务书重派（新会话，不 resume）
- 同一阶段重试 > 2 次 → 升级降级链（dsh → Codex → 人工）
- 回炉计数记录在 `/home/ubuntu/projects/hft_lob/.hermes/tasks/CHECKPOINT.md` 的 retry_count
