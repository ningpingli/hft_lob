"""模型层契约测试（需求文档 §18）：统一 ``forward(x) -> [B, 1]``。

覆盖：
- ``build_model`` 工厂对全部已注册模型（hlob 除外）实例化并对
  ``(2, 1, 100, 20)`` 前向得到 ``(2, 1)``；
- hlob 用最小假同调结构验证构造与前向；
- 契约注入断言（AxialLOB W/H、BiN d1/t1/t2、DLA num_snapshots）；
- 输入维度与构造契约不匹配时抛 ``ValueError``。
"""

from __future__ import annotations

import pytest
import torch

from hft_lob.configs.experiment import (
    EvaluationConfig,
    LoaderConfig,
    ModelConfig,
    ModelRunConfig,
    TrainingConfig,
)
from hft_lob.models import build_model

#: 5 档盘口契约（20 特征 / 100 快照 / 5 档）。
_FEATURES = 20
_HISTORY = 100
_LEVELS = 5

#: 可前向模型名（hlob 单列测试，其构造依赖同调结构）。
_FORWARD_NAMES = (
    "cnn1",
    "deeplob",
    "cnn2",
    "transformer",
    "itransformer",
    "lobtransformer",
    "axiallob",
    "dla",
    "binbtabl",
    "binctabl",
)


def _make_config(model_name: str) -> ModelRunConfig:
    return ModelRunConfig(
        experiment_id="test",
        loader=LoaderConfig(),
        model=ModelConfig(name=model_name),
        training=TrainingConfig(),
        evaluation=EvaluationConfig(),
    )


@pytest.fixture(scope="module")
def sample() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(2, _HISTORY, _FEATURES)


@pytest.mark.parametrize("name", _FORWARD_NAMES)
def test_forward_all_models(name: str, sample: torch.Tensor) -> None:
    model = build_model(
        _make_config(name),
        feature_columns=[f"f{i}" for i in range(_FEATURES)],
        history_snapshots=_HISTORY,
    )
    out = model(sample)
    assert out.shape == (2, 1)



def test_registered_model_output_width_follows_label_count(sample: torch.Tensor) -> None:
    model = build_model(
        _make_config("cnn1"),
        feature_columns=[f"f{i}" for i in range(_FEATURES)],
        history_snapshots=_HISTORY,
        target_count=3,
    )
    assert model(sample).shape == (2, 3)





def test_hlob_constructs_and_forwards_with_minimal_structures(
    sample: torch.Tensor,
) -> None:
    # 扁平索引布局：与 complete_homological_utils.execute_pipeline 的
    # chain.from_iterable 展平输出一致（前向要求扁平索引，嵌套会得到 5D 张量）。
    structures = {
        "tetrahedra": list(range(8)),
        "triangles": list(range(6)),
        "edges": list(range(4)),
    }
    model = build_model(
        _make_config("hlob"),
        feature_columns=[f"f{i}" for i in range(_FEATURES)],
        history_snapshots=_HISTORY,
        homological_structures=structures,
    )
    out = model(sample)
    assert out.shape == (2, 1)




def test_hlob_requires_homological_structures() -> None:
    with pytest.raises(ValueError, match="homological_structures"):
        build_model(
            _make_config("hlob"),
            feature_columns=[f"f{i}" for i in range(_FEATURES)],
            history_snapshots=_HISTORY,
        )


def test_contract_injections() -> None:
    axial = build_model(
        _make_config("axiallob"),
        feature_columns=[f"f{i}" for i in range(_FEATURES)],
        history_snapshots=_HISTORY,
    )
    assert axial.W == _FEATURES
    assert axial.H == _HISTORY

    btabl = build_model(
        _make_config("binbtabl"),
        feature_columns=[f"f{i}" for i in range(_FEATURES)],
        history_snapshots=_HISTORY,
    )
    assert btabl.BiN.t1 == _HISTORY
    assert btabl.BiN.d1 == _FEATURES
    assert btabl.BiN.t2 == _LEVELS

    dla = build_model(
        _make_config("dla"),
        feature_columns=[f"f{i}" for i in range(_FEATURES)],
        history_snapshots=_HISTORY,
    )
    assert dla.num_snapshots == _HISTORY


@pytest.mark.parametrize(
    ("name", "x"),
    [
        ("cnn1", torch.randn(2, _HISTORY, _FEATURES + 4)),
        ("deeplob", torch.randn(2, _HISTORY, _FEATURES + 4)),
        ("cnn2", torch.randn(2, _HISTORY, _FEATURES + 4)),
        ("transformer", torch.randn(2, _HISTORY, _FEATURES + 4)),
        ("lobtransformer", torch.randn(2, _HISTORY, _FEATURES + 4)),
        ("axiallob", torch.randn(2, _HISTORY, _FEATURES + 4)),
        ("dla", torch.randn(2, _HISTORY, _FEATURES + 4)),
        # iTransformer 只校验时间维（嵌入宽度绑定 history_length）。
        ("itransformer", torch.randn(2, _HISTORY // 2, _FEATURES)),
    ],
)
def test_input_dimension_mismatch_raises(name: str, x: torch.Tensor) -> None:
    model = build_model(
        _make_config(name),
        feature_columns=[f"f{i}" for i in range(_FEATURES)],
        history_snapshots=_HISTORY,
    )
    with pytest.raises(ValueError):
        model(x)
