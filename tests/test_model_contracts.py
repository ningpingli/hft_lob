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
from hft_lob.models.TABL import bin_tabl
from hft_lob.models.TABL.bin_tabl import enforce_weight_constraints

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


#: TABL 家族受范数约束的权重矩阵；显式列出，不依赖实现内部的模块遍历。
_TABL_WEIGHT_PATHS: dict[str, tuple[str, ...]] = {
    "binbtabl": ("BL.W1", "BL.W2", "TABL.W1", "TABL.W", "TABL.W2"),
    "binctabl": ("BL.W1", "BL.W2", "BL2.W1", "BL2.W2", "TABL.W1", "TABL.W", "TABL.W2"),
}

_MAX_WEIGHT_NORM = 10.0


def _build_tabl(name: str) -> torch.nn.Module:
    return build_model(
        _make_config(name),
        feature_columns=[f"f{i}" for i in range(_FEATURES)],
        history_snapshots=_HISTORY,
    )


def _tabl_weights(model: torch.nn.Module, name: str) -> list[torch.Tensor]:
    weights: list[torch.Tensor] = []
    for path in _TABL_WEIGHT_PATHS[name]:
        target: torch.nn.Module = model
        for attribute in path.split("."):
            target = getattr(target, attribute)
        weights.append(target)
    return weights


def _inflate(weights: list[torch.Tensor], norm: float) -> None:
    with torch.no_grad():
        for weight in weights:
            weight.mul_(norm / _norm(weight))


def _norm(weight: torch.Tensor) -> float:
    return float(torch.linalg.matrix_norm(weight.detach()))


@pytest.mark.parametrize("name", _TABL_WEIGHT_PATHS)
def test_tabl_construction_satisfies_weight_constraints(name: str) -> None:
    """构造即满足范数约束：等价于旧实现「第一次前向内钳制」。"""
    torch.manual_seed(0)
    model = _build_tabl(name)

    for weight in _tabl_weights(model, name):
        assert _norm(weight) <= _MAX_WEIGHT_NORM * (1.0 + 1e-6)


@pytest.mark.parametrize("name", _TABL_WEIGHT_PATHS)
def test_tabl_forward_does_not_mutate_weights(name: str) -> None:
    """前向不再原地钳制权重（旧实现每次前向都要 matrix_norm + device→CPU 同步）。"""
    torch.manual_seed(0)
    model = _build_tabl(name)
    _inflate(_tabl_weights(model, name), 20.0)
    before = [weight.detach().clone() for weight in _tabl_weights(model, name)]

    model.eval()
    with torch.no_grad():
        model(torch.randn(2, _HISTORY, _FEATURES))

    for original, current in zip(before, _tabl_weights(model, name), strict=True):
        assert torch.equal(original, current)


@pytest.mark.parametrize("name", _TABL_WEIGHT_PATHS)
def test_enforce_weight_constraints_caps_every_tabl_weight(name: str) -> None:
    torch.manual_seed(0)
    model = _build_tabl(name)
    _inflate(_tabl_weights(model, name), 20.0)

    enforce_weight_constraints(model)

    for weight in _tabl_weights(model, name):
        assert _norm(weight) == pytest.approx(_MAX_WEIGHT_NORM, rel=1e-6)


@pytest.mark.parametrize("name", _TABL_WEIGHT_PATHS)
def test_constraint_timing_matches_forward_checked_clamping(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """约束移到优化器步之后后，「前向实际看到的权重」与旧实现逐位一致。

    旧实现 = 原始初始化 + 每次前向入口钳制（forward 内的原地钳制等价于此）；
    新实现 = 构造时钳制 + 每个优化器步之后钳制。两者梯度相同，故逐位可比。
    """
    torch.manual_seed(0)
    inputs = torch.randn(8, _HISTORY, _FEATURES)
    targets = torch.randn(8, 1)
    loss_fn = torch.nn.MSELoss()

    def run(*, old_forward_clamping: bool) -> tuple[list[list[torch.Tensor]], list[bool]]:
        torch.manual_seed(0)
        if old_forward_clamping:
            # 旧实现没有构造期钳制：构造后保留原始初始化。
            monkeypatch.setattr(bin_tabl, "enforce_weight_constraints", lambda _model: None)
        model = _build_tabl(name)
        monkeypatch.undo()
        model.eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        seen: list[list[torch.Tensor]] = []
        clamped: list[bool] = []
        for _ in range(4):
            if old_forward_clamping:
                enforce_weight_constraints(model)
            seen.append([weight.detach().clone() for weight in _tabl_weights(model, name)])
            loss_fn(model(inputs), targets).backward()
            optimizer.step()
            clamped.append(
                any(_norm(weight) > _MAX_WEIGHT_NORM for weight in _tabl_weights(model, name))
            )
            if not old_forward_clamping:
                enforce_weight_constraints(model)
        return seen, clamped

    old_seen, _ = run(old_forward_clamping=True)
    new_seen, clamped_after_step = run(old_forward_clamping=False)

    # 非平凡：确实有步把某个权重推出上限，否则等价性是空断言。
    assert any(clamped_after_step)
    for old_snapshot, new_snapshot in zip(old_seen, new_seen, strict=True):
        for old_weight, new_weight in zip(old_snapshot, new_snapshot, strict=True):
            assert torch.equal(old_weight, new_weight)
