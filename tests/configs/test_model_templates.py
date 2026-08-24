from __future__ import annotations

from pathlib import Path

from hft_lob.configs.loader import load_model_config

EXPECTED_MODELS = {
    "axiallob",
    "binctabl",
    "binbtabl",
    "cnn1",
    "cnn2",
    "deeplob",
    "dla",
    "hlob",
    "itransformer",
    "lobtransformer",
    "transformer",
}


TEMPLATE_ROOT = Path(__file__).parents[2] / "configs" / "models"


def test_every_registered_model_has_a_loadable_default_template() -> None:
    template_paths = {path.stem: path for path in TEMPLATE_ROOT.glob("*.yaml")}

    assert set(template_paths) == EXPECTED_MODELS
    for model_name, path in template_paths.items():
        config = load_model_config(str(path), experiment_id=f"template-{model_name}")
        assert config.model.name == model_name
        assert config.model.output_dim == 1
