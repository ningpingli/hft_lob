from pathlib import Path

import pytest

from hft_lob.application.dataset import DatasetBuildRequest, build_dataset


def test_build_dataset_loads_config_and_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = object()
    seen: dict[str, object] = {}

    monkeypatch.setattr("hft_lob.application.dataset.load_data_config", lambda path: config)

    def fake_build(received_config: object, output_root: str) -> Path:
        seen["config"] = received_config
        seen["output_root"] = output_root
        return Path("published/dataset-id")

    monkeypatch.setattr("hft_lob.application.dataset.build_dataset_package", fake_build)

    result = build_dataset(DatasetBuildRequest("data.yaml", "published"))

    assert result == Path("published/dataset-id")
    assert seen == {"config": config, "output_root": "published"}
