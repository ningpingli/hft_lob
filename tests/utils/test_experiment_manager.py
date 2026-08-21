from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hft_lob.utils import experiment_manager
from hft_lob.utils.experiment_manager import (
    generate_experiment_id,
    resolve_experiment_id,
    resolve_log_dir,
    write_experiment_log,
)


def test_generate_experiment_id_creates_result_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiment_manager, "_RESULTS_ROOT", str(tmp_path))

    experiment_id = generate_experiment_id("DeepLOB", "TEST")

    assert experiment_id.startswith("TEST_DeepLOB_")
    assert len(experiment_id.rsplit("_", maxsplit=1)[1]) == 7
    assert (tmp_path / experiment_id).is_dir()


def test_resolve_experiment_id_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        experiment_manager, "generate_experiment_id", lambda model_name, ticker: "generated"
    )

    assert (
        resolve_experiment_id(
            model_name="model",
            ticker="TEST",
            override_id="override",
        )
        == "override"
    )
    assert resolve_experiment_id(model_name="model", ticker="TEST") == "generated"


def test_resolve_log_dir_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="single path component"):
        resolve_log_dir("../outside")


def test_write_experiment_log_creates_and_merges_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root = tmp_path / "results"
    monkeypatch.setattr(experiment_manager, "_RESULTS_ROOT", str(results_root))

    write_experiment_log("run-1", "dataset_info", {"rows": 100})
    write_experiment_log("run-1", "evaluation", {"ts_ic": 0.12})

    destination = results_root / "run-1" / "data.yaml"
    assert yaml.safe_load(destination.read_text(encoding="utf-8")) == {
        "dataset_info": {"rows": 100},
        "evaluation": {"ts_ic": 0.12},
    }


def test_write_experiment_log_replaces_same_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiment_manager, "_RESULTS_ROOT", str(tmp_path))
    write_experiment_log("run-1", "evaluation", {"ts_ic": 0.1})
    write_experiment_log("run-1", "evaluation", {"ts_ic": 0.2})

    contents = yaml.safe_load((tmp_path / "run-1" / "data.yaml").read_text(encoding="utf-8"))
    assert contents == {"evaluation": {"ts_ic": 0.2}}


def test_write_experiment_log_rejects_non_mapping_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiment_manager, "_RESULTS_ROOT", str(tmp_path))
    log_dir = tmp_path / "run-1"
    log_dir.mkdir()
    (log_dir / "data.yaml").write_text("- invalid\n- root\n", encoding="utf-8")

    with pytest.raises(ValueError, match="root must be a mapping"):
        write_experiment_log("run-1", "evaluation", {"ts_ic": 0.2})
