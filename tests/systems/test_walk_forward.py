from __future__ import annotations

from hft_lob.configs.experiment import WalkForwardConfig
from hft_lob.preprocessing.split import Fold, WalkForwardPlan
from hft_lob.systems.walk_forward import select_walk_forward_folds


def _plan() -> WalkForwardPlan:
    folds = tuple(
        Fold(
            index=index,
            train_dates=[f"2026-{month:02d}-02" for month in range(1, index + 3)],
            validation_dates=[f"2026-{index + 3:02d}-02"],
            test_dates=[f"2026-{index + 4:02d}-02", f"2026-{index + 4:02d}-27"],
        )
        for index in range(1, 6)
    )
    return WalkForwardPlan(dataset_version="dataset-v1", folds=folds)


def test_selects_three_folds_without_changing_fold_history() -> None:
    plan = _plan()

    selected = select_walk_forward_folds(
        plan,
        WalkForwardConfig(start_fold=2, num_folds=3),
    )

    assert [fold.index for fold in selected] == [2, 3, 4]
    assert selected[0] is plan.folds[1]
    assert selected[0].train_dates == plan.folds[1].train_dates


def test_start_fold_is_matched_by_fold_index() -> None:
    selected = select_walk_forward_folds(
        _plan(), WalkForwardConfig(start_fold=3, num_folds=1)
    )

    assert [fold.index for fold in selected] == [3]
