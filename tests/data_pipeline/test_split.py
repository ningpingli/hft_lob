from __future__ import annotations

import pytest

from hft_lob.configs.experiment import SplitConfig, WalkForwardConfig
from hft_lob.data_pipeline.split import (
    chronological_split,
    walk_forward_folds,
)


def test_chronological_ratio_split_operates_on_supplied_dates() -> None:
    dates = [f"2026-01-{day:02d}" for day in range(1, 11)]
    config = SplitConfig(train_ratio=0.6, validation_ratio=0.2)

    split = chronological_split(dates, config)

    assert split.train_dates == dates[:6]
    assert split.validation_dates == dates[6:8]
    assert split.test_dates == dates[8:]
    assert split.dates_for("training") == split.train_dates


def test_explicit_ranges_must_cover_selected_dates_exactly_once() -> None:
    dates = [f"2026-01-{day:02d}" for day in range(1, 7)]
    config = SplitConfig(
        train_dates=("2026-01-01", "2026-01-02"),
        validation_dates=("2026-01-03", "2026-01-04"),
        test_dates=("2026-01-05", "2026-01-06"),
    )

    split = chronological_split(dates, config)

    assert len(split.train_dates) == len(split.validation_dates) == len(split.test_dates) == 2


def test_walk_forward_uses_fixed_trade_day_windows() -> None:
    dates = [
        f"2026-{month:02d}-{day:02d}"
        for month in range(1, 7)
        for day in (2, 3)
    ]

    folds = walk_forward_folds(
        dates,
        WalkForwardConfig(
            train_window_days=6,
            validation_window_days=2,
            test_window_days=2,
            step_days=2,
        ),
    )

    assert len(folds) == 2
    assert folds[0].index == 1
    assert folds[0].train_dates == dates[:6]
    assert folds[0].validation_dates == dates[6:8]
    assert folds[0].test_dates == dates[8:10]
    assert folds[1].train_dates == dates[2:8]
    assert folds[1].validation_dates == dates[8:10]
    assert folds[1].test_dates == dates[10:12]


def test_split_rejects_unsorted_or_insufficient_dates() -> None:
    with pytest.raises(ValueError, match="sorted ascending"):
        chronological_split(
            ["2026-01-02", "2026-01-01", "2026-01-03"], SplitConfig()
        )
    with pytest.raises(ValueError, match="trade dates"):
        walk_forward_folds(
            ["2026-01-02", "2026-02-02", "2026-03-02", "2026-04-02"],
            WalkForwardConfig(
                train_window_days=3,
                validation_window_days=1,
                test_window_days=1,
            ),
        )
