from __future__ import annotations

import math

import polars as pl

from hft_lob.configs.experiment import RAW_FEATURE_COLUMNS, FeatureConfig
from hft_lob.data_pipeline.processor import FeatureTransformer, SessionSegment


def _feature_frame() -> pl.DataFrame:
    row: dict[str, object] = {
        "trade_date": "2026-01-05",
        "session_id": "AM",
        "book_valid": True,
        "mid_price": 10.0,
        "last": 10.0,
        "volume": 1_000.0,
        "amount": 10_000.0,
    }
    for level in range(1, 6):
        row[f"ASKp{level}"] = 10.01 + 0.01 * (level - 1)
        row[f"ASKs{level}"] = float(10 * level)
        row[f"BIDp{level}"] = 9.99 - 0.01 * (level - 1)
        row[f"BIDs{level}"] = float(20 * level)
    return pl.DataFrame([row]).select(
        "trade_date", "session_id", "mid_price", "book_valid", *RAW_FEATURE_COLUMNS
    )


def test_transform_computes_configured_features_in_declared_order() -> None:
    names = (
        "spread",
        "relative_spread",
        "mid_price",
        "microprice",
        "l1_imbalance",
        "l5_imbalance",
        "bid_depth",
        "ask_depth",
        "depth_imbalance",
        "price_slope",
        "volume_slope",
    )
    transformer = FeatureTransformer(FeatureConfig(use_derived=True, derived_features=names))
    segment = SessionSegment("2026-01-05", "AM", _feature_frame())

    result = transformer.transform(segment).frame.row(0, named=True)

    assert transformer.feature_columns() == [*RAW_FEATURE_COLUMNS, *names]
    assert math.isclose(result["spread"], 0.02)
    assert math.isclose(result["relative_spread"], 0.002)
    assert math.isclose(result["mid_price"], 10.0)
    assert math.isclose(result["microprice"], 10.003333333333334)
    assert math.isclose(result["l1_imbalance"], 1 / 3)
    assert math.isclose(result["l5_imbalance"], 1 / 3)
    assert result["bid_depth"] == 300.0
    assert result["ask_depth"] == 150.0
    assert math.isclose(result["depth_imbalance"], 1 / 3)
    assert math.isclose(result["price_slope"], 0.01)
    assert result["volume_slope"] == 30.0
    assert result["feature_valid"] is True


def test_transform_without_derived_features_keeps_raw_schema() -> None:
    transformer = FeatureTransformer(FeatureConfig(use_derived=False))
    result = transformer.transform(
        SessionSegment("2026-01-05", "AM", _feature_frame())
    ).frame

    assert transformer.feature_columns() == list(RAW_FEATURE_COLUMNS)
    assert result.get_column("feature_valid").to_list() == [True]
    assert "spread" not in result.columns


def test_transform_rejects_mixed_session_metadata() -> None:
    frame = pl.concat(
        [
            _feature_frame(),
            _feature_frame().with_columns(pl.lit("PM").alias("session_id")),
        ]
    )
    transformer = FeatureTransformer(FeatureConfig())

    try:
        transformer.transform(SessionSegment("2026-01-05", "AM", frame))
    except ValueError as exc:
        assert "session_id" in str(exc)
    else:
        raise AssertionError("mixed sessions must be rejected")
