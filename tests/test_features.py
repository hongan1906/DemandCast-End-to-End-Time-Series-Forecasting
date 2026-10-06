import numpy as np
import pandas as pd
import pytest
from demandcast.data import generate_demand
from demandcast.features import build_features, feature_columns


@pytest.fixture(scope="module")
def df():
    return generate_demand(n_products=2, start="2023-01-01", end="2023-09-30", seed=1)


@pytest.mark.parametrize("h", [1, 7, 14])
def test_features_use_only_the_past(df, h):
    """Rewrite all sales after the origin; the row's features must not change."""
    base = build_features(df, h).set_index(["product", "date"])
    row = base.loc[("SKU_01", pd.Timestamp("2023-08-15"))]
    origin = pd.Timestamp("2023-08-15") - pd.Timedelta(days=h)
    d2 = df.copy()
    d2.loc[d2["date"] > origin, "sales"] = 10_000
    changed = build_features(d2, h).set_index(["product", "date"]).loc[("SKU_01", pd.Timestamp("2023-08-15"))]
    cols = feature_columns(base.reset_index())
    pd.testing.assert_series_equal(row[cols], changed[cols])


def test_leaky_feature_is_detected_by_the_same_test(df):
    h = 1
    base = build_features(df, h, leak=True).set_index(["product", "date"])
    d2 = df.copy()
    d2.loc[d2["date"] > pd.Timestamp("2023-08-14"), "sales"] = 10_000
    changed = build_features(d2, h, leak=True).set_index(["product", "date"])
    key = ("SKU_01", pd.Timestamp("2023-08-15"))
    assert base.loc[key, "leaky_centered_mean_3"] != changed.loc[key, "leaky_centered_mean_3"]


def test_no_lag_shorter_than_horizon(df):
    cols = feature_columns(build_features(df, 7))
    assert "lag_1" not in cols and "lag_7" in cols
