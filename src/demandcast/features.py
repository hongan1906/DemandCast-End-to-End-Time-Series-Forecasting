"""Stage 3: tabular feature engineering, built so features only use the past.

One row = one (product, target date d). With horizon h the forecast *origin* is
t = d - h: the last day whose sales we know. Everything derived from sales uses
data up to t only; promotion / price / holiday / calendar are known in advance
for the target date, so they are legitimate inputs.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

LAGS = (1, 7, 14, 28)
NON_FEATURES = ("date", "product", "y")


def build_features(df: pd.DataFrame, horizon: int = 1, lags=LAGS, leak: bool = False) -> pd.DataFrame:
    """Return a frame with columns date, product, y (= sales at d) and features.

    leak=True adds a *centered* 3-day rolling mean (uses day d+1 and d itself)
    purely to demonstrate data leakage. Never use it for real evaluation.
    """
    h = int(horizon)
    df = df.sort_values(["product", "date"]).reset_index(drop=True)
    products = sorted(df["product"].unique())
    out = []
    for product, g in df.groupby("product", sort=False):
        g = g.reset_index(drop=True)
        gaps = g["date"].diff().dropna()
        if not (gaps == pd.Timedelta(days=1)).all():
            raise ValueError(f"{product}: dates must be contiguous daily")
        s = g["sales"].astype(float)
        f = pd.DataFrame({"date": g["date"], "product": product, "y": s})
        # origin-anchored summaries of the past (shifted by h -> known at time t)
        f["origin_last"] = s.shift(h)
        f["rolling_mean_7"] = s.rolling(7).mean().shift(h)
        f["rolling_mean_28"] = s.rolling(28).mean().shift(h)
        f["rolling_std_7"] = s.rolling(7).std().shift(h)
        # lags relative to the target date; only lags >= h are knowable
        for k in lags:
            if k >= h:
                f[f"lag_{k}"] = s.shift(k)
        # same weekday, most recent occurrence that is already known
        f["same_dow_last"] = s.shift(7 * math.ceil(h / 7))
        if leak:
            f["leaky_centered_mean_3"] = s.rolling(3, center=True, min_periods=1).mean()
        # known-in-advance features of the target date
        f["day_of_week"] = g["date"].dt.dayofweek
        f["month"] = g["date"].dt.month
        f["promotion"] = g["promotion"]
        f["price"] = g["price"]
        f["holiday"] = g["holiday"]
        f["product_id"] = products.index(product)
        out.append(f)
    return pd.concat(out, ignore_index=True).dropna().reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in NON_FEATURES]
