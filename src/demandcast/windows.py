"""Stage 2: turn a time series into supervised learning (window -> X, y).

For each product, a window of the previous L days (origin t is its last day)
predicts demand at t + h:

    days t-L+1 ... t   --->   demand at t+h

Same windows, two shapes:
    trees / linear : X.reshape(N, L*C)   (+ known target-day covariates)
    LSTM           : X as (N, L, C)

Sales and price are divided by their own window mean, so one global model can
serve products of different scale; multiply predictions by `scale` to undo it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

CHANNELS = ("sales", "price", "promotion", "holiday")
KNOWN_NAMES = ("dow_sin", "dow_cos", "month_sin", "month_cos", "promotion", "holiday", "price_ratio")


@dataclass
class Windows:
    X: np.ndarray          # (N, L, C)  scaled history
    known: np.ndarray      # (N, K)     known-in-advance covariates of the target day
    y: np.ndarray          # (N,)       raw demand at t+h
    y_scaled: np.ndarray   # (N,)       y / scale
    scale: np.ndarray      # (N,)       mean sales in the window
    naive_seasonal: np.ndarray  # (N,)  same weekday, latest known occurrence
    dates: np.ndarray      # (N,)       target dates
    products: np.ndarray   # (N,)

    @property
    def shape_seq(self):
        return self.X.shape

    def flat(self) -> np.ndarray:
        """(N, L*C + K): the 'tabular' view of the identical windows."""
        return np.hstack([self.X.reshape(len(self.X), -1), self.known])


def make_windows(df: pd.DataFrame, L: int, h: int = 1, channels=CHANNELS) -> Windows:
    Xs, Ks, ys, sc, nv, ds, ps = [], [], [], [], [], [], []
    off = 7 * math.ceil(h / 7) - h  # position of same-weekday day inside window
    if L <= off:
        raise ValueError("L too short to contain the seasonal-naive day")
    for product, g in df.sort_values(["product", "date"]).groupby("product", sort=False):
        arr = g[list(channels)].to_numpy(float)
        T = len(arr)
        if T < L + h:
            continue
        sw = sliding_window_view(arr, L, axis=0).transpose(0, 2, 1)  # (T-L+1, L, C)
        n = T - L - h + 1
        win = sw[:n].copy()  # window j ends at origin t = j+L-1
        tgt = np.arange(L - 1 + h, T)  # target indices d = t + h
        sales_mean = win[:, :, 0].mean(axis=1) + 1e-6
        price_mean = win[:, :, 1].mean(axis=1) + 1e-6
        raw_sales = win[:, :, 0].copy()
        win[:, :, 0] /= sales_mean[:, None]
        win[:, :, 1] /= price_mean[:, None]
        d = g["date"].iloc[tgt]
        dow, month = d.dt.dayofweek.to_numpy(), d.dt.month.to_numpy()
        known = np.column_stack([
            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * (month - 1) / 12), np.cos(2 * np.pi * (month - 1) / 12),
            g["promotion"].to_numpy()[tgt], g["holiday"].to_numpy()[tgt],
            g["price"].to_numpy()[tgt] / price_mean,
        ])
        y = g["sales"].to_numpy(float)[tgt]
        Xs.append(win); Ks.append(known); ys.append(y); sc.append(sales_mean)
        nv.append(raw_sales[:, L - 1 - off]); ds.append(d.to_numpy()); ps.append(np.full(n, product))
    y = np.concatenate(ys); scale = np.concatenate(sc)
    return Windows(
        X=np.concatenate(Xs), known=np.concatenate(Ks), y=y, y_scaled=y / scale, scale=scale,
        naive_seasonal=np.concatenate(nv), dates=np.concatenate(ds), products=np.concatenate(ps),
    )
