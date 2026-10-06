"""Forecast error metrics. All take array-likes of equal length."""
from __future__ import annotations

import numpy as np


def _a(x):
    return np.asarray(x, dtype=float)


def mae(y, yhat) -> float:
    """Mean absolute error, in units of demand."""
    return float(np.mean(np.abs(_a(y) - _a(yhat))))


def wape(y, yhat) -> float:
    """Weighted absolute % error: sum|e| / sum|y|. Safe when some y are 0."""
    y, yhat = _a(y), _a(yhat)
    return float(np.abs(y - yhat).sum() / np.abs(y).sum())


def mape(y, yhat) -> float:
    """Mean absolute % error, ignoring days where y == 0."""
    y, yhat = _a(y), _a(yhat)
    m = y > 0
    return float(np.mean(np.abs(y[m] - yhat[m]) / y[m]))


def bias(y, yhat) -> float:
    """Signed % bias: (sum yhat - sum y) / sum y. Positive = over-forecasting."""
    y, yhat = _a(y), _a(yhat)
    return float((yhat.sum() - y.sum()) / y.sum())


def summarize(y, yhat) -> dict:
    return {"MAE": mae(y, yhat), "WAPE": wape(y, yhat), "MAPE": mape(y, yhat), "Bias": bias(y, yhat)}
