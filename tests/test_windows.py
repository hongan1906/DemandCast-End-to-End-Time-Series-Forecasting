import numpy as np
import pandas as pd
from demandcast.windows import make_windows


def toy(T=60):
    d = pd.date_range("2024-01-01", periods=T)
    return pd.DataFrame({"date": d, "product": "A", "sales": np.arange(1, T + 1, dtype=float),
                         "price": 10.0, "promotion": 0, "holiday": 0})


def test_window_alignment_and_shapes():
    L, h = 14, 3
    W = make_windows(toy(), L, h)
    assert W.X.shape[1:] == (L, 4)
    assert W.flat().shape[1] == L * 4 + W.known.shape[1]
    for i in (0, 5, len(W.y) - 1):
        origin_sales = W.X[i, -1, 0] * W.scale[i]   # last day of window
        assert np.isclose(W.y[i], origin_sales + h)  # target is exactly h days later
        assert W.dates[i] == pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(W.y[i]) - 1)


def test_seasonal_naive_is_same_weekday():
    W = make_windows(toy(), 28, 3)
    # sales == day index, so same weekday latest known = y - 7
    assert np.allclose(W.naive_seasonal, W.y - 7)
