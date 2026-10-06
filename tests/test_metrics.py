import numpy as np
from demandcast import metrics


def test_basic_values():
    y, p = np.array([100, 200, 0]), np.array([110, 180, 10])
    assert metrics.mae(y, p) == (10 + 20 + 10) / 3
    assert np.isclose(metrics.wape(y, p), 40 / 300)
    assert np.isclose(metrics.bias(y, p), 0 / 300)  # errors +10 -20 +10 cancel
    assert np.isclose(metrics.mape(y, p), np.mean([0.1, 0.1]))  # zero-demand day ignored


def test_bias_sign():
    assert metrics.bias([100, 100], [120, 120]) > 0  # over-forecast
    assert metrics.bias([100, 100], [80, 80]) < 0
