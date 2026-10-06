"""Synthetic retail demand generator.

Columns: date, product, sales, price, promotion, holiday.

The data has the structure a real demand planner expects, so every stage of the
project has something to find:
  * weekly seasonality (weekend peak) and a weak yearly cycle
  * a slow-moving per-product level (mean-reverting random walk) -> forecast
    error genuinely grows with the horizon h
  * promotions that cut price and lift demand, plus price elasticity
  * holiday spikes, and overdispersed count noise
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATH = Path("data/demand.csv")


def _holiday_dates(years) -> set[pd.Timestamp]:
    days: set[pd.Timestamp] = set()
    for y in years:
        for m, d in [(1, 1), (5, 1), (12, 24), (12, 25), (12, 26)]:
            days.add(pd.Timestamp(y, m, d))
        fridays = pd.date_range(f"{y}-11-01", f"{y}-11-30", freq="W-FRI")
        days.add(fridays[-1])  # late-November sale day
    return days


def generate_demand(
    n_products: int = 12,
    start: str = "2023-01-01",
    end: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    T = len(dates)
    hol = _holiday_dates(sorted(set(dates.year)))
    is_hol = np.array([d in hol for d in dates], dtype=int)
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()
    week_profile = np.array([0.85, 0.90, 0.95, 1.00, 1.15, 1.35, 1.20])  # Mon..Sun

    frames = []
    for i in range(n_products):
        base = rng.uniform(40, 200)
        price0 = rng.uniform(5, 25)
        week = week_profile * rng.normal(1, 0.05, 7)
        year = 1 + rng.uniform(0.05, 0.25) * np.sin(
            2 * np.pi * (doy - rng.uniform(0, 365)) / 365.25
        )
        x = np.zeros(T)  # slow-moving level, half-life of roughly a month
        for t in range(1, T):
            x[t] = 0.98 * x[t - 1] + rng.normal(0, 0.04)
        level = np.exp(x)

        promo = np.zeros(T, dtype=int)
        t = 0
        while t < T:
            if rng.random() < 0.04:
                n = int(rng.integers(3, 8))
                promo[t : t + n] = 1
                t += n
            else:
                t += 1
        price = price0 * (1 - 0.12 * promo) * rng.normal(1, 0.01, T)
        lift = (price / price0) ** -1.2 * np.where(promo == 1, 1.10, 1.0)
        mean = base * week[dow] * year * level * lift * np.where(is_hol == 1, 1.3, 1.0)
        sales = rng.poisson(mean * rng.gamma(30, 1 / 30, T))

        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "product": f"SKU_{i + 1:02d}",
                    "sales": sales,
                    "price": price.round(2),
                    "promotion": promo,
                    "holiday": is_hol,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def load_or_generate(path: Path | str = DEFAULT_PATH) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    df = generate_demand()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
