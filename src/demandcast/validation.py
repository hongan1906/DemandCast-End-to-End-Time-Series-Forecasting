"""Time-aware splitting and walk-forward validation (never a random split)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics

# Chronological layout from the plan: TRAIN 2023 | VALIDATION 2024 H1 | TEST 2024 H2
VAL_START = pd.Timestamp("2024-01-01")
TEST_START = pd.Timestamp("2024-07-01")


@dataclass(frozen=True)
class Fold:
    k: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # exclusive


def time_split(dates: pd.Series, val_start=VAL_START, test_start=TEST_START):
    """Boolean masks (train, val, test) by target date."""
    d = pd.to_datetime(dates).reset_index(drop=True)
    return d < val_start, (d >= val_start) & (d < test_start), d >= test_start


def make_folds(first_test_start, fold_days: int, n_folds: int) -> list[Fold]:
    """Consecutive test blocks. Each fold trains on everything before its block."""
    s0 = pd.Timestamp(first_test_start)
    out = []
    for k in range(n_folds):
        a = s0 + pd.Timedelta(days=k * fold_days)
        out.append(Fold(k, a, a + pd.Timedelta(days=fold_days)))
    return out


VAL_FOLDS = make_folds(VAL_START, 30, 6)    # model selection / L / h experiments
TEST_FOLDS = make_folds(TEST_START, 30, 6)  # final, untouched comparison


def fold_masks(dates, folds: list[Fold]):
    """Yield (fold, train_mask, test_mask) with an expanding training window.

    Rows are indexed by *target* date. A training row's label lies strictly before
    the fold's test block, and its features only use data up to its own origin, so
    nothing from the test block reaches the model.
    """
    d = pd.to_datetime(pd.Series(np.asarray(dates))).reset_index(drop=True)
    for f in folds:
        yield f, (d < f.test_start).to_numpy(), ((d >= f.test_start) & (d < f.test_end)).to_numpy()


def summary_table(preds: pd.DataFrame, reference: str = "seasonal_naive") -> pd.DataFrame:
    """Pooled metrics per model + improvement vs a reference and fold consistency.

    `preds` columns: fold, model, y, yhat.
    """
    def fold_wape(g):
        return {k: metrics.wape(x["y"], x["yhat"]) for k, x in g.groupby("fold")}

    ref = preds[preds["model"] == reference]
    ref_pooled = metrics.wape(ref["y"], ref["yhat"]) if len(ref) else np.nan
    ref_fold = fold_wape(ref) if len(ref) else {}
    rows = []
    for model, g in preds.groupby("model", sort=False):
        s = metrics.summarize(g["y"], g["yhat"])
        fw = fold_wape(g)
        row = {"model": model, "MAE": s["MAE"], "WAPE %": 100 * s["WAPE"], "Bias %": 100 * s["Bias"]}
        if len(ref):
            row["vs naive (WAPE %)"] = 100 * (1 - s["WAPE"] / ref_pooled) if model != reference else 0.0
            wins = sum(fw[k] < ref_fold[k] for k in fw if k in ref_fold)
            row["folds beating naive"] = "baseline" if model == reference else f"{wins}/{len(fw)}"
        rows.append(row)
    return pd.DataFrame(rows).round(2)
