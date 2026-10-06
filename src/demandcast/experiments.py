"""One function per stage of the project. Each returns a DataFrame and saves a CSV (+ PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import metrics
from .features import build_features, feature_columns
from .lstm import LSTMForecaster, torch_available
from .models import available_models, make_xgboost
from .validation import TEST_FOLDS, VAL_FOLDS, fold_masks, summary_table

REPORTS = Path("reports")


def _save(df: pd.DataFrame, name: str) -> None:
    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / f"{name}.csv", index=False)


def _fig(name: str):
    REPORTS.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(REPORTS / f"{name}.png", dpi=130)
    plt.close()


# ---------------------------------------------------------------- helpers
def _naive_preds(frame: pd.DataFrame, col: str, name: str, folds) -> pd.DataFrame:
    parts = []
    for f, _, te in fold_masks(frame["date"], folds):
        t = frame.loc[te]
        parts.append(pd.DataFrame({"fold": f.k, "model": name, "date": t["date"].values,
                                   "product": t["product"].values, "y": t["y"].values, "yhat": t[col].values}))
    return pd.concat(parts, ignore_index=True)


def _model_preds(frame: pd.DataFrame, factory, name: str, folds, cols=None) -> pd.DataFrame:
    cols = cols or feature_columns(frame)
    parts = []
    for f, tr, te in fold_masks(frame["date"], folds):
        m = factory()
        m.fit(frame.loc[tr, cols], frame.loc[tr, "y"])
        p = np.clip(np.asarray(m.predict(frame.loc[te, cols])), 0, None)
        t = frame.loc[te]
        parts.append(pd.DataFrame({"fold": f.k, "model": name, "date": t["date"].values,
                                   "product": t["product"].values, "y": t["y"].values, "yhat": p}))
    return pd.concat(parts, ignore_index=True)


def _naive_set(frame, folds) -> pd.DataFrame:
    return pd.concat([
        _naive_preds(frame, "origin_last", "naive_last", folds),
        _naive_preds(frame, "same_dow_last", "seasonal_naive", folds),
        _naive_preds(frame, "rolling_mean_28", "mean_28d", folds),
    ], ignore_index=True)


# ---------------------------------------------------------------- stage 1
def run_baseline(df: pd.DataFrame, h: int = 1) -> pd.DataFrame:
    """Naive forecasts: y_hat = last value, same weekday last week, trailing 28-day mean."""
    frame = build_features(df, h)
    rows = []
    for region, folds in [("validation (2024 H1)", VAL_FOLDS), ("test (2024 H2)", TEST_FOLDS)]:
        p = _naive_set(frame, folds)
        for model, g in p.groupby("model", sort=False):
            s = metrics.summarize(g["y"], g["yhat"])
            rows.append({"region": region, "h": h, "model": model, "MAE": s["MAE"],
                         "WAPE %": 100 * s["WAPE"], "MAPE %": 100 * s["MAPE"], "Bias %": 100 * s["Bias"]})
    out = pd.DataFrame(rows).round(2)
    _save(out, f"stage1_baseline_h{h}")
    return out


# ---------------------------------------------------------------- stage 3
def run_leakage_demo(df: pd.DataFrame, h: int = 1) -> pd.DataFrame:
    """Centered rolling mean (peeks at the future) looks brilliant; past-only does not."""
    rows = []
    for label, leak in [("past-only features", False), ("LEAKY centered rolling mean", True)]:
        frame = build_features(df, h, leak=leak)
        cols = feature_columns(frame)
        zoo = {"ridge": available_models(cols)["ridge"], "xgboost": lambda: make_xgboost()}
        for name, factory in zoo.items():
            p = _model_preds(frame, factory, name, TEST_FOLDS, cols)
            s = metrics.summarize(p["y"], p["yhat"])
            rows.append({"features": label, "model": name, "MAE": s["MAE"], "WAPE %": 100 * s["WAPE"]})
    ref = _naive_preds(build_features(df, h), "same_dow_last", "seasonal_naive", TEST_FOLDS)
    rows.append({"features": "(reference)", "model": "seasonal_naive",
                 "MAE": metrics.mae(ref["y"], ref["yhat"]), "WAPE %": 100 * metrics.wape(ref["y"], ref["yhat"])})
    out = pd.DataFrame(rows).round(2)
    _save(out, f"stage3_leakage_h{h}")
    plt.figure(figsize=(6, 3.6))
    sub = out[out["model"] != "seasonal_naive"]
    for i, (label, g) in enumerate(sub.groupby("features")):
        plt.bar(np.arange(len(g)) + i * 0.38, g["WAPE %"], 0.38, label=label)
    plt.xticks(np.arange(2) + 0.19, ["ridge", "xgboost"])
    plt.axhline(out.loc[out["model"] == "seasonal_naive", "WAPE %"].iloc[0], ls="--", c="gray", label="seasonal naive")
    plt.ylabel("test WAPE %"); plt.title("Data leakage: too good to be true"); plt.legend(fontsize=8)
    _fig("stage3_leakage")
    return out


# ---------------------------------------------------------------- stage 5
def run_window_sweep(df: pd.DataFrame, Ls=(7, 14, 28, 56), h: int = 1) -> pd.DataFrame:
    """How much history does the model need? Flattened windows -> XGBoost, validated walk-forward."""
    from .windows import make_windows

    rows = []
    for L in Ls:
        W = make_windows(df, L, h)
        Xf = W.flat()
        parts = []
        for f, tr, te in fold_masks(W.dates, VAL_FOLDS):
            m = make_xgboost()
            m.fit(Xf[tr], W.y_scaled[tr])
            parts.append((W.y[te], np.clip(m.predict(Xf[te]), 0, None) * W.scale[te]))
        y = np.concatenate([a for a, _ in parts]); p = np.concatenate([b for _, b in parts])
        rows.append({"L": L, "MAE": metrics.mae(y, p), "WAPE %": 100 * metrics.wape(y, p)})
    ref = _naive_preds(build_features(df, h), "same_dow_last", "seasonal_naive", VAL_FOLDS)
    out = pd.DataFrame(rows).round(2)
    out.attrs["naive_mae"] = metrics.mae(ref["y"], ref["yhat"])
    _save(out, f"stage5_window_L_h{h}")
    plt.figure(figsize=(5.5, 3.6))
    plt.plot(out["L"], out["MAE"], "o-", label="XGBoost on windows")
    plt.axhline(out.attrs["naive_mae"], ls="--", c="gray", label="seasonal naive")
    plt.xlabel("window length L (days)"); plt.ylabel("validation MAE"); plt.xscale("log", base=2)
    plt.xticks(out["L"], out["L"]); plt.title(f"Stage 5: window length (h={h})"); plt.legend()
    _fig("stage5_window_L")
    return out


# ---------------------------------------------------------------- stage 6
def run_horizon_sweep(df: pd.DataFrame, hs=(1, 7, 14, 28)) -> pd.DataFrame:
    """Uncertainty grows with the horizon: error vs h for the model and for naive."""
    rows = []
    for h in hs:
        frame = build_features(df, h)
        p = _model_preds(frame, lambda: make_xgboost(), "xgboost", VAL_FOLDS)
        n = _naive_preds(frame, "same_dow_last", "seasonal_naive", VAL_FOLDS)
        rows.append({"h": h, "xgboost WAPE %": 100 * metrics.wape(p["y"], p["yhat"]),
                     "seasonal_naive WAPE %": 100 * metrics.wape(n["y"], n["yhat"]),
                     "xgboost MAE": metrics.mae(p["y"], p["yhat"])})
    out = pd.DataFrame(rows).round(2)
    _save(out, "stage6_horizon")
    plt.figure(figsize=(5.5, 3.6))
    plt.plot(out["h"], out["xgboost WAPE %"], "o-", label="XGBoost")
    plt.plot(out["h"], out["seasonal_naive WAPE %"], "s--", c="gray", label="seasonal naive")
    plt.xlabel("forecast horizon h (days)"); plt.ylabel("validation WAPE %"); plt.title("Stage 6: error vs horizon")
    plt.legend(); _fig("stage6_horizon")
    return out


# ---------------------------------------------------------------- stage 7
def run_compare(df: pd.DataFrame, h: int = 7) -> pd.DataFrame:
    """Naive -> Ridge -> XGBoost -> CatBoost on the untouched test folds."""
    frame = build_features(df, h)
    cols = feature_columns(frame)
    parts = [_naive_set(frame, TEST_FOLDS)]
    for name, factory in available_models(cols).items():
        parts.append(_model_preds(frame, factory, name, TEST_FOLDS, cols))
    preds = pd.concat(parts, ignore_index=True)
    out = summary_table(preds)
    _save(out, f"stage7_compare_h{h}")
    _save(_per_fold(preds), f"stage7_per_fold_h{h}")
    _plot_folds(preds, f"stage7_per_fold_h{h}", f"Test WAPE by walk-forward fold (h={h})")
    return out


def _per_fold(preds: pd.DataFrame) -> pd.DataFrame:
    rows = [{"fold": k, "model": m, "WAPE %": 100 * metrics.wape(g["y"], g["yhat"])}
            for (k, m), g in preds.groupby(["fold", "model"], sort=False)]
    return pd.DataFrame(rows).pivot(index="fold", columns="model", values="WAPE %").round(2).reset_index()


def _plot_folds(preds: pd.DataFrame, name: str, title: str) -> None:
    pf = _per_fold(preds).set_index("fold")
    plt.figure(figsize=(6.5, 3.8))
    for m in pf.columns:
        plt.plot(pf.index + 1, pf[m], "o-" if "naive" not in m and m != "mean_28d" else "x--", label=m)
    plt.xlabel("walk-forward fold (30 days each)"); plt.ylabel("WAPE %"); plt.title(title); plt.legend(fontsize=8)
    _fig(name)


# ---------------------------------------------------------------- V2
def run_lstm(df: pd.DataFrame, L: int = 28, h: int = 7) -> pd.DataFrame:
    """Same windows, two shapes: XGBoost on (N, L*C+K) vs LSTM on (N, L, C)."""
    from .windows import make_windows

    if not torch_available():
        raise RuntimeError("PyTorch not installed: pip install torch")
    W = make_windows(df, L, h)
    Xf = W.flat()
    parts = []
    for f, tr, te in fold_masks(W.dates, TEST_FOLDS):
        base = {"fold": f.k, "date": W.dates[te], "product": W.products[te], "y": W.y[te]}
        parts.append(pd.DataFrame({**base, "model": "seasonal_naive", "yhat": W.naive_seasonal[te]}))
        xg = make_xgboost().fit(Xf[tr], W.y_scaled[tr])
        parts.append(pd.DataFrame({**base, "model": "xgboost", "yhat": np.clip(xg.predict(Xf[te]), 0, None) * W.scale[te]}))
        ls = LSTMForecaster().fit(W.X[tr], W.known[tr], W.y_scaled[tr])
        parts.append(pd.DataFrame({**base, "model": "lstm", "yhat": np.clip(ls.predict(W.X[te], W.known[te]), 0, None) * W.scale[te]}))
    preds = pd.concat(parts, ignore_index=True)
    out = summary_table(preds)
    out.attrs["shapes"] = f"XGBoost: {Xf.shape}  |  LSTM: {W.X.shape}"
    _save(out, f"v2_lstm_L{L}_h{h}")
    _plot_folds(preds, f"v2_lstm_L{L}_h{h}", f"V2 LSTM vs XGBoost (L={L}, h={h})")
    return out


# ---------------------------------------------------------------- overview
def plot_data_overview(df: pd.DataFrame) -> None:
    g = df[df["product"] == "SKU_01"]
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=False)
    ax[0].plot(g["date"], g["sales"], lw=0.8)
    ax[0].scatter(g.loc[g["promotion"] == 1, "date"], g.loc[g["promotion"] == 1, "sales"], s=6, c="tab:red", label="promotion")
    ax[0].set_title("SKU_01 daily sales"); ax[0].legend(fontsize=8)
    w = df.assign(dow=df["date"].dt.dayofweek).groupby("dow")["sales"].mean()
    ax[1].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], w.values)
    ax[1].set_title("Average sales by weekday (all products)")
    _fig("data_overview")
