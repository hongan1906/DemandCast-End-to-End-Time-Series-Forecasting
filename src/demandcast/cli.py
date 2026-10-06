from __future__ import annotations

import argparse

import pandas as pd

from . import experiments as ex
from .data import DEFAULT_PATH, generate_demand, load_or_generate
from .lstm import torch_available

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


def _show(title, df):
    print(f"\n=== {title} ===")
    print(df.to_string(index=False))


def main(argv=None):
    p = argparse.ArgumentParser(prog="demandcast", description=__doc__)
    p.add_argument("command", choices=["data", "baseline", "leakage", "windows", "horizon", "compare", "lstm", "all"])
    p.add_argument("--h", type=int, default=None, help="forecast horizon in days")
    p.add_argument("--data", default=str(DEFAULT_PATH))
    a = p.parse_args(argv)

    if a.command == "data":
        df = generate_demand()
        DEFAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.data, index=False)
        ex.plot_data_overview(df)
        print(f"wrote {a.data}: {len(df)} rows, {df['product'].nunique()} products")
        return
    df = load_or_generate(a.data)
    cmds = [a.command] if a.command != "all" else ["baseline", "leakage", "windows", "horizon", "compare", "lstm"]
    for c in cmds:
        if c == "baseline":
            _show("Stage 1: naive baselines (h=%d)" % (a.h or 1), ex.run_baseline(df, a.h or 1))
        elif c == "leakage":
            _show("Stage 3: leakage demo", ex.run_leakage_demo(df, a.h or 1))
        elif c == "windows":
            _show("Stage 5: window length L", ex.run_window_sweep(df, h=a.h or 1))
        elif c == "horizon":
            _show("Stage 6: horizon h", ex.run_horizon_sweep(df))
        elif c == "compare":
            _show("Stage 7: model comparison on test (h=%d)" % (a.h or 7), ex.run_compare(df, a.h or 7))
        elif c == "lstm":
            if not torch_available():
                print("\n[skip] V2 LSTM needs PyTorch: pip install torch")
                continue
            out = ex.run_lstm(df, h=a.h or 7)
            _show("V2: same windows, two shapes -- " + out.attrs["shapes"], out)
