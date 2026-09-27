"""Robustness: rerun the leave-one-out experiment (local search, K=5, 30 seeds) at other coverage thresholds.
python -m sp.sensitivity"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from . import experiments as E
from .env import WORK, Config, Env


def run(thresholds=(-100.0, -105.0, -110.0)):
    rows = []
    base = Env()
    for thr in thresholds:
        env = base
        env.cfg = dataclasses.replace(Config(), threshold_dbm=thr)
        covs = E.coverage_matrices(env, ("terrain", "itm", "sionna"))
        df = E.e2_leave_one_out(env, covs, planners=("local_search",), k_values=(5,), seeds=30)
        s = E.summarize(df)
        s["threshold_dbm"] = thr
        # nesting: share of each judge's covered links that the other judges also cover
        for a in covs:
            for b in covs:
                if a != b:
                    s[f"nest_{a}_in_{b}"] = float((covs[a] & covs[b]).sum() / max(covs[a].sum(), 1))
        rows.append(s)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(WORK / "blackmesa" / "sensitivity_threshold.csv", index=False)
    return out


if __name__ == "__main__":
    o = run()
    cols = ["threshold_dbm", "held_out", "panel", "claimed", "heldout", "overclaim", "regret"]
    with pd.option_context("display.width", 200):
        print((o[cols].set_index(cols[:3]) * 100).round(1))
