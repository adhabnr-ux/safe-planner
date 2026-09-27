"""Collect every number the write-up quotes into results/.   python -m sp.report"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from .env import ROOT, WORK, Env

RES = ROOT / "results"


def main(tag="main"):
    RES.mkdir(exist_ok=True)
    env = Env()
    w = WORK / env.cfg.name
    r = w / f"results_{tag}"
    for f in ("e1_pressure.csv", "e2_leave_one_out.csv", "e2_summary.csv", "meta.json"):
        shutil.copy(r / f, RES / f)
    for f in ("sensitivity_threshold.csv", "reality_check.csv", "existing_sites.csv"):
        if (w / f).exists():
            shutil.copy(w / f, RES / f)
    env.save_manifest(RES)
    pl = {j: np.load(w / f"pl_{j}.npy").astype(np.float32) for j in ("terrain", "itm", "sionna")}
    np.savez_compressed(RES / "path_loss_matrices.npz", **pl)

    thr, eirp = env.cfg.threshold_dbm, env.cfg.eirp_re_dbm
    cov = {j: (eirp - p) >= thr for j, p in pl.items()}
    e1 = pd.read_csv(RES / "e1_pressure.csv")
    g = e1.groupby(["train", "n"]).mean(numeric_only=True)
    s = pd.read_csv(RES / "e2_summary.csv")
    ls = s[s.planner == "local_search"]
    rnd = s[s.planner == "random"].set_index(["held_out", "panel"])
    out = dict(
        population=float(env.pop.sum()), demand_cells=len(env.demand), candidates=len(env.candidates),
        links=int(pl["terrain"].size),
        sionna_links_with_any_path=float((pl["sionna"] < 250).mean()),
        coverage_links={j: int(c.sum()) for j, c in cov.items()},
        nesting={f"{a}_within_{b}": float((cov[a] & cov[b]).sum() / cov[a].sum())
                 for a in cov for b in cov if a != b},
        e1_goodhart={t: {j: dict(n1=float(g.loc[(t, 1), f"score_{j}"]), n10000=float(g.loc[(t, 10000), f"score_{j}"]))
                         for j in cov} for t in cov},
        e2_local_search=[dict(held_out=x.held_out, panel=x.panel, claimed=x.claimed, heldout=x.heldout,
                              overclaim=x.overclaim, overclaim_se=x.overclaim_se, regret=x.regret, regret_se=x.regret_se,
                              excess_overclaim_vs_random=x.overclaim - rnd.loc[(x.held_out, x.panel), "overclaim"])
                         for x in ls.itertuples()],
    )
    try:
        from .reality import tested_tiles
        t = tested_tiles(env)
        out["reality_check"] = dict(tiles=len(t), tests=int(t.tests.sum()),
                                    rows=pd.read_csv(RES / "reality_check.csv").to_dict("records"))
    except Exception as e:                      # coverage-truth results not present
        out["reality_check"] = dict(error=str(e))
    (RES / "summary.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1)[:3000])
