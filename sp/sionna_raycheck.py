"""How much does the Sionna judge depend on its ray count?

The planning experiments computed Sionna RT path loss with 300,000 rays per candidate site (sp/judges.py).
A later study (coverage-truth: ct/diffraction_ablation.py; mesa-twin: results/rt_convergence.json) showed
that Sionna RT finds more paths to terrain-shadowed receivers as the ray count grows. This re-runs a random
subset of candidate sites at 10 million rays and compares which demand points each site covers.

    python -m sp.sionna_raycheck      # ~1-2 h on 2 CPU cores -> results/sionna_raycheck.json
"""

from __future__ import annotations

import json

import numpy as np

from .env import Env
from .judges import NO_PATH_DB, WORK
from .pathsim import path_gain

RESULTS = WORK.parent / "results"


def main(n_sites=6, samples=10_000_000, seed=0):
    env = Env()
    cfg = env.cfg
    pl_old = np.load(RESULTS / "path_loss_matrices.npz")["sionna"]                  # [candidates, demand]
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(env.candidates), n_sites, replace=False))
    c = env.candidates.iloc[pick]
    towers = [dict(lat=r.lat, lon=r.lon, height_m=cfg.tx_height_m) for r in c.itertuples()]
    pg, t = path_gain(WORK / cfg.name / "sionna_scene", towers, env.demand.lat.values, env.demand.lon.values,
                      cfg.rx_height_m, cfg.freq_mhz * 1e6, max_depth=2, diffraction=True, samples=samples)
    pl_new = np.minimum(-pg, NO_PATH_DB)
    np.save(WORK / cfg.name / f"pl_sionna_raycheck_{samples}.npy", pl_new)
    old = pl_old[pick]
    thr_pl = cfg.eirp_re_dbm - cfg.threshold_dbm                                     # max path loss that covers
    pop = env.pop
    res = dict(sites=[str(s) for s in c.site], samples_old=300_000, samples_new=samples, seconds=round(t),
               links_with_path_old=int((old < NO_PATH_DB).sum()), links_with_path_new=int((pl_new < NO_PATH_DB).sum()),
               links=int(old.size),
               covered_links_old=int((old <= thr_pl).sum()), covered_links_new=int((pl_new <= thr_pl).sum()),
               covered_pop_old=float(sum(pop[(old[i] <= thr_pl)].sum() for i in range(len(pick)))),
               covered_pop_new=float(sum(pop[(pl_new[i] <= thr_pl)].sum() for i in range(len(pick)))))
    both = (old < NO_PATH_DB) & (pl_new < NO_PATH_DB)
    res["median_abs_change_db_where_both_found"] = float(np.median(np.abs(old - pl_new)[both])) if both.any() else None
    json.dump(res, open(RESULTS / "sionna_raycheck.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
