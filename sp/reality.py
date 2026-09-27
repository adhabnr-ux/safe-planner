"""Reality check: which judge agrees best with where people actually got service?

Ookla speed tests (Jul 2025 - Jun 2026) exist in 41 map tiles inside the study area. Every one of them is a
place where a phone demonstrably had a working connection. For the towers that serve the area today
(registered + claim-inferred sites of AT&T, Verizon and Smith Bagley, from Coverage Truth), each judge
predicts the best-server RSRP at those tiles. A judge that says 'no coverage' where tests happened is
wrong there (a miss). This measures only one error direction (tests never prove absence of coverage),
so it is reported as recall, alongside each judge's overall optimism (share of the area it calls covered).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .env import WORK, Env, haversine_km
from .itm import p2p_loss
from .judges import NO_PATH_DB, profile
from .terrain_model import path_loss_db

COVERAGE_TRUTH = WORK.parent.parent / "coverage"


def existing_sites(env: Env, dedupe_km=1.0) -> pd.DataFrame:
    s, w, n, e = env.cfg.bbox
    rows = []
    for c in ("att", "verizon", "smith"):
        f = COVERAGE_TRUTH / "results" / "blackmesa" / f"physics_{c}" / "sites.csv"
        d = pd.read_csv(f)
        d["carrier"] = c
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d = d[d.lat.between(s - 0.02, n + 0.02) & d.lon.between(w - 0.02, e + 0.02)].reset_index(drop=True)
    keep = []
    for i, r in d.iterrows():
        if all(haversine_km(r.lat, r.lon, d.lat[j], d.lon[j]) > dedupe_km for j in keep):
            keep.append(i)
    return d.loc[keep].reset_index(drop=True)


def tested_tiles(env: Env) -> pd.DataFrame:
    s, w, n, e = env.cfg.bbox
    t = pd.read_parquet(COVERAGE_TRUTH / "work" / "tiles_vs_claims.parquet")
    return t[t.lat.between(s, n) & t.lon.between(w, e)].reset_index(drop=True)


def run(env: Env) -> pd.DataFrame:
    cfg = env.cfg
    sites, tiles = existing_sites(env), tested_tiles(env)
    rx_lat, rx_lon = tiles.lat.values, tiles.lon.values
    pl = {}
    pl["terrain"] = np.array([path_loss_db(env.terrain, r.lat, r.lon, cfg.tx_height_m, rx_lat, rx_lon, cfg.rx_height_m,
                                           cfg.freq_mhz * 1e6, n=2000)[0] for r in sites.itertuples()])
    pl["itm"] = np.array([[p2p_loss(cfg.tx_height_m, cfg.rx_height_m, profile(env, r.lat, r.lon, a, o), cfg.freq_mhz)[0]
                           for a, o in zip(rx_lat, rx_lon)] for r in sites.itertuples()])
    from .pathsim import path_gain
    towers = [dict(lat=r.lat, lon=r.lon, height_m=cfg.tx_height_m) for r in sites.itertuples()]
    pg, _ = path_gain(WORK / cfg.name / "sionna_scene", towers, rx_lat, rx_lon, cfg.rx_height_m, cfg.freq_mhz * 1e6,
                      max_depth=2, diffraction=True, samples=300_000)
    pl["sionna"] = np.minimum(-pg, NO_PATH_DB)
    rows = []
    for j, m in pl.items():
        best = cfg.eirp_re_dbm - m.min(0)
        for thr in (-105, -115):
            rows.append(dict(judge=j, threshold_dbm=thr, tiles=len(tiles), sites=len(sites),
                             recall=float((best >= thr).mean()),
                             tests_recalled=float((tiles.tests * (best >= thr)).sum() / tiles.tests.sum())))
    out = pd.DataFrame(rows)
    d = WORK / cfg.name
    out.to_csv(d / "reality_check.csv", index=False)
    sites.to_csv(d / "existing_sites.csv", index=False)
    np.savez(d / "reality_pl.npz", **pl)
    return out


if __name__ == "__main__":
    print(run(Env()))
