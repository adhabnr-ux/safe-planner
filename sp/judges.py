"""Three independent 'judges' of radio coverage. Each returns path loss (dB) for every candidate
tower -> every demand point, cached to work/<env>/pl_<judge>.npy.

  terrain : free space + ITU-R P.526 Bullington knife-edge over the 30 m DEM profile (deterministic,
            geometric; built for Coverage Truth)
  itm     : NTIA Irregular Terrain Model (Longley-Rice) point-to-point, median (50/50/50) - semi-empirical,
            fitted to decades of field measurements; the US government reference model
  sionna  : NVIDIA Sionna RT ray tracing on a terrain mesh (LoS + specular reflection + first-order
            edge diffraction) - full-wave-inspired geometric optics

They share only the terrain data and the link budget, so their errors come from different physics.
"""
from __future__ import annotations

import time

import numpy as np

from .env import WORK, Env, haversine_km
from .itm import p2p_loss
from .terrain_model import path_loss_db

JUDGES = ("terrain", "itm", "sionna")
NO_PATH_DB = 250.0       # Sionna returns no path at all for many shadowed links: treat as 'no signal'


def cache_path(env: Env, judge: str):
    d = WORK / env.cfg.name
    d.mkdir(parents=True, exist_ok=True)
    return d / f"pl_{judge}.npy"


def pl_terrain(env: Env) -> np.ndarray:
    c, d, cfg = env.candidates, env.demand, env.cfg
    out = np.zeros((len(c), len(d)))
    for i, r in enumerate(c.itertuples()):
        dist = haversine_km(r.lat, r.lon, d.lat.values, d.lon.values).max()
        n = int(np.clip(dist * 1000 / 30, 200, 2500))
        out[i], _, _ = path_loss_db(env.terrain, r.lat, r.lon, cfg.tx_height_m, d.lat.values, d.lon.values,
                                    cfg.rx_height_m, cfg.freq_mhz * 1e6, n=n)
    return out


def profile(env: Env, lat1, lon1, lat2, lon2, step_m=30.0):
    dist_m = max(haversine_km(lat1, lon1, lat2, lon2) * 1000, 1.0)
    n = max(int(np.ceil(dist_m / step_m)), 10)
    s = np.linspace(0, 1, n + 1)
    z = env.terrain(lat1 + (lat2 - lat1) * s, lon1 + (lon2 - lon1) * s)
    return np.r_[n, dist_m / n, z]


def pl_itm(env: Env) -> np.ndarray:
    c, d, cfg = env.candidates, env.demand, env.cfg
    out = np.zeros((len(c), len(d)))
    for i, r in enumerate(c.itertuples()):
        for j, q in enumerate(d.itertuples()):
            pfl = profile(env, r.lat, r.lon, q.lat, q.lon)
            out[i, j], _ = p2p_loss(cfg.tx_height_m, cfg.rx_height_m, pfl, cfg.freq_mhz)
    return out


def pl_sionna(env: Env, scene_dir=None, samples=300_000, max_depth=2) -> np.ndarray:
    from .pathsim import path_gain
    from .terrain_scene import build
    cfg = env.cfg
    s, w, n, e = cfg.bbox
    scene_dir = scene_dir or (WORK / cfg.name / "sionna_scene")
    if not (scene_dir / "scene.xml").exists():
        build(scene_dir, WORK.parent / "data", (s - 0.03, w - 0.03, n + 0.03, e + 0.03),
              lat0=(s + n) / 2, lon0=(w + e) / 2, terrain_step=4, ue_step=16)
    towers = [dict(lat=r.lat, lon=r.lon, height_m=cfg.tx_height_m) for r in env.candidates.itertuples()]
    pg, _ = path_gain(scene_dir, towers, env.demand.lat.values, env.demand.lon.values, cfg.rx_height_m,
                      cfg.freq_mhz * 1e6, max_depth=max_depth, diffraction=True, samples=samples)
    return np.minimum(-pg, NO_PATH_DB)


def compute(env: Env, judge: str, force=False) -> np.ndarray:
    p = cache_path(env, judge)
    if p.exists() and not force:
        return np.load(p)
    t = time.time()
    pl = {"terrain": pl_terrain, "itm": pl_itm, "sionna": pl_sionna}[judge](env)
    np.save(p, pl)
    print(f"{judge}: {pl.shape} in {time.time() - t:.0f} s", flush=True)
    return pl


def rsrp(env: Env, pl: np.ndarray) -> np.ndarray:
    return env.cfg.eirp_re_dbm - pl


if __name__ == "__main__":
    import sys
    env = Env()
    for j in (sys.argv[1:] or JUDGES):
        compute(env, j)
