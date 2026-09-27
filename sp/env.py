"""The planning environment: Black Mesa / Pinon (Navajo Nation), real terrain and real population.

- Demand: Census 2020 block populations (internal points) aggregated to H3 res-8 cells (~0.7 km^2).
- Candidates: a regular grid of possible new tower sites over the area (default 3 km spacing).
- Radio assumptions shared by every judge: 750 MHz (700 MHz LTE band), 30 m towers, 1.5 m users,
  33 dBm EIRP per resource element, coverage = predicted RSRP >= -105 dBm.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .terrain_model import Terrain
from .terrain_scene import load_dem

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WORK = ROOT / "work"


@dataclass(frozen=True)
class Config:
    name: str = "blackmesa"
    bbox: tuple = (36.05, -110.45, 36.55, -109.80)       # south, west, north, east
    candidate_spacing_km: float = 3.0
    freq_mhz: float = 750.0
    tx_height_m: float = 30.0
    rx_height_m: float = 1.5
    eirp_re_dbm: float = 33.0
    threshold_dbm: float = -105.0


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


class Env:
    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        s, w, n, e = cfg.bbox
        self.demand = pd.read_parquet(DATA / "demand_blocks_h8.parquet").reset_index(drop=True)
        lat, lon, z = load_dem(DATA, s - 0.05, w - 0.05, n + 0.05, e + 0.05, step=1)
        self.grid = dict(lat=lat, lon=lon, z=z)
        self.terrain = Terrain(self.grid)
        self.candidates = self._candidate_grid()

    def _candidate_grid(self) -> pd.DataFrame:
        s, w, n, e = self.cfg.bbox
        dlat = self.cfg.candidate_spacing_km / 111.32
        dlon = dlat / np.cos(np.radians((s + n) / 2))
        la = np.arange(s + dlat / 2, n, dlat)
        lo = np.arange(w + dlon / 2, e, dlon)
        LA, LO = np.meshgrid(la, lo, indexing="ij")
        c = pd.DataFrame(dict(lat=LA.ravel(), lon=LO.ravel()))
        c["elev_m"] = self.terrain(c.lat.values, c.lon.values)
        c.insert(0, "site", [f"c{i:03d}" for i in range(len(c))])
        return c

    @property
    def pop(self) -> np.ndarray:
        return self.demand["pop"].to_numpy(float)

    def save_manifest(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        self.demand.to_csv(out / "demand.csv", index=False)
        self.candidates.to_csv(out / "candidates.csv", index=False)
        (out / "config.json").write_text(json.dumps(asdict(self.cfg), indent=2))
