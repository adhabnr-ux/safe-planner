"""Download and prepare inputs.   python -m sp.prepare

- Census 2020 blocks for Arizona (TIGER/Line TABBLOCK20, with POP20) -> population per H3 res-8 cell
- Copernicus GLO-30 DEM tiles (fetched on demand by sp.terrain_scene.load_dem)
- NTIA ITM shared library: scripts/build_itm.sh
"""
from __future__ import annotations

import urllib.request

import geopandas as gpd
import h3

from .env import DATA, Config

BLOCKS = "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_04_tabblock20.zip"


def demand(cfg: Config = Config()):
    DATA.mkdir(exist_ok=True)
    z = DATA / "tl_2020_04_tabblock20.zip"
    if not z.exists():
        urllib.request.urlretrieve(BLOCKS, z)
    s, w, n, e = cfg.bbox
    b = gpd.read_file(f"zip://{z}", bbox=(w, s, e, n), ignore_geometry=True,
                      columns=["GEOID20", "POP20", "HOUSING20", "INTPTLAT20", "INTPTLON20"])
    b["lat"], b["lon"] = b.INTPTLAT20.astype(float), b.INTPTLON20.astype(float)
    b = b[b.lat.between(s, n) & b.lon.between(w, e) & (b.POP20 > 0)].copy()
    b["h8"] = [h3.latlng_to_cell(a, o, 8) for a, o in zip(b.lat, b.lon)]
    d = b.groupby("h8").agg(pop=("POP20", "sum"), housing=("HOUSING20", "sum"),
                            lat=("lat", "mean"), lon=("lon", "mean")).reset_index()
    d.to_parquet(DATA / "demand_blocks_h8.parquet")
    return d


if __name__ == "__main__":
    d = demand()
    print(len(d), "demand cells,", int(d["pop"].sum()), "people")
