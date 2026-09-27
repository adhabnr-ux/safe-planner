import csv
from pathlib import Path

import numpy as np
import pytest

from sp.itm import p2p_loss
from sp.plan import PanelObjective, best_of_n, consensus_covered, greedy, local_search

HERE = Path(__file__).parent


ITM_BUILT = any((HERE.parent / "itm-src" / "build").glob("libitm.*"))


@pytest.mark.skipif(not ITM_BUILT, reason="run scripts/build_itm.sh first")
def test_itm_matches_ntia_reference_values():
    rows = list(csv.DictReader(open(HERE / "p2p.csv")))
    pfls = [list(map(float, l.split(","))) for l in open(HERE / "pfls.csv") if l.strip()]
    for r, p in zip(rows, pfls):
        a, _ = p2p_loss(float(r["h_tx__meter"]), float(r["h_rx__meter"]), p, float(r["f__mhz"]), int(r["climate"]),
                        float(r["N_0"]), int(r["pol"]), float(r["epsilon"]), float(r["sigma"]), int(r["mdvar"]),
                        float(r["time"]), float(r["location"]), float(r["situation"]))
        assert abs(a - float(r["A__db"])) < 0.01


def toy():
    # 4 sites x 6 users; judge A thinks site 0 covers everyone, judge B disagrees
    A = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]], bool)
    B = np.array([[1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]], bool)
    return A, B, np.ones(6)


def test_consensus_requires_every_judge():
    A, B, pop = toy()
    assert consensus_covered([A, B], (0,)).tolist() == [True, False, False, False, False, False]
    assert PanelObjective([A], pop)((0,)) == pytest.approx(1.0)
    assert PanelObjective([A, B], pop)((0,)) == pytest.approx(1 / 6)


def test_single_judge_optimizer_exploits_disagreement_referee_does_not():
    A, B, pop = toy()
    pa = greedy(PanelObjective([A], pop), 4, 1)
    pr = greedy(PanelObjective([A, B], pop), 4, 1)
    assert pa == (0,)                                   # A-only optimizer picks the site only A believes in
    assert PanelObjective([B], pop)(pa) < PanelObjective([B], pop)(pr)


def test_local_search_never_worse_than_greedy_and_optimal_on_toy():
    A, B, pop = toy()
    obj = PanelObjective([B], pop)
    g = greedy(obj, 4, 3)
    ls = local_search(obj, 4, 3, np.random.default_rng(0))
    assert obj(ls) >= obj(g) and obj(ls) == pytest.approx(1.0)


def test_best_of_n_monotone_in_expectation():
    rng = np.random.default_rng(0)
    cov = rng.random((40, 100)) < 0.1
    obj = PanelObjective([cov], np.ones(100))
    a = np.mean([obj(best_of_n(obj, 40, 3, 1, np.random.default_rng(s))) for s in range(30)])
    b = np.mean([obj(best_of_n(obj, 40, 3, 200, np.random.default_rng(s))) for s in range(30)])
    assert b > a


def test_mesh_elevation_matches_triangles():
    from sp.pathsim import elev_lookup
    lat = np.array([[1.0, 1.0], [0.0, 0.0]])
    lon = np.array([[0.0, 1.0], [0.0, 1.0]])
    z = np.array([[0.0, 10.0], [20.0, 40.0]])
    g = dict(lat=lat, lon=lon, z=z)
    # upper-left triangle (a,b,c): at centre-ish point u=0.25 (east), v=0.25 (south) -> 0 + .25*10 + .25*20
    assert elev_lookup(g, 0.75, 0.25) == pytest.approx(7.5)
    # lower-right triangle (b,c,d): u=0.75, v=0.75 -> 40 + .25*(20-40) + .25*(10-40)
    assert elev_lookup(g, 0.25, 0.75) == pytest.approx(27.5)


RES = HERE.parent / "results"


@pytest.mark.skipif(not (RES / "path_loss_matrices.npz").exists(), reason="results not generated")
def test_published_numbers_match_the_matrices():
    import json
    s = json.loads((RES / "summary.json").read_text())
    pl = np.load(RES / "path_loss_matrices.npz")
    cov = {j: (33.0 - pl[j]) >= -105.0 for j in ("terrain", "itm", "sionna")}
    assert {j: int(c.sum()) for j, c in cov.items()} == s["coverage_links"]
    assert (cov["itm"] & cov["terrain"]).sum() / cov["itm"].sum() > 0.999      # the nesting claim
    assert (cov["sionna"] & cov["itm"]).sum() / cov["sionna"].sum() > 0.95


@pytest.mark.skipif(not (RES / "e2_summary.csv").exists(), reason="results not generated")
def test_headline_referee_effect():
    import pandas as pd
    s = pd.read_csv(RES / "e2_summary.csv")
    ls = s[s.planner == "local_search"].set_index(["held_out", "panel"])
    assert ls.loc[("itm", "terrain"), "overclaim"] > 0.35
    assert ls.loc[("itm", "referee(terrain+sionna)"), "overclaim"] < 0
    assert ls.loc[("itm", "terrain"), "regret"] - ls.loc[("itm", "referee(terrain+sionna)"), "regret"] > 0.15
