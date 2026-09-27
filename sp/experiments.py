"""Experiments.

E1  Optimization-pressure curve (Goodhart): best-of-N random search on one judge, N = 1 .. 10^4;
    report the training judge's score and every other judge's score of the chosen plan.
E2  Leave-one-judge-out: for each held-out judge H and each planner, train on
      (a) each single other judge, (b) the referee panel of all other judges;
    evaluate on H. Metrics per plan:
      claimed   = training-panel score (what the planner reports)
      heldout   = coverage according to H
      overclaim = claimed - heldout
      regret    = oracle(H) - heldout, where oracle(H) is the best plan any planner finds when trained on H
    Seeds vary the candidate pool (random 70% of sites, as if land/permits exclude the rest) and the
    planner's randomness; K in {3, 5, 8}.
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pandas as pd

from .env import WORK, Env
from .plan import PanelObjective, best_of_n, greedy, local_search, random_plan

K_VALUES = (3, 5, 8)
SEEDS = 30
POOL_FRACTION = 0.7


def coverage_matrices(env: Env, judges) -> dict[str, np.ndarray]:
    thr = env.cfg.threshold_dbm
    return {j: (env.cfg.eirp_re_dbm - np.load(WORK / env.cfg.name / f"pl_{j}.npy")) >= thr for j in judges}


def run_planner(name, obj, n_sites, k, rng, pool):
    if name == "random":
        return random_plan(n_sites, k, rng, pool)
    if name == "best_of_1000":
        return best_of_n(obj, n_sites, k, 1000, rng, pool)
    if name == "greedy":
        return greedy(obj, n_sites, k, pool)
    if name == "local_search":
        return local_search(obj, n_sites, k, rng, pool, restarts=3)
    raise ValueError(name)


def e1_pressure(env: Env, covs, train: str, k=5, seeds=20, ns=(1, 3, 10, 30, 100, 300, 1000, 3000, 10000)):
    pop, m = env.pop, len(env.candidates)
    objs = {j: PanelObjective([c], pop) for j, c in covs.items()}
    rows = []
    for n in ns:
        for s in range(seeds):
            p = best_of_n(objs[train], m, k, n, np.random.default_rng(1000 * s + n))
            rows.append(dict(train=train, n=n, seed=s, **{f"score_{j}": o(p) for j, o in objs.items()}))
    return pd.DataFrame(rows)


def e2_leave_one_out(env: Env, covs, planners=("random", "best_of_1000", "greedy", "local_search"),
                     k_values=K_VALUES, seeds=SEEDS):
    pop, m = env.pop, len(env.candidates)
    judges = list(covs)
    rows = []
    for held in judges:
        others = [j for j in judges if j != held]
        panels = {j: [j] for j in others}
        panels["referee(" + "+".join(others) + ")"] = others
        panels["oracle(" + held + ")"] = [held]
        h_obj = PanelObjective([covs[held]], pop)
        for k, s in itertools.product(k_values, range(seeds)):
            rng = np.random.default_rng(10_000 * k + s)
            pool = np.sort(rng.choice(m, size=int(POOL_FRACTION * m), replace=False))
            for planner in planners:
                for pname, panel in panels.items():
                    obj = PanelObjective([covs[j] for j in panel], pop)
                    plan = run_planner(planner, obj, m, k, np.random.default_rng(s), pool)
                    rows.append(dict(held_out=held, planner=planner, panel=pname, k=k, seed=s,
                                     claimed=obj(plan), heldout=h_obj(plan), plan=list(map(int, plan))))
    df = pd.DataFrame(rows)
    # oracle = the best plan found for H itself (strongest planner trained directly on H, same pool)
    orc = df[df.panel.str.startswith("oracle")].groupby(["held_out", "k", "seed"]).heldout.max()
    df["oracle"] = df.set_index(["held_out", "k", "seed"]).index.map(orc).values
    df["overclaim"] = df.claimed - df.heldout
    df["regret"] = df.oracle - df.heldout
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df[~df.panel.str.startswith("oracle")].groupby(["held_out", "planner", "panel"])
    out = g.agg(n=("seed", "size"), claimed=("claimed", "mean"), heldout=("heldout", "mean"),
                overclaim=("overclaim", "mean"), overclaim_se=("overclaim", "sem"),
                regret=("regret", "mean"), regret_se=("regret", "sem"))
    return out.reset_index()


def main(judges=("terrain", "itm", "sionna"), tag="main"):
    env = Env()
    covs = coverage_matrices(env, judges)
    out = WORK / env.cfg.name / f"results_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    e1 = pd.concat([e1_pressure(env, covs, j) for j in judges], ignore_index=True)
    e1.to_csv(out / "e1_pressure.csv", index=False)
    print(f"E1 done {time.time() - t:.0f}s", flush=True)
    e2 = e2_leave_one_out(env, covs)
    e2.to_csv(out / "e2_leave_one_out.csv", index=False)
    s = summarize(e2)
    s.to_csv(out / "e2_summary.csv", index=False)
    print(f"E2 done {time.time() - t:.0f}s", flush=True)
    (out / "meta.json").write_text(json.dumps(dict(judges=list(judges), k_values=K_VALUES, seeds=SEEDS,
                                                  pool_fraction=POOL_FRACTION, n_candidates=len(env.candidates),
                                                  n_demand=len(env.demand), population=float(env.pop.sum())), indent=2))
    return s


if __name__ == "__main__":
    import sys
    js = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("terrain", "itm", "sionna")
    tag = sys.argv[2] if len(sys.argv) > 2 else "main"
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(main(js, tag))
