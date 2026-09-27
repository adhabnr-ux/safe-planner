"""Planners and the referee.

A plan is a set of K candidate sites. A judge turns path loss into a coverage matrix
cov[site, demand_point] (RSRP >= threshold). A plan's coverage under a *panel* of judges is the
population share covered according to EVERY judge in the panel (consensus) - that is the referee.
With a single-judge panel, the referee is just that judge.

Planners, in increasing optimization pressure:
  random        one random plan (no optimization)
  best_of_n     best of N random plans (pressure grows with N)
  greedy        add the site with the largest marginal gain, K times
  local_search  greedy start + first-improvement single swaps until no swap helps (+ random restarts)
"""
from __future__ import annotations

import numpy as np


def consensus_covered(covs: list[np.ndarray], plan) -> np.ndarray:
    """A point is covered by the plan under the panel if every judge sees it covered by some site in the plan.
    Note: this is per-judge 'any site', then AND across judges (a user may be served by different sites
    in different models), which is the fair definition of 'every model agrees this user is covered'."""
    return np.logical_and.reduce([c[list(plan)].any(0) for c in covs])


class PanelObjective:
    """Referee score: population share covered according to every judge in the panel (see consensus_covered)."""

    def __init__(self, covs: list[np.ndarray], pop: np.ndarray):
        self.covs = covs
        self.pop = pop / pop.sum()

    def __call__(self, plan) -> float:
        return float(self.pop @ consensus_covered(self.covs, plan))


def random_plan(n_sites, k, rng, allowed=None):
    pool = np.arange(n_sites) if allowed is None else np.asarray(allowed)
    return tuple(sorted(rng.choice(pool, size=k, replace=False)))


def best_of_n(obj, n_sites, k, n, rng, allowed=None):
    best, best_v = None, -1.0
    for _ in range(n):
        p = random_plan(n_sites, k, rng, allowed)
        v = obj(p)
        if v > best_v:
            best, best_v = p, v
    return best


def greedy(obj, n_sites, k, allowed=None, start=()):
    pool = list(range(n_sites)) if allowed is None else list(allowed)
    plan = list(start)
    while len(plan) < k:
        cand = [s for s in pool if s not in plan]
        vals = [obj(plan + [s]) for s in cand]
        plan.append(cand[int(np.argmax(vals))])
    return tuple(sorted(plan))


def local_search(obj, n_sites, k, rng, allowed=None, restarts=4, max_iter=50):
    pool = list(range(n_sites)) if allowed is None else list(allowed)
    starts = [greedy(obj, n_sites, k, allowed)] + [random_plan(n_sites, k, rng, allowed) for _ in range(restarts)]
    best, best_v = None, -1.0
    for plan in starts:
        plan, v = list(plan), obj(plan)
        for _ in range(max_iter):
            improved = False
            for i in range(k):
                others = plan[:i] + plan[i + 1:]
                for s in pool:
                    if s in plan:
                        continue
                    nv = obj(others + [s])
                    if nv > v + 1e-12:
                        plan, v, improved = others + [s], nv, True
                        break
                if improved:
                    break
            if not improved:
                break
        if v > best_v:
            best, best_v = tuple(sorted(plan)), v
    return best
