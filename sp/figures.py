"""Figures for the write-up.  python -m sp.figures [tag]"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402

from .env import DATA, ROOT, WORK, Env  # noqa: E402
from .experiments import coverage_matrices  # noqa: E402
from .plan import PanelObjective, local_search  # noqa: E402
from .terrain_scene import load_dem  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
NAMES = {"terrain": "Terrain model", "itm": "Longley-Rice (NTIA)", "sionna": "Sionna RT (NVIDIA)"}
COLORS = {"terrain": "#4c72b0", "itm": "#dd8452", "sionna": "#55a868"}


def goodhart(res):
    e1 = pd.read_csv(res / "e1_pressure.csv")
    judges = sorted(e1.train.unique(), key=list(NAMES).index)
    fig, axes = plt.subplots(1, len(judges), figsize=(4.6 * len(judges), 3.9), sharey=True)
    for ax, t in zip(np.atleast_1d(axes), judges):
        d = e1[e1.train == t]
        for j in judges:
            g = d.groupby("n")[f"score_{j}"]
            m, se = g.mean(), g.sem()
            lw, ls = (2.6, "-") if j == t else (1.8, "--")
            ax.plot(m.index, 100 * m, ls, color=COLORS[j], lw=lw, marker="o", ms=3,
                    label=f"{NAMES[j]}{' (training judge)' if j == t else ''}")
            ax.fill_between(m.index, 100 * (m - se), 100 * (m + se), color=COLORS[j], alpha=0.15)
        ax.set_xscale("log")
        ax.set_title(f"Optimized against: {NAMES[t]}", fontsize=10)
        ax.set_xlabel("optimization pressure (random plans tried, N)")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("population covered by the chosen 5-tower plan (%)")
    np.atleast_1d(axes)[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("The harder a planner optimizes against one judge, the more its score and the other judges' scores diverge",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "goodhart_curves.png", dpi=150)
    plt.close(fig)


def referee_bars(res, planner="local_search"):
    df = pd.read_csv(res / "e2_leave_one_out.csv")
    df = df[(df.planner == planner) & ~df.panel.str.startswith("oracle")]
    helds = sorted(df.held_out.unique(), key=list(NAMES).index)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.8 / 3
    for mi, (metric, label) in enumerate([("overclaim", "Over-claim: claimed − held-out coverage (points)"),
                                           ("regret", "Regret on held-out judge (points)")]):
        ax = axes[mi]
        for hi, h in enumerate(helds):
            d = df[df.held_out == h]
            panels = [p for p in d.panel.unique() if not p.startswith("referee")] + \
                     [p for p in d.panel.unique() if p.startswith("referee")]
            for pi, p in enumerate(panels):
                v = d[d.panel == p][metric] * 100
                ref = p.startswith("referee")
                color = "#333333" if ref else COLORS[p]
                ax.bar(hi + (pi - 1) * width, v.mean(), width, yerr=v.sem(), color=color, alpha=0.9 if ref else 0.75,
                       capsize=2, label=("referee (all other judges)" if ref else f"trained on {NAMES[p]}")
                       if hi == 0 or (not ref and p not in [q for q in df[df.held_out == helds[0]].panel]) else None)
        ax.set_xticks(range(len(helds)))
        ax.set_xticklabels([f"held out:\n{NAMES[h]}" for h in helds], fontsize=8.5)
        ax.set_ylabel(label, fontsize=9)
        ax.axhline(0, color="black", lw=0.8)
        ax.grid(axis="y", alpha=0.3)
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(f"Leave-one-judge-out ({planner.replace('_', ' ')}; K = 3, 5, 8; 30 seeds each; mean ± s.e.)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(FIG / "referee_leave_one_out.png", dpi=150)
    plt.close(fig)


def plans_map(env, judges, k=5):
    covs = coverage_matrices(env, judges)
    pop = env.pop
    s, w, n, e = env.cfg.bbox
    lat, lon, z = load_dem(DATA, s, w, n, e, step=6)
    hs = LightSource(azdeg=315, altdeg=35).hillshade(z, vert_exag=2, dx=180, dy=180)
    panels = [(j, [j]) for j in judges] + [("referee", list(judges))]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.8))
    for ax, (name, panel) in zip(axes, panels):
        obj = PanelObjective([covs[j] for j in panel], pop)
        plan = local_search(obj, len(env.candidates), k, np.random.default_rng(0))
        ax.imshow(hs, cmap="gray", extent=(lon.min(), lon.max(), lat.min(), lat.max()), alpha=0.5)
        ax.scatter(env.demand.lon, env.demand.lat, s=4 + env.demand["pop"] / 3, c="#e377c2", alpha=0.55,
                   edgecolors="none", label="people (Census 2020)")
        c = env.candidates.iloc[list(plan)]
        ax.scatter(c.lon, c.lat, marker="^", s=130, color="black", edgecolors="white", label="chosen towers")
        scores = " · ".join(f"{NAMES[j].split(' ')[0]} {100 * PanelObjective([covs[j]], pop)(plan):.0f}%" for j in judges)
        title = "Referee (all judges agree)" if name == "referee" else f"Optimized for {NAMES[name]}"
        ax.set_title(f"{title}\n{scores}", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1 / np.cos(np.radians((s + n) / 2)))
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Where each planner puts {k} towers around Black Mesa / Piñon, and how each judge scores the plan",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "plans_map.png", dpi=150)
    plt.close(fig)


def main(tag="main"):
    res = WORK / "blackmesa" / f"results_{tag}"
    judges = [j for j in NAMES if (WORK / "blackmesa" / f"pl_{j}.npy").exists()]
    goodhart(res)
    referee_bars(res)
    plans_map(Env(), judges)


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["main"]))
