"""Draw the README schematics for PRO_Utils into ``PRO_Utils/docs/``.

Run from the pipelines folder with the mosaic conda environment (~/miniconda3/envs/mosaic)::

    python docs/make_PRO_Utils_figures.py

Figures
-------
hto_decision_flow.png       Decision logic of ``call_single_hashtag``.
hto_simulated_fit.png       Real ``call_single_hashtag`` fit on simulated bimodal and unimodal tags.
silhouette_schematic.png    What the per-cell silhouette and ``homogeneity_like`` measure (simulated).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _style import (AQUA, BLUE, BLUE_T, GRAY_T, INK, INK2, MUTED, ORANGE, ORANGE_T, PIPELINES,  # noqa: E402
                    YELLOW_T, arrow, box, canvas, plt, save)

sys.path.insert(0, PIPELINES)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from PRO_Utils.homogeneity_silhouette import plot_protein_homogeneity_silhouette  # noqa: E402
from PRO_Utils.hto_calling import call_single_hashtag  # noqa: E402

PKG = "PRO_Utils"


# ── 1. Decision flow of call_single_hashtag ──────────────────────────────────────────────────────
def hto_decision_flow():
    fig, ax = canvas(10.5, 6.4, 21, 12.8, title="call_single_hashtag: present / absent decision for one hashtag")

    box(ax, 3.2, 11.2, 5.6, 1.1, "Raw counts of one hashtag\n(all cells, zeros included)", fc=GRAY_T)
    box(ax, 3.2, 9.3, 5.6, 0.9, "x = log1p(counts)", fc=GRAY_T)
    arrow(ax, [(3.2, 10.65), (3.2, 9.75)])

    box(ax, 1.6, 7.4, 2.6, 1.1, "GMM, k = 1", fc=GRAY_T, size=9)
    box(ax, 5.0, 7.4, 3.5, 1.1, "GMM, k = 2\nn_init=10, seed 42", fc=GRAY_T, size=8.8)
    arrow(ax, [(2.4, 8.85), (1.9, 7.95)])
    arrow(ax, [(4.0, 8.85), (4.7, 7.95)])

    box(ax, 3.2, 5.0, 6.0, 1.7,
        "Bimodal?\nBIC(2) < BIC(1)  AND\nmean gap >= 1.0 log units", fc=SURFACE_ALT, weight="bold", size=9.3)
    arrow(ax, [(1.6, 6.85), (1.6, 5.85)])
    arrow(ax, [(5.0, 6.85), (5.0, 5.85)])

    # No branch
    box(ax, 3.4, 1.6, 6.6, 2.1,
        "Unimodal: no split trusted (HTOResult)\nall cells 'Negative', p_pos = 0\nthreshold = None, bimodal = False\nnote = 'unimodal: ...'",
        fc=ORANGE_T, ec=ORANGE, size=8.8)
    arrow(ax, [(3.2, 4.15), (3.2, 2.65)], label="no", label_xy=(3.55, 3.4), ha="left")

    # Yes branch
    arrow(ax, [(6.2, 5.0), (8.3, 5.0)], label="yes", label_xy=(7.25, 5.3))
    box(ax, 11.2, 5.0, 5.9, 1.3, "p_pos = posterior of the\nhigher-mean component (per cell)", fc=BLUE_T, ec=BLUE, size=9)

    box(ax, 11.2, 8.4, 5.9, 1.3, "Positive  if p_pos >= 0.5\nNegative  otherwise", fc=BLUE_T, ec=BLUE, size=9.3)
    arrow(ax, [(11.2, 5.65), (11.2, 7.75)], label="posterior_cutoff = 0.5", label_xy=(11.45, 6.7), ha="left")

    box(ax, 17.7, 8.4, 6.0, 1.3, "uncertain_band = (0.05, 0.95):\n0.05 < p_pos < 0.95  ->  'Uncertain'",
        fc=GRAY_T, size=9)
    arrow(ax, [(14.15, 8.4), (14.7, 8.4)])
    note(ax, 14.8, 6.9, "uncertain_band=None keeps the\nhard Positive / Negative split")

    box(ax, 11.2, 1.6, 5.9, 2.1,
        "Reported threshold (reporting only)\nexpm1 of first upward crossing of\np_pos = 0.5 on a 2000-point\nlog1p grid; None if no crossing",
        fc=GRAY_T, size=8.8)
    arrow(ax, [(11.2, 4.35), (11.2, 2.65)])

    box(ax, 17.7, 4.0, 6.0, 2.3,
        "HTOResult\ncall, p_pos, threshold,\nbimodal, means [neg, pos],\nnote, gmm",
        fc=SURFACE_ALT, size=9)
    arrow(ax, [(17.7, 7.75), (17.7, 5.15)])
    arrow(ax, [(14.15, 1.6), (17.7, 1.6), (17.7, 2.85)])
    return save(fig, PKG, "hto_decision_flow.png")


SURFACE_ALT = "#ffffff"


def note(ax, x, y, text, size=8.6):
    ax.text(x, y, text, fontsize=size, color=INK2, va="bottom", linespacing=1.4, style="italic")


# ── 2. Real fit on simulated tags ────────────────────────────────────────────────────────────────
def _simulate_tags(seed=0):
    rng = np.random.default_rng(seed)
    n_neg, n_pos = 1400, 600
    neg = np.round(np.expm1(rng.normal(3.0, 0.6, n_neg))).clip(min=0)
    pos = np.round(np.expm1(rng.normal(6.0, 0.5, n_pos))).clip(min=0)
    bimodal = pd.Series(np.concatenate([neg, pos]), name="Simulated-bimodal")
    uni = pd.Series(np.round(np.expm1(rng.normal(3.6, 0.7, 2000))).clip(min=0), name="Simulated-unimodal")
    return bimodal, uni


def _draw_fit(ax, counts, res, show_posterior):
    logv = np.log1p(counts.to_numpy(dtype=float))
    xs = np.linspace(logv.min(), logv.max(), 500)
    ax.hist(logv, bins=60, density=True, color="#ffffff", ec=MUTED, lw=0.5)
    g = res.gmm
    sig = int(np.argmax(g.means_.flatten()))
    for k, (w, m, c) in enumerate(zip(g.weights_, g.means_.flatten(), g.covariances_.flatten())):
        col = BLUE if (k == sig and res.bimodal) else (ORANGE if res.bimodal else MUTED)
        lab = None
        if res.bimodal:
            lab = "signal component" if k == sig else "background component"
        ax.plot(xs, w * norm.pdf(xs, m, np.sqrt(c)), color=col, lw=1.8, label=lab)
    ax.set_xlabel("log1p(count)")
    ax.set_ylabel("density")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if show_posterior:
        p = g.predict_proba(xs.reshape(-1, 1))[:, sig]
        ax2 = ax.twinx()
        inside = (p > 0.05) & (p < 0.95)
        ax2.axvspan(xs[inside].min(), xs[inside].max(), color="#e4e2dc", lw=0, zorder=0,
                    label="'Uncertain' band: 0.05 < p_pos < 0.95")
        ax2.plot(xs, p, color=INK, lw=1.4, label="p_pos (posterior)")
        ax2.axhline(0.5, color=INK2, lw=0.8, ls=":", label="p_pos = 0.5 (posterior_cutoff)")
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_ylabel("p_pos")
        ax2.spines["top"].set_visible(False)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        t = res.threshold
        ax.axvline(np.log1p(t), color=AQUA, lw=2, ls="--", label=f"threshold ~ {t:.0f} counts")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), ncol=3)


def hto_simulated_fit():
    bim, uni = _simulate_tags()
    rb = call_single_hashtag(bim)
    ru = call_single_hashtag(uni)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.55, 1]})
    _draw_fit(axes[0], bim, rb, show_posterior=True)
    n_pos = int((rb.call == "Positive").sum())
    n_unc = int((rb.call == "Uncertain").sum())
    gap_b = rb.means[1] - rb.means[0]
    axes[0].set_title(f"Simulated bimodal tag: bimodal=True, mean gap {gap_b:.2f}\n"
                      f"{n_pos} Positive, {n_unc} Uncertain, {len(bim) - n_pos - n_unc} Negative",
                      loc="left", fontsize=9.5)

    _draw_fit(axes[1], uni, ru, show_posterior=False)
    gap_u = ru.means[1] - ru.means[0]
    axes[1].set_title(f"Simulated unimodal tag: bimodal=False\nmean gap {gap_u:.2f} -> all {len(uni)} cells Negative",
                      loc="left", fontsize=9.5)
    fig.suptitle("call_single_hashtag on simulated data (default parameters)", x=0.01, ha="left",
                 fontsize=11, weight="bold")
    fig.tight_layout()
    print(f"bimodal: thr={rb.threshold}, means={rb.means}, pos={n_pos}, unc={n_unc}")
    print(f"unimodal: means={ru.means}, note={ru.note}")
    return save(fig, PKG, "hto_simulated_fit.png")


# ── 3. Silhouette schematic ──────────────────────────────────────────────────────────────────────
def silhouette_schematic():
    rng = np.random.default_rng(3)
    centres = {"A": (0.0, 0.0), "B": (4.2, 0.4), "C": (2.6, 1.6)}
    spreads = {"A": 0.55, "B": 0.6, "C": 1.15}
    rows = []
    for lab, (cx, cy) in centres.items():
        n = 120
        xy = rng.normal(0, spreads[lab], size=(n, 2)) + (cx, cy)
        rows.append(pd.DataFrame({"label": lab, "CD34": xy[:, 0], "CD3": xy[:, 1]}))
    df = pd.concat(rows, ignore_index=True)

    per_label, _ = plot_protein_homogeneity_silhouette(df, label_col="label", scale=False, show_plot=False,
                                                       verbose=False)
    plt.close("all")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"width_ratios": [1.3, 1]})
    cols = {"A": BLUE, "B": ORANGE, "C": AQUA}
    for lab in centres:
        d = df[df.label == lab]
        ax0.scatter(d.CD34, d.CD3, s=9, color=cols[lab], alpha=0.55, lw=0, label=f"label {lab}")
    # one illustrative cell in A
    a_cells = df[df.label == "A"][["CD34", "CD3"]].to_numpy()
    i = int(np.argmin(((a_cells - (0.5, 0.3)) ** 2).sum(1)))
    p = a_cells[i]
    ax0.scatter(*p, s=60, color=INK, zorder=5)
    # mean distance from the cell to every cell of each label; b uses the nearest other label
    dist = {lab: np.linalg.norm(df[df.label == lab][["CD34", "CD3"]].to_numpy() - p, axis=1).mean()
            for lab in centres}
    a_val = dist["A"]
    nearest = min((lab for lab in centres if lab != "A"), key=dist.get)
    b_val = dist[nearest]
    near_cells = df[df.label == nearest][["CD34", "CD3"]].to_numpy()
    for q in a_cells[rng.choice(len(a_cells), 14, replace=False)]:
        ax0.plot([p[0], q[0]], [p[1], q[1]], color=BLUE, lw=0.6, alpha=0.8, zorder=3)
    for q in near_cells[rng.choice(len(near_cells), 14, replace=False)]:
        ax0.plot([p[0], q[0]], [p[1], q[1]], color=INK2, lw=0.6, ls="--", alpha=0.7, zorder=3)
    s_val = (b_val - a_val) / max(a_val, b_val)
    ax0.set_ylim(-2.6, 7.4)
    ax0.text(-1.7, 7.2, f"highlighted cell (label A):\n"
             f"a = mean distance to cells of A = {a_val:.2f}\n"
             f"b = mean distance to cells of {nearest} (nearest other label) = {b_val:.2f}\n"
             f"s = {s_val:.2f}", fontsize=8.2, va="top", color=INK,
             bbox=dict(fc="#ffffff", ec=MUTED, lw=0.6, boxstyle="round,pad=0.3"), zorder=6)
    ax0.set_title("Per-cell silhouette  s = (b - a) / max(a, b)\n"
                  "s near 1: well inside its label; s < 0: closer to another label",
                  loc="left", fontsize=9.5)
    ax0.set_xlabel("protein feature 1")
    ax0.set_ylabel("protein feature 2")
    ax0.legend(fontsize=8, frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax0.spines[s].set_visible(False)

    pl = per_label.sort_values("mean", ascending=False)
    x = np.arange(len(pl))
    ax1.bar(x, pl["homogeneity_like"], color=[cols[l] for l in pl["label"]], alpha=0.85, width=0.6)
    for xi, (m, h) in enumerate(zip(pl["mean"], pl["homogeneity_like"])):
        ax1.text(xi, h + 0.03, f"mean s = {m:.2f}", ha="center", fontsize=8.5)
    ax1.set_xticks(x, [f"label {l}" for l in pl["label"]])
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("homogeneity_like = max(mean s, 0)")
    ax1.set_title("Per-label summary returned by the function\n(simulated, scale=False)", loc="left", fontsize=9.5)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    fig.suptitle("plot_protein_homogeneity_silhouette: what the scores measure (simulated data)", x=0.01,
                 ha="left", fontsize=11, weight="bold")
    fig.tight_layout()
    print(pl)
    return save(fig, PKG, "silhouette_schematic.png")


if __name__ == "__main__":
    for f in (hto_decision_flow, hto_simulated_fit, silhouette_schematic):
        print(f())
