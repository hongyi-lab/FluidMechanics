#!/usr/bin/env python
"""Figures for the Gate 3 report (white background, PNG + PDF).

    python code/scripts/gate3_figures.py experiments/<gate3 run> docs/reports/figs/gate3
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from committor.plotting import C_ALT, C_FIT, C_PTS, C_REF, STYLE, _save  # noqa: E402

run, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
R = json.loads((run / "results.json").read_text())


def V(x):
    """results.json stores +inf as null."""
    return np.inf if x is None else x


def CI(x):
    return [np.nan, np.nan] if x is None else [V(x[0]), V(x[1])]

TH = R["threshold"]
CA = TH["candidates"]
SIG3 = [3e-9, 3e-8, 3e-7]
CS = {3e-9: C_PTS, 3e-8: C_ALT, 3e-7: C_FIT}

# ---------------- fig1: threshold-prediction error, all candidates ----------------
main = [k for k in CA if "@" not in k]
order = sorted(main, key=lambda k: (V(CA[k]["test"]["median_abs"]), V(CA[k]["test"]["q90_abs"])))
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 0.28 * len(order) + 1.6), sharey=True)
    y = np.arange(len(order))
    for a, key, title in ((ax[0], "median_abs", "median |ln(s_pred / s_c)|"), (ax[1], "q90_abs", "90% quantile of |ln(s_pred / s_c)|")):
        v = np.array([V(CA[k]["test"][key]) for k in order])
        ci = np.array([CI(CA[k]["test"].get(key + "_ci")) for k in order])
        fin = np.isfinite(v)
        col = [C_FIT if k.startswith("MLP") else (C_REF if k == "E" else (C_ALT if k.startswith("gain") or "gain" in k else C_PTS)) for k in order]
        lo = np.where(np.isfinite(ci[:, 0]), v - ci[:, 0], 0)
        hi = np.where(np.isfinite(ci[:, 1]), ci[:, 1] - v, 0)
        floor = np.array([CA[k].get("frac_below_grid", 0.0) > 0.5 for k in order])
        for i in np.where(fin)[0]:
            a.errorbar(v[i], y[i], xerr=[[max(lo[i], 0)], [max(hi[i], 0)]], fmt="o", color=col[i], ms=4, capsize=2,
                       mfc="none" if floor[i] else col[i])
        for i in np.where(~fin)[0]:
            a.annotate("inf (no crossing)", (1, y[i]), xycoords=("axes fraction", "data"), ha="right", va="center", fontsize=7, color=C_REF)
        a.set_xscale("log")
        a.set_xlabel(title)
        a.axvline(V(CA["E"]["test"][key]), color=C_REF, lw=0.8, ls="--")
        a.axvline(V(CA[TH["ceiling"]["name"]]["test"][key]), color=C_FIT, lw=0.8, ls="--")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([k + ("  [grid floor]" if CA[k].get("frac_below_grid", 0.0) > 0.5 else "") for k in order], fontsize=8)
    ax[0].invert_yaxis()
    ax[0].set_title(f"(a) test rays (n = {CA['E']['test']['n']}); dashed: E (black), ceiling {TH['ceiling']['name']} (red)", fontsize=9)
    ax[1].set_title("(b) tail of the error; open symbols [grid floor]: threshold met at the first grid point on every ray", fontsize=9)
    fig.tight_layout()
    _save(fig, out / "fig1_threshold_error")
    plt.close(fig)

# ---------------- fig2: accuracy vs |u| ----------------
sel = ["E", "|a3|", "pair (a2, a3)", "triple (1-a1, a2, a3)", "gain (i)", "gain (ii)", "triple + log gain (i)", "MLP (9D)", "MLP (9D, polar input)"]
sel = [k for k in sel if k in CA]
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    cmap = plt.get_cmap("tab10")
    for j, k in enumerate(sel):
        acc = CA[k]["accuracy_by_u"]
        u = np.array([float(x) for x in acc])
        a_ = np.array([acc[x] for x in acc])
        for a, msk, sgn in ((ax[0], u < 0, -1), (ax[1], u > 0, 1)):
            a.semilogx(sgn * u[msk], 1 - a_[msk], "o-", ms=3, lw=1, color=cmap(j % 10), label=k if sgn > 0 else None)
    irr = TH["irreducible"]["by_u"]
    ui = np.array([float(x) for x in irr])
    ax[1].semilogx(ui, [irr[x] for x in irr], "k--", lw=1.2, label="irreducible (decay windows)")
    for a, t in ((ax[0], "(a) u < 0 (state below the edge, label A)"), (ax[1], "(b) u > 0 (above the edge)")):
        a.set_ylim(bottom=0)
        a.set_xlabel("|u|,  state at s_c (1 + u)")
        a.set_ylabel("classification error on test offset states")
        a.set_title(t, fontsize=9)
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _save(fig, out / "fig2_accuracy_vs_u")
    plt.close(fig)

# ---------------- fig3: zero-fit test ----------------
ZF = R["zero_fit"]
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.4))
    xe = {0.0: 0, 1e-6: 1, 1e-4: 2, 1e-2: 3}
    for col, ver in enumerate(("i", "ii")):
        a = ax[0, col]
        pp = ZF[ver]["family_zero_fit"]["per_profile"]
        for k, sg in enumerate(SIG3):
            rows = [p for p in pp if p["sigma"] == sg]
            x = np.array([xe[p["eps"]] + (p["angle"] / 157.5 - 0.5) * 0.25 + (k - 1) * 0.08 for p in rows])
            e = np.array([V(p["ln_err"]) for p in rows])
            fin = np.isfinite(e)
            inr = fin & (np.abs(e) <= 1.0)
            a.plot(x[inr], e[inr], "o", ms=3.5, color=CS[sg], alpha=0.85, label=f"σ = {sg:.0e}")
            for msk, yv, mk in ((fin & (e > 1.0), 1.02, "^"), (fin & (e < -1.0), -1.02, "v")):   # finite, outside the axis
                if msk.any():
                    a.plot(x[msk], np.full(msk.sum(), yv), mk, ms=5, color=CS[sg])
            if (~fin).any():                                                                      # no crossing
                a.plot(x[~fin], np.full((~fin).sum(), 1.07), "x", ms=5, color=CS[sg])
        a.axhline(0, color=C_REF, lw=0.8)
        a.set_xticks(range(4))
        a.set_xticklabels(["ε = 0", "1e-6", "1e-4", "1e-2"])
        s_ = ZF[ver]["family_zero_fit"]["excluding_calibration"]
        a.set_title(f"(a{col + 1}) version ({ver}), zero-fit, family: median |ln| = {V(s_['median_abs']):.3f} (n = {s_['n']})\n"
                    "filled triangles: |ln| > 1 drawn at the axis edge; x: no crossing", fontsize=9)
        a.set_ylabel("ln(s_pred / s50)")
        a.set_ylim(-1.12, 1.12)
        a.legend(fontsize=7)
        a = ax[1, col]
        ev = np.load(run / f"gain_{ver}_at_edge.npy")
        ev = ev[np.isfinite(ev)]
        a.hist(ev, bins=100, color=C_PTS, alpha=0.8)
        a.axvline(ZF["c_star_zero_fit"][ver], color=C_FIT, lw=1.5, ls="--", label="c*_0 (pure-a3 calibration)")
        rf = CA[f"gain ({ver})"]["c_star"]
        a.axvline(rf, color=C_ALT, lw=1.2, ls=":", label="c* refitted on training rays")
        gzz = ZF[ver]["generic_zero_fit"][f"sigma={R['meta']['sigma_generic_primary']:g}"]
        gz = gzz["all_rays"]
        fb = gzz.get("frac_below_grid", 0.0)
        bound = f" ≥ {V(gz['median_abs']):.2f} ({fb:.1%} below the grid; extrapolated {V(gzz['all_rays_below_grid_extrapolated']['median_abs']):.2f})" if fb > 0 else f" = {V(gz['median_abs']):.3f}"
        a.set_title(f"(b{col + 1}) version ({ver}) at the generic edges; zero-fit median |ln|{bound}", fontsize=9)
        a.set_xlabel("c at s_c on the 10 000 random rays")
        a.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, out / "fig3_zero_fit")
    plt.close(fig)

# ---------------- fig4: p_eff of both versions on the pure-a3 ray ----------------
GA = R["gain_a3"]
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.2))
    ls = np.log10(SIG3)
    for ver, c in (("i", C_PTS), ("ii", C_FIT)):
        cc = np.array(GA[ver]["isotonic"]["c_at_s50"])
        se = np.array(GA[ver]["isotonic"]["c_se"])
        pe = GA[ver]["isotonic"]["p_eff"]
        ax.errorbar(ls, cc - cc[1], yerr=se, fmt="o-", color=c,
                    label=f"version ({ver}): p_eff = " + ", ".join(f"{p['p_eff']:.2f}±{p['se']:.2f}" for p in pe))
    ax.axhline(0, color=C_REF, lw=0.8, ls="--", label="prediction (p = 2): constant")
    ax.set_xlabel(r"$\log_{10}\sigma$")
    ax.set_ylabel(r"$\ln[\sigma^2 g(s_{50})]$ minus its value at σ = 3e-8")
    ax.legend(fontsize=8)
    ax.set_title("pure-a3 ray, isotonic s50 of Gate 2", fontsize=9)
    fig.tight_layout()
    _save(fig, out / "fig4_p_eff_versions")
    plt.close(fig)
print("figures written to", out)
