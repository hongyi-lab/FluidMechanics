#!/usr/bin/env python
"""Figures for the Gate 2 report (white background, PNG + PDF).

    python code/scripts/gate2_figures.py experiments/20261116_gate2_mfe_v1 docs/reports/figs/gate2
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
SC = np.load(run / "scan.npz")
SIG = [3e-9, 3e-8, 3e-7]
CS = {3e-9: C_PTS, 3e-8: C_ALT, 3e-7: C_FIT}

# ---------------- fig1: deterministic edge of the generic rays ----------------
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.2))
    Ec = SC["s_c"] ** 2
    Ec = Ec[np.isfinite(Ec)]
    a = ax[0]
    a.hist(np.log10(Ec), bins=80, color=C_PTS, alpha=0.8, edgecolor="white")
    q = R["scan"]["E_c_quantiles"]
    for k, ls in [("q05", ":"), ("q50", "--"), ("q95", ":")]:
        a.axvline(np.log10(q[k]), color=C_REF, lw=1.0, ls=ls)
    a.set_xlabel(r"$\log_{10} E_c$ (first deterministic transition)")
    a.set_ylabel("rays")
    a.set_title(f"(a) $E_c$ on {len(Ec)} random rays; max/min = {R['scan']['E_c_max_over_min']:.0f}", fontsize=10)

    a = ax[1]
    nm = R["scan"]["non_monotone_by_grid_points"]
    G = np.array([int(g) for g in nm]); F = np.array([nm[g] for g in nm])
    a.plot(G, 100 * F, "o-", color=C_PTS, label="10 000 rays (sub-sampled grid)")
    sr = R["scan_refine"]
    G2 = [sr["base_grid"], 2000, 4000]
    F2 = [sr["base_non_monotone"], sr["2000"]["non_monotone"], sr["4000"]["non_monotone"]]
    a.plot(G2, 100 * np.array(F2), "s-", color=C_FIT, label=f"{sr['n_rays']}-ray subset, refined grid")
    a.set_xscale("log")
    a.set_xlabel("grid points along the ray")
    a.set_ylabel("rays with non-monotone fate (%)")
    a.set_ylim(0, 100)
    a.legend(fontsize=8)
    a.set_title("(b) not converged under refinement", fontsize=10)

    a = ax[2]
    fr = R["offsets"]["fraction_B_per_u"]
    U = np.array([float(u) for u in fr]); P = np.array([fr[u] for u in fr])
    pos, neg = U > 0, U < 0
    a.semilogx(U[pos], 100 * (1 - P[pos]), "o-", color=C_FIT, label=r"$u>0$: fraction A (decay window)")
    a.semilogx(-U[neg], 100 * P[neg], "s-", color=C_PTS, label=r"$u<0$: fraction B")
    a.set_xlabel(r"$|u|$, state at $s_c(1+u)$")
    a.set_ylabel("% of 10 000 offset states")
    a.legend(fontsize=8)
    a.set_title("(c) labels of the offset states", fontsize=10)
    fig.tight_layout()
    _save(fig, out / "fig1_det_edge")
    plt.close(fig)

# ---------------- fig2: noise family ----------------
det = {(d["angle"], d["eps"], d["draw"]): d for d in R["family_det"]["rays"]}
PR = R["family_mc"]["profiles"]
EPS = [0.0, 1e-6, 1e-4, 1e-2]
xe = {0.0: 0, 1e-6: 1, 1e-4: 2, 1e-2: 3}
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for sg in SIG:
        for k, p in enumerate([p for p in PR if p["sigma"] == sg]):
            d = det[(p["angle"], p["eps"], p["draw"])]
            if d["E_c_det"] is None or "s50" not in p:
                continue
            jit = (p["angle"] / 157.5 - 0.5) * 0.25 + {3e-9: -0.08, 3e-8: 0.0, 3e-7: 0.08}[sg]
            mk = "o" if p["resolved"] else "x"
            ax[0].plot(xe[p["eps"]] + jit, p["s50"] / np.sqrt(d["E_c_det"]), mk, color=CS[sg], ms=4, alpha=0.8,
                       label=f"σ = {sg:.0e}" if k == 0 else None)
            ax[1].plot(xe[p["eps"]] + jit, p["width_rel"], mk, color=CS[sg], ms=4, alpha=0.8)
    for a in ax:
        a.set_xticks(range(4)); a.set_xticklabels(["ε = 0", "1e-6", "1e-4", "1e-2"])
    ax[0].axhline(1, color=C_REF, lw=0.8, ls="--")
    ax[0].set_ylabel(r"$s_{50}$ (MC) / $s_c$ (deterministic)")
    ax[0].set_title("(a) where noise moves the edge (angle 90°, ε = 0 has no det. edge: omitted)", fontsize=9)
    ax[0].legend(fontsize=8)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("relative layer width  (s₉₀ − s₁₀)/s₅₀")
    ax[1].set_title("(b) transition-layer width; × = flagged unresolved", fontsize=9)
    fig.tight_layout()
    _save(fig, out / "fig2_noise_family")
    plt.close(fig)

# ---------------- fig3: prediction test on the pure-a3 ray ----------------
gp = R["gain_pred"]["by_estimator"]
g = np.load(run / "gain_a3.npz")
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    a = ax[0]
    for i, sg in enumerate(SIG):
        z = np.load(run / f"fam_profile_s{i}_r0.npz")
        S, k, M = z["S"], z["k"], int(z["M"])
        qq = k / M
        se = np.sqrt(np.maximum(qq * (1 - qq), 1 / M) / M)
        a.errorbar(S * 1e3, qq, yerr=se, fmt="o-", ms=3, lw=0.8, color=CS[sg], label=f"σ = {sg:.0e}")
        a.axvline(gp["isotonic"]["rows"][i]["s50"] * 1e3, color=CS[sg], lw=0.7, ls=":")
    a.set_xlabel(r"amplitude $s$ along $e_3$ ($\times 10^{-3}$)")
    a.set_ylabel("MC committor q")
    a.legend(fontsize=8)
    a.set_title("(a) pure-a3 profiles (final zoom level, M = 400)", fontsize=10)
    a = ax[1]
    lsig = np.log10(SIG)
    for est, c, off in [("isotonic", C_PTS, -0.03), ("lapse_probit", C_ALT, 0.03)]:
        rows = gp[est]["rows"]
        y = np.array([r_["ln_sigma2_g"] for r_ in rows]); e = np.array([r_["ln_sigma2_g_se"] for r_ in rows])
        a.errorbar(lsig + off, y, yerr=e, fmt="o", color=c, label=f"{est}: p_eff = "
                   + ", ".join(f"{p['p_eff']:.2f}±{p['se']:.2f}" for p in gp[est]["p_eff"]))
    a.axhline(gp["isotonic"]["rows"][1]["ln_sigma2_g"], color=C_REF, lw=0.8, ls="--", label="prediction: constant (p = 2)")
    uv = R["gain_pred"]["user_ln_sigma2_g"]
    a.plot(lsig, [uv[k] for k in ["3e-09", "3e-08", "3e-07"]], "^", color=C_REF, mfc="none", label="user's s50 values")
    a.set_xlabel(r"$\log_{10}\sigma$")
    a.set_ylabel(r"$\ln[\sigma^2 g(s_{50})]$")
    a.legend(fontsize=7.5)
    a.set_title(r"(b) $\sigma^2 g(s_{50})$ rises by ~0.4 per decade", fontsize=10)
    fig.tight_layout()
    _save(fig, out / "fig3_noise_gain_prediction")
    plt.close(fig)
print("figures written to", out)
