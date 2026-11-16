#!/usr/bin/env python
"""Figures for the Gate 3c report (white background, PNG + PDF).

    python code/scripts/gate3c_figures.py experiments/<gate3c run> docs/reports/figs/gate3c
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from committor.plotting import C_ALT, C_FIT, C_PTS, C_REF, STYLE, _save  # noqa: E402

run, out = Path(sys.argv[1]), Path(sys.argv[2])
REPO = Path(__file__).resolve().parents[2]
out.mkdir(parents=True, exist_ok=True)
R = json.loads((run / "results.json").read_text())
YL = ["a4", "a5", "a6", "a7", "a8"]
TH = R["theory"]
Z = np.load(run / "theory.npz")


def heat(a, W, title):
    v = float(np.max(np.abs(W))) or 1.0
    im = a.imshow(W, cmap="coolwarm", vmin=-v, vmax=v)
    for i in range(5):
        for j in range(5):
            if abs(W[i, j]) >= 0.005 * v:
                a.text(j, i, f"{W[i, j]:.3g}", ha="center", va="center", fontsize=7)
    a.set_xticks(range(5), YL)
    a.set_yticks(range(5), YL)
    a.set_title(title, fontsize=9)
    return im


# ---------------- fig1: item 0, W(T) against M1's y-y form ----------------
with plt.rc_context(STYLE):
    fig = plt.figure(figsize=(15, 8.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
    a = fig.add_subplot(gs[0, :2])
    Ts, cs = Z["Ts"][1:], Z["cos_T"][1:]
    a.semilogx(Ts, cs, color=C_PTS, label="cos(W(T), W_M1)")
    a.axhline(TH["M1"]["cos_W_inf"], color=C_REF, ls=":", lw=1, label=f"T = inf: {TH['M1']['cos_W_inf']:.4f}")
    a.axhline(TH["M1"]["cos_F"], color=C_ALT, ls="--", lw=1, label=f"instantaneous F: {TH['M1']['cos_F']:.3f}")
    a.axvline(45, color=C_FIT, lw=0.8, ls="--")
    a.plot([45], [TH["M1"]["cos_W45"]], "o", color=C_FIT, label=f"T = 45: {TH['M1']['cos_W45']:.4f}")
    a.set_xlabel("T")
    a.set_ylabel("Frobenius cosine")
    a.set_ylim(0, 1.02)
    a.legend(fontsize=8, loc="lower right")
    a.set_title("(a) shape: cosine between W(T) and M1's y-y form (Q_yy / b3)", fontsize=9)
    b = fig.add_subplot(gs[0, 2:])
    b.loglog(Z["Ts"][1:], Z["normW_T"][1:], color=C_PTS, label="|W(T)|_F")
    b.axhline(TH["M1"]["W_M1_frobenius"], color=C_FIT, ls="--", lw=1, label=f"|W_M1|_F = {TH['M1']['W_M1_frobenius']:.1f}")
    b.axhline(TH["W_inf_frobenius"], color=C_REF, ls=":", lw=1, label=f"|W_inf|_F = {TH['W_inf_frobenius']:.1f}")
    if TH["M1"]["T_where_norm_W_T_equals_norm_W_M1"]:
        b.axvline(TH["M1"]["T_where_norm_W_T_equals_norm_W_M1"], color=C_FIT, lw=0.8)
    if "item1" in R:
        b.axvline(R["item1"]["c"]["T"], color=C_ALT, lw=1.2, ls="-.", label=f"item 1 (c) fitted T = {R['item1']['c']['T']:.1f}")
    b.set_xlabel("T")
    b.set_ylabel("Frobenius norm")
    b.legend(fontsize=8, loc="lower right")
    b.set_title(f"(b) magnitude: |W(T)| = |W_M1| at T = {TH['M1']['T_where_norm_W_T_equals_norm_W_M1']:.1f}", fontsize=9)
    for i, (W, t) in enumerate([(Z["F"], "(c) F (instantaneous, da3/dt)"), (Z["W45"], "(d) W(45)"),
                                (Z["W_inf"], "(e) W_inf (Lyapunov)"), (Z["W_M1"], "(f) M1: Q_yy / b3 (Gate 3b)")]):
        ax = fig.add_subplot(gs[1, i])
        heat(ax, W, t)
    fig.tight_layout()
    _save(fig, out / "fig1_theory_vs_M1")
    plt.close(fig)

# ---------------- fig2: item 1 ----------------
I1 = R["item1"]
C1 = np.load(run / "item1_curves.npz")
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0), gridspec_kw={"width_ratios": [1.15, 1, 1.25]})
    a = ax[0]
    for r_ in C1["rnd_meds"]:
        a.plot(C1["b_grid"], r_, color="0.75", lw=0.5, zorder=1)
    a.plot([], [], color="0.75", lw=0.8, label="100 random symmetric W")
    a.plot(C1["F_grid"], C1["F_meds"], color=C_ALT, lw=1.3, label="F (scaled to |W_inf|)")
    a.plot(C1["b_grid"], C1["b_meds"], color=C_PTS, lw=1.6, label="W_inf")
    a.axvline(I1["b"]["kappa"], color=C_PTS, ls="--", lw=0.8)
    a.set_xscale("symlog", linthresh=1e-3)
    a.set_xticks([-1e3, -1, -1e-3, 0, 1e-3, 1, 1e3], ["-1e3", "-1", "-1e-3", "0", "1e-3", "1", "1e3"])
    a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a.set_xlabel("kappa (W = kappa x matrix)")
    a.set_ylabel("train median |ln| (analytic search)")
    a.set_ylim(0, max(0.5, float(np.nanmax(C1["b_meds"][np.isfinite(C1["b_meds"])])) * 1.05))
    a.legend(fontsize=8)
    a.set_title(f"(a) kappa search: W_inf best kappa = {I1['b']['kappa']:.3f}", fontsize=9)
    b = ax[1]
    tg, tm = C1["c_grid"], C1["c_meds"]
    b.semilogx(tg[1:], tm[1:], color=C_PTS, lw=1.5, label="W(T)")
    b.axhline(tm[0], color=C_REF, ls=":", lw=1, label=f"T = 0 (|a3|): {tm[0]:.3f}")
    b.axvline(I1["c"]["T"], color=C_PTS, ls="--", lw=0.8)
    b.axhline(I1["b"]["train_median_analytic"], color=C_FIT, ls="--", lw=0.8, label=f"(b) kappa W_inf: {I1['b']['train_median_analytic']:.4f}")
    b.set_xlabel("T")
    b.set_ylabel("train median |ln| (analytic search)")
    b.legend(fontsize=8)
    b.set_title(f"(b) T search: best T = {I1['c']['T']:.1f}", fontsize=9)
    c = ax[2]
    rows = [("(a) W_inf", I1["a"]["test"]), ("(b) kappa W_inf", I1["b"]["test"]), ("(c) W(T)", I1["c"]["test"]),
            ("(d) kappa F", I1["d_F"]["test"])]
    refs = I1["references"]
    for k in ["Gate 3b: M1", "Gate 3b: M2", "Gate 3b: S1 (strict single quadric)", "Gate 3: |a3|", "Gate 3: gain (i) (refit c*)",
              "Gate 3: MLP (9D, polar input) = ceiling"]:
        if k in refs:
            rows.append((k.replace("Gate 3b: ", "3b ").replace("Gate 3: ", "3 "), refs[k]))
    for i, (lab, t) in enumerate(rows):
        col = C_PTS if i < 3 else (C_ALT if i == 3 else C_REF)
        ci = t.get("median_abs_ci", [t["median_abs"], t["median_abs"]])
        c.errorbar([t["median_abs"]], [i], xerr=[[t["median_abs"] - ci[0]], [ci[1] - t["median_abs"]]], fmt="o", color=col, ms=5, capsize=2)
    rm = np.array([r_["test_median_abs"] for r_ in I1["d_random"]["controls"]])
    jit = np.random.default_rng(0).uniform(-0.25, 0.25, len(rm))
    c.scatter(rm, len(rows) + jit, s=6, color="0.6")
    c.set_yticks(range(len(rows) + 1), [r_[0] for r_ in rows] + ["(d) 100 random W"], fontsize=8)
    c.axvline(0.05, color=C_FIT, ls="--", lw=0.8)
    c.axvline(0.10, color=C_FIT, ls=":", lw=0.8)
    c.set_xscale("log")
    c.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    c.set_xticks([0.02, 0.03, 0.05, 0.1, 0.2], ["0.02", "0.03", "0.05", "0.1", "0.2"])
    c.set_xlabel("test median |ln(s_pred/s_c)| (95% CI)")
    c.invert_yaxis()
    c.set_title(f"(c) test rays; criterion lines 0.05 / 0.10\n{I1['criterion']['verdict']}", fontsize=8)
    fig.tight_layout()
    _save(fig, out / "fig2_theory_quantity")
    plt.close(fig)

# ---------------- fig3: item 2 ----------------
I2 = R["item2"]
E2 = np.load(run / "item2_errors.npz")
E1 = np.load(run / "item1_test_errors.npz")
split = E2["split"]
te = split == 2
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.4))
    bins = np.linspace(-3, 3, 121)

    def hist(a, e, lab, col, **kw):
        e = np.asarray(e)
        a.hist(np.clip(e[np.isfinite(e)], -3, 3), bins=bins, histtype="step", color=col, label=lab, **kw)

    a = ax[0, 0]
    hist(a, E2["e_fwd"][te], f"merged, forward c* = {I2['forward']['c_star']:.2f}: median |ln| {I2['forward']['test']['median_abs']:.3f}", C_PTS, lw=1.5)
    g3T = np.load(REPO / R["meta"]["data"]["gate3_run"] / "test_errors.npz")
    names = list(g3T["names"])
    pos = {int(r): i for i, r in enumerate(g3T["test_rays"])}
    te_ids = E2["rays"][te]
    if all(int(r) in pos for r in te_ids):
        g_ref = g3T["E"][names.index("gain (i)")][[pos[int(r)] for r in te_ids]]
        hist(a, g_ref, f"Gate 3 gain (i), refit c*: {np.median(np.abs(g_ref)):.3f}", C_REF, lw=1.0)
    hist(a, E1["E"][list(E1["names"]).index("b")], f"item 1 (b) |u_th| = c*: {I1['b']['test']['median_abs']:.3f}", C_ALT, lw=1.0, ls="--")
    a.set_xlabel("ln(s_pred / s_c), clipped to [-3, 3]")
    a.set_ylabel("test rays")
    a.legend(fontsize=7)
    a.set_title("(a) forward: c* fitted on random train rays", fontsize=9)
    a = ax[0, 1]
    hist(a, E2["e_rev"][te], f"merged, c*_0 = {I2['reverse']['c_star_0']:.3f}: median |ln| {I2['reverse']['test']['median_abs']:.3f}", C_PTS, lw=1.5)
    hist(a, E2["zf_g3"][te], f"Gate 3 gain (i), c*_0: {np.median(np.abs(E2['zf_g3'][te])):.3f}", C_REF, lw=1.0)
    a.set_xlabel("ln(s_pred / s_c), clipped to [-3, 3]")
    a.set_ylabel("test rays")
    a.legend(fontsize=7)
    a.set_title("(b) reverse: c*_0 from the pure-a3 ray (Gate 3), no fit", fontsize=9)
    a = ax[0, 2]
    use = E2["fam_use"]
    eps_l = sorted(set(E2["fam_eps"][use].tolist()))
    xpos = {e: i for i, e in enumerate(eps_l)}
    for lab, e, col, off in (("merged, forward c*", E2["e_fam_fwd"], C_PTS, -0.12), ("merged, c*_0 (descriptive)", E2["e_fam_rev"], C_ALT, 0.12)):
        xs = np.array([xpos[v] for v in E2["fam_eps"][use]]) + off
        a.scatter(xs, np.clip(e[use], -1.5, 1.5), s=12, color=col, label=lab, alpha=0.8)
    fr = I2["family_references_gate3b"]
    for lab, d, col in (("Gate 3b M1 (signed median)", fr["M1"], C_REF), ("Gate 3 gain (i)@0, refit c* (signed median)", fr["gain (i)@0 (Gate 3 refit c*)"], "0.5")):
        a.plot([xpos[e] for e in eps_l], [d[f"{e:g}"] for e in eps_l], "s--", color=col, ms=4, lw=1, label=lab)
    a.axhline(0, color="k", lw=0.5)
    a.set_xticks(range(len(eps_l)), [f"{e:g}" for e in eps_l])
    a.set_xlabel("eps (noise family, deterministic s_c)")
    a.set_ylabel("signed ln error (clipped to +-1.5)")
    tr = I2["family_forward"]["eps_trend"]["median"]
    a.set_title(f"(c) family eps > 0; forward eps trend (1e-6 minus 1e-2) = {tr:+.3f}", fontsize=9)
    a.legend(fontsize=7)
    a = ax[1, 0]
    V = I2["vs_a3_fraction"]
    for key, lab, col, ls in (("forward_test", "merged forward (test)", C_PTS, "-"), ("reverse_all", "merged reverse (all rays)", C_ALT, "-"),
                              ("gate3_gain_i_refit_test", "Gate 3 gain (i) refit (test)", C_REF, "--"),
                              ("gate3_gain_i_zero_fit_all", "Gate 3 gain (i) c*_0 (all rays)", "0.5", "--")):
        if key in V:
            a.semilogx([b_["x_median"] for b_ in V[key]], [b_["signed_median"] for b_ in V[key]], ls, color=col, marker="o", ms=3, label=lab)
    a.axhline(0, color="k", lw=0.5)
    a.set_xlabel("a3 energy fraction of the ray direction")
    a.set_ylabel("binned signed median ln error")
    a.legend(fontsize=7)
    a.set_title("(d) error vs a3 share (20 equal-count bins)", fontsize=9)
    a = ax[1, 1]
    P = E2["left_peak"]
    hist(a, E2["e_fwd"][P], f"merged forward: {np.median(np.abs(E2['e_fwd'][P])):.3f}" if P.any() else "merged forward", C_PTS, lw=1.5)
    hist(a, E2["e_rev"][P], f"merged reverse: {np.median(np.abs(E2['e_rev'][P])):.3f}" if P.any() else "merged reverse", C_ALT, lw=1.2)
    hist(a, E2["zf_g3"][P], f"Gate 3 gain (i) c*_0: {np.median(np.abs(E2['zf_g3'][P])):.3f}" if P.any() else "Gate 3", C_REF, lw=1.0)
    a.set_xlabel("ln(s_pred / s_c), clipped to [-3, 3]")
    a.set_ylabel("rays")
    a.legend(fontsize=7)
    a.set_title(f"(e) the {int(P.sum())} left-peak rays of Gate 3 fig3 (b1) (median |ln|)", fontsize=9)
    a = ax[1, 2]
    ec = E2["edge_c"]
    a.hist(ec[np.isfinite(ec)], bins=100, color=C_PTS, alpha=0.7, label="merged c at the edge (all rays)")
    a.axvline(I2["reverse"]["c_star_0"], color=C_ALT, lw=1.2, label=f"c*_0 = {I2['reverse']['c_star_0']:.2f}")
    a.axvline(I2["forward"]["c_star"], color=C_FIT, lw=1.2, ls="--", label=f"forward c* = {I2['forward']['c_star']:.2f}")
    a.set_xlabel("ln max_t tr P at s_c (sigma = 3e-8)")
    a.set_ylabel("rays")
    a.legend(fontsize=7)
    a.set_title("(f) merged coordinate at the random-ray edges", fontsize=9)
    fig.suptitle(f"Item 2: {I2['criterion']['verdict']}", fontsize=9)
    fig.tight_layout()
    _save(fig, out / "fig3_seed_x_gain")
    plt.close(fig)
print("figures written to", out)
