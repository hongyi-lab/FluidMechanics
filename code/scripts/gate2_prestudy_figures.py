#!/usr/bin/env python
"""Figure for the Gate 2 pre-study page (white background, PNG + PDF).

    python code/scripts/gate2_prestudy_figures.py experiments/20261115_gate2_prestudy_v2
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from committor.plotting import C_ALT, C_FIT, C_PTS, C_REF, STYLE, _save  # noqa: E402

run = Path(sys.argv[1])
supp = Path(sys.argv[2]) if len(sys.argv) > 2 else None
R = json.loads((run / "results.json").read_text())
SU = json.loads((supp / "results.json").read_text()) if supp else None
H = np.load(run / "histograms.npz")
Ed = np.load(run / "edge.npz")
Ry = np.load(run / "rays.npz")
r_A = R["edge"]["r_A"]
E_B = R["histograms"]["E_B"]
s_lo, s_hi = R["sigma"]["sigma_lo"], R["sigma"]["sigma_hi"]
trS = R["lyapunov"]["trace_S1"]

with plt.rc_context(STYLE):
    fig, axx = plt.subplots(2, 4, figsize=(20.0, 8.4))
    ax = axx[:, :3]
    # (a) E histogram
    a = ax[0, 0]
    cE = 0.5 * (H["eE"][1:] + H["eE"][:-1])
    a.bar(cE, H["hE"] / H["hE"].sum() / np.diff(H["eE"]), width=np.diff(H["eE"]), color=C_PTS, alpha=0.75, edgecolor="white")
    a.axvline(E_B, color=C_FIT, lw=1.3, ls="--", label=f"valley: E_B = {E_B}")
    a.set_yscale("log")
    a.set_xlabel(r"$E=\sum_i (a_i-a_i^{lam})^2$")
    a.set_ylabel("occupation density")
    a.set_title("(a) E histogram, 1e4 deterministic trajectories, t ≤ 1e4", fontsize=9)
    a.legend(fontsize=8)
    # (b) a1 histogram
    a = ax[0, 1]
    ca = 0.5 * (H["ea"][1:] + H["ea"][:-1])
    a.bar(ca, H["ha"] / H["ha"].sum() / np.diff(H["ea"]), width=np.diff(H["ea"]), color=C_PTS, alpha=0.75, edgecolor="white")
    a.axvline(R["histograms"]["a1_valley"], color=C_ALT, lw=1.3, ls="--", label=f"valley: a1 = {R['histograms']['a1_valley']:.3f}")
    a.set_yscale("log")
    a.set_xlabel(r"$a_1$")
    a.set_title("(b) a1 histogram (same trajectories)", fontsize=9)
    a.legend(fontsize=8)
    # (c) energies: E_c distribution, r_A, E_ss(sigma)
    a = ax[0, 2]
    Ec = Ed["E_c"]
    bins = np.logspace(-8, 0, 81)
    a.hist(Ec, bins=bins, color=C_PTS, alpha=0.75, edgecolor="white", label=f"E_c, {len(Ec)} random rays")
    a.axvline(R["edge"]["E_c_min"], color=C_REF, lw=1.0, ls=":", label=f"min E_c (refined) {R['edge']['E_c_min']:.2g}")
    a.axvline(r_A, color=C_FIT, lw=1.4, label=f"r_A = {r_A:g}")
    for s, c in ((s_hi, C_ALT), (s_lo, "#55A868")):
        a.axvline(s ** 2 * trS, color=c, lw=1.2, ls="--", label=rf"$E_{{ss}}$, $\sigma$ = {s:g}")
    a.set_xscale("log")
    a.set_xlabel("E")
    a.set_ylabel("rays")
    a.set_title("(c) deterministic edge energy along rays; r_A; noise floor", fontsize=9)
    a.legend(fontsize=7)
    # (d) Lyapunov E_ss vs sigma, MC mean E, spontaneous fraction
    a = ax[1, 0]
    sg = np.array(sorted(float(s) for s in R["lyapunov"]["per_sigma"]))
    a.loglog(sg, sg ** 2 * trS, color=C_PTS, lw=1.4, label=r"$E_{ss}=\sigma^2\,\mathrm{tr}\,S_1$ (Lyapunov)")
    sp = R["spontaneous"]
    xs = np.array([float(k) for k in sp])
    a.loglog(xs, [sp[k]["mean_E_mc"] for k in sp], "o", ms=5, mfc="white", mec=C_REF, label="MC mean E (from laminar)")
    a.axhline(r_A, color=C_FIT, lw=1.2, label=f"r_A = {r_A:g}")
    a.axhline(r_A / 10, color=C_FIT, lw=1.0, ls=":", label="r_A / 10")
    a.set_xlabel(r"$\sigma$")
    a.set_ylabel("E")
    a2 = a.twinx()
    a2.semilogx(xs, [sp[k]["fraction"] for k in sp], "s-", color=C_ALT, ms=4, lw=1.0, label="spontaneous to B (right)")
    a2.set_ylabel("fraction reaching B within T_max", color=C_ALT)
    a2.set_ylim(-0.02, 1.02)
    a2.spines["right"].set_visible(True)
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    a.set_title("(d) noise floor and spontaneous transitions", fontsize=9)
    # (e, f) q profiles along rays for the two sigma
    names = R["rays"]["names"]
    cols = [C_PTS, C_FIT, C_ALT, "#55A868", "#8172B2", C_REF]
    for a, sname in ((ax[1, 1], "sigma_lo"), (ax[1, 2], "sigma_hi")):
        P = R["rays"]["profiles"][sname]
        Sf, q = Ry[f"{sname}_S_f"], Ry[f"{sname}_q_dt"]
        for i, nm in enumerate(names):
            row = P["rays"][nm]
            if not row.get("transition_located", True) or not row["width_rel"]:
                continue                                   # pure-a3 ray: see panel (g)
            x = (Sf[i] - row["s50"]) / row["s50"]
            a.plot(x, q[i], "o-", ms=2.5, lw=0.9, color=cols[i],
                   label=f"{nm}: E50 {row['E50']:.2g}, rel. width {row['width_rel']:.1e}")
        a.set_xlabel(r"$(s - s_{50}) / s_{50}$  along the ray")
        a.set_ylabel(r"$q = n_B / M$  (M = 200)")
        a.set_title(f"({'e' if sname == 'sigma_lo' else 'f'}) q profiles, sigma = {P['sigma']:g}", fontsize=9)
        a.legend(fontsize=6.5, loc="upper left")
        a.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
    if SU is not None:
        a = axx[0, 3]
        for sname, c in (("sigma_lo", "#55A868"), ("sigma_hi", C_ALT)):
            x = SU["S1"][sname]
            sc = SU["S1"]["s_c_det"]
            a.plot(np.asarray(x["coarse_s_over_sc"]), x["coarse_q"], "o", ms=3, color=c, alpha=0.5)
            a.plot(np.asarray(x["final_s"]) / sc, x["final_q"], "-", lw=1.3, color=c,
                   label=rf"$\sigma$ = {x['sigma']:g}: E50 = {x['E50']:.2g}, rel. width {x['width_rel']:.2f}")
        a.axvline(1.0, color=C_REF, ls=":", lw=1.0, label="deterministic s_c (stays in a4..a8 = 0)")
        a.set_xscale("log")
        a.set_xlabel(r"$s / s_c^{det}$ along the pure-$a_3$ ray (linear optimal growth)")
        a.set_ylabel(r"$q$")
        a.set_title("(g) pure-a3 ray: noise breaks the invariant subspace", fontsize=9)
        a.legend(fontsize=7, loc="upper left")
        a = axx[1, 3]
        per = SU["S2"]["per_start"]
        labs = list(per)
        y = np.arange(len(labs))
        f = np.array([per[k]["fraction_decayed_within_follow"] for k in labs])
        lo = np.array([per[k]["cp95"][0] for k in labs])
        hi = np.array([per[k]["cp95"][1] for k in labs])
        a.errorbar(f, y, xerr=[f - lo, hi - f], fmt="o", color=C_PTS, ms=4, capsize=2,
                   label="decayed within 1000 after entering B (95% CI)")
        a.axvline(SU["S2"]["reference_turbulence"], color=C_FIT, ls="--", lw=1.2,
                  label=f"turbulence reference {SU['S2']['reference_turbulence']:.2f}")
        a.set_yticks(y)
        nice = {"optimal_growth": "a3 ray at 0.9 s_c(det)", "a3_ray_S1": "a3 ray at its s50"}
        a.set_yticklabels([f"{nice.get(k, k)} (n={per[k]['n_entries']})" for k in labs], fontsize=7)
        a.set_xlim(0, 0.2)
        a.set_xlabel("fraction")
        a.set_title(rf"(h) is B committed? entries from q~0.5 states, $\sigma$ = {SU['S2']['sigma']:g}", fontsize=9)
        a.legend(fontsize=7, loc="upper right")
    else:
        axx[0, 3].axis("off")
        axx[1, 3].axis("off")
    fig.tight_layout()
    (run / "figures").mkdir(exist_ok=True)
    _save(fig, run / "figures" / "figP_prestudy")
print("saved", run / "figures" / "figP_prestudy.png")
