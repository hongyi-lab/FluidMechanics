#!/usr/bin/env python
"""Figures for the Gate 3b report (white background, PNG + PDF).

    python code/scripts/gate3b_figures.py experiments/<gate3b run> docs/reports/figs/gate3b
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import SymLogNorm  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

from committor.plotting import C_ALT, C_FIT, C_PTS, C_REF, STYLE, _save  # noqa: E402

run, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
R = json.loads((run / "results.json").read_text())
MODES = [f"a{i}" for i in range(1, 10)]


def V(x):
    return np.inf if x is None else x


# ---------------- fig1: mechanism planes (item 1, descriptive) ----------------
M = np.load(run / "mech_edge.npz")
q = {"|a3|": M["abs_a3"], "|a2|": M["abs_a2"], "|y|": M["abs_y"], "1-a1": M["one_minus_a1"], "a9": M["a9"]}
planes = [("|a3|", "|y|"), ("|a2|", "|y|"), ("|a3|", "|a2|")]
D = R["describe"]["planes"]
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 9.8))
    for row, cn in enumerate(("1-a1", "a9")):
        cv = q[cn]
        lt = float(np.quantile(np.abs(cv), 0.05))
        vmax = float(np.quantile(np.abs(cv), 0.99))
        for col, (p, o) in enumerate(planes):
            a = ax[row, col]
            order = np.argsort(np.abs(cv))
            sc = a.scatter(q[p][order], q[o][order], c=cv[order], s=2.0, cmap="coolwarm",
                           norm=SymLogNorm(linthresh=lt, vmin=-vmax, vmax=vmax, base=10), rasterized=True)
            a.set_xscale("log")
            a.set_yscale("log")
            a.xaxis.set_minor_formatter(NullFormatter())
            a.yaxis.set_minor_formatter(NullFormatter())
            a.set_xlabel(f"{p} at s_c")
            a.set_ylabel(f"{o} at s_c")
            st = D[f"({p}, {o})"]
            s1, s2 = st[f"ln {o} given ln {p}"], st[f"ln {p} given ln {o}"]
            r1 = s1["spearman_resid_with"]
            a.set_title(f"({'abc'[col]}{row + 1}) colour: {cn}\nIQR ratio (within bin / overall): ln {o} | ln {p} "
                        f"{s1['ratio_within_to_overall']:.2f},\nln {p} | ln {o} {s2['ratio_within_to_overall']:.2f}", fontsize=8)
            cb = fig.colorbar(sc, ax=a, pad=0.01)
            cb.set_label(f"{cn} at s_c (symlog)", fontsize=8)
            # binned medians (descriptive)
            a.plot(np.exp(s1["bin_centres"]), np.exp(s1["bin_medians"]), "k-", lw=1.0, label=f"bin median of {o}")
            if row == 0 and col == 0:
                a.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"States at the edge of the {R['describe']['n_rays']} random rays (description only, no fit)", fontsize=10)
    fig.tight_layout()
    _save(fig, out / "fig1_mechanism_planes")
    plt.close(fig)

# ---------------- fig2: complexity ladder ----------------
EV = R["evaluate"]
names = list(EV["models"])
short = {"S1: single quadric threshold (k = 1 strict)": "S1\n(strict k=1)",
         "quadratic logistic (9D, convex BCE; diagnostic)": "convex\nquad. log.", "M1": "M1", "M2": "M2", "M3": "M3", "M4": "M4",
         "phys2: g(a3^2, |y|^2)": "phys k=2\ng(a3², |y|²)", "phys2 additive (theta search)": "additive\n(θ search)",
         "phys2 additive (BCE, signs free)": "additive\n(BCE)"}
crit = R["meta"]["criterion"]
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8))
    x = np.arange(len(names))
    for a, key, title in ((ax[0], "median_abs", "median |ln(s_pred / s_c)|, test rays"), (ax[1], "q90_abs", "90% quantile of |ln(s_pred / s_c)|")):
        for i, n in enumerate(names):
            t = EV["models"][n]["test"]
            v = V(t[key])
            lo, hi = (V(t[key + "_ci"][0]), V(t[key + "_ci"][1]))
            col = C_FIT if (n.startswith("M") or n.startswith("S1")) else (C_ALT if n.startswith("phys") else C_PTS)
            if np.isfinite(v):
                a.errorbar(i, v, yerr=[[v - lo], [hi - v if np.isfinite(hi) else 0]], fmt="o", color=col, ms=6, capsize=3)
            if key == "median_abs":
                fam = n if n.startswith("M") else ("phys2" if n.startswith("phys2:") else ("S1" if n.startswith("S1") else None))
                if fam is not None:
                    for tag, rr in EV["restarts"].items():
                        if tag.startswith(fam + "_"):
                            a.plot(i + 0.18, rr["test_median_abs"], ".", color="0.55", ms=4)
        refs = EV["references_gate3"]
        for rn, ls in (("E", ":"), ("|a3|", "--"), ("triple (1-a1, a2, a3)", "-."), ("MLP (9D, polar input)", "-")):
            if rn in refs:
                a.axhline(V(refs[rn][key]), color=C_REF, lw=0.8, ls=ls, label=f"Gate 3 {rn}" + (" (ceiling)" if "MLP" in rn else ""))
        if key == "median_abs":
            a.axhline(crit["pass_median"], color=C_FIT, lw=1.2, ls="--", alpha=0.6, label=f"criterion {crit['pass_median']}")
            a.axhline(crit["fail_median_at_k4"], color=C_ALT, lw=1.2, ls="--", alpha=0.6, label=f"k = 4 fail line {crit['fail_median_at_k4']}")
        a.set_yscale("log")
        a.set_xticks(x)
        a.set_xticklabels([short.get(n, n) for n in names], fontsize=8)
        a.set_ylabel(title)
        a.legend(fontsize=7, loc="upper right" if key == "median_abs" else "lower left")
    ax[0].set_title(f"(a) test rays (n = {EV['models']['M1']['test']['n']}); grey dots: every restart (diagnostic)\n"
                    f"{EV['criterion']['verdict']}", fontsize=8.5)
    ax[1].set_title("(b) tail of the error", fontsize=9)
    fig.tight_layout()
    _save(fig, out / "fig2_ladder")
    plt.close(fig)

# ---------------- fig3: k = 2 planes ----------------
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.6))
    P = np.load(run / "plane_M2.npz")
    a = ax[0]
    b = P["boundary"]
    a.plot(b[:, 0], b[:, 1], ".", ms=2.5, color=C_PTS, alpha=0.6, label="test boundary states (x at s_c)", rasterized=True)
    Z1, Z2 = np.meshgrid(P["g1"], P["g2"], indexing="ij")
    a.contour(Z1, Z2, P["C"], levels=[float(P["c_star"])], colors=[C_FIT], linewidths=1.6, linestyles="solid")
    a.plot([], [], "-", color=C_FIT, label="learned critical curve g = c*")
    a.set_xlabel(r"asinh$(u_1/\tau_1)$")
    a.set_ylabel(r"asinh$(u_2/\tau_2)$")
    a.set_title(f"(a) M2: learned u1, u2 (side {int(P['side']):+d}: B where side·g ≥ side·c*)\n"
                f"test median |ln| = {EV['models']['M2']['test']['median_abs']:.3f}", fontsize=9)
    a.legend(fontsize=7)
    P = np.load(run / "plane_phys2.npz")
    a = ax[1]
    b = P["boundary"]
    a.plot(b[:, 0] / np.log(10), b[:, 1] / np.log(10), ".", ms=2.5, color=C_PTS, alpha=0.6, label="test boundary states", rasterized=True)
    Z1, Z2 = np.meshgrid(P["g1"], P["g2"], indexing="ij")
    a.contour(Z1 / np.log(10), Z2 / np.log(10), P["C"], levels=[float(P["c_star"])], colors=[C_FIT], linewidths=1.6, linestyles="solid")
    a.plot([], [], "-", color=C_FIT, label="g free: g = c*")
    u1 = np.exp(P["g1"])
    for w, cst, c0, col, lab in ((P["add_theta_w"], float(P["add_theta_cstar"]), 0.0, C_ALT, "additive (θ search)"),
                                 (P["add_bce_w"], float(P["add_bce_cstar"]), float(P["add_bce_const"]), C_REF, "additive (BCE)")):
        u2 = (cst - c0 - w[0] * u1) / w[1] if w[1] != 0 else np.full_like(u1, np.nan)
        ok = u2 > 0
        a.plot(np.log10(u1[ok]), np.log10(u2[ok]), "--", color=col, lw=1.3, label=lab)
    a.set_xlim(P["g1"][0] / np.log(10), P["g1"][-1] / np.log(10))
    a.set_ylim(P["g2"][0] / np.log(10), P["g2"][-1] / np.log(10))
    a.set_xlabel(r"$\log_{10} a_3^2$ at s_c")
    a.set_ylabel(r"$\log_{10} |y|^2$ at s_c")
    e1, e2, e3 = (EV["models"][k]["test"]["median_abs"] for k in ("phys2: g(a3^2, |y|^2)", "phys2 additive (theta search)", "phys2 additive (BCE, signs free)"))
    a.set_title(f"(b) physics k = 2: u1 = a3², u2 = |y|²; test median |ln|: g free {e1:.3f}, additive θ {e2:.3f}, BCE {e3:.3f}", fontsize=9)
    a.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, out / "fig3_k2_planes")
    plt.close(fig)

# ---------------- fig4: quadratic forms (canonical basis) ----------------
S = R["structure"]
mods = ["S1: single quadric threshold (k = 1 strict)"] + [k for k in ("M1", "M2", "M3", "M4") if k in S]
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(1, len(mods), figsize=(4.4 * len(mods), 6.6), gridspec_kw={"wspace": 1.05})
    for a, mn in zip(ax, mods):
        rec = S[mn]
        forms = rec.get("canonical_forms", rec["raw"])
        share = rec.get("canonical", {}).get("J_share", [1.0])
        rows, labels = [], []
        for j, f in enumerate(forms):
            for i, axv in enumerate(f["axes_top"][:3]):
                rows.append([axv["composition"][m] for m in MODES])
                labels.append(f"u{j + 1} ({share[j]:.2f})  λ{i + 1} {axv['eigenvalue_rel']:+.2f}")
            rows.append([f["b_composition"][m] for m in MODES])
            labels.append(f"u{j + 1}  b (lin. share {f['median_linear_share_at_boundary']:.2f})")
        im = a.imshow(np.array(rows), cmap="Greys", vmin=0, vmax=1, aspect="auto")
        a.set_xticks(range(9))
        a.set_xticklabels(MODES, fontsize=7)
        a.set_yticks(range(len(rows)))
        a.set_yticklabels(labels, fontsize=6.5)
        a.set_title(short.get(mn, mn).replace("\n", " "), fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.88)
    fig.colorbar(im, ax=ax, shrink=0.6, label="squared component of the eigenvector")
    fig.suptitle("Principal axes of the quadratic forms (canonical basis; in brackets: share of grad-c energy; λ relative to the largest |λ|);\n"
                 "row 'b': modal composition of the linear term, with its median share |b.x| / (|b.x| + |x^T Q x|) at the train boundary states", fontsize=9)
    _save(fig, out / "fig4_quadratic_forms")
    plt.close(fig)

# ---------------- fig5: noise family, frozen models ----------------
F = R["family"]
fn = list(F["models"]) + [f"Gate 3: {k}" for k in F["references_gate3"]]
get = lambda n: F["references_gate3"][n[8:]] if n.startswith("Gate 3: ") else F["models"][n]  # noqa: E731
EPS = sorted({p["eps"] for p in F["models"]["M1"]["per_ray"]})
EC = {EPS[0]: C_PTS, EPS[1]: C_ALT, EPS[-1]: C_FIT} if len(EPS) >= 3 else {e: c for e, c in zip(EPS, (C_PTS, C_ALT, C_FIT))}
with plt.rc_context(STYLE):
    fig, a = plt.subplots(1, 1, figsize=(14.0, 5.0))
    for i, n in enumerate(fn):
        pr = get(n)["per_ray"]
        for k, e in enumerate(EPS):
            ee = np.array([V(p["ln_err"]) for p in pr if p["eps"] == e])
            xx = i + (k - 1) * 0.22 + np.linspace(-0.06, 0.06, len(ee))
            fin = np.isfinite(ee)
            inr = fin & (np.abs(ee) <= 1.5)
            a.plot(xx[inr], ee[inr], "o", ms=3, color=EC[e], alpha=0.8, label=f"ε = {e:g}" if i == 0 else None)
            for msk, yv, mk in ((fin & (ee > 1.5), 1.55, "^"), (fin & (ee < -1.5), -1.55, "v")):
                if msk.any():
                    a.plot(xx[msk], np.full(msk.sum(), yv), mk, ms=5, color=EC[e])
            if (~fin).any():
                a.plot(xx[~fin], np.full((~fin).sum(), 1.62), "x", ms=5, color=EC[e])
    a.axhline(0, color=C_REF, lw=0.8)
    a.set_xticks(range(len(fn)))
    a.set_xticklabels([short.get(n, n).replace("\n", " ") if not n.startswith("Gate 3") else n.replace("Gate 3: ", "G3: ") for n in fn],
                      fontsize=7, rotation=25, ha="right")
    a.set_ylim(-1.7, 1.7)
    a.set_ylabel("ln(s_pred / s_c,det)")
    a.set_title("Noise family, ε > 0 rays (48), deterministic s_c; frozen models and c* from the random rays; "
                "triangles: |ln| > 1.5 drawn at the edge; x: no crossing", fontsize=9)
    a.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    _save(fig, out / "fig5_family")
    plt.close(fig)

# ---------------- fig6: gain (i) failure ----------------
GF = np.load(run / "gain_failure.npz")
GR = R["gain_failure"]
cut = list(GR["left_peak"])[0]
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(2, 3, figsize=(19.0, 9.0))
    panels = ((ax[0, 0], "fy", GF["zero_fit_err"], np.arange(len(GF["fy"])), "(a) zero-fit c*_0, all rays", "zero_fit_vs_fy_all_rays", "initial y energy fraction |y0|² / |x0|²"),
              (ax[0, 1], "fy", GF["refit_err_test"], GF["test_rows"], "(b) refitted c*, test rays", "refit_vs_fy_test_rays", "initial y energy fraction |y0|² / |x0|²"),
              (ax[1, 0], "f3", GF["zero_fit_err"], np.arange(len(GF["fy"])), "(d) zero-fit c*_0, all rays", "zero_fit_vs_f3_all_rays", "a3 energy fraction of the ray direction"),
              (ax[1, 1], "f3", GF["refit_err_test"], GF["test_rows"], "(e) refitted c*, test rays", "refit_vs_f3_test_rays", "a3 energy fraction of the ray direction"))
    for a, xk, err, rows, title, bk, xl in panels:
        if len(err) == 0:
            continue
        fy = GF[xk][rows]
        e = np.asarray(err)
        fin = np.isfinite(e)
        a.plot(fy[fin], np.clip(e[fin], -2, 2), ".", ms=2, color=C_PTS, alpha=0.4, rasterized=True)
        if (~fin).any():
            a.plot(fy[~fin], np.full((~fin).sum(), 2.08), "x", ms=4, color=C_FIT, label="no crossing")
        bb = GR[bk]["bins"]
        a.plot([b_["x_median"] for b_ in bb], [b_["signed_median"] for b_ in bb], "k-", lw=1.3, label="bin median (signed)")
        a.plot([b_["x_median"] for b_ in bb], [b_["median_abs"] for b_ in bb], "k--", lw=1.0, label="bin median |ln|")
        a.axhline(0, color=C_REF, lw=0.6)
        a.set_xscale("log")
        a.set_xlabel(xl)
        a.set_ylabel("gain (i): ln(s_pred / s_c)  (clipped at ±2)")
        a.set_title(f"{title}; Spearman with |ln| {GR[bk]['spearman_abs_inf_as_large']:+.2f}", fontsize=9)
        a.legend(fontsize=7)
    a = ax[0, 2]
    ce = GF["c_edge"]
    f3 = GF["f3"]
    fin = np.isfinite(ce)
    bins = np.linspace(np.min(ce[fin]), np.quantile(ce[fin], 0.995), 100)
    a.hist([ce[fin & (f3 < 1e-2)], ce[fin & (f3 >= 1e-2)]], bins=bins, stacked=True, color=[C_FIT, C_PTS],
           label=["a3 energy fraction < 0.01", "≥ 0.01"])
    a.axvline(float(cut.split("<")[1]), color=C_REF, ls=":", lw=1.0, label=f"left-peak cut ({cut})")
    a.set_xlabel("gain (i) c at s_c (σ = 3e-8), random rays")
    a.set_ylabel("rays")
    lp = GR["left_peak"][cut]
    a.set_title(f"(c) Gate 3 fig3 (b1) histogram split by a3 content\nleft peak: {lp['n']} rays, "
                f"{lp['frac_peak_with_f3_below_0.01']:.1%} with a3 fraction < 0.01, {lp['frac_peak_with_f3_below_0.001']:.1%} < 0.001", fontsize=9)
    a.legend(fontsize=7)
    a = ax[1, 2]
    w = 0.38
    xx = np.arange(9)
    a.bar(xx - w / 2, [lp["mean_modal_fraction_all"][m] for m in MODES], w, color=C_PTS, label="all rays")
    a.bar(xx + w / 2, [lp["mean_modal_fraction_peak"][m] for m in MODES], w, color=C_FIT, label=f"left peak ({cut})")
    a.set_xticks(xx)
    a.set_xticklabels(MODES)
    a.set_ylabel("mean energy fraction of the ray direction")
    a.set_title(f"(f) modal composition; |c(σ=3e-7) − c(σ=0)| median in the peak: "
                f"{lp['median_abs_c_sigma_3e-7_minus_c_sigma_0_peak']:.1e}", fontsize=9)
    a.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, out / "fig6_gain_failure")
    plt.close(fig)
print("figures written to", out)
