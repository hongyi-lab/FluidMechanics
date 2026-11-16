#!/usr/bin/env python
"""Figures for the Gate 3d report (white background, PNG + PDF).

    python code/scripts/gate3d_figures.py experiments/<gate3d run> docs/reports/figs/gate3d
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
REPO = Path(__file__).resolve().parents[2]
out.mkdir(parents=True, exist_ok=True)
R = json.loads((run / "results.json").read_text())
cfg = json.loads(json.dumps(__import__("yaml").safe_load((run / "config_used.yaml").read_text())))
g3c = REPO / cfg["data"]["gate3c_run"]
YL = ["a4", "a5", "a6", "a7", "a8"]
TH = R["theory"]
Z = np.load(run / "theory.npz")
C_GRAY = "0.55"


def heat(a, W, title):
    v = float(np.max(np.abs(W))) or 1.0
    a.imshow(W, cmap="coolwarm", vmin=-v, vmax=v)
    for i in range(5):
        for j in range(5):
            if abs(W[i, j]) >= 0.005 * v:
                a.text(j, i, f"{W[i, j]:.3g}", ha="center", va="center", fontsize=7)
    a.set_xticks(range(5), YL)
    a.set_yticks(range(5), YL)
    a.set_title(title, fontsize=9)


def fig1():
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(16, 8.4))
        gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0])
        a = fig.add_subplot(gs[0, 0:3])
        ts, L = TH["t_star"], Z["L_tau"]
        a.plot(Z["tau"], -L, color=C_PTS, lw=1.6, label=r"$-L(\tau)$  (a2 response to a unit a3)")
        a.plot(Z["tau"], np.exp(-TH["d2"] * Z["tau"]) * abs(TH["L_t_star"]), color=C_ALT, lw=1.0, ls="--",
               label=r"$e^{-d_2\tau}\,|L(t^*)|$ (scaled, for reference)")
        a.axvline(ts, color=C_FIT, ls="--", lw=0.9)
        a.text(ts * 1.03, 0.1 * abs(TH["L_t_star"]), f"t* = {ts:.1f}\n|L(t*)| = {abs(TH['L_t_star']):.1f}", fontsize=8, color=C_FIT)
        a.set_xlim(0, 600)
        a.set_xlabel(r"$\tau$")
        a.legend(fontsize=8, frameon=False)
        a.set_title(f"(a) lift-up kernel: d2 = {TH['d2']:.5f}, d3 = {TH['d3']:.5f}, J23 = {TH['J23']:.4f}", fontsize=9)
        b = fig.add_subplot(gs[0, 3:6])
        rows = [r for r in TH["ode_check"]]
        for kind, col in (("no a1, a9", C_PTS), ("all 9 modes", C_ALT)):
            for key, ls, mk in (("resid_S_over_eps2", "-", "o"), ("resid_Shat_over_eps2", ":", "s")):
                eps = sorted(set(r["eps"] for r in rows))
                mx = [max(abs(r[key]) for r in rows if r["kind"] == kind and r["eps"] == e) for e in eps]
                lab = f"{kind}: |a2(t*) - {'S' if key == 'resid_S_over_eps2' else 'S - m-m term'}| / eps^2"
                if kind == "no a1, a9":
                    lab += " (m-m term = 0: the two coincide)"
                b.loglog(eps, mx, ls=ls, marker=mk, color=col, label=lab)
        b.set_xlabel("eps (initial amplitude, full nonlinear ODE)")
        b.set_ylabel("max over 5 directions")
        b.legend(fontsize=7.5, frameon=False)
        b.set_title("(b) S against the full ODE: residual / eps^2 -> 0 unless a1 or a9 is excited (m-m term, not in S)", fontsize=9)
        L_ = TH["L_t_star"]
        mats = [(np.array(TH["F2"]), "(c) F2 (da2/dt, y-y)"), (Z["V_roll"] / L_, "(d) V_roll / L(t*)"),
                (Z["V_dir"] / L_, "(e) V_dir / L(t*)"), (Z["V"] / L_, "(f) V / L(t*)"), (Z["W45"], "(g) W(45) (Gate 3c)"),
                (Z["W_M1"], "(h) W_M1 (Gate 3b M1)")]
        for k, (W, t) in enumerate(mats):
            heat(fig.add_subplot(gs[1, k]), W, t)
        so = TH["S_over_L"]
        fig.suptitle(f"S / L(t*) = a3 + {so['a2_coefficient']:.4f} a2 + y^T (V/L) y;  cos(V/L, W(45)) = {so['cos_W45']:.4f}, "
                     f"|V/L| / |W(45)| = {so['norm_ratio_to_W45']:.3f};  cos(V/L, W_M1) = {so['cos_W_M1']:.4f}", fontsize=10)
        fig.tight_layout()
        _save(fig, out / "fig1_liftup_theory")


def fig2():
    I1 = R["item1"]
    ZS = np.load(run / "item1_t_scan.npz")
    with plt.rc_context(STYLE):
        fig, (a, b, c) = plt.subplots(1, 3, figsize=(17, 5.0), gridspec_kw={"width_ratios": [1.2, 1.0, 1.0]})
        t, tr, te = ZS["t"], ZS["train_median"], ZS["test_median"]
        k = t > 0
        a.semilogx(t[k], tr[k], color=C_PTS, lw=1.5, label="train median (analytic S* per t)")
        a.semilogx(t[k], te[k], color=C_ALT, lw=1.0, ls="--", label="test median (same S*)")
        a.axvline(TH["t_star"], color=C_FIT, ls="--", lw=0.9, label=f"t* = {TH['t_star']:.1f} (no fit)")
        sc = I1["t_scan_(descriptive)"]
        a.axvline(sc["t_at_min_train_median"], color=C_GRAY, ls=":", lw=0.9, label=f"train minimum at t = {sc['t_at_min_train_median']:.1f}")
        a.axhline(I1["criterion"]["3c_c_test_median"], color=C_REF, ls=":", lw=0.9, label=f"Gate 3c (c) test {I1['criterion']['3c_c_test_median']:.4f}")
        a.axhline(tr[0], color=C_GRAY, ls="-.", lw=0.8, label=f"t = 0 (|a2|): {tr[0]:.3f}")
        a.set_ylim(0, min(0.3, float(np.nanmax(tr[k])) * 1.05))
        a.set_xlabel("t used in place of t* in S")
        a.set_ylabel("median |ln(s_pred/s_c)|")
        a.legend(fontsize=7.5, frameon=False)
        a.set_title("(a) t scan (descriptive)", fontsize=9)
        rows = [("S (t = t*, no fitted time)", I1["S"]["test"], C_PTS), ("S + m-m term (descriptive)", I1["S_hat_(descriptive)"]["test"], C_ALT)]
        for name, r in I1["references"].items():
            rows.append((name, r, C_REF))
        for i, (n_, r, col) in enumerate(rows):
            lo, hi = r["median_abs_ci"]
            b.errorbar(r["median_abs"], i, xerr=[[r["median_abs"] - lo], [hi - r["median_abs"]]], fmt="o", color=col, capsize=3)
        b.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
        b.axvline(0.05, color=C_FIT, ls="--", lw=0.8)
        b.set_xscale("log")
        b.set_xticks([0.02, 0.05, 0.1, 0.2], ["0.02", "0.05", "0.1", "0.2"])
        b.minorticks_off()
        b.invert_yaxis()
        b.set_xlabel("test median |ln| (95% CI)")
        b.set_title(f"(b) {I1['criterion']['verdict']}", fontsize=8)
        E = np.load(run / "item1_errors.npz")
        sp = E["split"]
        e3c = np.load(g3c / "item1_test_errors.npz")
        pos = {int(r): i for i, r in enumerate(e3c["test_rays"])}
        ec = e3c["E"][list(e3c["names"]).index("c")][[pos[int(r)] for r in E["rays"][sp == 2]]]          # same test rays
        bins = np.linspace(-1.5, 1.5, 121)
        c.hist(np.clip(E["e_S"][sp == 2], -1.5, 1.5), bins=bins, histtype="step", color=C_PTS, lw=1.5, label=f"S: {I1['S']['test']['median_abs']:.4f}")
        c.hist(np.clip(ec, -1.5, 1.5), bins=bins, histtype="step", color=C_REF, lw=1.0, label=f"Gate 3c (c): {I1['criterion']['3c_c_test_median']:.4f}")
        c.set_xlabel("ln(s_pred/s_c) on the test rays (clipped to +-1.5)")
        c.legend(fontsize=8, frameon=False)
        c.set_title("(c) test errors", fontsize=9)
        fig.tight_layout()
        _save(fig, out / "fig2_item1_S")


def fig3():
    I2 = R["item2"]
    E = np.load(run / "item2_errors.npz")
    te = E["split"] == 2
    with plt.rc_context(STYLE):
        fig, (a, b, c) = plt.subplots(1, 3, figsize=(17, 5.0))
        bins = np.linspace(-1.5, 1.5, 121)
        for key, lab, col, lw in (("e_c", "composite max(s_i, s_ii)", C_PTS, 1.6), ("e_i", "(i) |S| >= S* alone", C_ALT, 1.0),
                                  ("e_ii", "(ii) merged gain >= c*_0 alone", C_REF, 1.0)):
            e = E[key][te]
            a.hist(np.clip(e[np.isfinite(e)], -1.5, 1.5), bins=bins, histtype="step", color=col, lw=lw,
                   label=f"{lab}: {np.median(np.abs(e)):.4f}")
        a.set_xlabel("ln(s_pred/s_c), random test rays (clipped)")
        a.legend(fontsize=8, frameon=False)
        bt = I2["random"]["binding_test"]
        a.set_title(f"(a) random test rays; (i) binds on {bt['frac_i_binds']:.1%}, (ii) on {bt['frac_ii_binds']:.1%}", fontsize=9)
        use = E["fam_use"]
        eps, ang = E["fam_eps"], E["fam_ang"]
        xs = np.log10(eps[use])
        for key, lab, col, dx in (("fam_e_c", "composite", C_PTS, 0.0), ("fam_e_i", "(i) alone", C_ALT, -0.15), ("fam_e_ii", "(ii) alone", C_REF, 0.15)):
            e = E[key][use]
            b.scatter(xs + dx, np.clip(e, -1.5, 1.5), s=14, color=col, alpha=0.8, label=lab)
        n90 = use & (ang == 90.0)
        b.scatter(np.log10(eps[n90]), np.clip(E["fam_e_c"][n90], -1.5, 1.5), s=60, facecolors="none", edgecolors=C_FIT, label="90 deg rays (composite)")
        b.axhline(0, color="0.3", lw=0.6)
        b.set_xticks([-6, -4, -2], ["1e-6", "1e-4", "1e-2"])
        b.set_xlabel("eps (noise family, deterministic s_c, sigma = 0)")
        b.set_ylabel("signed ln error (clipped to +-1.5)")
        b.legend(fontsize=8, frameon=False)
        cf = I2["family"]["composite"]
        b.set_title(f"(b) noise family: composite median {cf['all']['median_abs']:.4f}, eps trend {cf['eps_trend']['median']:+.4f}", fontsize=9)
        labs, fi, fii = ["random test"], [bt["frac_i_binds"]], [bt["frac_ii_binds"]]
        for k_, v in I2["family_binding_by_eps"].items():
            labs.append(f"family eps={k_}")
            fi.append(v["frac_i_binds"])
            fii.append(v["frac_ii_binds"])
        x = np.arange(len(labs))
        c.bar(x, fi, color=C_ALT, label="(i) binds (s_i > s_ii)")
        c.bar(x, fii, bottom=fi, color=C_REF, label="(ii) binds (s_ii > s_i)")
        c.set_xticks(x, labs, fontsize=8)
        c.set_ylim(0, 1.05)
        c.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        cr = I2["criterion"]
        c.set_title(f"(c) binding; item 2: random {cr['random_test_median']:.4f}, family {cr['family_median']:.4f}, "
                    f"eps trend {cr['eps_trend']:+.4f} (each <= 0.05: {all(cr['parts_ok'])})", fontsize=8)
        fig.tight_layout()
        _save(fig, out / "fig3_item2_composite")


def fig4():
    I3 = R["item3"]
    E2 = np.load(run / "item2_errors.npz")
    E3 = np.load(run / "item3_errors.npz")
    with plt.rc_context(STYLE):
        fig, (a, b, c) = plt.subplots(1, 3, figsize=(17, 5.2))
        use = E2["fam_use"]
        for (eps, e_c, e_i, e_ii, sel, ang, tag) in (
                (E2["fam_eps"], E2["fam_e_c"], E2["fam_e_i"], E2["fam_e_ii"], use, E2["fam_ang"], "noise family"),
                (E3["eps"], E3["e_c"], E3["e_i"], E3["e_ii"], E3["main"] | E3["sep"], E3["angle"], "transition family")):
            ok = sel & (ang != 90.0)
            a.scatter(eps[ok] * np.exp(np.random.default_rng(0).uniform(-0.15, 0.15, ok.sum())), np.clip(e_c[ok], -1.5, 1.5), s=14,
                      color=C_PTS, alpha=0.8, label="composite, 7 angles" if tag == "noise family" else None)
            n9 = sel & (ang == 90.0)
            a.scatter(eps[n9], np.clip(e_c[n9], -1.5, 1.5), s=50, facecolors="none", edgecolors=C_FIT,
                      label="composite, 90 deg" if tag == "noise family" else None)
        all_eps = sorted(set(E2["fam_eps"][use].tolist()) | set(E3["eps"][E3["main"]].tolist()))

        def med_by(eps_arr, e_arr, sel_, ang_):
            return [float(np.median(e_arr[sel_ & (ang_ != 90.0) & (eps_arr == x)])) if np.any(sel_ & (ang_ != 90.0) & (eps_arr == x)) else np.nan
                    for x in all_eps]
        for key, lab, col, ls in (("c", "composite (signed median)", C_PTS, "-"), ("i", "(i) alone", C_ALT, "--"), ("ii", "(ii) alone", C_REF, "--")):
            m2 = med_by(E2["fam_eps"], E2[f"fam_e_{key}"], use, E2["fam_ang"])
            m3 = med_by(E3["eps"], E3[f"e_{key}"], E3["main"], E3["angle"])
            mm = [x if np.isfinite(x) else y for x, y in zip(m2, m3)]
            a.plot(all_eps, np.clip(mm, -1.5, 1.5), ls=ls, color=col, lw=1.2, label=lab)
        a.axhline(0, color="0.3", lw=0.6)
        for y_ in (-0.05, 0.05):
            a.axhline(y_, color=C_FIT, ls=":", lw=0.6)
        a.axvspan(0.02, 4, color="0.93", zorder=-1)
        a.set_xscale("log")
        a.set_xlabel("eps (noise family 1e-6..1e-2; transition family 0.03..3, shaded)")
        a.set_ylabel("signed ln(s_pred/s_c) (clipped to +-1.5)")
        a.legend(fontsize=7.5, frameon=False)
        cr = I3["criterion"]
        a.set_title(f"(a) frozen composite, both families; item 3 pooled median {cr['pooled_median']:.4f} (criterion 0.05), "
                    f"eps over 0.05: {', '.join(cr['eps_over']) or 'none'}", fontsize=8)
        # s_c and the two predicted amplitudes vs eps (medians over the 7 angles x 2 draws)
        sc2 = np.load(REPO / cfg["data"]["gate2_run"] / "family_det.npz")["s_c"]
        lsc2, lsc3 = np.log(np.where(np.isfinite(sc2), sc2, 1.0)), np.log(np.where(np.isfinite(E3["s_c"]), E3["s_c"], 1.0))

        def med_s(eps_arr, v, sel_, ang_):
            return [float(np.exp(np.median(v[sel_ & (ang_ != 90.0) & (eps_arr == x)]))) if np.any(sel_ & (ang_ != 90.0) & (eps_arr == x)) else np.nan for x in all_eps]
        for v2, v3, lab, col, ls in ((lsc2, lsc3, "s_c (deterministic)", "k", "-"), (E2["fam_ln_i"], E3["ln_i"], "s_i: |S| = S*", C_ALT, "--"),
                                     (E2["fam_ln_ii"], E3["ln_ii"], "s_ii: merged gain = c*_0", C_REF, "--")):
            m2, m3 = med_s(E2["fam_eps"], v2, use, E2["fam_ang"]), med_s(E3["eps"], v3, E3["main"], E3["angle"])
            b.loglog(all_eps, [x if np.isfinite(x) else y for x, y in zip(m2, m3)], ls=ls, color=col, marker="o", ms=3, label=lab)
        b.axvspan(0.02, 4, color="0.93", zorder=-1)
        b.set_xlabel("eps")
        b.set_ylabel("amplitude s (median over 7 angles x 2 draws)")
        b.legend(fontsize=8, frameon=False)
        b.set_title("(b) where each condition is met along the ray", fontsize=9)
        fi, fii = [], []
        for x in all_eps:
            if x in [float(k) for k in R["item2"]["family_binding_by_eps"]]:
                v = R["item2"]["family_binding_by_eps"][f"{x:g}"]
            else:
                v = I3["binding_main_by_eps"][f"{x:g}"]
            fi.append(v["frac_i_binds"])
            fii.append(v["frac_ii_binds"])
        xx = np.arange(len(all_eps))
        c.bar(xx, fi, color=C_ALT, label="(i) binds")
        c.bar(xx, fii, bottom=fi, color=C_REF, label="(ii) binds")
        c.set_xticks(xx, [f"{x:g}" for x in all_eps], fontsize=8)
        c.set_xlabel("eps (all 8 angles for the noise family; 7 for the transition family)")
        c.set_ylim(0, 1.05)
        c.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
        c.set_title("(c) which condition binds", fontsize=9)
        fig.tight_layout()
        _save(fig, out / "fig4_item3_transition")


fig1()
fig2()
fig3()
fig4()
print("figures written to", out)
