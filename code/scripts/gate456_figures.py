#!/usr/bin/env python
"""Figures for Gate 4b / 5 / 6 (DNS), white background, PNG + PDF. Each figure is drawn only if its inputs exist.

    python code/scripts/gate456_figures.py --g4b <dir> [--g5 <dir>] [--g6 <dir>] --out docs/reports/figs/gate456
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from committor.plotting import STYLE, _save  # noqa: E402

COL = {"F": "#4C72B0", "R": "#55A868", "W": "#C44E52", "ref": "#222222", "alt": "#DD8452", "grey": "#888888"}


def fig_4b(g4b, out):
    R = json.loads((g4b / "results.json").read_text())
    pk = [k for k in R if k.startswith("pilot_Re") and R[k].get("decision") == "ok"]
    pk = pk[0] if pk else next((k for k in R if k.startswith("pilot_Re")), None)
    if pk is None:
        return
    P = R[pk]
    tag = pk.split("_dt")[0].replace("pilot_", "")
    H = np.load(g4b / f"pilot_{tag}_hist.npz")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        e = H["edges"]
        c = 0.5 * (e[1:] + e[:-1])
        bw = e[1] - e[0]
        dens = H["counts"] / (H["counts"].sum() * bw)
        ax[0].semilogy(c, np.maximum(dens, 1e-6), color=COL["grey"], lw=0.8, label="density (200 bins)")
        ax[0].semilogy(c, np.maximum(H["smooth"], 1e-6), color=COL["F"], lw=1.6, label="5-bin moving average")
        h = P["histogram"]
        ax[0].axvline(h["valley_E"], color=COL["W"], ls="--", lw=1.0, label=f"valley E = {h['valley_E']:.3g} -> E_B = {h['E_B']:g}")
        ax[0].axvline(h["turbulent_mode_E"], color=COL["R"], ls=":", lw=1.0, label=f"turbulent mode {h['turbulent_mode_E']:.3g}")
        ax[0].set_xlabel("E = ||u - U||^2 / ||U||^2")
        ax[0].set_ylabel("density of samples")
        ax[0].set_title(f"(a) pilot {tag}, {P['n_dirs_used']} directions x 8 energies, t <= 3000", fontsize=9)
        ax[0].legend(fontsize=7.5)
        if "undecided_fraction_by_T" in P:
            T = np.array([float(k) for k in P["undecided_fraction_by_T"]])
            f = np.array(list(P["undecided_fraction_by_T"].values()))
            ax[1].plot(T, f, "o-", color=COL["F"], ms=3)
            ax[1].axhline(0.01, color=COL["ref"], ls=":", lw=0.8, label="1% (rule)")
            ax[1].axvline(P["T_det"], color=COL["W"], ls="--", lw=1.0, label=f"T_det = {P['T_det']:g} (pilot rule; 8000 used, user decision)")
            ax[1].set_yscale("symlog", linthresh=1e-3)
            ax[1].set_xlabel("T")
            ax[1].set_ylabel("fraction of pilot runs undecided at T")
            ax[1].legend(fontsize=8)
            ax[1].set_title(f"(b) r_A = {P['r_A']:g}, E_B = {h['E_B']:g}", fontsize=9)
        lad = [k for k in R if k.startswith("sc_")]
        if lad:
            names = R[lad[0]]["names"]
            x = np.arange(len(names))
            for j, k in enumerate(sorted(lad)):
                v = np.array([np.nan if s is None else s for s in R[k]["s_c"]])
                ax[2].plot(x + 0.08 * (j - len(lad) / 2), v, "o", ms=5, label=k.replace("sc_", ""))
            ax[2].set_xticks(x)
            ax[2].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
            ax[2].set_yscale("log")
            ax[2].set_ylabel("s_c (DNS)")
            ax[2].set_title("(c) check rays: resolution and dt ladder", fontsize=9)
            ax[2].legend(fontsize=7.5)
        fig.tight_layout()
        _save(fig, out / "fig1_gate4b_pilot_ladder")


def fig_5(g5, out):
    R = json.loads((g5 / "results.json").read_text())
    if "analysis" not in R:
        return
    A = R["analysis"]
    rows = A["rays"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
        for st in ("F", "R", "W"):
            x = [r["s_c_mfe"] for r in rows if r["set"] == st and r["s_c_mfe"] and r["s_c_dns"]]
            y = [r["s_c_dns"] for r in rows if r["set"] == st and r["s_c_mfe"] and r["s_c_dns"]]
            ax[0].loglog(x, y, "o", color=COL[st], ms=5, label=f"{st} ({len(x)} rays)")
        lim = ax[0].get_xlim() + ax[0].get_ylim()
        lo, hi = min(lim), max(lim)
        ax[0].plot([lo, hi], [lo, hi], color=COL["ref"], lw=0.8, ls=":", label="DNS = MFE")
        ax[0].set_xlabel("s_c MFE (Gate 2 / 3d / here)")
        ax[0].set_ylabel("s_c DNS")
        ax[0].legend(fontsize=8)
        ax[0].set_title("(a) edge amplitude: DNS vs nine-mode model", fontsize=9)
        pdw = A["5.1"]["per_draw"]
        for k, v in pdw.items():
            e = np.array([float(x) for x in v["ln_A3c"]])
            l = np.array([np.nan if y is None else y for y in v["ln_A3c"].values()])
            ax[1].semilogx(e, np.exp(l), "o-", ms=4, color=COL["F"] if str(k) == "0" else COL["alt"], label=f"draw {k}")
        ax[1].set_yscale("log")
        ax[1].set_xlabel("eps (relative wave seed)")
        ax[1].set_ylabel("A3_c = s_c d_3")
        ax[1].set_title(f"(b) F rays, 5.1: (a) {A['5.1']['a']}, (b) {A['5.1']['b']}, (c) {A['5.1']['c']}", fontsize=9)
        ax[1].legend(fontsize=8)
        names = ["S (first gate)", "first order only", "|a3| only", "energy only"]
        for j, st in enumerate(("R", "W")):
            vals = []
            for nm in names:
                v = [abs(r[f"ln_err[{nm}]"]) if not isinstance(r[f"ln_err[{nm}]"], str) else np.inf
                     for r in rows if r["set"] == st and r.get("role") == "test" and r[f"ln_err[{nm}]"] is not None]
                vals.append(np.array(v))
            pos = np.arange(len(names)) + (j - 0.5) * 0.35
            for p_, v in zip(pos, vals):
                vv = np.where(np.isfinite(v), v, 10.0)
                ax[2].plot(np.full(len(vv), p_), np.maximum(vv, 1e-4), "o", ms=3, alpha=0.6, color=COL[st])
                if len(v):
                    ax[2].plot([p_ - 0.12, p_ + 0.12], [max(np.median(np.where(np.isfinite(v), v, np.inf)), 1e-4)] * 2, color=COL[st], lw=2)
            ax[2].plot([], [], "o", color=COL[st], label=f"{st} test rays (bar = median)")
        ax[2].axhline(0.10, color=COL["ref"], ls=":", lw=0.8, label="0.10 / 0.25")
        ax[2].axhline(0.25, color=COL["ref"], ls=":", lw=0.8)
        ax[2].set_yscale("log")
        ax[2].set_xticks(np.arange(len(names)))
        ax[2].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax[2].set_ylabel("|ln s_pred / s_c|  (no crossing plotted at 10)")
        ax[2].set_title(f"(c) 5.2 first gate: {A['5.2']['verdict']} (median {A['5.2']['median']:.3f})", fontsize=9)
        ax[2].legend(fontsize=7.5)
        fig.tight_layout()
        _save(fig, out / "fig2_gate5_truth_first_gate")
        # S(s) curves
        Z = np.load(g5 / "coords.npz")
        S, s = Z["S"], Z["s"]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
        cs = A["S (first gate)"]["c_star"]
        for j, st in enumerate(("F", "R", "W")):
            idx = [i for i, r in enumerate(rows) if r["set"] == st]
            for i in idx:
                ax[j].loglog(s, np.maximum(S[i], 1e-12), color=COL[st], lw=0.8, alpha=0.7)
                if rows[i]["s_c_dns"]:
                    ax[j].plot(rows[i]["s_c_dns"], max(rows[i]["S_at_edge"], 1e-12), "k.", ms=5)
            ax[j].axhline(cs, color=COL["ref"], ls="--", lw=0.9, label=f"S* = {cs:.3g} (fit on R first 12)")
            ax[j].set_xlabel("s")
            ax[j].set_title(f"({'abc'[j]}) {st} rays: S(s) (dots: DNS edge)", fontsize=9)
            ax[j].legend(fontsize=8)
        ax[0].set_ylabel("S(s) = max_t ||streak(s P0 delta + s^2 q)||")
        fig.tight_layout()
        _save(fig, out / "fig3_gate5_S_curves")


def fig_6(g6, g5, out, offs_half=0.9, K=13):
    if not (g6 / "results.json").exists():
        return
    R = json.loads((g6 / "results.json").read_text())
    if "analysis" not in R:
        return
    A = R["analysis"]
    ZF = np.load(g6 / "gain_F.npz")
    offs = np.linspace(-offs_half, offs_half, K)
    R5 = json.loads((g5 / "results.json").read_text())["analysis"]["rays"]
    F = [r for r in R5 if r["set"] == "F"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
        cmap = plt.get_cmap("viridis")
        epsl = sorted(set(r["eps"] for r in F))
        for j, i in enumerate(ZF["idx"]):
            e = F[i]["eps"]
            ax[0].plot(offs, ZF["c"][j], color=cmap(epsl.index(e) / max(1, len(epsl) - 1)), lw=1.0, ls="-" if F[i]["draw"] == 0 else "--")
        ax[0].axhline(A["c_star"], color=COL["ref"], ls=":", lw=0.9, label=f"c* = {A['c_star']:.3g} (eps = 1e-6 rays)")
        ax[0].axvline(0, color=COL["grey"], lw=0.6)
        for e in epsl:
            ax[0].plot([], [], color=cmap(epsl.index(e) / max(1, len(epsl) - 1)), label=f"eps = {e:g}")
        ax[0].set_xlabel("ln(s / s_c)")
        ax[0].set_ylabel("c(s) = ln max_t ||y||^2")
        ax[0].legend(fontsize=7, ncol=2)
        ax[0].set_title("(a) F rays: seed gain (solid draw 0, dashed draw 1)", fontsize=9)
        rf = A["rays_F"]
        for key, col, lab in (("ln_err_composite", COL["ref"], "composite"), ("ln_err_i", COL["F"], "(i) first gate"), ("ln_err_ii", COL["W"], "(ii) seed gain")):
            x = [r["eps"] for r in rf if r[key] is not None and not r["used_for_c_star"]]
            y = [(r[key] if not isinstance(r[key], str) else (1.0 if r[key].startswith("+") else -1.0)) for r in rf
                 if r[key] is not None and not r["used_for_c_star"]]                      # +-inf drawn at +-1 (axis clipped)
            ax[1].semilogx(x, y, "o", ms=4, color=col, alpha=0.8, label=lab)
        for yv in (-0.25, -0.10, 0.10, 0.25):
            ax[1].axhline(yv, color=COL["grey"], ls=":", lw=0.7)
        ax[1].set_ylim(-1.05, 1.05)
        ax[1].set_xlabel("eps")
        ax[1].set_ylabel("ln(s_pred / s_c)  (no crossing drawn at +1)")
        ax[1].set_title(f"(b) 6.1: {A['6.1']['verdict']}", fontsize=9)
        ax[1].legend(fontsize=8)
        s62 = A["6.2"]
        ln_sc = {(r["eps"], r["draw"]): math.log(r["s_c_dns"]) for r in F if r["s_c_dns"]}
        for k, col in ((0, COL["F"]), (1, COL["alt"])):
            pts = [(e, ln_sc.get((e, k))) for e in (1e-6, 1e-4, 1e-2)]
            ax[2].semilogx([p[0] for p in pts], [np.nan if p[1] is None else p[1] for p in pts], "o-", color=col, label=f"DNS, draw {k}")
            if pts[0][1] is not None:
                pr = s62["per_draw"][str(k)]["predicted"]
                ee = np.array([1e-6, 1e-2])
                ax[2].semilogx(ee, pts[0][1] + pr * np.log(ee / 1e-6), ":", color=col, label=f"predicted slope {pr:.3f}")
        ax[2].set_xlabel("eps")
        ax[2].set_ylabel("ln s_c")
        ax[2].set_title(f"(c) 6.2: ratio {s62['ratio']:.3f} -> {s62['verdict']}", fontsize=9)
        ax[2].legend(fontsize=8)
        fig.tight_layout()
        _save(fig, out / "fig4_gate6_seed_gain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g4b", required=True)
    ap.add_argument("--g5")
    ap.add_argument("--g6")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_4b(Path(a.g4b), out)
    if a.g5:
        fig_5(Path(a.g5), out)
    if a.g6 and a.g5:
        fig_6(Path(a.g6), Path(a.g5), out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
