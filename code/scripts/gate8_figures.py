#!/usr/bin/env python
"""Figures for Gate 8 (DNS), white background, PNG + PDF. Each figure is drawn only if its inputs exist.

    python code/scripts/gate8_figures.py --g8 experiments/20261118_gate8_dns_v1 --out docs/reports/figs/gate8
"""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from committor.plotting import STYLE, _save  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate7_figures  # noqa: E402

KZ = ["kz=gamma", "kz=2gamma", "kz>=3gamma"]
KZC = {"kz=gamma": "#4C72B0", "kz=2gamma": "#C44E52", "kz>=3gamma": "#888888"}
RAYC = {"W#1": "#C44E52", "W#3": "#DD8452", "R#22": "#2CA02C", "W#11": "#7F7F7F", "X#1": "#6A3D9A"}


# label offsets (points) in fig9 (a), so that the labels of nearby points do not overlap
OFF_A = {"W#2": (-24, 3), "W#10": (5, -9), "R#22": (-28, 4), "W#1": (5, 4), "R#14": (-28, -3), "W#9": (5, -1), "W#5": (5, -9),
         "W#7": (-24, 1), "X#1": (-22, 5), "R#15": (5, -9), "W#3": (-24, 2), "W#8": (-10, 7), "W#12": (5, -9), "W#6": (-8, -12)}


def fig_blind(g8, out):
    R = json.loads((g8 / "results.json").read_text())
    B = R["blind"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))
        a = ax[0]
        lim = [0.007, 0.15]
        a.plot(lim, lim, color="k", lw=0.8)
        a.fill_between(lim, [x / 1.105 for x in lim], [x * 1.105 for x in lim], color="#DDDDDD", lw=0)
        for r in B["rows"] + [R["X#1"]] + R["not_blind"]:
            if r.get("s_c_96") is None:
                continue
            blind = r["blind"]
            c = "#C44E52" if r["route_pred"] == "fast" else "#4C72B0"
            m = "*" if r["ray"] == "X#1" else "o"
            a.scatter(r["s_c_96"], r["s_pred"], marker=m, s=110 if m == "*" else 45, color=c if blind else "white", edgecolor=c, lw=1.4, zorder=3)
            a.annotate(r["ray"], (r["s_c_96"], r["s_pred"]), fontsize=7, xytext=OFF_A.get(r["ray"], (4, 3)), textcoords="offset points")
            if blind:
                a.scatter(r["s_c_96"], r["s_slow"], marker="x", s=25, color="#555555", zorder=2)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlim(lim)
        a.set_ylim(lim)
        a.set_xlabel("truth s_c at 96x48x48")
        a.set_ylabel("predicted s_pred (three conditions, frozen 8a)")
        a.scatter([], [], color="#C44E52", label="route predicted fast")
        a.scatter([], [], color="#4C72B0", label="route predicted slow")
        a.scatter([], [], color="white", edgecolor="#555555", label="not blind (open)")
        a.scatter([], [], marker="x", color="#555555", label="two conditions only (blind)")
        a.legend(fontsize=7, loc="upper left")
        a.set_title(f"(a) blind test: median |ln| {B['median_abs_err_3cond']:.3f} -> {B['verdict']}", fontsize=9)
        a = ax[1]
        rows = [r for r in B["rows"] if r.get("s_c_96") is not None]
        x = np.arange(len(rows))
        w = 0.27
        for k, (key, lab, c) in enumerate((("err_3cond", "three conditions", "#C44E52"), ("err_2cond", "two conditions", "#4C72B0"),
                                           ("err_energy", "energy only", "#AAAAAA"))):
            a.bar(x + (k - 1) * w, [r[key] for r in rows], width=w, color=c,
                  label=f"{lab} (median |ln| {B['median_abs_err_' + key.split('_')[1]]:.3f})")
        a.axhspan(-0.10, 0.10, color="#EEEEEE", lw=0, zorder=0)
        for y in (0.25, -0.25):
            a.axhline(y, color="k", ls=":", lw=0.8)
        a.axhline(0, color="k", lw=0.6)
        a.set_xticks(x)
        a.set_xticklabels([r["ray"] for r in rows], fontsize=8)
        a.set_ylabel("ln(s_pred / s_c(96))")
        a.legend(fontsize=7)
        a.set_title("(b) per blind ray (grey band 0.10, dotted 0.25)", fontsize=9)
        a = ax[2]
        cluster = []
        for r in B["rows"] + [R["X#1"]] + R["not_blind"]:
            if r.get("t_hit_hi") is None:
                continue
            c = "#C44E52" if r["route_pred"] == "fast" else "#4C72B0"
            m = "*" if r["ray"] == "X#1" else "o"
            xv = math.log(r["s_fast"] / r["s_slow"])
            a.scatter(xv, r["t_hit_hi"], marker=m, s=110 if m == "*" else 45, color=c if r["blind"] else "white", edgecolor=c, lw=1.4, zorder=3)
            if r["t_hit_hi"] >= 300:   # the fast cluster (t_hit_hi 110-170) is named in one text line instead
                a.annotate(r["ray"], (xv, r["t_hit_hi"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
            else:
                cluster.append(r["ray"])
        a.text(0.03, 0.40, "lower-left cluster (t_hit_hi " + f"{min(r['t_hit_hi'] for r in B['rows'] + [R['X#1']] + R['not_blind'] if r.get('t_hit_hi') is not None):.0f}-"
               + f"{max(r['t_hit_hi'] for r in B['rows'] + [R['X#1']] + R['not_blind'] if r.get('t_hit_hi') is not None and r['t_hit_hi'] < 300):.0f}):\n"
               + ", ".join(cluster), transform=a.transAxes, fontsize=7, va="bottom", bbox=dict(facecolor="white", edgecolor="none", pad=1))
        a.axvline(0, color="k", lw=0.8)
        a.axhline(600, color="k", ls="--", lw=0.8)
        a.set_yscale("log")
        a.set_xlabel("ln(s_fast / s_slow)  (< 0: route predicted fast)")
        a.set_ylabel("t_hit_hi at 96 (< 600: truth fast)")
        a.set_title(f"(c) route: {B['route_right']}/{B['route_n']} blind rays right", fontsize=9)
        fig.suptitle("Gate 8b: blind test of the third condition (filled = blind, star = X#1 pure a8)", fontsize=10)
        fig.tight_layout()
        _save(fig, out / "fig9_gate8_blind")
        plt.close(fig)


def fig_posthoc(g8, out):
    P = json.loads((g8 / "posthoc.json").read_text())
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))
        a = ax[0]
        T = P["table52"]
        for r in T:
            st = r["ray"][0]
            c = {"F": "#4C72B0", "R": "#55A868", "W": "#C44E52", "X": "#6A3D9A"}[st]
            m = "*" if st == "X" else ("o" if r["fast_64"] else "s")
            a.scatter(r["a8_sq"], r["G_96"], color=c if r["fast_64"] or st == "X" else "white", edgecolor=c, marker=m, s=90 if st == "X" else 35, lw=1.2)
        k = P["obs1"]["fit_G10_over_a8sq_lsq"]
        xx = np.linspace(0, 1, 50)
        a.plot(xx, k * xx, color="k", lw=0.8, ls="--", label=f"G(t=10) fit {k:.1f} a8^2 (W rays, 64)")
        for st, c in (("F", "#4C72B0"), ("R", "#55A868"), ("W", "#C44E52"), ("X", "#6A3D9A")):
            a.scatter([], [], color=c, label=st)
        a.scatter([], [], color="white", edgecolor="k", marker="s", label="slow at 64 (open)")
        a.set_xlabel("a8^2")
        a.set_ylabel("G = max_{t<=20} ||Pw delta||^2  (96)")
        a.legend(fontsize=7)
        a.set_title("(a) the 52 rays and X#1: G vs a8^2 (filled = fast at 64)", fontsize=9)
        a = ax[1]
        cols = {"W#1": "#C44E52", "W#3": "#DD8452", "W#7": "#8C564B", "W#9": "#E377C2", "R#22": "#2CA02C", "R#13": "#17BECF",
                "R#15": "#4C72B0", "W#11": "#7F7F7F"}
        for n, rr in P["obs3"].items():
            t = [int(k_.split("=")[1]) for k_ in rr["lo"]]
            a.plot(t, [v["ratio_alpha"] for v in rr["lo"].values()], marker="o", ms=3, color=cols[n], label=n)
        a.set_yscale("log")
        a.axhline(1, color="k", lw=0.6)
        a.set_xlabel("t")
        a.set_ylabel("DNS kx=alpha energy / Gate 6 linear wave ||y||^2")
        a.legend(fontsize=7, ncol=2)
        a.set_title("(b) observation 3 (64, lo case at s_c(64))", fontsize=9)
        a = ax[2]
        o2 = P["obs2"]
        names = list(o2.keys())
        a.barh(range(len(names)), [o2[k_]["max_kx_alpha_t_le_20"] for k_ in names], color=["#2CA02C" if "R#22" in k_ else "#C44E52" for k_ in names])
        a.set_yticks(range(len(names)))
        a.set_yticklabels(names, fontsize=7)
        a.set_xticks([0, 0.5e-3, 1e-3, 1.5e-3, 2e-3])
        a.set_xticklabels(["0", "0.5e-3", "1e-3", "1.5e-3", "2e-3"])
        a.set_xlabel("max kx=alpha energy, t <= 20 (highest level, at the edge)")
        a.set_title("(c) observation 2", fontsize=9)
        fig.suptitle("Gate 8: post-hoc observations 1-3 checked (descriptive)", fontsize=10)
        fig.tight_layout()
        _save(fig, out / "fig10_gate8_posthoc")
        plt.close(fig)


def fig_d8(g8, out):
    f1, f2, f3 = g8 / "d8_theory_kz.npz", g8 / "d8_diag_kz.npz", g8 / "d8_gain_alpha.npz"
    if not (f1.exists() and f2.exists() and f3.exists()):
        return
    Th, Dg, Ga = np.load(f1), np.load(f2), np.load(f3)
    names = [str(x) for x in Th["names"]]
    PP = [str(p) for p in Dg["parts"]]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(2, len(names), figsize=(3.3 * len(names), 6.6), sharex=True, squeeze=False)
        for i, n in enumerate(names):
            s = float(Th["s_c"][i])
            t = Th["t"]
            a = ax[0, i]
            A2 = s ** 2 * Th["streak_aa"][i] + 2 * s ** 3 * Th["streak_ab"][i] + s ** 4 * Th["streak_bb"][i]
            for j, b in enumerate(KZ):
                a.semilogy(t, np.maximum(A2[j], 1e-30), color=KZC[b], ls="--", lw=1.0, label=f"theory {b}" if i == 0 else None)
            j = [k for k in range(len(Dg["names"])) if str(Dg["names"][k]) == n and str(Dg["which"][k]) == "hi"][0]
            jl = [k for k in range(len(Dg["names"])) if str(Dg["names"][k]) == n and str(Dg["which"][k]) == "lo"][0]
            for b in KZ:
                a.semilogy(Dg["t"], Dg["rec"][jl, :, PP.index(f"streak {b}")], color=KZC[b], lw=1.2, label=f"DNS lo {b}" if i == 0 else None)
                a.semilogy(Dg["t"], Dg["rec"][j, :, PP.index(f"streak {b}")], color=KZC[b], lw=0.6, alpha=0.6)
            a.set_ylim(1e-14, 1)
            a.set_title(f"{n}: streak energy by kz (s = s_c(96))", fontsize=8.5)
            a = ax[1, i]
            a.semilogy(Ga["t"], Ga["ya"][i], color="k", ls="--", lw=1.2, label="linear wave ||y||^2 (Gate 6 eq.)" if i == 0 else None)
            a.semilogy(Dg["t"], Dg["rec"][jl, :, PP.index("kx=alpha")], color="#C44E52", lw=1.2, label="DNS kx=alpha, lo" if i == 0 else None)
            a.semilogy(Dg["t"], Dg["rec"][j, :, PP.index("kx=alpha")], color="#C44E52", lw=0.6, alpha=0.6, label="DNS kx=alpha, hi" if i == 0 else None)
            a.set_xlabel("t")
            a.set_title(f"{n}: kx = alpha energy", fontsize=8.5)
        ax[0, 0].set_ylabel("streak energy (u, kx = 0)")
        ax[1, 0].set_ylabel("energy")
        ax[0, 0].legend(fontsize=6)
        ax[1, 0].legend(fontsize=6)
        fig.suptitle("Gate 8d (96, record only): streak energy by kz (solid DNS lo, thin DNS hi, dashed theory) and the linear wave vs the DNS",
                     fontsize=10)
        fig.tight_layout()
        _save(fig, out / "fig11_gate8_8d")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g8", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g8, out = Path(a.g8), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (g8 / "posthoc.json").exists():
        fig_posthoc(g8, out)
    if (g8 / "results.json").exists():
        fig_blind(g8, out)
    fig_d8(g8, out)
    sub = g8 / "8c_gate7_128"
    if (sub / "results.json").exists():                                  # the Gate 7 levels figure for the 8c re-analysis
        tmp = out / "_8c"
        tmp.mkdir(exist_ok=True)
        gate7_figures.fig_levels(sub, tmp)
        for ext in ("png", "pdf"):
            shutil.move(str(tmp / f"fig6_gate7_levels.{ext}"), str(out / f"fig12_gate7_levels_8c.{ext}"))
        tmp.rmdir()


if __name__ == "__main__":
    main()
