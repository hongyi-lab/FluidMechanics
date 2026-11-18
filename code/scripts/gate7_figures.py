#!/usr/bin/env python
"""Figures for Gate 7 (DNS) and the fast/slow post-hoc check, white background, PNG + PDF. Each figure is drawn only if its
inputs exist.

    python code/scripts/gate7_figures.py --g7 experiments/20261118_gate7_dns_v1 \
        --posthoc experiments/20261118_gate56_posthoc_fast_v1 --out docs/reports/figs/gate7
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from committor.plotting import STYLE, _save  # noqa: E402

COL = {"F": "#4C72B0", "R": "#55A868", "W": "#C44E52", "ref": "#222222", "alt": "#DD8452", "grey": "#888888"}
LV = ["64", "96", "128"]
RAYCOL = {"W#1": "#C44E52", "W#3": "#DD8452", "W#7": "#8C564B", "W#9": "#E377C2", "R#22": "#2CA02C", "R#13": "#17BECF",
          "R#15": "#4C72B0", "W#11": "#7F7F7F"}


def fig_posthoc(ph, out):
    R = json.loads((ph / "results.json").read_text())
    O = json.loads((ph / "mfe_offsets.json").read_text())
    j = O["deltas"].index(1e-3)
    T = O["t_hit"]["E_B=0.29"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for r in R["rays"]:
            st = r["ray"][0]
            ax[0].scatter(r["t_hit_hi"], abs(r["ln_err"]), s=22, color=COL[st], alpha=0.85, edgecolor="none")
        for st in "FRW":
            ax[0].scatter([], [], s=22, color=COL[st], label={"F": "F (composite)", "R": "R (composite; calibration: (i))", "W": "W (composite)"}[st])
        ax[0].axvline(R["fast_line_t_hit_hi"], color=COL["ref"], ls="--", lw=1.0, label="t_hit_hi = 600")
        ax[0].axhline(0.24, color=COL["grey"], ls=":", lw=1.0, label="|error| = 0.24")
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_xlabel("DNS t_hit_hi at 64 (time to B of the smallest B case)")
        ax[0].set_ylabel("|ln(s_pred / s_c)|")
        ax[0].set_title(f"(a) 52 rays: fast {R['n_fast']} (|err| >= {R['fast_abs_err_min']:.3f}), slow {R['n_slow']} "
                        f"(median {R['slow_abs_err_median']:.4f})", fontsize=9)
        ax[0].legend(fontsize=7.5)
        for r in R["rays"]:
            st = r["ray"][0]
            ax[1].scatter(r["t_hit_hi"], T[r["ray"]][j], s=22, color=COL[st], alpha=0.85, edgecolor="none")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("DNS t_hit_hi at 64")
        ax[1].set_ylabel("MFE time to B at s = 1.001 s_c(MFE)")
        ax[1].set_title("(b) MFE has no fast channel (all 52 rays 880-1190)", fontsize=9)
        ax[1].axvline(R["fast_line_t_hit_hi"], color=COL["ref"], ls="--", lw=1.0)
        for a in ax:
            a.set_xticks([150, 300, 600, 1200, 2400])
            a.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        fig.tight_layout()
        _save(fig, out / "fig5_posthoc_fast_slow")
        plt.close(fig)


def fig_levels(g7, out):
    R = json.loads((g7 / "results.json").read_text())
    rays = list(R["rays"].keys())
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.4))
        for k, n in enumerate(rays):
            Q = R["rays"][n]
            xs = [i for i, lv in enumerate(LV) if Q[lv].get("s_c")]
            if not xs:
                continue
            s64 = Q["64"]["s_c"]
            y = [math.log(Q[LV[i]]["s_c"] / s64) for i in xs]
            lo = [math.log(Q[LV[i]]["lo"] / s64) for i in xs]
            hi = [math.log(Q[LV[i]]["hi"] / s64) for i in xs]
            off = (k - len(rays) / 2) * 0.03
            ax[0].errorbar(np.array(xs) + off, y, yerr=[np.array(y) - lo, np.array(hi) - y], marker="o", ms=4, lw=1.2,
                           color=RAYCOL.get(n, "k"), label=n, capsize=2)
            th = [Q[LV[i]].get("t_hit_hi") for i in xs]
            ax[1].plot(np.array(xs) + off, th, marker="o", ms=4, lw=1.2, color=RAYCOL.get(n, "k"), label=n)
            e = [Q[LV[i]].get("err_i") for i in xs]
            ax[2].plot(np.array(xs) + off, e, marker="o", ms=4, lw=1.2, color=RAYCOL.get(n, "k"), label=n)
        for a in ax:
            a.set_xticks(range(3))
            a.set_xticklabels(["64x32x32", "96x48x48", "128x64x64"])
        ax[0].set_ylabel("ln(s_c / s_c(64))  (bar: final bracket)")
        ax[0].set_title("(a) edge amplitude vs resolution", fontsize=9)
        ax[0].legend(fontsize=7.5, ncol=2)
        ax[1].set_yscale("log")
        ax[1].axhline(600, color=COL["ref"], ls="--", lw=1.0)
        ax[1].set_ylabel("t_hit_hi (time to B of the smallest B case)")
        ax[1].set_title("(b) fast channel (< 600) vs resolution", fontsize=9)
        ax[2].axhspan(-0.10, 0.10, color="#DDDDDD", alpha=0.6, lw=0, label="|err(i)| <= 0.10 (artefact)")
        ax[2].axhline(0.25, color=COL["ref"], ls=":", lw=1.0, label="|err(i)| = 0.25 (real route)")
        ax[2].axhline(-0.25, color=COL["ref"], ls=":", lw=1.0)
        ax[2].set_ylabel("err(i) = ln(s_i / s_c), same resolution")
        ax[2].set_title(f"(c) first gate at the same resolution (S* = {R['S_star']:.4f})", fontsize=9)
        ax[2].legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        fig.suptitle(f"Gate 7: verdict = {R['verdict']['verdict']}; controls {'pass' if R['controls']['pass'] else 'FAIL'}", fontsize=10)
        fig.tight_layout()
        _save(fig, out / "fig6_gate7_levels")
        plt.close(fig)


def _diag_files(g7):
    D = {}
    for f in sorted(g7.glob("diag_*.npz")):
        Z = np.load(f)
        lv = f.name.split("_")[1]
        for i, (n, w) in enumerate(zip(Z["names"], Z["which"])):
            D[(str(n), lv, str(w))] = (Z["t"], Z["rec"][i], [str(p) for p in Z["parts"]])
    return D


def fig_diag(g7, out):
    D = _diag_files(g7)
    if not D:
        return
    R = json.loads((g7 / "results.json").read_text()) if (g7 / "results.json").exists() else None
    rays = ["W#1", "W#3", "W#7", "W#9", "R#22", "R#13", "R#15", "W#11"]
    rays = [n for n in rays if any(k[0] == n for k in D)]
    parts_show = [("streak kx=0", "#4C72B0"), ("roll kx=0", "#55A868"), ("kx=alpha", "#C44E52"), ("kx>=2alpha", "#DD8452")]
    for which in ("hi", "lo"):
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(2, len(rays), figsize=(2.6 * len(rays), 6.4), sharex=True, squeeze=False)
            for k, n in enumerate(rays):
                lvs = [lv for lv in LV if (n, lv, which) in D]
                top = lvs[-1] if lvs else None
                for lv in dict.fromkeys(["64", top]):
                    if lv is None or (n, lv, which) not in D:
                        continue
                    ls = "-" if lv == "64" else "--"
                    t, rec, P = D[(n, lv, which)]
                    E = rec[:, 0]
                    for p, c in parts_show:
                        ax[0, k].semilogy(t, rec[:, P.index(p)], color=c, ls=ls, lw=1.0, label=f"{p} ({lv})" if k == 0 else None)
                    ax[0, k].semilogy(t, E, color="k", ls=ls, lw=1.2, label=f"E ({lv})" if k == 0 else None)
                    for p, c in (("high third x", "#4C72B0"), ("high third y", "#55A868"), ("high third z", "#C44E52")):
                        ax[1, k].semilogy(t, rec[:, P.index(p)] / E, color=c, ls=ls, lw=1.0, label=f"{p} ({lv})" if k == 0 else None)
                ax[0, k].set_ylim(bottom=max(ax[0, k].get_ylim()[0], 1e-12))  # hide the t = 0 start-up spike
                ax[0, k].set_title(f"{n} ({which}); - 64, -- {top}", fontsize=8.5)
                ax[1, k].set_xlabel("t")
            ax[0, 0].set_ylabel("energy")
            ax[1, 0].set_ylabel("fraction in highest third of resolved k")
            ax[0, 0].legend(fontsize=6)
            ax[1, 0].legend(fontsize=6)
            fig.suptitle(f"7c: energy parts of the final {which} case of the bisection, 64 (solid) vs highest level (dashed)", fontsize=10)
            fig.tight_layout()
            _save(fig, out / f"fig7_gate7_diag_{which}")
            plt.close(fig)


def fig_scan(g7, out):
    S = {lv: json.loads((g7 / f"scan_{lv}.json").read_text()) for lv in ("64", "96") if (g7 / f"scan_{lv}.json").exists()}
    if not S:
        return
    R = json.loads((g7 / "results.json").read_text()) if (g7 / "results.json").exists() else None
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.2))
        mk = {0: ("o", "A"), 1: ("s", "B"), -1: ("x", "undecided")}
        for lv, c in (("64", COL["F"]), ("96", COL["W"])):
            if lv not in S:
                continue
            s, f, t = np.array(S[lv]["s"]), np.array(S[lv]["fate"]), np.array([np.nan if x is None else x for x in S[lv]["t_hit"]], dtype=float)
            for fv, (m, lab) in mk.items():
                sel = f == fv
                if sel.any():
                    ax.scatter(s[sel], t[sel] if fv != -1 else np.full(sel.sum(), 8000.0), marker=m, color=c,
                               facecolor="none" if fv == 0 else c, s=30, label=f"{lv}: {lab}")
            if R and R["rays"]["W#1"][lv].get("s_c"):
                ax.axvline(R["rays"]["W#1"][lv]["s_c"], color=c, ls="--", lw=0.9)
            ax.text(0.99, 0.62 - 0.07 * ("64", "96").index(lv), f"{lv}: {'monotone' if S[lv]['monotone'] else 'NOT monotone'} "
                    f"({S[lv]['n_switches']} switch{'es' if S[lv]['n_switches'] != 1 else ''})", transform=ax.transAxes, fontsize=8, color=c, ha="right")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("s (W#1)")
        ax.set_ylabel("time to A (open) / B (filled)")
        ax.set_title("7c: W#1 fine scan, 24 points in [0.012, 0.030]; dashed: s_c", fontsize=9)
        ax.legend(fontsize=7.5, loc="lower left")
        fig.tight_layout()
        _save(fig, out / "fig8_gate7_W1_scan")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g7", default=None)
    ap.add_argument("--posthoc", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.posthoc:
        fig_posthoc(Path(a.posthoc), out)
    if a.g7:
        g7 = Path(a.g7)
        if (g7 / "results.json").exists():
            fig_levels(g7, out)
        fig_diag(g7, out)
        fig_scan(g7, out)


if __name__ == "__main__":
    main()
