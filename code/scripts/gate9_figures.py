#!/usr/bin/env python
"""Gate 9 figures (white background) from <run>/results.json and predictions_9.json.

    python code/scripts/gate9_figures.py --g9 experiments/20261119_gate9_dns_v1 --out docs/reports/figs/gate9
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STYLE = {"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "font.size": 9,
         "axes.spines.top": False, "axes.spines.right": False}
CC = {"G": "#4C72B0", "O": "#C44E52", "V": "#55A868", "M": "#8172B2"}
BASE = {"rule": ("this round's rule", "k"), "gate8a": ("Gate 8a rule", "#C44E52"), "two_cond": ("two conditions", "#4C72B0"),
        "energy": ("energy only", "#999999")}


def _save(fig, stem):
    fig.savefig(f"{stem}.png", dpi=200)
    fig.savefig(f"{stem}.pdf")


def fig_scatter(R, P, out):
    rows = [r for r in R["rows"] if r.get("s_c") is not None and r["s_pred"] is not None]
    cens = [r for r in R["rows"] if "censored" in r and r["s_pred"] is not None]
    nopred = [r for r in R["rows"] if r["s_pred"] is None]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.4))
        allv = [r["s_c"] for r in rows] + [r["s_pred"] for r in rows + cens] + [r["censored"]["bound"] for r in cens]
        lim = [0.6 * min(allv), 1.6 * max(allv)]
        for k, a in enumerate(ax):
            sel = [r for r in rows if r["in_range"] == (k == 0)]
            for r in cens:                                                       # (h18) censored truth: arrow from the bound
                if r["in_range"] != (k == 0):
                    continue
                b, sp = r["censored"]["bound"], r["s_pred"]
                a.annotate("", xy=(b * (2.2 if r["censored"]["side"] == ">" else 1 / 2.2), sp), xytext=(b, sp),
                           arrowprops={"arrowstyle": "->", "color": CC[r["cls"]], "lw": 1.2})
                a.plot(b, sp, "x", color=CC[r["cls"]], ms=7, mew=1.5)
                a.annotate(f"{r['ray']} (s_c {r['censored']['side']} {b:.3g}, not bracketed)", (b, sp), fontsize=6.5, xytext=(4, -9), textcoords="offset points")
            if k == 0 and nopred:
                a.text(0.98, 0.02, "no prediction: " + "; ".join(f"{r['ray']} (s_c {r['s_c']:.3g})" if r.get("s_c") else r["ray"] for r in nopred),
                       transform=a.transAxes, ha="right", va="bottom", fontsize=6.5)
            a.plot(lim, lim, color="k", lw=0.8)
            if k == 1:
                a.plot(lim, [x / 0.8 for x in lim], color="k", ls=":", lw=0.8, label="bound s_c = 0.8 s_pred")
            a.axhline(0.03, color="#BBBBBB", lw=0.8, ls="--")
            for r in sel:
                c = CC[r["cls"]]
                m = "o" if r["route_pred"] == "fast" else "s"
                xerr = [[r["s_c"] * (1 - math.exp(-r["u"]))], [r["s_c"] * (math.exp(r["u"]) - 1)]]
                if k == 0:
                    iv8, iv5 = r["pred"]["intervals"]["p80"], r["pred"]["intervals"]["p50"]
                    a.plot([r["s_c"]] * 2, iv8, color=c, lw=0.8, alpha=0.6)
                    a.plot([r["s_c"]] * 2, iv5, color=c, lw=2.5, alpha=0.8)
                a.errorbar(r["s_c"], r["s_pred"], xerr=xerr, fmt=m, ms=6, color=c, mfc=c if r["route_pred"] == "fast" else "white", mew=1.3,
                           elinewidth=0.8, capsize=2, zorder=3)
                a.annotate(r["ray"], (r["s_c"], r["s_pred"]), fontsize=6.5, xytext=(4, 3), textcoords="offset points")
            a.set_xscale("log")
            a.set_yscale("log")
            a.set_xlim(lim)
            a.set_ylim(lim)
            a.set_xlabel("truth s_c at 96x48x48 (bar: u)")
            a.set_ylabel("frozen prediction s_pred (9.0 rule)")
            a.set_title(("(a) in range (s_pred <= 0.03): bars = 50% (thick) and 80% (thin) intervals" if k == 0 else
                         "(b) out of range: only s_c >= 0.8 s_pred is claimed"), fontsize=9)
            for cl, c in CC.items():
                a.plot([], [], "o", color=c, label=cl)
            a.plot([], [], "o", color="k", label="predicted fast (filled)")
            a.plot([], [], "s", color="k", mfc="white", label="predicted slow (open)")
            a.legend(fontsize=7, loc="upper left")
        fig.suptitle("Gate 9: frozen predictions against the truth at 96 (28 rays outside the nine patterns)", fontsize=10)
        fig.tight_layout()
        _save(fig, out / "fig13_gate9_pred_vs_truth")
        plt.close(fig)


def fig_errdist(R, out):
    rows = [r for r in R["rows"] if "err" in r]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
        for k, (a, rng) in enumerate(zip(ax[:2], (None, True))):
            sel = [r for r in rows if rng is None or r["in_range"] == rng]
            for key, (lab, c) in BASE.items():
                e = np.sort([abs(r["err"]) if key == "rule" else abs(r["err_base"][key]) for r in sel if (key == "rule" or r["err_base"][key] is not None)])
                if len(e) == 0:
                    continue
                a.step(e, np.arange(1, len(e) + 1) / len(e), where="post", color=c, lw=2.0 if key == "rule" else 1.2,
                       label=f"{lab} (median {np.median(e):.3f})")
            a.set_xscale("log")
            a.set_xlabel("|ln(prediction / s_c(96))|")
            a.set_ylabel("fraction of rays")
            a.set_ylim(0, 1.02)
            a.legend(fontsize=7, loc="upper left")
            a.set_title(f"({'ab'[k]}) empirical distribution, {'all rays with a truth' if rng is None else 'in range only'} (n = {len(sel)})", fontsize=9)
        a = ax[2]
        cls = ["G", "O", "V", "M"]
        for i, cl in enumerate(cls):
            sel = [r for r in rows if r["cls"] == cl]
            for r in sel:
                x = i + (0.12 if r["in_range"] else -0.12)
                a.errorbar(x, r["err"], yerr=r["u"], fmt="o" if r["in_range"] else "^", color=CC[cl], ms=5, mfc=CC[cl] if r["in_range"] else "white",
                           elinewidth=0.8, capsize=2)
                a.annotate(r["ray"].split("#")[1], (x, r["err"]), fontsize=6, xytext=(3, 0), textcoords="offset points")
        a.axhline(0, color="k", lw=0.6)
        a.set_xticks(range(len(cls)))
        a.set_xticklabels(cls)
        a.set_ylabel("ln(s_pred / s_c(96))  (bar: u)")
        a.set_title("(c) signed error by class (filled: in range; open triangle: out of range)", fontsize=9)
        fig.suptitle("Gate 9: error distributions of this round's rule and the three baselines", fontsize=10)
        fig.tight_layout()
        _save(fig, out / "fig14_gate9_errors")
        plt.close(fig)


def fig_M(R, out):
    rows = [r for r in R["rows"] if r["cls"] == "M" and r.get("M_xy")]
    with plt.rc_context(STYLE):
        fig, a = plt.subplots(figsize=(5.6, 5.0))
        xs = [r["M_xy"]["x"] for r in rows if r["M_xy"]["x"] is not None and r["M_xy"]["y"] is not None]
        ys = [r["M_xy"]["y"] for r in rows if r["M_xy"]["x"] is not None and r["M_xy"]["y"] is not None]
        hi = max([1.3] + [v * 1.15 for v in xs + ys])
        lo = min([0.5] + [v * 0.85 for v in xs + ys])
        a.plot([lo, 1, 1], [1, 1, lo], color="k", lw=1.2)
        a.fill_between([lo, 1], [lo, lo], [1, 1], color="#EEEEEE", lw=0, zorder=0)
        a.text(lo * 1.03, lo * 1.03, "inside the right angle:\ns_c below both routes", fontsize=7, va="bottom")
        for r in rows:
            x, y = r["M_xy"]["x"], r["M_xy"]["y"]
            th = r["pred"].get("theta_deg")
            if x is None or y is None:
                a.annotate(f"{r['ray']} (theta {th:g}): no fast route, x = {x:.3f}" if x else r["ray"], (0.02, 0.95 - 0.05 * rows.index(r)),
                           xycoords="axes fraction", fontsize=7)
                continue
            a.errorbar(x, y, xerr=[[x * (1 - math.exp(-r["u"]))], [x * (math.exp(r["u"]) - 1)]], yerr=[[y * (1 - math.exp(-r["u"]))], [y * (math.exp(r["u"]) - 1)]],
                       fmt="o", color=CC["M"], ms=6, capsize=2, elinewidth=0.8)
            a.annotate(f"{th:g} deg", (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlim(lo, hi)
        a.set_ylim(lo, hi)
        a.set_xlabel("s_c / s_slow")
        a.set_ylabel("s_c / s_fast")
        a.set_title("Gate 9 M class (cos(theta) a3 + sin(theta) a8): min rule = the right angle", fontsize=9)
        fig.tight_layout()
        _save(fig, out / "fig15_gate9_M")
        plt.close(fig)


def fig_routes(R, out):
    rows = [r for r in R["rows"] if r.get("t_hit_hi") is not None]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
        a = ax[0]
        th = [r["t_hit_hi"] for r in rows]
        bins = np.geomspace(max(1.0, 0.8 * min(th)), 1.25 * max(th), 26)
        vv = [[r["t_hit_hi"] for r in rows if r["route_pred"] == route] for route in ("fast", "slow")]
        a.hist(vv, bins=bins, stacked=True, color=["#C44E52", "#4C72B0"], edgecolor="white", lw=0.5,
               label=[f"predicted {route} (n = {len(v)})" for route, v in zip(("fast", "slow"), vv)])
        a.axvline(600, color="k", ls="--", lw=0.9)
        a.set_xscale("log")
        a.set_xlabel("t_hit_hi at 96 (time to B of the smallest B case)")
        a.set_ylabel("rays")
        a.legend(fontsize=7)
        a.set_title("(a) t_hit_hi; dashed: 600 (truth fast below)", fontsize=9)
        a = ax[1]
        a.axis("off")
        T = R["routes"]
        cell = [[str(T["pred fast / true fast"]), str(T["pred fast / true slow"])], [str(T["pred slow / true fast"]), str(T["pred slow / true slow"])]]
        tb = a.table(cellText=cell, rowLabels=["pred fast", "pred slow"], colLabels=["true fast", "true slow"], loc="center", cellLoc="center")
        tb.scale(1, 2)
        a.set_title("(b) route table (rays with a truth)", fontsize=9)
        fig.tight_layout()
        _save(fig, out / "fig16_gate9_routes")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g9", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g9, out = Path(a.g9), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = json.loads((g9 / "results.json").read_text())
    P = json.loads((g9 / "predictions_9.json").read_text())
    fig_scatter(R, P, out)
    fig_errdist(R, out)
    fig_M(R, out)
    fig_routes(R, out)


if __name__ == "__main__":
    main()
