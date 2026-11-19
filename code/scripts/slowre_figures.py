#!/usr/bin/env python
"""Figure and table for the slow route at another Re (from <run>/results.json of slowre_dns.py).

    python code/scripts/slowre_figures.py --run experiments/20261119_slowre500_dns_v1 --out docs/reports/figs/slowre
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402

COL = {"G": "#4C72B0", "V": "#55A868", "M": "#8172B3"}


def f(x, d=3, sign=False):
    if x is None:
        return "—"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def table(R):
    out = [f"Re = {R['Re']:g} against Re = {R['Re0']:g}; S* = {R['S_star']:.5g}, c* = {R['c_star']:.5g} fixed; T_det {R['T_det']:g}.\n",
           "| ray | s_i | s_ii | s_slow (bind) | s_c(500, 64) | t_hit_hi | err(500) | s_slow(400) | s_c(400, 96) | err(400) | shift ln(s_c500/s_c400) | "
           "S(s_c)/S* 400 → 500 | c(s_c) − c* 400 → 500 | gain t_max to crossing |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in R["rows"]:
        sc = f"{r['s_c']:.5g}" if r.get("s_c") is not None else r["status"]
        S5 = None if r.get("S_at_sc") is None else r["S_at_sc"] / R["S_star"]
        S4 = r["S_at_sc_400"] / R["S_star"]
        c5 = None if r.get("c_at_sc") is None else r["c_at_sc"] - R["c_star"]
        c4 = None if r.get("c_at_sc_400") is None else r["c_at_sc_400"] - R["c_star"]
        out.append(f"| {r['ray']} | {r['s_i']:.5g} | {r['s_ii']:.5g} | {r['s_slow']:.5g} ({r['binding']}) | {sc} | "
                   f"{f(r.get('t_hit_hi'), 0)} | {f(r.get('err'), 3, True)} | {r['s_slow_400']:.5g} | {r['s_c_400']:.5g} | {f(r['err_400'], 3, True)} | "
                   f"{f(r.get('shift'), 3, True)} | {f(S4, 3)} → {f(S5, 3)} | {f(c4, 3, True)} → {f(c5, 3, True)} | "
                   f"{f(r.get('gain_t_at_max_upto_cross'), 0)}{' (flag)' if r.get('gain_flag_upto_cross') else ''} |")
    S = R["summary"]
    out.append("")
    out.append(f"- n with s_c: {S['n']}; not ok: {S['not_ok']}")
    out.append(f"- |err(500)| median {f(S['abs_err_median'], 4)}, max {f(S['abs_err_max'], 4)}; |err(400)| median {f(S['abs_err_400_median'], 4)}, "
               f"max {f(S['abs_err_400_max'], 4)}")
    out.append(f"- shift median {f(S['shift_median'], 4, True)}, range {S['shift_range']}; rule closer than 'no change': "
               f"{S['n_rule_closer_than_no_change']} / {S['n']}")
    c96 = [r for r in R["rows"] if r.get("s_c_96") is not None]
    for r in c96:
        out.append(f"- 96 control {r['ray']}: s_c(96) {r['s_c_96']:.5g}, ln(96/64) {r['dln_96_64']:+.4f}; err_96 {r['err_96']:+.4f}, "
                   f"shift_96 {r['shift_96']:+.4f}")
    if R.get("pilot"):
        out.append("\nPilot (s = 3 s_c(400), T = 2000, no absorption):\n")
        out.append("| Re | ray | t first E ≥ E_B | E min after | E median after | fraction below E_B after | non-finite | dt used | Courant max |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for p in R["pilot"]:
            out.append(f"| {p['Re']:g} | {p['ray']} | {f(p.get('t_first_EB'), 0)} | {f(p.get('E_min_after'))} | {f(p.get('E_median_after'))} | "
                       f"{f(p.get('frac_below_EB_after'))} | {p['n_nonfinite']} | {p['dt_used']:g} | {p['courant_max']:.2f} |")
    out.append(f"\ncounts 64: {R.get('counts_64')}")
    out.append(f"96 files: {R.get('truth_96_files')}")
    return "\n".join(out)


def figure(R, out):
    rows = [r for r in R["rows"] if r.get("s_c") is not None]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.0))
    # (a) prediction against truth at both Re
    a = ax[0]
    for r in rows:
        c = COL[r["ray"].split("#")[0]]
        a.plot(r["s_c_400"], r["s_slow_400"], "o", mfc="white", mec=c, ms=7)
        a.plot(r["s_c"], r["s_slow"], "o", color=c, ms=7)
        a.annotate("", xy=(r["s_c"], r["s_slow"]), xytext=(r["s_c_400"], r["s_slow_400"]), arrowprops=dict(arrowstyle="->", color=c, lw=0.8, alpha=0.6))
        a.annotate(r["ray"], (r["s_c"], r["s_slow"]), textcoords="offset points", xytext=(-4, 6), fontsize=7, ha="right")
    v = [x for r in rows for x in (r["s_c"], r["s_slow"], r["s_c_400"], r["s_slow_400"])]
    lim = (min(v) / 1.3, max(v) * 1.3)
    a.plot(lim, lim, "k-", lw=0.8)
    a.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="truth s_c", ylabel="slow rule s_slow = max(s_i, s_ii)",
          title=f"(a) open: Re = {R['Re0']:g} (truth 96); filled: Re = {R['Re']:g} (truth 64)")
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for k, c in COL.items():
        a.plot([], [], "o", color=c, label=k)
    a.legend(frameon=True, fontsize=8, loc="upper left")
    # (b) errors per ray
    a = ax[1]
    x = np.arange(len(rows))
    a.bar(x, [-r["shift"] for r in rows], color="#BBBBBB", width=0.6, label="'no change with Re': ln(s_c(400)/s_c(500))")
    a.plot(x, [r["err_400"] for r in rows], "o", mfc="white", mec="k", ms=7, label=f"rule at Re = {R['Re0']:g}")
    a.plot(x, [r["err"] for r in rows], "o", color="k", ms=6, label=f"rule at Re = {R['Re']:g}")
    a.axhline(0, color="k", lw=0.8)
    a.set_xticks(x)
    a.set_xticklabels([r["ray"] for r in rows], rotation=45, fontsize=8)
    a.set(ylabel="ln(prediction / s_c)", title="(b) error of the slow rule and of 'no change'")
    a.legend(frameon=True, fontsize=8, loc="best")
    # (c) each condition alone, in s (ln(s_cond / s_c)); Re = 400 values from the Gate 9 predictions
    a = ax[2]
    P9 = {q["ray"]: q for q in json.loads((Path(__file__).resolve().parents[2] / "experiments/20261119_gate9_dns_v1/predictions_9.json").read_text())["rays"]}
    ei4 = [math.log(P9[r["ray"]]["s_i"] / r["s_c_400"]) for r in rows]
    eii4 = [math.log(P9[r["ray"]]["s_ii"] / r["s_c_400"]) for r in rows]
    a.plot(x - 0.12, ei4, "s", mfc="white", mec="#C44E52", ms=7, label=f"s_i (S* fixed), Re = {R['Re0']:g}")
    a.plot(x - 0.12, [r["err_i"] for r in rows], "s", color="#C44E52", ms=6, label=f"s_i (S* fixed), Re = {R['Re']:g}")
    a.plot(x + 0.12, eii4, "^", mfc="white", mec="#4C72B0", ms=7, label=f"s_ii (c* fixed), Re = {R['Re0']:g}")
    a.plot(x + 0.12, [r["err_ii"] for r in rows], "^", color="#4C72B0", ms=6, label=f"s_ii (c* fixed), Re = {R['Re']:g}")
    a.axhline(0, color="k", lw=0.8)
    a.set_xticks(x)
    a.set_xticklabels([r["ray"] for r in rows], rotation=45, fontsize=8)
    a.set(ylabel="ln(s_condition / s_c)", title="(c) each condition alone (s_slow = the larger)")
    a.legend(frameon=True, fontsize=7, loc="best")
    for a_ in (ax[1], ax[2]):
        for k, r in enumerate(rows):
            if r.get("t_hit_hi") is not None and r["t_hit_hi"] < 600:
                a_.axvspan(k - 0.45, k + 0.45, color="#F2C14E", alpha=0.25, lw=0)
    for a_ in (ax[1], ax[2]):
        a_.set_xlabel(f"shaded: truth fast at Re = {R['Re']:g} (t_hit_hi < 600)", fontsize=8)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Slow route at Re = {R['Re']:g}: S* and c* kept from Re = {R['Re0']:g}, theory recomputed", fontsize=12)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig17_slowre{int(R['Re'])}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = json.loads((Path(a.run) / "results.json").read_text())
    (Path(a.run) / "tables.md").write_text(table(R) + "\n")
    print(table(R))
    figure(R, Path(a.out))


if __name__ == "__main__":
    main()
