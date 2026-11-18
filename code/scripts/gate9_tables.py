#!/usr/bin/env python
"""Gate 9.4 tables (Markdown) from <run>/results.json, for the report; no new statistics are computed here.

    python code/scripts/gate9_tables.py --g9 experiments/20261119_gate9_dns_v1 > <run>/tables_9_4.md
"""
import argparse
import json
from pathlib import Path


def f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def ci(c, d=3):
    return "—" if c is None else f"[{c[0]:.{d}f}, {c[1]:.{d}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g9", required=True)
    a = ap.parse_args()
    R = json.loads((Path(a.g9) / "results.json").read_text())
    G = R["groups"]
    out = []
    out.append("### (1) ln(s_pred / s_c(96)) by class and range\n")
    out.append("| group | n | median | IQR [q25, q75] | 90% quantile of abs | max abs | abs median (bootstrap 95%) | censored |")
    out.append("|---|---|---|---|---|---|---|---|")
    for k, b in G.items():
        s = b["rule"]
        cen = ", ".join(f"{c['ray']} (s_c {c['side']} {c['bound']:.4g})" for c in b["censored"]) or "—"
        if s["n"] == 0:
            out.append(f"| {k} | 0 | — | — | — | — | — | {cen} |")
            continue
        out.append(f"| {k} | {s['n']} | {f(s['median'])} | {f(s['IQR'])} [{f(s['q25'])}, {f(s['q75'])}] | {f(s['abs_q90'])} | {f(s['abs_max'])} | "
                   f"{f(s['abs_median'])} {ci(s['abs_median_boot95'])} | {cen} |")
    out.append("\n### (2) coverage (in range; censored rays count as not covered)\n")
    out.append("| group | 50%: k / n (rate) | binomial p vs 0.5 | 80%: k / n (rate) | binomial p vs 0.8 | censored in n |")
    out.append("|---|---|---|---|---|---|")
    for k, b in G.items():
        if not k.endswith("/in"):
            continue
        c5, c8 = b["coverage_p50"], b["coverage_p80"]
        if c5["n"] == 0:
            out.append(f"| {k} | 0 | — | 0 | — | 0 |")
            continue
        out.append(f"| {k} | {c5['k']} / {c5['n']} ({c5['rate']:.2f}) | {c5['binom_p']:.3g} | {c8['k']} / {c8['n']} ({c8['rate']:.2f}) | {c8['binom_p']:.3g} | "
                   f"{c5['n_censored_in_n']} |")
    out.append("\n### (3) paired |err_rule| − |err_baseline| (negative = the rule is closer)\n")
    out.append("| group | baseline | n | median (bootstrap 95%) | rule closer | baseline abs median | no baseline value |")
    out.append("|---|---|---|---|---|---|---|")
    for k, b in G.items():
        for bl in ("gate8a", "two_cond", "energy"):
            p = b[f"paired_vs_{bl}"]
            if p["n"] == 0:
                continue
            out.append(f"| {k} | {bl} | {p['n']} | {f(p['median'])} {ci(p['boot95'])} | {p['n_rule_better']} / {p['n']} | {f(b[bl].get('abs_median'))} | "
                       f"{b[bl]['n_no_prediction']} |")
    out.append("\n### (4) range rule (out of range: s_c >= 0.8 s_pred)\n")
    out.append("| group | k / n |")
    out.append("|---|---|")
    for k, b in G.items():
        if k.endswith("/out"):
            out.append(f"| {k} | {b['bound']['k']} / {b['bound']['n']} |")
    out.append("\n### (5) route table (truth fast = t_hit_hi < 600)\n")
    out.append("| | true fast | true slow |")
    out.append("|---|---|---|")
    T = R["routes"]
    for a_ in ("fast", "slow"):
        out.append(f"| pred {a_} | {T[f'pred {a_} / true fast']} | {T[f'pred {a_} / true slow']} |")
    out.append("\n### per ray\n")
    out.append("| ray | range | route pred / true | s_pred | s_c(96) [lo, hi] | t_hit_hi | ln err | u (d_grid source) | 50% / 80% | bound | "
               "8a / 2-cond / energy ln err | status |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in R["rows"]:
        if r.get("s_c") is not None:
            sc = f"{r['s_c']:.5g} [{r['lo']:.5g}, {r['hi']:.5g}]"
        elif "censored" in r:
            sc = f"{r['censored']['side']} {r['censored']['bound']:.4g} (censored)"
        else:
            sc = "—"
        cov = "/".join(("yes" if r["cover"][p] else "no") for p in ("p50", "p80")) if "cover" in r else "—"
        eb = r.get("err_base")
        ebs = " / ".join(f(eb[k]) for k in ("gate8a", "two_cond", "energy")) if eb else "—"
        sp = "—" if r["s_pred"] is None else f"{r['s_pred']:.4g}"
        th = "—" if r.get("t_hit_hi") is None else f"{r['t_hit_hi']:.0f}"
        out.append(f"| {r['ray']} | {'in' if r['in_range'] else 'out'} | {r['route_pred']} / {r.get('route_true') or '—'} | {sp} | {sc} | {th} | "
                   f"{f(r.get('err'))} | {f(r.get('u'), 4)} ({r.get('d_grid_source', '—')}) | {cov} | "
                   f"{'—' if r.get('bound_ok') is None else ('ok' if r['bound_ok'] else 'violated')} | {ebs} | {r['status']} |")
    out.append("\n### grid 96 -> 128\n")
    for c, d in R["grid_128"].items():
        out.append(f"- {c}: {d['ray']} s_c 96 {d['s_c_96']:.5g}, 128 {d['s_c_128']:.5g}, Δln {d['dln']:+.4f}")
    out.append("\n### flagged / non-monotone / no prediction\n")
    out.append(f"- flagged: {R['flagged']}")
    out.append(f"- non-monotone: {R['non_monotone']}")
    out.append(f"- no prediction: {R['no_prediction']}")
    if R.get("budget"):
        out.append(f"- budget decisions: {R['budget'].get('decisions')}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
