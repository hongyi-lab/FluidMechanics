#!/usr/bin/env python
"""One-page project overview for non-specialists (English, white background). Descriptive only: every number is read from
the run folders; nothing is fitted here. Re-run after Gate 7 finishes to refresh panel 6.

    python code/scripts/overview_figure.py --out docs/reports/figs/overview/overview_en
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from committor.dns_theory import A_of  # noqa: E402
from committor.plotting import STYLE  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments"
PH = EXP / "20261118_gate56_posthoc_fast_v1"
G5 = EXP / "20261117_gate5_dns_v1"
G4B = EXP / "20261117_gate4b_dns_v2"
G7 = EXP / "20261118_gate7_dns_v1"
TR = EXP / "20261118_overview_traces_v1"
G7_DONE = (G7 / "results.json").exists()

S_STAR = 0.16552964560447833
R_A, E_B = 5e-7, 0.24
C_SLOW, C_FAST = "#4C72B0", "#DD8452"
C_A, C_B = "#3B75AF", "#C44E52"
MK = {"F": "o", "R": "s", "W": "^"}
LADDER_NAMES = ["F eps=1e-06 draw 0", "F eps=0.01 draw 0", "F eps=1 draw 0", "R #1", "R #13", "W #1"]
DECISION = ["W#1", "W#3", "W#7", "W#9", "R#22"]
CONTROLS = ["R#15", "W#11"]
RAYCOL = {"W#1": "#C44E52", "W#3": "#DD8452", "W#7": "#8C564B", "W#9": "#E377C2", "R#22": "#2CA02C",
          "R#15": "#4C72B0", "W#11": "#7F7F7F"}
FONT = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Noto Sans CJK SC"]}
BOX = dict(fc="white", ec="#DDDDDD", lw=0.6, alpha=0.92)


def pct(e):
    """|ln(pred/true)| -> deviation in percent: larger/smaller - 1."""
    return 100.0 * (math.exp(abs(e)) - 1.0)


def s_i_from_theory(f, name):
    Z = np.load(f)
    names = [str(n) for n in Z["names"]]
    if name not in names:
        return None
    i = names.index(name)
    sg = np.sqrt(np.geomspace(R_A * (1 + 1e-9), E_B * (1 - 1e-9), 4000))
    S = A_of(Z["aa"][i][None], Z["ab"][i][None], Z["bb"][i][None], sg)[0].max(axis=1)
    k = np.nonzero(S >= S_STAR)[0]
    if len(k) == 0 or k[0] == 0:
        return None
    j = k[0]  # log-linear interpolation between sg[j-1] and sg[j]
    x0, x1, y0, y1 = math.log(sg[j - 1]), math.log(sg[j]), math.log(S[j - 1]), math.log(S[j])
    return math.exp(x0 + (math.log(S_STAR) - y0) * (x1 - x0) / (y1 - y0)), sg, S


def gate7_state():
    """ray -> level -> dict(lo, hi, done). 64 from Gate 5, 96 W#1 from the Gate 4b ladder, the rest from Gate 7 files
    (finished, or the current bracket of a search still running)."""
    T5 = np.load(G5 / "truth_RW.npz")
    lad = np.load(G4B / "ckpt_sc_96x48x48_dt0.0625.npz")
    out = {}
    for n in DECISION + CONTROLS:
        s, k = n.split("#")
        i = int(k) - 1 if s == "R" else 24 + int(k) - 1
        out[n] = {"64": {"lo": float(T5["lo"][i]), "hi": float(T5["hi"][i]), "done": True}}
    j = LADDER_NAMES.index("W #1")
    out["W#1"]["96"] = {"lo": float(lad["lo"][j]), "hi": float(lad["hi"][j]), "done": True}
    for f in sorted(G7.glob("truth_*.json")):
        lv = f.stem.split("_")[1]
        d = json.loads(f.read_text())
        for r in d["rays"]:
            n = r["name"]
            if n not in out:
                continue
            if r.get("s_c") is not None:
                out[n][lv] = {"lo": r["lo"], "hi": r["hi"], "done": True}
            elif r.get("lo") is not None and r.get("hi") is not None:
                out[n][lv] = {"lo": r["lo"], "hi": r["hi"], "done": False}
    return out


def timeline(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")
    items = [
        ("Gate 0", "Method tested on a 2D toy\nmodel with a known answer", "done"),
        ("Gate 1", "9-mode reduced model (MFE)\nbuilt; matches the literature", "done"),
        ("Gate 2–3d", "Criterion found in MFE:\n\"two gates\", median error ≈ 4%", "done"),
        ("Gate 4", "Full 3D simulation (DNS)\nbuilt; solver verified", "done"),
        ("Gate 5–6", "Criterion tested in DNS:\n38 of 52 shapes, median error 1%", "part"),
        (("Gate 7", "The other 14 shapes: grid\nartefact or a real second route?", "undet") if G7_DONE else
         ("Gate 7 (running)", "The other 14 shapes: grid\nartefact or a real second route?", "run")),
    ]
    col = {"done": ("#E8F3E8", "#2E7D32", "✓ done"), "part": ("#FFF4E0", "#E08A00", "◐ partly holds"),
           "run": ("#E8F0FB", "#1F5FAF", "… running"), "undet": ("#F1EEF6", "#5E4B8B", "? undetermined")}
    w, gap, x = 14.6, 2.3, 0.6
    for k, (head, body, st) in enumerate(items):
        fc, ec, tag = col[st]
        ax.add_patch(FancyBboxPatch((x, 0.3), w, 9.2, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, 8.2, head, ha="center", va="center", fontsize=12.5, weight="bold", color=ec)
        ax.text(x + w / 2, 5.0, body, ha="center", va="center", fontsize=10.3, color="#222222", linespacing=1.45)
        ax.text(x + w / 2, 1.5, tag, ha="center", va="center", fontsize=10, color=ec)
        if k < len(items) - 1:
            ax.annotate("", xy=(x + w + gap - 0.15, 4.9), xytext=(x + w + 0.35, 4.9),
                        arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.4))
        x += w + gap


def panel_traces(ax):
    if not (TR / "traces_64.npz").exists():
        ax.text(0.5, 0.5, "(traces running)", transform=ax.transAxes, ha="center")
        return
    Z = np.load(TR / "traces_64.npz")
    t, E = Z["t"], Z["rec"][:, :, 0]
    names, which, s = [str(x) for x in Z["names"]], [str(x) for x in Z["which"]], Z["s"]
    sty = {("R#15", "lo"): (C_A, "-", 2.0), ("R#15", "hi"): (C_B, "-", 2.0),
           ("W#1", "lo"): (C_A, "--", 1.4), ("W#1", "hi"): (C_B, "--", 1.4)}
    for k in range(len(names)):
        c, ls, lw = sty[(names[k], which[k])]
        e = np.where(E[k] > 0, E[k], np.nan)
        ax.plot(t, e, color=c, ls=ls, lw=lw)
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 10)
    ax.set_xlim(0, t[-1])
    ax.axhline(E_B, color="#999999", lw=0.8, ls=":")
    ax.axhline(R_A, color="#999999", lw=0.8, ls=":")
    ax.text(t[-1] * 0.99, 1.6, "turbulent", ha="right", va="bottom", fontsize=10, color=C_B, weight="bold")
    ax.text(2000, R_A / 1.5, "back to laminar", ha="left", va="top", fontsize=10, color=C_A, weight="bold")
    idx = {(names[k], which[k]): k for k in range(len(names))}
    d1 = 100 * (s[idx[("R#15", "hi")]] - s[idx[("R#15", "lo")]]) / s[idx[("R#15", "lo")]]
    d2 = 100 * (s[idx[("W#1", "hi")]] - s[idx[("W#1", "lo")]]) / s[idx[("W#1", "lo")]]
    ax.text(0.97, 0.80, f"Solid: one shape, two starting\namplitudes only {d1:.2f}% apart:\none decays to laminar (blue),\n"
            f"the other turns turbulent (red).\nDashed: another shape, also\n{d2:.2f}% apart; the larger one turns\nturbulent much sooner.",
            transform=ax.transAxes, ha="right", va="top", multialignment="left", fontsize=9.2, linespacing=1.4, bbox=BOX)
    ax.set_xlabel("time")
    ax.set_ylabel("disturbance energy (log scale)")


def panel_rays(ax):
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.12, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    th = np.linspace(0, 2 * np.pi, 400)
    r = 0.68 + 0.11 * np.sin(3 * th) + 0.07 * np.cos(5 * th + 0.5)
    ax.add_patch(plt.Rectangle((-1.25, -1.12), 2.5, 2.57, fc="#FBE3E3", ec="none", zorder=0))
    ax.add_patch(Polygon(np.c_[r * np.cos(th), r * np.sin(th)], closed=True, fc="#DDEAF6", ec="#333333", lw=1.8, zorder=1))
    ax.text(0.24, 0.36, "back to\nlaminar", ha="center", va="center", fontsize=10.5, color=C_A, weight="bold", zorder=5, linespacing=1.1)
    ax.text(1.0, 0.62, "turbulent", ha="center", fontsize=10.5, color=C_B, weight="bold", zorder=5)
    ax.text(-1.18, -1.08, "boundary = the \"edge\"\n(any shape)", fontsize=9.3, color="#333333", zorder=5, va="bottom")
    for a in [20, 95, 165, 250, 305]:
        u = np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
        ax.annotate("", xy=1.12 * u, xytext=0 * u, arrowprops=dict(arrowstyle="-|>", color="#666666", lw=1.1), zorder=2)
    a = math.radians(20)
    u = np.array([math.cos(a), math.sin(a)])
    rc = 0.68 + 0.11 * math.sin(3 * a) + 0.07 * math.cos(5 * a + 0.5)
    for rr, f in [(0.40, "A"), (1.45, "B"), (0.85, "A"), (1.18, "B"), (0.95, "A"), (1.05, "B")]:
        ax.plot(*(rr * rc * u), "o", ms=7, color=C_A if f == "A" else C_B, mec="white", mew=0.8, zorder=6)
    ax.plot(*(rc * u), marker="*", ms=17, color="#FFC107", mec="#333333", mew=0.8, zorder=7)
    ax.annotate("critical\namplitude $s_c$", xy=rc * u, xytext=(0.62, -0.42), fontsize=10, zorder=8, linespacing=1.1,
                arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9))
    ax.plot(0, 0, "o", ms=9, color="black", zorder=8)
    ax.annotate("no disturbance", xy=(-0.03, 0.03), xytext=(-1.15, 0.72), ha="left", va="top", fontsize=9.3, zorder=8,
                arrowprops=dict(arrowstyle="-", color="#333333", lw=0.7, shrinkB=4))
    ax.text(-1.18, 1.40, "Each arrow = one disturbance shape (52 in total).\nRaise the amplitude along it: blue = back to laminar,\n"
            "red = turbulent; bisect to 0.1%.\nEvery dot is one full 3D simulation.",
            fontsize=9.2, va="top", linespacing=1.4, zorder=8, bbox=BOX)


def panel_criterion(ax, rec):
    f = G7 / "theory_64.npz"
    for n, c, lab in (("R#15", C_SLOW, "shape R#15 (typical)"), ("W#1", C_B, "shape W#1 (fast channel)")):
        si, sg, S = s_i_from_theory(f, n)
        ax.plot(sg, S, color=c, lw=2.0, label=lab)
        sc = rec[n]["s_c_dns"]
        Sc = float(np.interp(math.log(sc), np.log(sg), S))
        ax.plot(sc, Sc, "*", ms=15, color="#FFC107", mec="#333333", mew=0.8, zorder=6)
        ax.plot(si, S_STAR, "o", ms=8, color=c, mec="white", mew=1.0, zorder=6)
    ax.axhline(S_STAR, color="#333333", lw=1.0, ls="--", label="threshold S* (a constant, set once)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2e-3, 0.2)
    ax.set_ylim(3e-3, 25)
    ax.set_xlabel("initial disturbance amplitude s")
    ax.set_ylabel("peak streak strength S")
    ax.plot([], [], "*", ms=12, color="#FFC107", mec="#333333", label="true critical point (DNS bisection)")
    ax.plot([], [], "o", ms=7, color="#888888", label="criterion's prediction (S reaches S*)")
    ax.legend(loc="lower right", fontsize=8.8, frameon=False)
    ax.text(0.03, 0.97, "Physics (lift-up): streamwise vortices lift slow fluid up\nand push fast fluid down, forming \"streaks\";\n"
            "strong enough streaks break down into turbulence.\nThe criterion needs one cheap calculation, no full simulation.\n"
            "R#15: prediction = truth. W#1: streaks still weak at the truth.",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.0, linespacing=1.4, bbox=BOX)


def panel_scatter(ax, rays):
    for r in rays:
        x = r["s_c_dns"]
        y = x * math.exp(r["ln_err"])
        ax.plot(x, y, MK[r["ray"][0]], ms=7, color=C_FAST if r["fast"] else C_SLOW, mec="white", mew=0.6, alpha=0.95)
    lim = [2e-3, 0.3]
    ax.plot(lim, lim, color="#333333", lw=1.0)
    ax.fill_between(lim, [v / 1.1 for v in lim], [v * 1.1 for v in lim], color="#BBBBBB", alpha=0.35, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("true critical amplitude (full 3D simulation)")
    ax.set_ylabel("critical amplitude predicted by the criterion")
    slow = [r for r in rays if not r["fast"]]
    fast = [r for r in rays if r["fast"]]
    ms = np.median([pct(r["ln_err"]) for r in slow])
    fr = [math.exp(abs(r["ln_err"])) for r in fast]
    ax.plot([], [], "o", color=C_SLOW, ms=7, label=f"{len(slow)} shapes: median error {ms:.0f}%")
    ax.plot([], [], "o", color=C_FAST, ms=7, label=f"{len(fast)} shapes: off by a factor {min(fr):.1f}–{max(fr):.1f}")
    ax.plot([], [], color="#BBBBBB", lw=6, alpha=0.6, label="±10% band")
    for k, lab in (("o", "F: mostly a streamwise vortex"), ("s", "R: random mix"), ("^", "W: pure waves (no streamwise vortex)")):
        ax.plot([], [], k, color="#777777", ms=6, label=lab)
    ax.legend(loc="upper left", fontsize=8.8, frameon=False)
    ax.text(0.97, 0.03, "on the diagonal = exact prediction", transform=ax.transAxes, ha="right", fontsize=9.2, color="#333333")


def panel_fast(ax, rays, mfe_range):
    for r in rays:
        ax.plot(r["t_hit_hi"], max(pct(r["ln_err"]), 0.03), MK[r["ray"][0]], ms=7, color=C_FAST if r["fast"] else C_SLOW,
                mec="white", mew=0.6, alpha=0.95)
    ax.axvline(600, color="#333333", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(120, 3000)
    ax.set_ylim(0.03, 3000)
    ax.set_xticks([150, 300, 600, 1200, 2400])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([0.1, 1, 10, 100, 1000])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("time for a just-supercritical disturbance to become turbulent")
    ax.set_ylabel("prediction error (%)")
    slow = [r for r in rays if not r["fast"]]
    fast = [r for r in rays if r["fast"]]
    fmin = min(pct(r["ln_err"]) for r in fast)
    sm, sx = np.median([pct(r["ln_err"]) for r in slow]), max(pct(r["ln_err"]) for r in slow)
    ax.text(0.03, 0.97, f"fast (< 600): {len(fast)} shapes\nevery error ≥ {fmin:.0f}%", transform=ax.transAxes, ha="left", va="top",
            fontsize=10, color=C_FAST, weight="bold")
    ax.text(0.97, 0.97, f"slow (> {min(r['t_hit_hi'] for r in slow):.0f}): {len(slow)} shapes\nmedian {sm:.0f}%, max {sx:.0f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color=C_SLOW, weight="bold")
    ax.text(0.03, 0.04, f"For comparison: the 9-mode model\nhas no fast channel. The same 12 W\nshapes take {mfe_range[0]:.0f}–{mfe_range[1]:.0f} there.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color="#444444", linespacing=1.4)


def panel_gate7(ax):
    st = gate7_state()
    lvls = ["64", "96", "128"]
    si = {}
    for lv in lvls:
        f = G7 / f"theory_{lv}.npz"
        if f.exists():
            for n in st:
                v = s_i_from_theory(f, n)
                if v is not None:
                    si[(n, lv)] = v[0]
    notes, pending = [], []
    for n in DECISION + CONTROLS:
        xs, ys, yl, yh, done = [], [], [], [], []
        for k, lv in enumerate(lvls):
            if lv not in st[n]:
                continue
            q = st[n][lv]
            if not q["done"] and (q["hi"] - q["lo"]) / q["lo"] > 0.10:
                pending.append(n)
                continue
            s_i = si.get((n, lv))
            if s_i is None:  # theory at this level not finished: use the nearest lower level (listed in the printout)
                lower = [l2 for l2 in lvls[:k][::-1] if (n, l2) in si]
                s_i = si[(n, lower[0])]
                notes.append(f"{n} {lv}")
            e_lo, e_hi = math.log(s_i / q["hi"]), math.log(s_i / q["lo"])
            e_mid = math.log(s_i / (0.5 * (q["lo"] + q["hi"])))
            xs.append(k)
            ys.append(pct(e_mid) * np.sign(e_mid))
            yl.append(pct(e_lo) * np.sign(e_lo))
            yh.append(pct(e_hi) * np.sign(e_hi))
            done.append(q["done"])
        ctrl = n in CONTROLS
        c = RAYCOL[n]
        off = (DECISION + CONTROLS).index(n) * 0.035 - 0.1
        xa = np.array(xs) + off
        ax.plot(xa, ys, color=c, lw=1.0 if ctrl else 1.8, ls="--" if ctrl else "-", zorder=3, label=n + (" (control, slow)" if ctrl else ""))
        for x, y, a, b, d in zip(xa, ys, yl, yh, done):
            ax.plot([x, x], [a, b], color=c, lw=1.2, zorder=3)
            ax.plot(x, y, "o", ms=7, color=c if d else "white", mec=c, mew=1.5, zorder=4)
    ax.axhspan(-100 * (math.exp(0.10) - 1), 100 * (math.exp(0.10) - 1), color="#DDDDDD", alpha=0.7, lw=0, zorder=1)
    ax.axhline(100 * (math.exp(0.25) - 1), color="#333333", ls=":", lw=1.0)
    ax.text(1.05, 17, "if it were a grid artefact, the error would drop into the grey band (≤ 10%)", fontsize=8.8, ha="center",
            va="center", color="#444444")
    ax.text(2.45, 100 * (math.exp(0.25) - 1) + 4, "\"real route\" needs ≥ 28% and a threshold\nthat no longer moves on a finer grid",
            fontsize=8.8, ha="right", va="bottom", color="#444444")
    no128 = [n for n in DECISION if "128" not in st[n]]
    if G7_DONE and no128:
        ax.text(2.05, 112, "no fine-grid run\n(time budget):\n" + ", ".join(no128), fontsize=9, ha="center", va="center",
                color="#666666", style="italic")
    if pending:
        ax.text(2.05, 100, "finest grid:\n" + ", ".join(sorted(set(pending))) + "\nstill running", fontsize=9, ha="center",
                va="center", color="#666666", style="italic")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["coarse\n64×32×32", "medium\n96×48×48", "fine\n128×64×64"])
    ax.set_xlim(-0.35, 2.5)
    ax.set_ylim(-25, 175)
    ax.set_ylabel("prediction error (%)")
    ax.set_xlabel("computational grid (finer to the right)")
    if pending:
        ax.plot([], [], "o", ms=7, color="white", mec="#555555", mew=1.5, label="open: still bisecting\n(bar = current bracket)")
    ax.legend(loc="upper right", fontsize=8.4, frameon=False, ncol=2)
    return st, si, notes + [f"pending {p}" for p in pending]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "docs/reports/figs/overview/overview_en"))
    ap.add_argument("--stamp", default="")
    a = ap.parse_args()
    P = json.loads((PH / "results.json").read_text())
    rays = P["rays"]
    rec = {r["ray"]: r for r in rays}
    O = json.loads((PH / "mfe_offsets.json").read_text())
    j = O["deltas"].index(1e-3)
    tm = [v[j] for k, v in O["t_hit"]["E_B=0.29"].items() if k.startswith("W")]  # MFE, the 12 W rays at 1.001 s_c
    assert len(tm) == 12
    with plt.rc_context({**STYLE, **FONT}):
        fig = plt.figure(figsize=(20, 13.8))
        gs = fig.add_gridspec(4, 3, height_ratios=[0.95, 2.6, 2.6, 0.45], hspace=0.44, wspace=0.27,
                              left=0.05, right=0.985, top=0.9, bottom=0.035)
        fig.text(0.5, 0.977, "When does a shear flow turn turbulent? One simple criterion instead of costly 3D simulations: project overview",
                 ha="center", va="center", fontsize=18, weight="bold")
        fig.text(0.5, 0.944, "Problem: the flow between two plates (Re = 400) is smooth and laminar. A small disturbance dies out; a large enough one "
                 "makes the flow turbulent. The critical size depends on the disturbance's shape.\nGoal: a physics-based formula that gives "
                 "the critical size for any shape directly, without running a full 3D simulation each time.",
                 ha="center", va="center", fontsize=11.5, color="#333333", linespacing=1.5)
        timeline(fig.add_subplot(gs[0, :]))
        axs = [fig.add_subplot(gs[1, k]) for k in range(3)] + [fig.add_subplot(gs[2, k]) for k in range(3)]
        panel_traces(axs[0])
        panel_rays(axs[1])
        panel_criterion(axs[2], rec)
        panel_scatter(axs[3], rays)
        panel_fast(axs[4], rays, (min(tm), max(tm)))
        st, si, notes = panel_gate7(axs[5])
        titles = ["① What happens: a tiny difference, opposite outcomes", "② Method: find the critical amplitude in 52 directions",
                  "③ The criterion: turbulence needs strong streaks", "④ Result: most shapes are predicted well",
                  "⑤ Pattern: the misses are the \"fast\" transitions",
                  "⑥ Gate 7: is it a coarse-grid artefact?" if G7_DONE else "⑥ Now (Gate 7): is it a coarse-grid artefact?"]
        for ax, tt in zip(axs, titles):
            ax.set_title(tt, fontsize=13, weight="bold", loc="left", pad=10)
        up = []
        for n in DECISION:
            if "96" in st[n] and st[n]["96"]["done"]:
                q64, q96 = st[n]["64"], st[n]["96"]
                up.append(0.5 * (q96["lo"] + q96["hi"]) / (0.5 * (q64["lo"] + q64["hi"])) - 1)
        if G7_DONE:
            V = json.loads((G7 / "results.json").read_text())
            top = [V["rays"][n][V["rays"][n]["highest"]]["err_i"] for n in DECISION]
            e64 = [V["rays"][n]["64"]["err_i"] for n in DECISION]
            d128 = [abs(V["rays"][n]["dln_top2"]) for n in DECISION if V["rays"][n]["highest"] == "128"]
            nreal, verdict = len(V["verdict"]["real_route_rays"]), V["verdict"]["verdict"]
            no128 = [n for n in DECISION if V["rays"][n]["highest"] != "128"]
            concl = ("Result: the criterion works for most disturbance shapes (about 1% error). It misses one class, the shapes that turn turbulent "
                     f"fast. On a finer grid their true threshold moves up by {100 * min(up):.0f}–{100 * max(up):.0f}%, then stays put\n"
                     f"(the {len(d128)} shapes run on the finest grid move by ≤ {100 * (math.exp(max(d128)) - 1):.1f}% more). The prediction is then "
                     f"{math.exp(min(top)):.2f}–{math.exp(max(top)):.2f}× the truth instead of {math.exp(min(e64)):.2f}–{math.exp(max(e64)):.2f}×: "
                     "smaller, but far from the grid-artefact band.\n"
                     f"Verdict by the rules fixed in advance: {verdict}. {nreal} of the {len(DECISION)} test shapes meet \"real second route\" "
                     f"(4 needed); {', '.join(no128)} had no finest-grid run within the time budget. Both control shapes pass.")
        else:
            concl = ("So far: the criterion works for most disturbance shapes (about 1% error). It misses one class, the shapes that turn turbulent fast. "
                 "On a finer grid their true threshold\nmoves up" + (f" by {100 * min(up):.0f}–{100 * max(up):.0f}%" if up else "")
                 + " and the error roughly halves, but it does not drop to the artefact level. For now it looks partly like a grid effect and partly "
                 "like a real second route\nto turbulence. Gate 7 is still running; the final verdict will follow the criteria fixed in advance.")

        ax_c = fig.add_subplot(gs[3, :])
        ax_c.axis("off")
        ax_c.text(0.5, 0.5, concl, ha="center", va="center", fontsize=12, color="#1F3B5C", linespacing=1.6,
                  bbox=dict(boxstyle="round,pad=0.6", fc="#EEF4FB", ec="#1F5FAF", lw=1.2), transform=ax_c.transAxes)
        fig.text(0.05, 0.004, "Flow: sinusoidal shear flow driven by a body force, free-slip walls, Re = 400, domain 4π × 2 × 2π (the flow of the "
                 "Moehlis–Faisst–Eckhardt 9-mode model). Error = larger / smaller of (prediction, truth) − 1. "
                 "Data: Gate 5–7 run folders under experiments/.", ha="left", va="bottom", fontsize=8.5, color="#777777")
        if a.stamp:
            fig.text(0.985, 0.004, a.stamp, ha="right", va="bottom", fontsize=8.5, color="#777777")
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.with_suffix(".png"), dpi=170, facecolor="white")
        fig.savefig(out.with_suffix(".pdf"), facecolor="white")
        plt.close(fig)
    print(json.dumps({"gate7_state": st, "s_i": {f"{k[0]} {k[1]}": v for k, v in si.items()}, "s_i_fallback_and_pending": notes,
                      "shift_96_vs_64": up}, indent=1, default=str))


if __name__ == "__main__":
    main()
