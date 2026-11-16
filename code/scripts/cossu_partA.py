#!/usr/bin/env python
"""Part A: threshold curves of Cossu, Brandt, Bagheri & Henningson (2011), Phys. Fluids 23, 074103.

Source: the authors' preprint (dated 16 June 2011) from mech.kth.se/~luca/papers/AUAW.pdf, stored as
literature/papers/Cossu_Brandt_Bagheri_Henningson_2011_PoF_23_074103.pdf. All plots in it are gnuplot vector graphics.

Step 1 (extraction, vector graphics, no raster digitisation):
  pdftocairo -svg renders pages 4, 6 and 8 losslessly. gnuplot draws each 'triangle' point as a closed
  triangle plus a zero-length round-capped dot ('Pnt') at the exact data position; the dot is taken as the
  data point (the triangle is used only to confirm it is a marker). The lower-branch saddle (filled circle)
  is a filled Bezier circle; its centre is taken. The axes are calibrated on the panels' own major ticks:
  each tick on the left/bottom frame edge is paired with the nearest numeric tick label found by
  `pdftotext -bbox` (text positions in the same page coordinates), and a straight line is fitted.
  Panels:
    fig2   toy model eq. (1), R = 300          : x = 100 u (streak, u = U1), y = 100 w_c
    fig5a  plane Couette, Re = 500             : x = 100 u_rms,  y = 100 w_rms
    fig5b  plane Couette, Re = 500             : x = A_U (%),    y = A_W (%)     [eq. (3)]
    fig9a  Blasius boundary layer, Re_d* = 700  : x = 100 u_rms,  y = 100 w_rms
    fig9b  Blasius boundary layer, Re_d* = 700  : x = A_U (%),    y = A_W (%)     [eq. (3)]
Step 2 (description): straight-line fits of ln A_W,crit against A_U on upper A_U ranges (every lower cut-off
  is listed), a continuous two-segment fit with a free breakpoint, the small-A_U end, and the point where
  A_W,crit = A_U (the paper's 'cross-over').
Step 3: figures (white background, PNG + PDF).

    python -I code/scripts/cossu_partA.py experiments/20261117_cossu_partA_v1
"""
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PDF = REPO / "literature/papers/Cossu_Brandt_Bagheri_Henningson_2011_PoF_23_074103.pdf"
# same style as committor.plotting (copied so the script runs with python -I)
STYLE = {"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False}
C_CO, C_BL, C_TOY, C_REF = "#4C72B0", "#C44E52", "#55A868", "#222222"

PANELS = {  # page -> panel names in top-to-bottom order of the frames that hold triangle markers
    4: ["fig2"], 6: ["fig5a", "fig5b"], 8: ["fig9a", "fig9b"]}
EXPECT = {  # tick labels as printed (checked against the rendered pages)
    "fig2": (list(range(0, 11)), [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]),
    "fig5a": (list(range(0, 17, 2)), list(range(0, 6))),
    "fig5b": (list(range(0, 36, 5)), list(range(0, 11, 2))),
    "fig9a": (list(range(0, 11)), list(range(0, 4))),
    "fig9b": (list(range(0, 36, 5)), list(range(0, 11, 2)))}
META = {
    "fig2": dict(flow="Waleffe toy model eq. (1)", Re="R = 300", x="100 u (= 100 U1, streak)", y="100 w_c"),
    "fig5a": dict(flow="plane Couette", Re="Re = 500", x="100 u_rms", y="100 w_rms"),
    "fig5b": dict(flow="plane Couette", Re="Re = 500", x="A_U (%)", y="A_W (%)"),
    "fig9a": dict(flow="Blasius boundary layer", Re="Re_delta* = 700", x="100 u_rms", y="100 w_rms"),
    "fig9b": dict(flow="Blasius boundary layer", Re="Re_delta* = 700", x="A_U (%)", y="A_W (%)")}


# ----------------------------------------------------------------------------------------------- extraction
def svg_paths(svg):
    """All drawn paths (outside <defs>) as dicts with style and subpaths in page coordinates (pt, y down)."""
    body = svg[svg.find("</defs>"):]
    out = []
    for p in re.findall(r"<path ([^>]*)/>", body):
        st = re.search(r'style="([^"]*)"', p)
        st = st.group(1) if st else ""
        d = re.search(r' d="([^"]*)"', p).group(1)
        tr = re.search(r'transform="matrix\(([^)]*)\)"', p)
        a, b, c, dd, e, f = [float(v) for v in tr.group(1).split(",")] if tr else (1, 0, 0, 1, 0, 0)
        toks = re.findall(r"[MLCZ]|-?[\d.]+(?:e-?\d+)?", d)
        subs, cur, i, has_c = [], [], 0, False
        while i < len(toks):
            t = toks[i]
            if t == "M":
                if cur:
                    subs.append(cur)
                cur = [(float(toks[i + 1]), float(toks[i + 2]))]
                i += 3
            elif t == "L":
                cur.append((float(toks[i + 1]), float(toks[i + 2])))
                i += 3
            elif t == "C":
                cur.append((float(toks[i + 5]), float(toks[i + 6])))
                has_c = True
                i += 7
            else:
                i += 1
        if cur:
            subs.append(cur)
        subs = [np.array([[a * x + c * y + e, b * x + dd * y + f] for x, y in s]) for s in subs]
        lw = re.search(r"stroke-width:([\d.]+)", st)
        out.append({"stroke": "stroke:none" not in st and "stroke:" in st, "fill": "fill:none" not in st,
                    "round": "stroke-linecap:round" in st, "curve": has_c, "subs": subs,
                    "lw": float(lw.group(1)) * abs(a) if lw else 0.0})
    return out


def numeric_words(page):
    xml = subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), "-bbox", str(PDF), "-"],
                         check=True, capture_output=True, text=True).stdout
    words = []
    for m in re.finditer(r'xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)<', xml):
        if re.fullmatch(r"-?\d+(\.\d+)?", m.group(5)):
            x0, y0, x1, y1 = (float(m.group(k)) for k in range(1, 5))
            words.append({"v": float(m.group(5)), "x0": x0, "x1": x1, "xc": 0.5 * (x0 + x1), "yc": 0.5 * (y0 + y1),
                          "y0": y0})
    return words


def extract_page(page, svg):
    P = svg_paths(svg)
    words = numeric_words(page)
    # frames: stroked axis-aligned rectangles (gnuplot draws the border twice -> dedupe)
    frames = {}
    for p in P:
        if not p["stroke"]:
            continue
        for s in p["subs"]:
            u = np.unique(np.round(s, 3), axis=0)
            if len(u) == 4 and len(np.unique(u[:, 0])) == 2 and len(np.unique(u[:, 1])) == 2 and np.ptp(u[:, 0]) > 50:
                frames[tuple(np.round([u[:, 0].min(), u[:, 0].max(), u[:, 1].min(), u[:, 1].max()], 2))] = 1
    tri, dots, circ, segs = [], [], [], []
    for p in P:
        for s in p["subs"]:
            if p["stroke"] and p["round"] and len(s) == 2 and np.allclose(s[0], s[1], atol=1e-9):
                dots.append(s[0])
            elif p["stroke"] and len(s) >= 3 and len(np.unique(np.round(s, 4), axis=0)) == 3 and not p["curve"]:
                tri.append(s)
            elif p["stroke"] and len(s) == 2 and not p["round"]:
                segs.append(s)
        if p["fill"] and not p["stroke"] and p["curve"]:
            pts = np.vstack(p["subs"])
            if np.ptp(pts[:, 0]) < 8 and np.ptp(pts[:, 1]) < 8:
                circ.append(0.5 * (pts.min(0) + pts.max(0)))
    tri_apex_x = np.array([t[0, 0] for t in tri])
    out = []
    for (x0, x1, y0, y1) in sorted(frames, key=lambda f: f[2]):
        pad = 6.0
        inside = lambda q: x0 - pad <= q[0] <= x1 + pad and y0 - pad <= q[1] <= y1 + pad  # noqa: E731
        md = [d for d in dots if inside(d) and len(tri_apex_x) and np.min(np.abs(tri_apex_x - d[0])) < 1e-3]
        if not md:
            continue
        # major ticks on the left (y) and bottom (x) edges
        yt, xt = {}, {}
        for s in segs:
            (xa, ya), (xb, yb) = s
            if abs(ya - yb) < 1e-6 and abs(min(xa, xb) - x0) < 0.05 and y0 - 0.1 <= ya <= y1 + 0.1:
                yt[round(ya, 4)] = max(yt.get(round(ya, 4), 0), abs(xb - xa))
            if abs(xa - xb) < 1e-6 and abs(max(ya, yb) - y1) < 0.05 and x0 - 0.1 <= xa <= x1 + 0.1:
                xt[round(xa, 4)] = max(xt.get(round(xa, 4), 0), abs(yb - ya))
        Ly, Lx = max(yt.values()), max(xt.values())
        yt = sorted(y for y, L in yt.items() if abs(L - Ly) < 0.05)
        xt = sorted(x for x, L in xt.items() if abs(L - Lx) < 0.05)
        wy = [w for w in words if x0 - 30 <= w["x1"] <= x0 + 0.5 and y0 - 10 <= w["yc"] <= y1 + 10]
        wx = [w for w in words if y1 - 0.5 <= w["y0"] <= y1 + 20 and x0 - 10 <= w["xc"] <= x1 + 10]

        def pair(ticks, ws, key):
            vals = []
            for t in ticks:
                j = int(np.argmin([abs(w[key] - t) for w in ws]))
                if abs(ws[j][key] - t) > 7:
                    raise RuntimeError(f"page {page}: no label near tick {t}")
                vals.append(ws[j]["v"])
            return vals
        vy, vx = pair(yt, wy, "yc"), pair(xt, wx, "xc")
        py, px = np.polyfit(yt, vy, 1), np.polyfit(xt, vx, 1)
        res_y = float(np.abs(np.polyval(py, yt) - vy).max())
        res_x = float(np.abs(np.polyval(px, xt) - vx).max())
        md = np.array(sorted(md, key=lambda d: d[0]))
        mc = [c for c in circ if inside(c)]
        out.append({"frame_pt": [x0, x1, y0, y1], "x_ticks_pt": xt, "x_tick_labels": vx, "y_ticks_pt": yt,
                    "y_tick_labels": vy, "x_cal_max_residual": res_x, "y_cal_max_residual": res_y,
                    "units_per_pt": [float(abs(px[0])), float(abs(py[0]))],
                    "points": np.c_[np.polyval(px, md[:, 0]), np.polyval(py, md[:, 1])].tolist(),
                    "points_pt": md.tolist(), "n_triangles_matched": len(md),
                    "lower_branch": [[float(np.polyval(px, c[0])), float(np.polyval(py, c[1]))] for c in mc],
                    "on_frame_edge": [bool(abs(d[1] - y0) < 0.3 or abs(d[0] - x1) < 0.3) for d in md]})
    return out


def extract():
    res = {}
    with tempfile.TemporaryDirectory() as td:
        for page, names in PANELS.items():
            subprocess.run(["pdftocairo", "-svg", "-f", str(page), "-l", str(page), str(PDF), f"{td}/p.svg"],
                           check=True)
            pan = extract_page(page, Path(f"{td}/p.svg").read_text())
            if len(pan) != len(names):
                raise RuntimeError(f"page {page}: found {len(pan)} marker panels, expected {len(names)}")
            for nm, p in zip(names, pan):
                ex, ey = EXPECT[nm]
                if sorted(p["x_tick_labels"]) != sorted(map(float, ex)) or sorted(p["y_tick_labels"]) != sorted(map(float, ey)):
                    raise RuntimeError(f"{nm}: tick labels {p['x_tick_labels']} {p['y_tick_labels']} unexpected")
                res[nm] = {"page": page, **META[nm], **p}
    return res


# ----------------------------------------------------------------------------------------------- description
def linfit(x, y):
    """OLS y = a + b x; slope standard error; R^2."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    A = np.c_[np.ones(n), x]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    ss_res, ss_tot = float(r @ r), float(((y - y.mean()) ** 2).sum())
    out = {"n": n, "intercept": float(coef[0]), "slope": float(coef[1]),
           "R2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")}
    if n > 2:
        s2 = ss_res / (n - 2)
        cov = s2 * np.linalg.inv(A.T @ A)
        out["slope_se"] = float(np.sqrt(cov[1, 1]))
        out["intercept_se"] = float(np.sqrt(cov[0, 0]))
        out["max_abs_resid"] = float(np.abs(r).max())
    return out


def two_segment(x, y, ngrid=2000):
    """Continuous two-segment fit y = a + b1 (x - xb) for x < xb, a + b2 (x - xb) for x >= xb; xb on a grid
    strictly between the 2nd and the 2nd-to-last point (each segment then holds >= 2 points)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    best = None
    for xb in np.linspace(x[1], x[-2], ngrid)[1:-1]:
        A = np.c_[np.ones_like(x), np.minimum(x - xb, 0), np.maximum(x - xb, 0)]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        ss = float(((y - A @ coef) ** 2).sum())
        if best is None or ss < best[0]:
            best = (ss, xb, coef)
    ss, xb, coef = best
    lin = linfit(x, y)
    ss_lin = (1 - lin["R2"]) * float(((y - y.mean()) ** 2).sum())
    return {"breakpoint_x": float(xb), "y_at_breakpoint": float(coef[0]), "slope_left": float(coef[1]),
            "slope_right": float(coef[2]), "ss_two_segment": ss, "ss_single_line": ss_lin}


def describe(x_pct, y_pct, err_x, err_y, label, rng):
    """x_pct, y_pct: amplitudes in % as plotted; ln is taken of the y amplitude as a fraction (y/100), so the
    slope d ln y / d x is quoted per percentage point of x. err_x, err_y: digitisation half-widths (%)."""
    o = np.argsort(x_pct)
    X, Y = np.asarray(x_pct, float)[o], np.asarray(y_pct, float)[o]
    L = np.log(Y / 100.0)
    d = {"label": label, "x_pct": X.tolist(), "y_pct": Y.tolist(), "lnY": L.tolist(),
         "local_slopes_per_pctpt": (np.diff(L) / np.diff(X)).tolist(),
         "local_dY_dX": (np.diff(Y) / np.diff(X)).tolist(),
         "local_slope_midpoints_pct": (0.5 * (X[1:] + X[:-1])).tolist(),
         "smallest_x_point_pct": [float(X[0]), float(Y[0])]}
    # the paper's 'cross-over' (y = x): linear interpolation of ln y - ln x between the bracketing points
    g = L - np.log(X / 100.0)
    cross = []
    for i in range(len(X) - 1):
        if g[i] * g[i + 1] < 0:
            t = g[i] / (g[i] - g[i + 1])
            cross.append(float(X[i] + t * (X[i + 1] - X[i])))
    d["Y_equals_X_at_pct"] = cross
    xc = cross[0] if cross else -np.inf
    d["two_segment"] = ts = two_segment(X, L)
    ts["breakpoint_y_pct"] = float(100 * np.exp(ts["y_at_breakpoint"]))
    ts["n_points_right_of_break"] = int((X > ts["breakpoint_x"] + 0.05).sum())
    xb = ts["breakpoint_x"] + 0.05            # a point within 0.05 %-pt of the break belongs to both segments
    sel = {"all": np.ones_like(X, bool), "below_break": X <= xb, "right_of_crossover": X > xc,
           "right_of_crossover_below_break": (X > xc) & (X <= xb)}
    d["fits"] = {}
    for k, m in sel.items():
        if m.sum() < 3:
            continue
        f = linfit(X[m], L[m])
        f["x_min_pct"], f["x_max_pct"] = float(X[m].min()), float(X[m].max())
        # digitisation: uniform errors of +-err in x and y (in %), 4000 draws; spread of the slope
        sl = []
        for _ in range(4000):
            xx = X[m] + rng.uniform(-err_x, err_x, m.sum())
            yy = np.log((Y[m] + rng.uniform(-err_y, err_y, m.sum())) / 100.0)
            sl.append(np.polyfit(xx, yy, 1)[0])
        f["slope_sd_digitisation"] = float(np.std(sl))
        f["slope_per_unit_fraction"] = 100 * f["slope"]
        f["efold_x_pctpt"] = float(-1 / f["slope"])
        d["fits"][k] = f
    # straight line on every upper range x >= x_k (>= 3 points)
    d["upper_range_fits"] = []
    for k in range(len(X) - 2):
        f = linfit(X[k:], L[k:])
        f["x_min_pct"], f["x_max_pct"] = float(X[k]), float(X[-1])
        d["upper_range_fits"].append(f)
    return d


def toy_lower_branch(R=300.0):
    """Lower-branch fixed point of eq. (1) with the paper's coefficients (independent check of Fig. 2's axes)."""
    km, ku, kv, kw = 1.57, 2.28, 2.77, 2.67
    su, sv, sw = 1.29, 0.22, 0.68                     # sigma_m = 0

    def parts(W2):
        v = sv * W2 * R / kv ** 2                     # (1b) = 0
        u = (kw ** 2 / R + sv * v) / sw               # (1c) = 0, w != 0
        m = -su * u * v * R / km ** 2                 # (1d) = 0
        return u, v, m, -(ku ** 2 / R) * u + su * v - sw * W2 + su * m * v   # (1a)
    W = np.linspace(1e-9, 1e-3, 100001)
    F = np.array([parts(w)[3] for w in W])
    i = int(np.where(np.sign(F[:-1]) != np.sign(F[1:]))[0][0])
    a, b = W[i], W[i + 1]
    for _ in range(100):
        c = 0.5 * (a + b)
        if np.sign(parts(c)[3]) == np.sign(parts(a)[3]):
            a = c
        else:
            b = c
    u, v, m, _ = parts(0.5 * (a + b))
    return {"100u": 100 * u, "100w": 100 * np.sqrt(0.5 * (a + b)), "100v": 100 * v, "m": m,
            "100Uc": 100 * kw ** 2 / (R * sw)}


# ----------------------------------------------------------------------------------------------- figures
HEADLINE = "right_of_crossover_below_break"   # fit drawn and quoted first in NOTES.md (all fits are in description.json)


def figures(R, D, out):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.0, 4.4))
        ax.axvspan(25, 35, color="0.93", lw=0)
        ax.text(25.4, np.log(0.125), "linearly unstable\nstreaks (paper:\n$A_U$ > 25%, Couette)", fontsize=6.6,
                va="top", color="0.35")
        xx = np.linspace(1.5, 35, 100)
        ax.plot(xx, np.log(xx / 100), ":", color=C_REF, lw=1, label=r"$A_W = A_U$ (the paper's 'cross-over')")
        for nm, col, mk, lab in (("fig5b", C_CO, "^", "plane Couette, Re = 500 (Fig. 5b)"),
                                 ("fig9b", C_BL, "v", r"Blasius, Re$_{\delta^*}$ = 700 (Fig. 9b)")):
            pts = np.array(R[nm]["points"])
            ax.plot(pts[:, 0], np.log(pts[:, 1] / 100), mk, color=col, ms=6.5, mfc="white", mew=1.4, label=lab,
                    zorder=3)
            f = D[nm]["fits"][HEADLINE]
            xs = np.linspace(f["x_min_pct"], f["x_max_pct"], 50)
            ax.plot(xs, f["intercept"] + f["slope"] * xs, "-", color=col, lw=1.3, alpha=0.9,
                    label=fr"   line fit, $A_U$ = {f['x_min_pct']:.1f}–{f['x_max_pct']:.1f}%: "
                          fr"{f['slope']:.3f}$\pm${f['slope_se']:.3f} per %-pt, $R^2$ = {f['R2']:.3f}")
            xe = np.linspace(pts[:, 0].min() - 1, pts[:, 0].max() + 1, 50)
            ax.plot(xe, f["intercept"] + f["slope"] * xe, "--", color=col, lw=0.7, alpha=0.6)
            xc = D[nm]["Y_equals_X_at_pct"][0]
            ax.plot(xc, np.log(xc / 100), "x", color=col, ms=8, mew=1.6, zorder=4)
            for lb in R[nm]["lower_branch"]:
                ax.plot(lb[0], np.log(lb[1] / 100), "o", color=col, ms=6.5, zorder=4,
                        label="   lower-branch saddle (filled circle in Fig. 5b)")
        ax.plot([], [], "x", color="0.3", ms=7, mew=1.5, label=r"interpolated $A_W = A_U$ point")
        ax.plot([], [], "--", color="0.3", lw=0.7, label="fitted line extended over all points")
        ax.set_xlabel(r"$A_U$ (%)   [eq. (3a)]")
        ax.set_ylabel(r"$\ln A_{W,\mathrm{crit}}$  ($A_W$ as a fraction, eq. 3b)")
        ax.set_xlim(0, 35)
        ax.set_ylim(np.log(0.0035), np.log(0.13))
        sec = ax.secondary_yaxis("right", functions=(lambda v: 100 * np.exp(v),
                                                     lambda p: np.log(np.maximum(p, 1e-9) / 100)))
        sec.set_yticks([0.5, 1, 2, 5, 10])
        sec.set_yticklabels(["0.5", "1", "2", "5", "10"])
        sec.set_ylabel(r"$A_{W,\mathrm{crit}}$ (%)")
        ax.spines["right"].set_visible(True)
        ax.legend(fontsize=6.6, loc="lower left")
        ax.set_title("Cossu, Brandt, Bagheri & Henningson (2011): threshold points (vector-extracted)",
                     fontsize=9)
        fig.savefig(out / "cossu_threshold_lnAW.png", dpi=200, bbox_inches="tight")
        fig.savefig(out / "cossu_threshold_lnAW.pdf", bbox_inches="tight")
        plt.close(fig)

        # all five panels: ln of the y amplitude against the x amplitude
        fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.0))
        for ax, items, ttl, xl, yl in (
                (axes[0], (("fig5b", C_CO, "^"), ("fig9b", C_BL, "v")), "eq. (3) amplitudes (Figs. 5b, 9b)",
                 r"$A_U$ (%)", r"$\ln A_W$"),
                (axes[1], (("fig5a", C_CO, "^"), ("fig9a", C_BL, "v")), "rms amplitudes (Figs. 5a, 9a)",
                 r"100 $u_{rms}$", r"$\ln w_{rms}$"),
                (axes[2], (("fig2", C_TOY, "^"),), "toy model eq. (1), R = 300 (Fig. 2)",
                 r"100 $u$ (= 100 $U_1$)", r"$\ln w_c$")):
            for nm, col, mk in items:
                pts = np.array(R[nm]["points"])
                ax.plot(pts[:, 0], np.log(pts[:, 1] / 100), mk, color=col, ms=5, mfc="white", mew=1.3,
                        label=f"{R[nm]['flow']}, {R[nm]['Re']}")
                if nm != "fig2":            # the toy-model curve is smoothly curved; no line drawn
                    f = D[nm]["fits"][HEADLINE]
                    xs = np.linspace(f["x_min_pct"], f["x_max_pct"], 20)
                    ax.plot(xs, f["intercept"] + f["slope"] * xs, "-", color=col, lw=1, alpha=0.8,
                            label=f"   line, x = {f['x_min_pct']:.1f}–{f['x_max_pct']:.1f}: slope {f['slope']:.3f}, "
                                  f"R² {f['R2']:.3f}")
                for lb in R[nm]["lower_branch"]:
                    ax.plot(lb[0], np.log(lb[1] / 100), "o", color=col, ms=5)
            xr = np.linspace(0.3, ax.get_xlim()[1], 100)
            ax.plot(xr, np.log(xr / 100), ":", color=C_REF, lw=1, label="y = x")
            ax.set_title(ttl, fontsize=9.5)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl + " (amplitude as a fraction)")
            ax.legend(fontsize=6.5, loc="lower left")
        fig.text(0.5, -0.03, "lines: straight fits of ln y on the points right of the y = x cross-over and left of "
                 "the top-end change of slope (two-segment fit); "
                 "filled circles: lower-branch saddle", ha="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "cossu_all_panels_lnY.png", dpi=200, bbox_inches="tight")
        fig.savefig(out / "cossu_all_panels_lnY.pdf", bbox_inches="tight")
        plt.close(fig)

        # check figure: extracted points on the paper's own linear axes (compare with the PDF by eye)
        fig, axes = plt.subplots(1, 5, figsize=(17, 3.2))
        for ax, nm in zip(axes, ["fig2", "fig5a", "fig5b", "fig9a", "fig9b"]):
            pts = np.array(R[nm]["points"])
            ax.plot(pts[:, 0], pts[:, 1], "^", ms=5, mfc="white", color="k")
            for lb in R[nm]["lower_branch"]:
                ax.plot(*lb, "ko")
            xt, yt = R[nm]["x_tick_labels"], R[nm]["y_tick_labels"]
            ax.set_xlim(min(xt), max(xt))
            ax.set_ylim(min(yt), max(yt))
            ax.set_xlabel(R[nm]["x"])
            ax.set_ylabel(R[nm]["y"])
            ax.set_title(f"{nm} (paper's axes)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "check_linear_axes.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    R = extract()
    (out / "extracted_points.json").write_text(json.dumps(
        {"source": "Cossu, Brandt, Bagheri & Henningson, Phys. Fluids 23, 074103 (2011); authors' preprint "
                   "dated 16 June 2011 (mech.kth.se/~luca/papers/AUAW.pdf), vector graphics of pages 4, 6, 8",
         "method": "gnuplot point = zero-length round-capped dot at the data position (inside each triangle "
                   "marker); axes calibrated on major ticks paired with pdftotext -bbox tick labels",
         "units": "x, y in percent as plotted (100*u etc.); lower_branch = filled-circle centre",
         "panels": R}, indent=1))
    with open(out / "extracted_points.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "flow", "Re", "kind", "x_quantity", "y_quantity", "x_pct", "y_pct", "ln_y_fraction"])
        for nm, p in R.items():
            for x, y in p["points"]:
                w.writerow([nm, p["flow"], p["Re"], "threshold", p["x"], p["y"], f"{x:.4f}", f"{y:.4f}",
                            f"{np.log(y / 100):.5f}"])
            for x, y in p["lower_branch"]:
                w.writerow([nm, p["flow"], p["Re"], "lower_branch_saddle", p["x"], p["y"], f"{x:.4f}", f"{y:.4f}",
                            f"{np.log(y / 100):.5f}"])
    rng = np.random.default_rng(20261117)
    D = {}
    for nm, p in R.items():
        pts = np.array(p["points"])
        # digitisation half-width: calibration residual + 1/256 pt coordinate quantisation of the SVG, in %
        ex = p["x_cal_max_residual"] + p["units_per_pt"][0] / 256
        ey = p["y_cal_max_residual"] + p["units_per_pt"][1] / 256
        D[nm] = describe(pts[:, 0], pts[:, 1], ex, ey, nm, rng)
        D[nm]["digitisation_halfwidth_pct"] = [ex, ey]
    # same bisection cases in panels (a) and (b)? ratios by rank (the paper does not list the pairing)
    pair = {}
    for a, b in (("fig5a", "fig5b"), ("fig9a", "fig9b")):
        A, B = np.array(R[a]["points"]), np.array(R[b]["points"])
        pair[f"{a}/{b}"] = {"urms_over_AU": (A[:, 0] / B[:, 0]).tolist(), "wrms_over_AW": (A[:, 1] / B[:, 1]).tolist()}
    yy = np.linspace(-1, 1, 200001)
    pair["couette_mode_rms_over_halfpp"] = float(np.sqrt(np.mean((1 - yy ** 2) ** 2) * 0.5))   # sqrt(8/15 * 1/2)
    toy = toy_lower_branch()
    (out / "description.json").write_text(json.dumps({"curves": D, "pairing_by_rank": pair,
                                                      "toy_lower_branch_computed": toy}, indent=1))
    figures(R, D, out)
    ver = subprocess.run(["pdftocairo", "-v"], capture_output=True, text=True).stderr.splitlines()[0]
    (out / "config_used.json").write_text(json.dumps({
        "script": "code/scripts/cossu_partA.py", "pdf": str(PDF.relative_to(REPO)),
        "pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(), "pages": PANELS, "expected_tick_labels": EXPECT,
        "headline_fit": HEADLINE, "two_segment_grid": 2000, "digitisation_mc_draws": 4000, "seed": 20261117,
        "tools": {"pdftocairo": ver, "python": sys.version.split()[0], "numpy": np.__version__,
                  "matplotlib": matplotlib.__version__}}, indent=1))
    log = open(out / "log.txt", "w")

    def say(*a):  # echo the summary to log.txt
        print(*a)
        print(*a, file=log)
    for nm, p in R.items():
        say(f"{nm} {p['flow']} {p['Re']} n={len(p['points'])} cal.res {p['x_cal_max_residual']:.1e} "
              f"{p['y_cal_max_residual']:.1e} digit.halfwidth {np.round(D[nm]['digitisation_halfwidth_pct'], 4)} "
              f"lower branch {np.round(p['lower_branch'], 3).tolist()}")
        d = D[nm]
        say("   local slopes", np.round(d["local_slopes_per_pctpt"], 3).tolist())
        say("   y=x at", np.round(d["Y_equals_X_at_pct"], 3).tolist(), " two-seg",
              {k: round(v, 4) for k, v in d["two_segment"].items()})
        for k, f in d["fits"].items():
            say(f"   {k:32s} x={f['x_min_pct']:.2f}-{f['x_max_pct']:.2f} n={f['n']} slope {f['slope']:.4f} "
                  f"+- {f['slope_se']:.4f} (digit {f['slope_sd_digitisation']:.1e}) R2 {f['R2']:.4f} "
                  f"maxres {f['max_abs_resid']:.3f}")
        for f in d["upper_range_fits"]:
            say(f"      upper x>={f['x_min_pct']:.2f} n={f['n']} slope {f['slope']:.4f} +- {f.get('slope_se', 0):.4f} "
                  f"R2 {f['R2']:.4f}")
    say("pairing", json.dumps({k: (np.round(v["urms_over_AU"], 3).tolist(), np.round(v["wrms_over_AW"], 3).tolist())
                                 if isinstance(v, dict) else v for k, v in pair.items()}))
    say("toy lower branch (computed)", toy)
    log.close()


if __name__ == "__main__":
    main(sys.argv[1])
