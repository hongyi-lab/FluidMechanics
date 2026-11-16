#!/usr/bin/env python
"""Extract the Re = 400 survival curve of NJP 2004 figure 10 from the PDF's vector graphics.

pdftocairo -svg renders page 16 losslessly. The axes are calibrated on the figure's own major tick
marks (x: t = 0, 2000, ..., 10000; y: P = 0.01, 0.1, 1). Each survival curve is one polyline of
points (t_i, P_i). The Re = 400 curve is the one with the largest P just before t = 10000.
The paper states 1000 initial conditions. The plotted levels are multiples of 1/1000 only to within
+-0.4/1000 (the curve coordinates in the PDF are quantised), so each drop is rounded to whole
trajectories. One more drop is drawn exactly at t = 10000, where the curve also falls to the axis;
it is counted as censored here (P(10^4-) = 0.410), or as a decay (0.409) by another reading.

This replaces the raster digitisation (digitise_njp_fig10.py) for comparisons. It was written after
the Gate 1 run and is used for description only. An independent verifier extracted the same curve
separately; agreement is recorded in the Gate 1 report.

    python code/scripts/extract_njp_fig10_vector.py literature/notes/MFE2004_fig10_re400_vector.json
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PDF = REPO / "literature/papers/Moehlis_Faisst_Eckhardt_2004_NJP_6_56.pdf"


def paths(svg):
    out = []
    for p in re.findall(r"<path ([^>]*)/>", svg):
        st = re.search(r'style="([^"]*)"', p)
        st = st.group(1) if st else ""
        d = re.search(r' d="([^"]*)"', p)
        if not d or "stroke:none" in st:
            continue
        tr = re.search(r'transform="([^"]*)"', p)
        if not tr or tr.group(1) != "matrix(1,0,0,-1,0,841)":
            raise ValueError("unexpected transform; coordinates below assume PDF space")
        col = re.search(r"stroke:([^;]*)", st).group(1)
        toks = re.findall(r"[MLCZ]|-?[\d.]+(?:e-?\d+)?", d.group(1))
        segs, cur, i = [], [], 0
        while i < len(toks):
            if toks[i] == "M":
                if cur:
                    segs.append(np.array(cur))
                cur = [[float(toks[i + 1]), float(toks[i + 2])]]
                i += 3
            elif toks[i] == "L":
                cur.append([float(toks[i + 1]), float(toks[i + 2])])
                i += 3
            elif toks[i] == "C":
                cur.append([float(toks[i + 5]), float(toks[i + 6])])
                i += 7
            else:
                i += 1
        if cur:
            segs.append(np.array(cur))
        out.append((col, segs))
    return out


def main(out_path):
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdftocairo", "-svg", "-f", "16", "-l", "16", str(PDF), f"{td}/p16.svg"], check=True)
        svg = Path(f"{td}/p16.svg").read_text()
    P = paths(svg)
    # figure 10 frame: the black rectangle in the upper half of the page
    frame = None
    for col, segs in P:
        for sg in segs:
            if col == "rgb(0%,0%,0%)" and len(sg) in (4, 5) and np.ptp(sg[:, 0]) > 150 and np.ptp(sg[:, 1]) > 100 \
                    and sg[:, 1].min() > 500:
                frame = sg
    fx0, fx1 = frame[:, 0].min(), frame[:, 0].max()
    fy0, fy1 = frame[:, 1].min(), frame[:, 1].max()
    # major ticks (longest short segments on the left / bottom frame)
    hticks, vticks = {}, {}
    for col, segs in P:
        for sg in segs:
            if len(sg) != 2:
                continue
            (xa, ya), (xb, yb) = sg
            if abs(ya - yb) < 1e-6 and abs(min(xa, xb) - fx0) < 0.05 and fy0 - 0.1 <= ya <= fy1 + 0.1:
                hticks.setdefault(round(ya, 3), set()).add(round(abs(xb - xa), 3))
            if abs(xa - xb) < 1e-6 and abs(min(ya, yb) - fy0) < 0.05 and fx0 - 0.1 <= xa <= fx1 + 0.1:
                vticks.setdefault(round(xa, 3), set()).add(round(abs(yb - ya), 3))
    Lmaj = max(v for vs in hticks.values() for v in vs if v < 20)                  # major tick length
    ymaj = sorted(y for y, vs in hticks.items() if any(abs(v - Lmaj) < 0.05 for v in vs))   # P = 0.01, 0.1, 1
    xmaj = sorted(x for x, vs in vticks.items() if any(abs(v - Lmaj) < 0.05 for v in vs))   # t = 0, ..., 10000
    if len(ymaj) != 3 or len(xmaj) != 6:
        raise RuntimeError(f"tick calibration failed: {ymaj} {xmaj}")
    px = np.polyfit(xmaj, np.arange(6) * 2000.0, 1)
    py = np.polyfit(ymaj, [-2.0, -1.0, 0.0], 1)
    x_res = float(np.abs(np.polyval(px, xmaj) - np.arange(6) * 2000.0).max())
    y_res = float(np.abs(np.polyval(py, ymaj) - np.array([-2.0, -1.0, 0.0])).max())
    t_of = lambda x: np.polyval(px, x)
    P_of = lambda y: 10 ** np.polyval(py, y)

    # survival polylines: black, starting at (t = 0, P = 1)
    curves = []
    for col, segs in P:
        if col != "rgb(0%,0%,0%)":
            continue
        for sg in segs:
            if len(sg) > 20 and abs(t_of(sg[0, 0])) < 1 and abs(P_of(sg[0, 1]) - 1) < 1e-3:
                curves.append(sg)
    def p_before_end(sg):
        t = t_of(sg[:, 0])
        k = np.where(t < 9999.0)[0]
        return P_of(sg[k[-1], 1]) if len(k) else 0.0
    sg = max(curves, key=p_before_end)
    t = t_of(sg[:, 0])
    Pv = P_of(sg[:, 1])
    inside = t < 9999.0
    t_in, P_in = t[inside], Pv[inside]
    N = 1000
    k = np.rint((1 - P_in) * N)
    level_res = float(np.abs((1 - P_in) * N - k).max())
    drops = np.diff(k)
    if not (k[0] == 0 and np.all(drops >= 0)):
        raise RuntimeError("curve is not a non-increasing survival function")
    lifetimes = np.repeat(t_in[1:], drops[:].astype(int))               # one entry per decay
    n_dec = len(lifetimes)
    lifetimes = np.r_[lifetimes, np.full(N - n_dec, np.inf)]            # censored at t = 10000
    out = {"source": "NJP 6, 56 (2004) figure 10 (proof version, page 16), Re = 400 curve, vector graphics",
           "calibration": {"x_major_ticks_pt": xmaj, "y_major_ticks_pt": ymaj, "x_fit_max_residual_t": x_res,
                           "y_fit_max_residual_log10P": y_res},
           "n_points": int(len(sg)), "N_initial_conditions": N, "n_decayed_before_1e4": int(n_dec),
           "max_level_residual_in_units_of_1_over_N": level_res,
           "P_at_1e4_minus": float(1 - n_dec / N),
           "note": "one further drop is drawn exactly at t = 10000 (decay at t_max or censoring: P(1e4) = 0.409 or 0.410)",
           "lifetimes": [None if not np.isfinite(x) else float(x) for x in lifetimes]}
    Path(out_path).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "lifetimes"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
