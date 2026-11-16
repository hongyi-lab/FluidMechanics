#!/usr/bin/env python
"""Digitise the Re = 400 curve of NJP 2004 figure 10 (Moehlis, Faisst, Eckhardt; Lx = 4 pi, Lz = 2 pi).

Renders page 16 of the local PDF at 400 dpi (pdftoppm), crops the figure, locates the axes from the
frame and the major ticks (P = 1, 0.1, 0.01) and reads:
  - the red exponential-fit line of the Re = 400 curve (least squares in (t, log10 P), t > 3500 so the
    Re = 200 fit line is excluded) -> tau, the time where the line meets P = 1, P(10^4);
  - the centre of the topmost curve (data + fit, drawn on top of each other) at times between ticks.
The fit-line numbers were read before the Gate 1 lifetime run (they are in gate1_mfe.yaml);
the data points were read after it, for display only.

    python code/scripts/digitise_njp_fig10.py literature/notes/MFE2004_fig10_digitised.json
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PDF = REPO / "literature/papers/Moehlis_Faisst_Eckhardt_2004_NJP_6_56.pdf"

with tempfile.TemporaryDirectory() as td:
    subprocess.run(["pdftoppm", "-f", "16", "-l", "16", "-r", "400", "-png", str(PDF), f"{td}/p"], check=True)
    page = mpimg.imread(next(Path(td).glob("p*.png")))[..., :3]
h, w, _ = page.shape
im = page[int(0.12 * h):int(0.36 * h), int(0.27 * w):int(0.70 * w)]
dark = im.sum(2) < 0.6
cols = np.where(dark.sum(0) > 0.5 * im.shape[0])[0]
x0, x1 = 0.5 * (cols[cols < im.shape[1] // 2].min() + cols[cols < im.shape[1] // 2].max()), \
    0.5 * (cols[cols > im.shape[1] // 2].min() + cols[cols > im.shape[1] // 2].max())
ticks = []
for y in range(im.shape[0]):
    n = 0
    for x in range(int(x0) + 3, int(x0) + 100):
        if dark[y, x]:
            n += 1
        else:
            break
    if n >= 30:
        ticks.append(y)
groups = np.split(np.array(ticks), np.where(np.diff(ticks) > 1)[0] + 1)
centres = [g.mean() for g in groups]            # top frame, P = 1, 0.1, 0.01 (= bottom frame)
yP1, yP01, yP001 = centres[1], centres[2], centres[3]
dec = 0.5 * (yP001 - yP1)
x2t = lambda x: (x - x0) / (x1 - x0) * 1e4
t2x = lambda t: x0 + t / 1e4 * (x1 - x0)
y2P = lambda y: 10 ** (-(y - yP1) / dec)

red = (im[..., 0] > 0.8) & (im[..., 1] < 0.35) & (im[..., 2] < 0.35)
ys, xs = np.nonzero(red)
m = (ys > yP1 - 15) & (ys < yP1 + 0.6 * dec) & (x2t(xs) > 3500)
p = np.polyfit(x2t(xs[m]), -(ys[m] - yP1) / dec, 1)
fit = {"tau": float(-1 / (p[0] * np.log(10))), "t_onset": float(-p[1] / p[0]),
       "P_at_1e4": float(10 ** np.polyval(p, 1e4))}

curve = (dark | red | (im.sum(2) < 1.2)) & ~((im[..., 0] > 0.6) & (im[..., 1] < 0.4) & ~red)
pts = []
for t in [2250, 2750, 3250, 3750, 4250, 4750, 5250, 5750, 6250, 6750, 7250, 7750, 8250, 8750, 9250, 9750]:
    cs = []
    for xx in range(int(round(t2x(t))) - 3, int(round(t2x(t))) + 4):
        col = np.nonzero(curve[int(yP1) + 4:int(yP1 + dec), xx])[0] + int(yP1) + 4
        lab = (xx > t2x(8500)) & (xx < t2x(9300))           # the "400" / "350" labels
        if lab:
            col = col[(col < yP1 + 0.18 * dec) | (col > yP1 + 0.32 * dec)]
        if not len(col):
            continue
        run = [col[0]]
        for c in col[1:]:
            if c != run[-1] + 1:
                break
            run.append(c)
        cs.append(0.5 * (run[0] + run[-1]))
    pts.append([t, float(y2P(np.median(cs)))])
out = {"source": "NJP 6, 56 (2004) figure 10, Re = 400 curve (proof version, page 16)",
       "axes_px": {"x_t0": float(x0), "x_t1e4": float(x1), "y_P1": float(yP1), "y_P0.1": float(yP01),
                   "y_P0.01": float(yP001)},
       "red_fit_line": fit, "data_points_t_P": pts,
       "note": "data points = centre of the topmost curve (data and red fit overlap); paper used 1000 ICs"}
Path(sys.argv[1]).write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
