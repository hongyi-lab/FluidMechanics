#!/usr/bin/env python
"""Probit maximum-likelihood fits of the pre-study ray profiles: q(s) = Phi((s - s50) / w), width(5-95%) = 2*1.6449*w.
Parametric bootstrap SE (200 replicates). Writes probit.json in the pre-study run directory.

    python code/scripts/gate2_prestudy_probit.py experiments/20261115_gate2_prestudy_v2
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

run = Path(sys.argv[1])
R = json.loads((run / "results.json").read_text())
Z = np.load(run / "rays.npz")
names = R["rays"]["names"]
M = 200
rng = np.random.default_rng(0)


def fit(s, k):
    s0, sc = s.mean(), s.std()
    x = (s - s0) / sc
    def nll(p):
        q = np.clip(norm.cdf((x - p[0]) / np.exp(p[1])), 1e-12, 1 - 1e-12)
        return -(k * np.log(q) + (M - k) * np.log1p(-q)).sum()
    r = minimize(nll, [0.0, np.log(0.3)], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 4000})
    return s0 + sc * r.x[0], sc * np.exp(r.x[1])


out = {}
for sname in ("sigma_lo", "sigma_hi"):
    S, q1, q2 = Z[f"{sname}_S_f"], Z[f"{sname}_q_dt"], Z[f"{sname}_q_dt2"]
    rows = {}
    for i, nm in enumerate(names):
        if not R["rays"]["profiles"][sname]["rays"][nm].get("transition_located", True) or np.ptp(S[i]) == 0:
            rows[nm] = None
            continue
        s50, w = fit(S[i], np.rint(q1[i] * M))
        h50, hw = fit(S[i], np.rint(q2[i] * M))
        bs = []
        for _ in range(200):
            kb = rng.binomial(M, norm.cdf((S[i] - s50) / w))
            bs.append(fit(S[i], kb))
        bs = np.array(bs)
        W = 2 * 1.6449 * w
        sc = R["rays"]["profiles"][sname]["rays"][nm]["s_c_det_heun_dt"]
        rows[nm] = {"s50": s50, "s50_se": float(bs[:, 0].std()), "width": W, "width_se": float(2 * 1.6449 * bs[:, 1].std()),
                    "width_rel": W / s50, "dt2_s50": h50, "dt2_width": 2 * 1.6449 * hw,
                    "dt2_shift_over_width": (h50 - s50) / W, "dt2_shift_z": (h50 - s50) / (np.sqrt(2) * bs[:, 0].std()),
                    "dt2_width_ratio": hw / w, "s50_minus_det_over_width": (s50 - sc) / W}
    out[sname] = rows
(run / "probit.json").write_text(json.dumps(out, indent=1))
for sname, rows in out.items():
    for nm, v in rows.items():
        if v:
            print(f"{sname} {nm:14s} rel.width {v['width_rel']:.2e} (+-{v['width_se'] / v['s50']:.1e})  dt2 shift/w {v['dt2_shift_over_width']:+.3f} "
                  f"(z {v['dt2_shift_z']:+.2f})  dt2 width ratio {v['dt2_width_ratio']:.3f}  (s50-sc)/w {v['s50_minus_det_over_width']:+.3f}")
