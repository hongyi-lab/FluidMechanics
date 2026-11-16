"""Gate 3c post-hoc (descriptive, NOT a criterion): item-1 state thresholds (b), (c) frozen, applied to the noise family.

Same family rays, eps set, deterministic s_c and ray grid as Gate 3b section 6 / Gate 3c item 2; c*, kappa, T frozen from
the Gate 3c full run. Question: does the state threshold also drift with eps on the family (as M1 did in Gate 3b)?
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code" / "src"))
from committor.feedback import Feedback, u_th  # noqa: E402
from committor.systems.mfe import MFE  # noqa: E402
from committor.threshold import RayCoordinate  # noqa: E402

run3c = REPO / "experiments/20261116_gate3c_mfe_v1"
g2 = REPO / "experiments/20261116_gate2_mfe_v1"
out_dir = REPO / "experiments/20261116_gate3c_posthoc_family_v1"
R = json.load(open(run3c / "results.json"))
R2 = json.load(open(g2 / "results.json"))
FD = np.load(g2 / "family_det.npz")
meta = R2["family_det"]["rays"]
dirs, sc = FD["d"], FD["s_c"]
eps = np.array([f["eps"] for f in meta])
ang = np.array([f["angle"] for f in meta])
grid = np.sqrt(np.geomspace(5e-7 * (1 + 1e-9), 0.29 * (1 - 1e-9), 4000))      # as gate3c_mfe.py / gate3b family
use = np.isin(eps, [1e-6, 1e-4, 1e-2]) & np.isfinite(sc)
fb = Feedback(MFE(Re=400.0))
dev = "cuda" if torch.cuda.is_available() else "cpu"
X = grid[None, :, None] * dirs[:, None, :]                                       # x = a - laminar
cands = {"(b) kappa W_inf": (R["item1"]["b"]["kappa"] * fb.W_inf, R["item1"]["b"]),
         "(c) W(T)": (fb.W(R["item1"]["c"]["T"]), R["item1"]["c"])}
res = {"note": "post-hoc, descriptive; frozen item-1 thresholds on the noise family (eps > 0, 48 rays); signed ln(s_pred/s_c)",
       "models": {}}
for name, (W, rec) in cands.items():
    C = np.abs(u_th(X, W))
    rc = RayCoordinate(C, grid, dev)
    ln_p, _ = rc.ln_s_pred([rec["c_star"]], rec["side"])
    e = ln_p[:, 0].cpu().numpy() - np.log(np.where(np.isfinite(sc), sc, 1.0))
    e = np.where(np.isnan(e), np.inf, e)                                         # no crossing on the grid -> +inf
    by_eps = {}
    for ep in (1e-6, 1e-4, 1e-2):
        sel = use & (eps == ep)
        ee = e[sel]
        fin = np.isfinite(ee)
        by_eps[f"{ep:g}"] = {"n": int(sel.sum()), "signed_median": float(np.median(ee)),
                             "signed_median_finite_only": float(np.median(ee[fin])),
                             "no_crossing_angles": sorted(set(ang[sel][~fin].tolist())),
                             "median_abs": float(np.median(np.abs(ee))), "frac_no_crossing": float((~fin).mean())}
    pairs = []
    for a in np.unique(ang[use]):
        for d_ in (0, 1):
            i1 = [i for i in range(len(meta)) if use[i] and ang[i] == a and meta[i]["draw"] == d_ and eps[i] == 1e-6]
            i2 = [i for i in range(len(meta)) if use[i] and ang[i] == a and meta[i]["draw"] == d_ and eps[i] == 1e-2]
            if i1 and i2 and np.isfinite(e[i1[0]]) and np.isfinite(e[i2[0]]):
                pairs.append(float(e[i1[0]] - e[i2[0]]))
    res["models"][name] = {"c_star": rec["c_star"], "by_eps": by_eps,
                           "eps_trend_median_1e-6_minus_1e-2": float(np.median(pairs)), "n_pairs_both_finite": len(pairs),
                           "per_angle_signed_median": {f"{a:g}": float(np.median(e[use & (ang == a)])) for a in np.unique(ang[use])},
                           "all_median_abs": float(np.median(np.abs(e[use])))}
    print(name, {k: (round(v["signed_median"], 4), round(v["signed_median_finite_only"], 4), v["no_crossing_angles"]) for k, v in by_eps.items()},
          "trend", round(float(np.median(pairs)), 4), len(pairs))
res["reference_gate3b_M1_signed_by_eps"] = R["item2"]["family_references_gate3b"]["M1"]
json.dump(res, open(out_dir / "results.json", "w"), indent=1)
