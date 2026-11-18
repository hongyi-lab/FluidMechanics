#!/usr/bin/env python
"""Post-hoc, descriptive (user observation of 2026-11-18): fast vs slow transitions in the Gate 5 truth and the composite error,
and the MFE side of the same W rays (transition time just above the MFE edge; MFE criterion with frozen MFE constants predicting
the MFE W thresholds). No criterion, no refit; reads the Gate 5 / Gate 6 runs, recomputes only MFE ODE quantities.

    python code/scripts/gate56_posthoc_fast.py --out experiments/20261118_gate56_posthoc_fast_v1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code" / "src"))
from committor.dns_rays import build_rays  # noqa: E402
from committor.threshold import RayCoordinate  # noqa: E402
from committor.utils import atomic_write_json  # noqa: E402

G5 = REPO / "experiments/20261117_gate5_dns_v1"
G6 = REPO / "experiments/20261117_gate6_dns_v1"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def mfe_pred(D, sg, cfg5, dev, S_star):
    """Gate 5 (f7) MFE block with the frozen MFE constants: ln s_i (|S_MFE| >= S*), ln s_ii (merged gain >= c*_0), on grid sg."""
    from committor.feedback import Feedback, u_th
    from committor.gain import gain_versions
    from committor.liftup import LiftUp, ray_pq_S
    from committor.systems.mfe import LAMINAR, MFE
    R3c = json.loads((REPO / cfg5["data"]["gate3c_run"] / "results.json").read_text())
    R3d = json.loads((REPO / cfg5["data"]["gate3d_run"] / "results.json").read_text())
    cfg3d = yaml.safe_load((REPO / cfg5["data"]["gate3d_run"] / "config_used.yaml").read_text())
    m = MFE(Re=400.0)
    lu, fb = LiftUp(m), Feedback(m)
    kappa_b = float(R3c["item1"]["b"]["kappa"])
    c0 = float(cfg5["mfe_constants"]["c_star_0"])
    side = int(R3d["item1"]["S"]["side"])
    p, q = ray_pq_S(D, lu.coeffs())
    Smfe = np.abs(p[:, None] * sg[None, :] + q[:, None] * sg[None, :] ** 2)
    ln_i = RayCoordinate(Smfe, sg, dev).ln_s_pred([side * S_star], side)[0][:, 0].cpu().numpy()
    Xa = LAMINAR[None, None, :] + sg[None, :, None] * D[:, None, :]
    Xm = np.array(Xa, copy=True)
    Xm[..., 2] = u_th(Xa, fb.W_inf, kappa_b)
    gI = cfg3d["item3"]["gain"]
    r = gain_versions(m, Xm.reshape(-1, 9), [0.0], float(gI["T_g"]), float(gI["dt"]), dev, chunk=250_000)
    MT = r["max_trP"][:, 0].reshape(len(D), len(sg))
    with np.errstate(divide="ignore"):
        ln_ii = RayCoordinate(np.log(MT), sg, dev).ln_s_pred([c0], 1)[0][:, 0].cpu().numpy()
    return ln_i, ln_ii


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fast-line", type=float, default=600.0)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg5 = yaml.safe_load((G5 / "config_used.yaml").read_text())
    rays = build_rays(cfg5, REPO)
    F_d, F_meta, _ = rays["F"]
    R_d, R_meta, R_mfe = rays["R"]
    W_d, W_meta, _ = rays["W"]
    nF, nR, nW = len(F_d), len(R_d), len(W_d)
    names = [f"F eps={m['eps']:g} d{m['draw']}" for m in F_meta] + [f"R#{m['k']}" for m in R_meta] + [f"W#{m['k']}" for m in W_meta]
    TF, TRW = np.load(G5 / "truth_F.npz"), np.load(G5 / "truth_RW.npz")
    th = np.concatenate([TF["t_hit_hi"], TRW["t_hit_hi"]])
    tl = np.concatenate([TF["t_hit_lo"], TRW["t_hit_lo"]])
    sc = np.concatenate([TF["s_c"], TRW["s_c"]])
    R5 = json.loads((G5 / "results.json").read_text())["analysis"]
    R6 = json.loads((G6 / "results.json").read_text())["analysis"]
    # error per ray: composite (Gate 6) where it exists; R calibration rays: (i) (Gate 5 first gate)
    err, src = np.full(len(names), np.nan), [""] * len(names)
    for i, r in enumerate(R6["rays_F"]):
        err[i], src[i] = r["ln_err_composite"], "composite (Gate 6, F)"
    for i in range(12):
        err[nF + i], src[nF + i] = R5["rays"][nF + i]["ln_err[S (first gate)]"], "(i) (Gate 5, R calibration)"
    for r in R6["rays_RW"]:
        j = nF + r["k"] - 1 if r["set"] == "R" else nF + nR + r["k"] - 1
        err[j], src[j] = r["ln_err_composite"], f"composite (Gate 6, {r['set']} test)"
    assert np.all(np.isfinite(err)), "missing error"
    fast = th < args.fast_line
    a = np.abs(err)
    res = {"note": "post-hoc, descriptive (user observation 2026-11-18); no criterion", "fast_line_t_hit_hi": args.fast_line,
           "n_fast": int(fast.sum()), "n_slow": int((~fast).sum()),
           "fast_rays": [names[i] for i in np.where(fast)[0]],
           "fast_t_hit_hi_max": float(th[fast].max()), "slow_t_hit_hi_min": float(th[~fast].min()),
           "fast_abs_err_min": float(a[fast].min()), "fast_abs_err_min_ray": names[int(np.where(fast)[0][np.argmin(a[fast])])],
           "fast_abs_err_median": float(np.median(a[fast])),
           "slow_abs_err_median": float(np.median(a[~fast])), "slow_abs_err_max": float(a[~fast].max()),
           "slow_abs_err_max_ray": names[int(np.where(~fast)[0][np.argmax(a[~fast])])],
           "n_slow_abs_err_ge_0.24": int(np.sum(a[~fast] >= 0.24)), "n_fast_abs_err_lt_0.24": int(np.sum(a[fast] < 0.24)),
           "rays": [{"ray": names[i], "t_hit_hi": float(th[i]), "t_hit_lo": float(tl[i]), "s_c_dns": float(sc[i]), "ln_err": float(err[i]),
                     "err_source": src[i], "fast": bool(fast[i])} for i in range(len(names))]}
    log(f"fast {res['n_fast']}: {res['fast_rays']}; t_hit_hi fast max {res['fast_t_hit_hi_max']:.0f}, slow min {res['slow_t_hit_hi_min']:.0f}")
    log(f"|err| fast min {res['fast_abs_err_min']:.4f} ({res['fast_abs_err_min_ray']}); slow median {res['slow_abs_err_median']:.4f}, "
        f"max {res['slow_abs_err_max']:.4f} ({res['slow_abs_err_max_ray']})")
    # ---------------- MFE side of the W rays
    from committor import edge
    from committor.systems.mfe import MFE, rhs_paper
    cfg2 = rays["gate2_cfg"]
    D2 = cfg2["deterministic"]
    m0 = MFE(Re=cfg2["system"]["Re"], Lx_over_pi=cfg2["system"]["Lx_over_pi"], Lz_over_pi=cfg2["system"]["Lz_over_pi"])
    rA2, EB2 = float(cfg2["sets"]["r_A"]), float(cfg2["sets"]["E_B"])
    grid2 = np.sqrt(np.geomspace(rA2 * (1 + 1e-9), EB2 * (1 - 1e-9), int(D2["dense_grid"]["n_points"])))
    t0 = time.time()
    ALL = np.concatenate([F_d, R_d, W_d])
    r = edge.first_crossing(lambda x: rhs_paper(x, m0.Re, m0.Lx, m0.Lz), ALL, grid2, int(D2["n_bisect"]), float(D2["dt"]), float(D2["T_det"]),
                            EB2, rA2, "rk4", device=args.device, chunk=1 << 20, progress=False, desc="MFE")
    log(f"MFE first crossing on all 52 rays: {time.time() - t0:.0f} s")
    mW = np.load(G5 / "mfe_W.npz")["s_c"]
    scW_mfe = r["s_c"][nF + nR:]
    res["mfe"] = {"settings": "Gate 2 routine (rk4, dt 0.25, T_det 8000, 1000-point grid, 24 bisections, r_A 5e-7, E_B 0.29)",
                  "W_s_c_recomputed": scW_mfe.tolist(), "W_s_c_gate5_mfe_W": mW.tolist(),
                  "W_s_c_max_rel_diff_vs_gate5": float(np.nanmax(np.abs(scW_mfe / mW - 1))),
                  "W_t_hit_above_edge": r["t_hit_above"][nF + nR:].tolist(), "W_t_hit_below_edge": r["t_hit_below"][nF + nR:].tolist(),
                  "W_t_hit_above_min": float(np.min(r["t_hit_above"][nF + nR:])), "W_t_hit_above_max": float(np.max(r["t_hit_above"][nF + nR:])),
                  "all_rays_t_hit_above_edge": {names[i]: float(r["t_hit_above"][i]) for i in range(len(names))},
                  "all_rays_s_c_mfe_recomputed": {names[i]: (float(r["s_c"][i]) if np.isfinite(r["s_c"][i]) else None) for i in range(len(names))}}
    log(f"MFE W: s_c max rel diff vs Gate 5 {res['mfe']['W_s_c_max_rel_diff_vs_gate5']:.2e}; t_hit just above edge "
        f"{res['mfe']['W_t_hit_above_min']:.0f}-{res['mfe']['W_t_hit_above_max']:.0f}")
    # MFE criterion (frozen constants) predicting the MFE W thresholds; Gate 5 (f7) grid (DNS r_A, E_B, 1000 points)
    sg = np.sqrt(np.geomspace(5e-7 * (1 + 1e-9), 0.24 * (1 - 1e-9), 1000))
    S_gate3d = float(cfg5["mfe_constants"]["S_star"])
    blocks = {}
    for tag, Sst in (("S*=0.12333 (Gate 3d item 1, as gate5_dns.yaml)", S_gate3d), ("S*=0.1236 (value in the user's message)", 0.1236)):
        ln_i, ln_ii = mfe_pred(W_d, sg, cfg5, args.device, Sst)
        ln_c = np.maximum(ln_i, ln_ii)
        lsc = np.log(scW_mfe)
        e = {"(i)": ln_i - lsc, "(ii)": ln_ii - lsc, "composite": ln_c - lsc}
        blocks[tag] = {k: {"median_abs": float(np.median(np.abs(v))), "max_abs": float(np.max(np.abs(v))), "per_ray": v.tolist()} for k, v in e.items()}
        blocks[tag]["binds"] = ["(i)" if a_ > b_ else ("(ii)" if b_ > a_ else "tie") for a_, b_ in zip(ln_i, ln_ii)]
        log(f"MFE criterion on MFE W thresholds [{tag}]: composite median |ln| {blocks[tag]['composite']['median_abs']:.4f} "
            f"((i) {blocks[tag]['(i)']['median_abs']:.4f}, (ii) {blocks[tag]['(ii)']['median_abs']:.4f})")
    res["mfe"]["criterion_on_mfe_W"] = blocks
    atomic_write_json(out_dir / "results.json", res)
    log("done")


if __name__ == "__main__":
    main()
