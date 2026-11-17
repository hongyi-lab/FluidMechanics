#!/usr/bin/env python
"""Gate 5 (DNS): ground-truth s_c on the F, R, W rays, the second-order theory quantity S(s) and the first gate S(s) >= S*.
Config: code/configs/gate5_dns.yaml (criteria fixed before any run); A, B, T_det, Re, resolution and dt from Gate 4b (choice.json).

    python code/scripts/gate5_dns.py --config code/configs/gate5_dns.yaml --gate4 code/configs/gate4_dns.yaml \
        --choice <gate4b run>/choice.json --run-dir <dir> --stage rays|truth_F|truth_RW|theory|second_order|mfe|analysis|all
"""
import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from committor.dns import Grid, find_sc_det, step_nl_c, _kmax_t
from committor.dns_rays import build_rays
from committor.dns_theory import A_of, run_delta_q
from committor.threshold import RayCoordinate, fit_cstar, summarize
from committor.utils import atomic_save_npz, atomic_write_json, git_info, read_json

REPO = Path(__file__).resolve().parents[2]
SHARED = ["mirror y about 1 (u,-v,w)(x,2-y,z)", "shift-reflect (u,v,-w)(x+Lx/2,y,-z)", "shift-rotate (-u,-v,w)(-x,-y,z+Lz/2)"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def make_grid(cfg4, N, Re):
    S = cfg4["system"]
    g = Grid(*N, Re=Re, Lx=S["Lx_over_pi"] * math.pi, Lz=S["Lz_over_pi"] * math.pi, Ly=S["Ly_ext"], device=cfg4["device"])
    g.set_enforced(SHARED)
    return g


def per_case_step(g, dt, n=32, steps=30):
    rng = np.random.default_rng(0)
    d = rng.standard_normal((n, 9))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    u = g.clean(g.from_coeffs(0.05 * d))
    E1, E2 = g.lap_factor(dt), g.lap_factor(0.5 * dt)
    km = _kmax_t(g)
    for _ in range(3):
        u, _c = step_nl_c(g, u, dt, E1, E2, km)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        u, _c = step_nl_c(g, u, dt, E1, E2, km)
    torch.cuda.synchronize()
    return (time.time() - t0) / (n * steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gate4", required=True)
    ap.add_argument("--choice", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stage", default="all")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg4 = yaml.safe_load(Path(args.gate4).read_text())
    ch = json.loads(Path(args.choice).read_text())
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    if not (run / "config_used.yaml").exists():
        (run / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        atomic_write_json(run / "gate4b_choice_used.json", ch)
    man = read_json(run / "manifest.json", {"invocations": []})
    man["invocations"].append({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv, "git": git_info(REPO),
                               "python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})
    atomic_write_json(run / "manifest.json", man)
    res = read_json(run / "results.json", {}) or {}

    def save():
        atomic_write_json(run / "results.json", res)

    Re, N, dt = float(ch["Re"]), ch["grid"], float(ch["dt"])
    r_A, E_B, T_det = float(ch["r_A"]), float(ch["E_B"]), float(ch["T_det"])
    R5 = cfg4["revised_5"]
    climit, lv = float(R5["5b"]["courant_limit"]), int(R5["5b"]["max_rerun_levels"])
    SC = cfg4["s_c"]
    g = make_grid(cfg4, N, Re)
    rays = build_rays(cfg, REPO)
    F_d, F_meta, F_mfe = rays["F"]
    R_d, R_meta, R_mfe = rays["R"]
    W_d, W_meta, _ = rays["W"]
    res.setdefault("meta", {"Re": Re, "grid": N, "dt": dt, "r_A": r_A, "E_B": E_B, "T_det": T_det,
                            "F_direction_check_max_abs": rays["F_direction_check_max_abs"], "n_F": len(F_d), "n_R": len(R_d), "n_W": len(W_d)})
    save()
    stages = ["mfe", "truth_F", "truth_RW", "theory", "second_order", "analysis"] if args.stage == "all" else [args.stage]

    def timed(name, fn):
        if name in res and res[name].get("done"):
            log(f"--- {name}: done earlier")
            return
        log(f"--- {name}")
        t0 = time.time()
        out = fn()
        out["seconds"] = time.time() - t0
        out["done"] = True
        res[name] = out
        save()
        log(f"--- {name}: {out['seconds'] / 60:.1f} min")

    # ------------------------------------------------------------ MFE s_c of the W rays (f2) with the Gate 2 routine
    def st_mfe():
        from committor import edge
        from committor.systems.mfe import rhs_paper, MFE
        cfg2 = rays["gate2_cfg"]
        D2 = cfg2["deterministic"]
        m0 = MFE(Re=cfg2["system"]["Re"], Lx_over_pi=cfg2["system"]["Lx_over_pi"], Lz_over_pi=cfg2["system"]["Lz_over_pi"])
        rA2, EB2 = float(cfg2["sets"]["r_A"]), float(cfg2["sets"]["E_B"])
        grid2 = np.sqrt(np.geomspace(rA2 * (1 + 1e-9), EB2 * (1 - 1e-9), int(D2["dense_grid"]["n_points"])))
        r = edge.first_crossing(lambda a: rhs_paper(a, m0.Re, m0.Lx, m0.Lz), W_d, grid2, int(D2["n_bisect"]), float(D2["dt"]), float(D2["T_det"]),
                                EB2, rA2, "rk4", device=cfg4["device"], chunk=1 << 20, progress=False, desc="MFE W")
        atomic_save_npz(run / "mfe_W.npz", s_c=r["s_c"], no_crossing=r["no_crossing"], below_grid=r["below_grid"])
        return {"W_s_c_mfe": [None if not np.isfinite(v) else float(v) for v in r["s_c"]],
                "W_no_crossing": np.asarray(r["no_crossing"]).tolist()}

    # ------------------------------------------------------------ ground truth
    def truth(dirs, tag):
        t_c = per_case_step(g, dt, n=min(64, 12 * len(dirs)))
        ub = len(dirs) * (int(SC["n_coarse"]) + 12) * (T_det / dt) * t_c / 60
        log(f"    1% timing: {t_c * 1e3:.3f} ms/case-step; upper bound {ub:.0f} min (every run to T_det)")
        r = find_sc_det(g, dirs, r_A, E_B, T_det, dt, int(SC["n_coarse"]), float(SC["rel_tol"]), courant_limit=climit, max_levels=lv,
                        ckpt=run / f"ckpt_truth_{tag}.npz", log=log)
        atomic_save_npz(run / f"truth_{tag}.npz", s_c=r["s_c"], lo=r["lo"], hi=r["hi"], no_crossing=r["no_crossing"], below_grid=r["below_grid"],
                        flag_undecided=r["flag_undecided"],
                        n_switches=r["n_switches"], coarse_fate=r["coarse_fate"], coarse_t_hit=r["coarse_t_hit"], s_grid=r["s_grid"],
                        t_hit_lo=r["t_hit_lo"], t_hit_hi=r["t_hit_hi"], out_hi=r["out_hi"])
        return {"s_c": [None if not np.isfinite(v) else float(v) for v in r["s_c"]], "n_unresolved": int(np.sum(~np.isfinite(r["s_c"]))),
                "n_flagged_undecided": int(np.sum(r["flag_undecided"])), "flag_undecided": [bool(x) for x in r["flag_undecided"]],
                "n_no_crossing": int(r["no_crossing"].sum()), "n_below_grid": int(r["below_grid"].sum()), "n_runs": r["n_runs"],
                "n_undecided_runs": r["n_undecided_runs"], "rerun_counts": r["counts"], "per_case_step_ms": t_c * 1e3, "upper_bound_min": ub,
                "n_switches_coarse": r["n_switches"].tolist()}

    def st_truth_F():
        return truth(F_d, "F")

    def st_truth_RW():
        return truth(np.concatenate([R_d, W_d]), "RW")

    # ------------------------------------------------------------ theory quantities (f4)
    TH = cfg["theory"]
    SO = cfg["second_order_check"]
    chk_names = ["F eps=0.01 draw 0", "R #13", "W #1"]
    jF = [i for i, m in enumerate(F_meta) if m["eps"] == 1e-2 and m["draw"] == 0][0]
    chk_dirs = np.stack([F_d[jF], R_d[12], W_d[0]])
    ALL = np.concatenate([F_d, R_d, W_d])

    def st_theory():
        t0 = time.time()
        _ = run_delta_q(g, ALL[:4], 10.0, dt, every=float(TH["output_every"]))
        tt = (time.time() - t0) / 10.0 * float(TH["T_end"]) * (len(ALL) / 4)
        log(f"    1% timing: estimate {tt / 60:.1f} min for {len(ALL)} rays to t = {TH['T_end']}")
        st_times = tuple(np.arange(0.0, float(SO["T"]) + 1e-9, float(SO["every"])))
        r = run_delta_q(g, ALL, float(TH["T_end"]), dt, every=float(TH["output_every"]), max_batch=64, progress=True)
        rc = run_delta_q(g, chk_dirs, float(SO["T"]), dt, every=float(TH["output_every"]), store_times=st_times)
        atomic_save_npz(run / "theory.npz", aa=r["aa"], ab=r["ab"], bb=r["bb"], t=r["t"])
        torch.save({t: (v[0], v[1]) for t, v in rc["stored"].items()}, run / "theory_check_fields.pt")
        return {"n_rays": int(len(ALL)), "estimate_min": tt / 60, "check_rays": chk_names,
                "check_consistency_aa_max_rel": float(np.max(np.abs(rc["aa"][:, :int(SO["T"]) + 1] - r["aa"][[jF, len(F_d) + 12, len(F_d) + len(R_d)], :int(SO["T"]) + 1])
                                                        / np.maximum(np.abs(r["aa"][[jF, len(F_d) + 12, len(F_d) + len(R_d)], :int(SO["T"]) + 1]), 1e-300)))}

    # ------------------------------------------------------------ second-order check (f6)
    def st_second_order():
        F = torch.load(run / "theory_check_fields.pt")
        times = sorted(F.keys())
        out = {"rays": chk_names, "s": [float(s) for s in SO["s"]], "times": times, "diff_over_s2": {}, "pred_norm_over_s": {}}
        E1, E2 = g.lap_factor(dt), g.lap_factor(0.5 * dt)
        km = _kmax_t(g)
        cmax = 0.0
        for s in SO["s"]:
            s = float(s)
            u = g.clean(g.from_coeffs(s * chk_dirs))
            rows = []
            k_out = {int(round(t / dt)): t for t in times}
            for k in range(0, int(round(float(SO["T"]) / dt)) + 1):
                if k in k_out:
                    t = k_out[k]
                    P0d, q = F[t]
                    pred = s * P0d.to(g.dev) + s * s * q.to(g.dev)
                    rows.append((g.energy(u * g.mask_kx0 - pred).sqrt() / s ** 2).cpu().numpy())
                if k < int(round(float(SO["T"]) / dt)):
                    u, c = step_nl_c(g, u, dt, E1, E2, km)
                    cmax = max(cmax, float(torch.nan_to_num(c, nan=float("inf")).max()))
            out["diff_over_s2"][f"{s:g}"] = np.array(rows).T.tolist()                        # (3 rays, n_times)
        D = np.array([out["diff_over_s2"][f"{float(s):g}"] for s in SO["s"]])               # (n_s, 3, n_t)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["ratio_successive_s"] = (D[1:] / D[:-1]).tolist()
            out["ratio_successive_s_expected"] = [float(SO["s"][i + 1] / SO["s"][i]) for i in range(len(SO["s"]) - 1)]  # O(s) residual
        out["courant_max"] = cmax
        return out

    # ------------------------------------------------------------ analysis: 5.1, 5.2, controls, MFE descriptive
    def st_analysis():
        TF, TRW = np.load(run / "truth_F.npz"), np.load(run / "truth_RW.npz")
        scF = TF["s_c"]
        scR, scW = TRW["s_c"][:len(R_d)], TRW["s_c"][len(R_d):]
        out = {}
        # 5.1 shape (f3)
        Cs = cfg["criteria"]["shape"]
        eps = np.array([m["eps"] for m in F_meta])
        draw = np.array([m["draw"] for m in F_meta])
        lnA = np.log(scF * F_d[:, 2])
        per_draw = {}
        for k in sorted(set(draw.tolist())):
            L = {e: lnA[(eps == e) & (draw == k)][0] for e in sorted(set(eps.tolist()))}
            fin = lambda *es: all(np.isfinite(L[e]) for e in es)  # noqa: E731
            a = bool(L[1e-6] > L[1e-4] > L[1e-2]) if fin(1e-6, 1e-4, 1e-2) else None
            b_off = float(abs(L[1e-4] - 0.5 * (L[1e-6] + L[1e-2]))) if fin(1e-6, 1e-4, 1e-2) else None
            s_lo = float(abs(L[1e-2] - L[1e-6]) / math.log(1e4)) if fin(1e-6, 1e-2) else None
            s_hi = float(abs(L[1.0] - L[0.1]) / math.log(10)) if fin(0.1, 1.0) else None
            per_draw[k] = {"ln_A3c": {f"{e:g}": (float(v) if np.isfinite(v) else None) for e, v in L.items()},
                           "a_monotone_decrease": a, "b_midpoint_off_chord": b_off, "b_ok": (b_off <= float(Cs["collinear_tol"])) if b_off is not None else None,
                           "slope_1e-6_1e-2": s_lo, "slope_0.1_1": s_hi,
                           "c_ok": (s_hi < float(Cs["slope_ratio_max"]) * s_lo) if (s_lo is not None and s_hi is not None) else None}
        Lm = {f"{e:g}": float(np.mean(lnA[eps == e])) for e in sorted(set(eps.tolist()))}

        def both(key):
            v = [per_draw[k][key] for k in per_draw]
            return "no (unresolved)" if any(x is None for x in v) else ("yes" if all(v) else "no")
        out["5.1"] = {"per_draw": per_draw, "draw_mean_ln_A3c": Lm, "a": both("a_monotone_decrease"), "b": both("b_ok"), "c": both("c_ok")}
        log(f"    5.1: (a) {out['5.1']['a']}, (b) {out['5.1']['b']}, (c) {out['5.1']['c']}")
        # coordinates on the grid (f5)
        G = int(TH["s_grid_points"])
        sg = np.sqrt(np.geomspace(r_A * (1 + 1e-9), E_B * (1 - 1e-9), G))
        Z = np.load(run / "theory.npz")
        nS = int(round(float(TH["T_S"]) / float(TH["output_every"]))) + 1
        aa, ab, bb = Z["aa"][:, :nS], Z["ab"][:, :nS], Z["bb"][:, :nS]
        S = np.stack([A_of(aa[i:i + 1], ab[i:i + 1], bb[i:i + 1], sg)[0].max(axis=1) for i in range(len(ALL))])     # (R, G)
        S1 = sg[None, :] * np.sqrt(np.maximum(aa, 0)).max(axis=1)[:, None]
        En = np.broadcast_to(sg[None, :] ** 2, S.shape).copy()
        a3 = sg[None, :] * np.abs(ALL[:, 2])[:, None]
        atomic_save_npz(run / "coords.npz", S=S, S1=S1, s=sg)
        sc_all = np.concatenate([scF, scR, scW])
        und_all = np.concatenate([TF["flag_undecided"], TRW["flag_undecided"]])
        noc_all = np.concatenate([TF["no_crossing"], TRW["no_crossing"]])
        bel_all = np.concatenate([TF["below_grid"], TRW["below_grid"]])
        out["n_flagged_undecided"] = {"F": int(und_all[:len(F_d)].sum()), "R": int(und_all[len(F_d):len(F_d) + len(R_d)].sum()),
                                      "W": int(und_all[len(F_d) + len(R_d):].sum())}
        nF, nR = len(F_d), len(R_d)
        cal = np.arange(nF, nF + int(cfg["rays"]["R"]["n_cal"]))
        test_R = np.arange(nF + int(cfg["rays"]["R"]["n_cal"]), nF + nR)
        test_W = np.arange(nF + nR, nF + nR + len(W_d))
        test = np.concatenate([test_R, test_W])
        ok = np.isfinite(sc_all)
        cal_ok, test_ok = cal[ok[cal]], test[ok[test]]
        out["n_cal_used"], out["n_test_used"] = int(len(cal_ok)), int(len(test_ok))
        out["unresolved_test_rays"] = [int(i) for i in test[~ok[test]]]
        out["unresolved_cal_rays"] = [int(i) for i in cal[~ok[cal]]]
        dev = cfg["device"]
        coords = {"S (first gate)": S, "energy only": En, "|a3| only": a3, "first order only": S1}
        preds = {}
        for name, C in coords.items():
            rc = RayCoordinate(C, sg, dev)
            fit = fit_cstar(rc, cal_ok, np.log(sc_all[cal_ok]))
            ln, below = rc.ln_s_pred([fit["thr"]], fit["side"])
            ln = ln[:, 0].cpu().numpy()
            e = ln - np.log(np.where(ok, sc_all, 1.0))
            preds[name] = ln
            blk = {"c_star": fit["c_star"], "side": fit["side"], "cal_median_abs": fit["train_median_abs"],
                   "test_all": summarize(e[test_ok]), "test_R": summarize(e[test_R[ok[test_R]]]) if ok[test_R].any() else None,
                   "test_W": summarize(e[test_W[ok[test_W]]]) if ok[test_W].any() else None,
                   "F_all": summarize(e[:nF][ok[:nF]]) if ok[:nF].any() else None,
                   "below_grid_test": int(below[test_ok, 0].sum().item()),
                   "unresolved_test_rays_prediction_ln_s": [float(ln[i]) for i in test[~ok[test]]]}
            out[name] = blk
            log(f"    {name}: c* {fit['c_star']:.4g} (side {fit['side']}), test median {blk['test_all']['median_abs']:.4f} "
                f"(R {blk['test_R']['median_abs'] if blk['test_R'] else None}, W {blk['test_W']['median_abs'] if blk['test_W'] else None})")
        Cf = cfg["criteria"]["first_gate"]
        med = out["S (first gate)"]["test_all"]["median_abs"]
        med_E = out["energy only"]["test_all"]["median_abs"]
        if med <= float(Cf["holds_median"]) and med < float(Cf["energy_ratio_max"]) * med_E:
            v = "holds"
        elif med >= float(Cf["fails_median"]):
            v = "does not hold"
        else:
            v = "partly holds"
        out["5.2"] = {"median": med, "energy_only_median": med_E, "verdict": v,
                      "R_median": out["S (first gate)"]["test_R"]["median_abs"] if out["S (first gate)"]["test_R"] else None,
                      "W_median": out["S (first gate)"]["test_W"]["median_abs"] if out["S (first gate)"]["test_W"] else None}
        log(f"    5.2: median {med:.4f}, energy-only {med_E:.4f} -> {v}")
        # per-ray table
        mfe = np.concatenate([F_mfe, R_mfe, np.array([np.nan if v_ is None else v_ for v_ in res.get("mfe", {}).get("W_s_c_mfe", [None] * len(W_d))])])
        meta = F_meta + R_meta + W_meta
        rows = []
        for i in range(len(ALL)):
            rr = dict(meta[i])
            rr["s_c_dns"] = float(sc_all[i]) if ok[i] else None
            rr["dns_status"] = ("undecided at T_det (no s_c)" if und_all[i] else ("no crossing below E_B" if noc_all[i] else
                                ("below the coarse grid" if bel_all[i] else "ok")))
            rr["s_c_mfe"] = float(mfe[i]) if np.isfinite(mfe[i]) else None
            rr["dns_over_mfe"] = float(sc_all[i] / mfe[i]) if ok[i] and np.isfinite(mfe[i]) else None
            rr["S_at_edge"] = float(np.interp(np.log(sc_all[i]), np.log(sg), S[i])) if ok[i] else None
            for name in coords:
                rr[f"ln_s_pred[{name}]"] = float(preds[name][i]) if np.isfinite(preds[name][i]) else "+inf (no crossing)"
                v_ = float(preds[name][i] - np.log(sc_all[i])) if ok[i] else None
                rr[f"ln_err[{name}]"] = ("+inf (no crossing)" if v_ > 0 else "-inf") if (v_ is not None and not np.isfinite(v_)) else v_
            rows.append(rr)
        out["rays"] = rows
        keys = {"S (first gate)": "pred_S", "energy only": "pred_E", "|a3| only": "pred_a3", "first order only": "pred_S1"}
        atomic_save_npz(run / "analysis.npz", sc=sc_all, mfe=mfe, **{keys[k]: v for k, v in preds.items()})
        # MFE formula and constants applied to the DNS s_c (f7, descriptive)
        try:
            out["mfe_constants_descriptive"] = mfe_descriptive(ALL, sg, sc_all, ok, cfg, dev, nF, test_R, test_W)
        except Exception as ex:  # descriptive only
            out["mfe_constants_descriptive"] = {"error": repr(ex)}
        return out

    fns = {"mfe": st_mfe, "truth_F": st_truth_F, "truth_RW": st_truth_RW, "theory": st_theory, "second_order": st_second_order,
           "analysis": st_analysis}
    for s in stages:
        timed(s, fns[s])
    log("done")


def mfe_descriptive(D, sg, sc, ok, cfg, dev, nF, test_R, test_W):
    """MFE |S| = S* (Gate 3d formula, t* = 103.5) and Gate 3c merged gain >= c*_0 (sigma = 0) along the rays on the DNS grid,
    composite max; errors against the DNS s_c (no fit)."""
    from committor.feedback import Feedback, u_th
    from committor.gain import gain_versions
    from committor.liftup import LiftUp, ray_pq_S
    from committor.systems.mfe import LAMINAR, MFE
    R3c = json.loads((REPO / cfg["data"]["gate3c_run"] / "results.json").read_text())
    R3d = json.loads((REPO / cfg["data"]["gate3d_run"] / "results.json").read_text())
    cfg3d = yaml.safe_load((REPO / cfg["data"]["gate3d_run"] / "config_used.yaml").read_text())
    m = MFE(Re=400.0)
    lu = LiftUp(m)
    fb = Feedback(m)
    kappa_b = float(R3c["item1"]["b"]["kappa"])
    S_star = float(cfg["mfe_constants"]["S_star"])
    c0 = float(cfg["mfe_constants"]["c_star_0"])
    side = int(R3d["item1"]["S"]["side"])                                     # Gate 3d item 1 (review fix: key path)
    assert abs(float(R3d["item1"]["S"]["c_star"]) - S_star) < 1e-12
    p, q = ray_pq_S(D, lu.coeffs())
    Smfe = np.abs(p[:, None] * sg[None, :] + q[:, None] * sg[None, :] ** 2)
    rc = RayCoordinate(Smfe, sg, dev)
    ln_i = rc.ln_s_pred([side * S_star], side)[0][:, 0].cpu().numpy()
    Xa = LAMINAR[None, None, :] + sg[None, :, None] * D[:, None, :]
    Xm = np.array(Xa, copy=True)
    Xm[..., 2] = u_th(Xa, fb.W_inf, kappa_b)
    gI = cfg3d["item3"]["gain"]
    r = gain_versions(m, Xm.reshape(-1, 9), [0.0], float(gI["T_g"]), float(gI["dt"]), dev, chunk=250_000)
    MT = r["max_trP"][:, 0].reshape(len(D), len(sg))
    with np.errstate(divide="ignore"):
        rcg = RayCoordinate(np.log(MT), sg, dev)
    ln_ii = rcg.ln_s_pred([c0], 1)[0][:, 0].cpu().numpy()
    ln_c = np.maximum(ln_i, ln_ii)
    lsc = np.log(np.where(ok, sc, 1.0))
    out = {"S_star": S_star, "S_side": side, "c_star_0": c0, "kappa_b": kappa_b}
    for name, ln in (("|S_MFE| >= S*", ln_i), ("merged gain >= c*_0", ln_ii), ("composite", ln_c)):
        e = ln - lsc
        out[name] = {grp: summarize(e[idx[ok[idx]]]) if ok[idx].any() else None
                     for grp, idx in (("F", np.arange(nF)), ("R test", test_R), ("W", test_W))}
    return out


if __name__ == "__main__":
    main()
