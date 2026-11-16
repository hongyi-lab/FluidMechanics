#!/usr/bin/env python
"""Gate 3d for the MFE model: is the edge two conditions at once, a lift-up streak condition |S| >= S* AND the Gate 3c
seed x gain condition (merged gain >= c*_0)? (code/configs/gate3d_mfe.yaml; criteria fixed there before the run.)

Stages (cached in results.json):
  theory       item 0: d2, J23, L, t*, F2, V_roll, V_dir (closed form vs quadrature), comparison with W(45) / W_M1,
               m-m term (not in S), ODE check of S at small amplitude, edge decomposition of S
  item1        |S| = S* on the random rays (fit on train), test metric, comparison with Gate 3c (c); S_hat (descriptive);
               t* scan (descriptive); CRITERION
  item2        composite max(s_i, s_ii) with frozen S* and c*_0 on the random rays and the noise family; binding fractions;
               (i) alone and (ii) alone; CRITERION
  item3_det    transition family (eps = 0.03 .. 3): directions from Gate 2's RNG stream, deterministic s_c (Gate 2 routine)
  item3_gain   merged gain on the transition-family grid (sigma = 0 and 3e-8)
  item3        frozen composite on the transition family, per eps; 90 deg separately; CRITERION

    python code/scripts/gate3d_mfe.py --config code/configs/gate3d_mfe.yaml --run-dir <dir> [--frac 0.01]
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
from scipy.integrate import quad_vec
from scipy.linalg import expm

from committor import edge  # noqa: E402
from committor.feedback import Feedback, analytic_cstar, first_crossing_abs, u_th  # noqa: E402
from committor.gain import gain_versions  # noqa: E402
from committor.liftup import LiftUp, S_of, mm_of, ray_pq_S  # noqa: E402
from committor.systems.mfe import LAMINAR, MFE, rhs_paper  # noqa: E402
from committor.threshold import RayCoordinate, fit_cstar, paired_diff, qinf, summarize  # noqa: E402
from committor.utils import (Status, atomic_save_npz, atomic_write_json, derive_seed, git_info, read_json,  # noqa: E402
                             stable_hash, tree_hash)

REPO = Path(__file__).resolve().parents[2]
SIM_CODE = ["liftup.py", "feedback.py", "threshold.py", "gain.py", "edge.py", "utils.py", "systems/mfe.py"]
Y = slice(3, 8)
M_IDX = [0, 1, 2, 8]


def cosF(A, B):
    return float(np.sum(A * B) / (np.linalg.norm(A) * np.linalg.norm(B)))


def trend_pairs(e, ang, draw, eps, sel, e_lo, e_hi):
    """Gate 3c (c10): per (angle, draw) pair e(eps = e_lo) - e(eps = e_hi), inf - inf := 0."""
    diffs = []
    for a in np.unique(ang[sel]):
        for dr in np.unique(draw[sel]):
            i1 = np.where(sel & (ang == a) & (draw == dr) & (eps == e_lo))[0]
            i2 = np.where(sel & (ang == a) & (draw == dr) & (eps == e_hi))[0]
            if len(i1) == 1 and len(i2) == 1:
                x1, x2 = e[i1[0]], e[i2[0]]
                diffs.append(0.0 if (np.isinf(x1) and x1 == x2) else x1 - x2)
    return np.array(diffs)


def group_block(e, sel, eps, ang, draw, trend_pair=None):
    """Summary over the selected rays, per eps and per angle; optional eps trend (Gate 3c c10)."""
    rec = {"all": summarize(e[sel]), "by_eps": {}, "by_angle": {}}
    for ep_ in sorted(set(eps[sel].tolist())):
        s_ = sel & (eps == ep_)
        r_ = summarize(e[s_])
        r_["max_abs"] = float(np.max(np.abs(e[s_])))
        rec["by_eps"][f"{ep_:g}"] = r_
    for a in np.unique(ang[sel]):
        rec["by_angle"][f"{a:g}"] = float(np.median(np.abs(e[sel & (ang == a)])))
    if trend_pair is not None:
        d = trend_pairs(e, ang, draw, eps, sel, *trend_pair)
        sm = [rec["by_eps"][k]["signed_median"] for k in rec["by_eps"]]
        rec["eps_trend"] = {"definition": f"median over (angle, draw) pairs of e(eps={trend_pair[0]:g}) - e(eps={trend_pair[1]:g})",
                            "n_pairs": int(len(d)), "median": float(qinf(d, 0.5)) if len(d) else None, "pairs": d.tolist(),
                            "range_of_signed_medians_by_eps": float(max(sm) - min(sm))}
    rec["per_ray"] = [{"ray": int(i), "angle": float(ang[i]), "eps": float(eps[i]), "draw": int(draw[i]), "ln_err": float(e[i])}
                      for i in np.where(sel)[0]]
    return rec


def binding(ln_i, ln_ii, rows=None):
    a, b = (ln_i, ln_ii) if rows is None else (ln_i[rows], ln_ii[rows])
    tie = (a == b)
    return {"n": int(len(a)), "frac_i_binds": float(np.mean(a > b)), "frac_ii_binds": float(np.mean(b > a)),
            "frac_tie": float(np.mean(tie)), "frac_i_no_crossing": float(np.mean(np.isposinf(a))),
            "frac_ii_no_crossing": float(np.mean(np.isposinf(b)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--allow-code-change", action="store_true")
    args = ap.parse_args()
    progress = not args.no_progress
    cfg = yaml.safe_load(Path(args.config).read_text())
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    device = torch.device(cfg["device"])
    src = REPO / "code" / "src" / "committor"
    ident = {"config_hash": stable_hash(cfg), "frac": args.frac,
             "sim_code_hash": tree_hash([src / f for f in SIM_CODE] + [Path(__file__).resolve()])}
    old = read_json(run / "run_identity.json")
    if old is None:
        atomic_write_json(run / "run_identity.json", ident)
    elif old["config_hash"] != ident["config_hash"] or old["frac"] != ident["frac"]:
        sys.exit(f"{run} was started with {old}, now {ident}; use a new run dir")
    elif old["sim_code_hash"] != ident["sim_code_hash"] and not args.allow_code_change:
        sys.exit(f"code changed since {run} was started; new run dir or --allow-code-change")
    if not (run / "config_used.yaml").exists():
        (run / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    man = read_json(run / "manifest.json", {"invocations": []})
    man["invocations"].append({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv, "git": git_info(REPO),
                               "sim_code_hash": ident["sim_code_hash"], "python": platform.python_version(),
                               "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})
    atomic_write_json(run / "manifest.json", man)
    status = Status(run / "status.json")
    res = read_json(run / "results.json", {}) or {}
    seed = int(cfg["seed"])
    seed3 = int(cfg["data"]["gate3_seed"])
    TPc = cfg["threshold_prediction"]
    NL, NR = int(TPc["n_levels"]), int(TPc["n_refine"])

    def save():
        atomic_write_json(run / "results.json", res)

    def timed(name, fn):
        if name in res:
            return
        print(f"--- {name}", flush=True)
        t0 = time.perf_counter()
        out = fn()
        out["seconds"] = time.perf_counter() - t0
        res[name] = out
        status.update(name, done=True, seconds=out["seconds"])
        save()
        print(f"    {name}: {out['seconds']:.1f} s", flush=True)

    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=device)  # noqa: E731

    # ------------------------------------------------------------ data (Gate 2 rays, Gate 3 split) --------------------
    g2, g3, g3b, g3c = (REPO / cfg["data"][k] for k in ("gate2_run", "gate3_run", "gate3b_run", "gate3c_run"))
    cfg2 = yaml.safe_load((REPO / cfg["data"]["gate2_config"]).read_text())
    cfg3c = yaml.safe_load((g3c / "config_used.yaml").read_text())
    R2, R3c = (json.loads((d / "results.json").read_text()) for d in (g2, g3c))
    RY, SC = np.load(g2 / "rays.npz"), np.load(g2 / "scan.npz")
    rng_sub = np.random.default_rng(derive_seed(seed3, "frac"))
    N_all = len(RY["dirs"])
    rays = np.arange(N_all) if args.frac >= 1 else np.sort(rng_sub.choice(N_all, max(30, int(round(args.frac * N_all))), replace=False))
    dirs, split = RY["dirs"][rays], RY["split"][rays]
    s_c, s_grid = SC["s_c"][rays], SC["s_grid"]
    N, G = len(rays), len(s_grid)
    ln_sc = np.log(s_c)
    tr_r, va_r, te_r = (np.where(split == k)[0] for k in (0, 1, 2))
    s_lo, s_hi = float(s_grid[0]), float(s_grid[-1])
    boot = np.random.default_rng(derive_seed(seed3, "boot")).integers(0, len(te_r), (int(TPc["bootstrap"]), len(te_r)))
    Xedge = s_c[:, None] * dirs
    res.setdefault("meta", {"n_rays": int(N), "split_counts": [int(len(tr_r)), int(len(va_r)), int(len(te_r))],
                            "grid_points": int(G), "grid_range": [s_lo, s_hi], "data": cfg["data"],
                            "criteria": {k: cfg[k]["criterion"] for k in ("item1", "item2", "item3")}})
    save()
    m = MFE(Re=400.0)
    lu = LiftUp(m)
    fb = Feedback(m)
    cS = lu.coeffs()
    T_c = float(R3c["item1"]["c"]["T"])
    kappa_b = float(R3c["item1"]["b"]["kappa"])
    I2 = cfg["item2"]
    c0 = float(I2["c_star_0"])
    assert abs(c0 - float(cfg3c["item2"]["c_star_0"])) < 1e-12

    # references on the test rays (Gate 3, 3b, 3c), aligned with te_r
    te_ids = rays[te_r]

    def ref_from(Z, name, key="names"):
        names = list(Z[key])
        pos = {int(r): i for i, r in enumerate(Z["test_rays"])}
        if name not in names or not all(int(r) in pos for r in te_ids):
            return None
        return Z["E"][names.index(name)][[pos[int(r)] for r in te_ids]]

    G3T, G3bT, G3cT = np.load(g3 / "test_errors.npz"), np.load(g3b / "test_errors.npz"), np.load(g3c / "item1_test_errors.npz")
    REF = {"Gate 3c: (c) W(T)": ref_from(G3cT, "c"), "Gate 3c: (b) kappa W_inf": ref_from(G3cT, "b"),
           "Gate 3b: M1": ref_from(G3bT, "M1"), "Gate 3b: M2": ref_from(G3bT, "M2"), "Gate 3: |a3|": ref_from(G3T, "|a3|"),
           "Gate 3: MLP (9D, polar input) = ceiling": ref_from(G3T, "MLP (9D, polar input)")}
    REF = {k: v for k, v in REF.items() if v is not None}

    # ------------------------------------------------------------ item 0 ----------------------------------------------
    def st_theory():
        beta, gamma = np.pi / 2, 2 * np.pi / m.Lz
        ts = lu.t_star
        tt = np.linspace(0.0, 1000.0, 200001)
        Vr, Vd = lu.V_roll(ts), lu.V_dir(ts)
        Gm = lambda X, s: expm(lu.Ay.T * s) @ X @ expm(lu.Ay * s)  # noqa: E731
        Vr_q, _ = quad_vec(lambda s: lu.L(ts - s) * Gm(lu.F, s), 0.0, ts, epsabs=1e-13, epsrel=1e-12)
        Vd_q, _ = quad_vec(lambda s: np.exp(-lu.d2 * (ts - s)) * Gm(lu.F2, s), 0.0, ts, epsabs=1e-13, epsrel=1e-12)
        V = cS[2]
        Vn = V / cS[1]                                          # S / L(t*) = a3 + (e^{-d2 t*}/L) a2 + y^T Vn y
        W45, Wc = fb.W(45.0), fb.W(T_c)
        th3c = np.load(g3c / "theory.npz")
        Wm = th3c["W_M1"]
        Ts = np.concatenate([np.geomspace(0.1, 1e4, 801)])
        cos_T = np.array([cosF(fb.W(t), Vn) for t in Ts])
        nW = np.array([np.linalg.norm(fb.W(t)) for t in Ts])
        nVn = float(np.linalg.norm(Vn))
        j = int(np.argmax(nW >= nVn)) if (nW >= nVn).any() else -1
        T_match = float(np.interp(nVn, nW[j - 1:j + 1], Ts[j - 1:j + 1])) if j > 0 else None
        Mmm = lu.M_mm()
        J = lu.J
        Q2 = m.Q[1]
        Yi = list(range(3, 8))
        struct = {"J_row_a2": J[1].tolist(), "a2_eq_linear_except_a2_a3": float(np.abs(np.delete(J[1], [1, 2])).max()),
                  "a2_eq_my_max": float(np.abs(Q2[np.ix_(M_IDX, Yi)]).max()),
                  "a2_eq_mm_nonzero": [[MODES_[i], MODES_[k], float(Q2[i, k] + Q2[k, i]) if i != k else float(Q2[i, k])]
                                       for i in range(9) for k in range(i, 9) if i in M_IDX and k in M_IDX and abs(Q2[i, k]) > 0],
                  "a2_eq_yy_nonzero_products": [[MODES_[i], MODES_[k], float(2 * lu.F2[i - 3, k - 3]) if i != k else float(lu.F2[i - 3, k - 3])]
                                                for i in range(3, 8) for k in range(i, 8) if abs(lu.F2[i - 3, k - 3]) > 0]}
        out = {"d2": lu.d2, "d2_formula_(4beta^2/3+gamma^2)/Re": float((4 * beta ** 2 / 3 + gamma ** 2) / m.Re),
               "d3": lu.d3, "J23": lu.J23, "J23_formula_-sqrt(3/2)beta*gamma/kappa_bg": float(-np.sqrt(1.5) * beta * gamma / np.sqrt(beta ** 2 + gamma ** 2)),
               "t_star": ts, "t_star_numeric_argmax_absL": float(tt[np.argmax(np.abs(lu.L(tt)))]),
               "L_t_star": float(lu.L(ts)), "e_minus_d2_t_star": cS[0], "structure": struct,
               "F2": lu.F2.tolist(), "V_roll": Vr.tolist(), "V_dir": Vd.tolist(), "V": V.tolist(),
               "norm_V_roll": float(np.linalg.norm(Vr)), "norm_V_dir": float(np.linalg.norm(Vd)),
               "closed_minus_quadrature_maxabs": {"V_roll": float(np.abs(Vr - Vr_q).max()), "V_dir": float(np.abs(Vd - Vd_q).max())},
               "S_over_L": {"a2_coefficient": cS[0] / cS[1], "V_over_L_frobenius": nVn,
                            "cos_W45": cosF(Vn, W45), "norm_ratio_to_W45": nVn / float(np.linalg.norm(W45)),
                            "cos_W_Tc": cosF(Vn, Wc), "T_c": T_c, "cos_W_M1": cosF(Vn, Wm), "norm_ratio_to_W_M1": nVn / float(np.linalg.norm(Wm)),
                            "cos_W_inf": cosF(Vn, fb.W_inf), "cos_F": cosF(Vn, fb.F),
                            "T_at_max_cos_W_T": float(Ts[int(np.argmax(cos_T))]), "max_cos_W_T": float(np.max(cos_T)),
                            "T_where_norm_W_T_equals_norm_V_over_L": T_match,
                            "kappa_of_V_over_L_on_W45": float(np.sum(W45 * Vn) / np.sum(W45 ** 2)),
                            "cos_V_dir_over_L_with_V_over_L": cosF(Vd / cS[1], Vn),
                            "share_V_dir_in_V_frobenius": float(np.linalg.norm(Vd) / np.linalg.norm(V))},
               "M_mm": Mmm.tolist()}
        # ODE check at small amplitude: a2(t*) vs S (and S_hat = S + m-m term)
        oc = cfg["item0"]["ode_check"]
        f = lambda z: rhs_paper(z, m.Re, m.Lx, m.Lz)  # noqa: E731

        def rk4(a, dt, n):
            for _ in range(n):
                k1 = f(a)
                k2 = f(a + 0.5 * dt * k1)
                k3 = f(a + 0.5 * dt * k2)
                k4 = f(a + dt * k3)
                a = a + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            return a
        rng = np.random.default_rng(derive_seed(seed, "ode_check"))
        n_steps = int(round(ts / float(oc["dt"])))
        dt_ = ts / n_steps
        rows = []
        for kind in ("no a1, a9", "all 9 modes"):
            for trial in range(int(oc["n_dirs"])):
                d = rng.standard_normal(9)
                if kind == "no a1, a9":
                    d[[0, 8]] = 0.0
                d /= np.linalg.norm(d)
                for eps in oc["eps"]:
                    a = rk4(LAMINAR + eps * d, dt_, n_steps)
                    Sp = float(S_of(eps * d, cS))
                    mm = float(mm_of(eps * d, Mmm))
                    rows.append({"kind": kind, "trial": trial, "eps": float(eps), "a2_ode": float(a[1]), "S": Sp, "mm": mm,
                                 "resid_S_over_eps2": (float(a[1]) - Sp) / eps ** 2, "resid_Shat_over_eps2": (float(a[1]) - Sp - mm) / eps ** 2,
                                 "mm_over_eps2": mm / eps ** 2, "rel_resid_S": abs(float(a[1]) - Sp) / abs(float(a[1]))})
        out["ode_check"] = rows
        summ = {}
        for kind in ("no a1, a9", "all 9 modes"):
            for eps in oc["eps"]:
                rr = [r for r in rows if r["kind"] == kind and r["eps"] == float(eps)]
                summ[f"{kind} | eps={float(eps):g}"] = {"max_abs_resid_S_over_eps2": max(abs(r["resid_S_over_eps2"]) for r in rr),
                                                        "max_abs_resid_Shat_over_eps2": max(abs(r["resid_Shat_over_eps2"]) for r in rr),
                                                        "max_rel_resid_S": max(r["rel_resid_S"] for r in rr)}
        out["ode_check_summary"] = summ
        # decomposition of S at the random-ray edges (all rays, descriptive)
        Se = S_of(Xedge, cS)
        parts = {"a2 term": cS[0] * Xedge[:, 1], "a3 term (L a3)": cS[1] * Xedge[:, 2],
                 "y-y term": np.einsum("ni,ij,nj->n", Xedge[:, Y], V, Xedge[:, Y]),
                 "m-m term (not in S)": mm_of(Xedge, Mmm)}
        out["edge_decomposition_median_abs_over_abs_S"] = {k: float(np.median(np.abs(v) / np.abs(Se))) for k, v in parts.items()}
        out["edge_quantiles_abs_S_5_50_95"] = [float(x) for x in np.quantile(np.abs(Se), [0.05, 0.5, 0.95])]
        uc = u_th(Xedge, Wc)
        out["edge_corr_S_over_L_vs_u_th_c"] = float(np.corrcoef(Se / cS[1], uc)[0, 1])
        atomic_save_npz(run / "theory.npz", V_roll=Vr, V_dir=Vd, V=V, F2=lu.F2, F=lu.F, M_mm=Mmm, W45=W45, W_M1=Wm,
                        Ts=Ts, cos_T=cos_T, normW_T=nW, tau=tt[::100], L_tau=lu.L(tt[::100]))
        return out

    MODES_ = [f"a{i}" for i in range(1, 10)]
    timed("theory", st_theory)

    # ------------------------------------------------------------ item 1 ---------------------------------------------
    p_S, q_S = ray_pq_S(dirs, cS)
    Mmm = np.array(res["theory"]["M_mm"])
    q_mm = np.einsum("ni,ij,nj->n", dirs[:, M_IDX], Mmm, dirs[:, M_IDX])      # x_m = s d_m (a1 enters as a1 - 1 = s d1)

    def eval_grid(p, q, with_boot=True):
        C = np.abs(p[:, None] * s_grid[None, :] + q[:, None] * s_grid[None, :] ** 2)
        rc = RayCoordinate(C, s_grid, device)
        fit = fit_cstar(rc, tr_r, ln_sc[tr_r], NL, NR)
        thr = [fit["side"] * fit["c_star"]]
        ln_all, below = rc.ln_s_pred(thr, fit["side"])
        ln_all = ln_all[:, 0].cpu().numpy()
        e = ln_all - ln_sc
        row = {"side": fit["side"], "c_star": fit["c_star"], "train_median_abs": fit["train_median_abs"],
               "val_median_abs": float(np.median(np.abs(e[va_r]))),
               "frac_below_grid_test": float(below[te_r, 0].float().mean().item()),
               "test": summarize(e[te_r], boot if with_boot else None)}
        del rc
        return row, e, ln_all

    def st_item1():
        out = {}
        row, e_S, ln_S = eval_grid(p_S, q_S)
        out["S"] = {"label": "|S| = S*, t = t* (no fitted time)", "t_star": lu.t_star, **row}
        row_h, e_h, _ = eval_grid(p_S, q_S + q_mm)
        out["S_hat_(descriptive)"] = {"label": "|S + x_m^T M x_m| = S* (m-m term added; not the user's S)", **row_h}
        # t scan (analytic, descriptive)
        tc = cfg["item1"]["t_scan"]
        tg = np.unique(np.concatenate([[0.0], np.geomspace(float(tc["min"]), float(tc["max"]), int(tc["n"])), [lu.t_star]]))
        tr_ln, te_ln = T(ln_sc[tr_r]), T(ln_sc[te_r])
        med_tr, med_te, cst = np.empty(len(tg)), np.empty(len(tg)), np.empty(len(tg))
        for i, t in enumerate(tg):
            ct = lu.coeffs(t) if t > 0 else (1.0, 0.0, np.zeros((5, 5)))
            p_, q_ = ray_pq_S(dirs, ct)
            med_tr[i], cst[i] = analytic_cstar(T(p_[tr_r]), T(q_[tr_r]), tr_ln, s_lo, s_hi, NL, NR)
            s_ = first_crossing_abs(T(p_[te_r]), T(q_[te_r]), torch.tensor(cst[i], dtype=torch.float64, device=device), s_lo, s_hi)
            ee = (torch.log(s_) - te_ln).abs()
            ee = torch.where(torch.isnan(ee), torch.full_like(ee, float("inf")), ee)
            med_te[i] = float(np.median(ee.cpu().numpy()))
        k = int(np.argmin(med_tr))
        kt = int(np.where(tg == lu.t_star)[0][0])
        out["t_scan_(descriptive)"] = {"t_at_min_train_median": float(tg[k]), "min_train_median": float(med_tr[k]),
                                        "test_median_at_that_t": float(med_te[k]), "train_median_at_t_star": float(med_tr[kt]),
                                        "test_median_at_t_star_analytic": float(med_te[kt]),
                                        "t_range_train_median_within_0.005_of_min": [float(tg[med_tr <= med_tr[k] + 0.005].min()), float(tg[med_tr <= med_tr[k] + 0.005].max())],
                                        "train_median_at_t0_(|a2|)": float(med_tr[0])}
        atomic_save_npz(run / "item1_t_scan.npz", t=tg, train_median=med_tr, test_median=med_te, c_star=cst)
        # references and paired differences
        out["references"] = {k_: summarize(v, boot) for k_, v in REF.items()}
        allerr = {"S": e_S[te_r], "S_hat": e_h[te_r], **REF}
        pairs = [("S", "Gate 3c: (c) W(T)"), ("S", "Gate 3c: (b) kappa W_inf"), ("S", "Gate 3b: M1"), ("S", "Gate 3: |a3|"),
                 ("S", "Gate 3: MLP (9D, polar input) = ceiling"), ("S_hat", "S")]
        out["paired"] = [{"a": a_, "b": b_, **paired_diff(allerr[a_], allerr[b_], boot)} for a_, b_ in pairs if a_ in allerr and b_ in allerr]
        Cc = cfg["item1"]["criterion"]
        mS = out["S"]["test"]["median_abs"]
        mc = float(np.median(np.abs(REF["Gate 3c: (c) W(T)"]))) if "Gate 3c: (c) W(T)" in REF else None
        okm = mS <= float(Cc["holds_median"])
        okc = mc is not None and (mS - mc) <= float(Cc["max_worse_than_3c_c"])
        if okm and okc:
            verdict = f"T explained by lift-up: test median {mS:.4f} <= {Cc['holds_median']} and {mS - mc:+.4f} vs 3c (c) <= {Cc['max_worse_than_3c_c']}"
        else:
            fail = ([f"median {mS:.4f} > {Cc['holds_median']}"] if not okm else []) + \
                   ([f"{mS - mc:+.4f} vs 3c (c) > {Cc['max_worse_than_3c_c']}"] if (mc is not None and not okc) else [])
            verdict = "not explained by lift-up: " + "; ".join(fail)
        out["criterion"] = {"S_test_median": mS, "3c_c_test_median": mc, "difference": (mS - mc) if mc is not None else None,
                            "verdict": verdict, "ci_S": out["S"]["test"]["median_abs_ci"]}
        atomic_save_npz(run / "item1_errors.npz", e_S=e_S, ln_S=ln_S, e_S_hat=e_h, split=split, rays=rays)
        print("    ITEM 1 CRITERION:", verdict, flush=True)
        print(f"    S: side {row['side']}, S* {row['c_star']:.5g}, test {mS:.4f} q90 {out['S']['test']['q90_abs']:.4f}; "
              f"S_hat test {row_h['test']['median_abs']:.4f}; t scan min train at t = {tg[k]:.1f}", flush=True)
        return out

    timed("item1", st_item1)
    S_side, S_star = int(res["item1"]["S"]["side"]), float(res["item1"]["S"]["c_star"])
    S_thr = S_side * S_star

    # ------------------------------------------------------------ item 2 ---------------------------------------------
    sig3c = [float(s) for s in cfg3c["item2"]["sigmas"]]
    FD = np.load(g2 / "family_det.npz")
    fam_meta = R2["family_det"]["rays"]
    fam_dirs, fam_sc = FD["d"], FD["s_c"]
    fam_eps = np.array([f["eps"] for f in fam_meta])
    fam_ang = np.array([f["angle"] for f in fam_meta])
    fam_draw = np.array([f["draw"] for f in fam_meta])
    r_A, E_B = float(cfg2["sets"]["r_A"]), float(cfg2["sets"]["E_B"])
    fam_grid = np.sqrt(np.geomspace(r_A * (1 + 1e-9), E_B * (1 - 1e-9), int(I2["family"]["grid_points"])))
    use = np.isin(fam_eps, [float(e) for e in I2["family"]["eps_used"]]) & np.isfinite(fam_sc)
    trend_pair = tuple(float(x) for x in I2["family"]["trend_pair"])

    def S_grid_ln(dirs_, grid_):
        p_, q_ = ray_pq_S(dirs_, cS)
        rc = RayCoordinate(np.abs(p_[:, None] * grid_[None, :] + q_[:, None] * grid_[None, :] ** 2), grid_, device)
        ln_, _ = rc.ln_s_pred([S_thr], S_side)
        return ln_[:, 0].cpu().numpy()

    def gain_ln(MTk, grid_):
        with np.errstate(divide="ignore"):
            rc = RayCoordinate(np.log(MTk), grid_, device)
        ln_, _ = rc.ln_s_pred([c0], 1)
        return ln_[:, 0].cpu().numpy()

    def st_item2():
        out = {"S_star": S_star, "S_side": S_side, "c_star_0": c0, "kappa_b_3c": kappa_b,
               "sigma_random": float(I2["sigma_random"]), "sigma_family": float(I2["sigma_family"])}
        # random rays
        ln_i = np.load(run / "item1_errors.npz")["ln_S"]
        files = sorted(g3c.glob("gain_grid_*.npz"))
        kr = sig3c.index(float(I2["sigma_random"]))
        MT = np.full((N, G), np.nan)
        pos = {int(r): i for i, r in enumerate(rays)}
        for f in files:
            Z = np.load(f)
            for j, r in enumerate(Z["rows"]):
                if int(r) in pos:
                    MT[pos[int(r)]] = Z["max_trP"][j, :, kr]
        assert np.isfinite(MT).any(axis=1).all()
        ln_ii = gain_ln(MT, s_grid)
        del MT
        ln_c = np.maximum(ln_i, ln_ii)
        e_i, e_ii, e_c = ln_i - ln_sc, ln_ii - ln_sc, ln_c - ln_sc
        e3c = np.load(g3c / "item2_errors.npz")
        er3c = e3c["e_rev"][rays]
        ok = np.isfinite(er3c) & np.isfinite(e_ii)
        out["check_ii_vs_3c_reverse"] = {"max_abs_diff": float(np.max(np.abs(er3c[ok] - e_ii[ok]))), "n_nonfinite_mismatch": int(np.sum(np.isfinite(er3c) != np.isfinite(e_ii)))}
        out["random"] = {"composite": {"test": summarize(e_c[te_r], boot), "all_rays": summarize(e_c),
                                       "train_median_abs": float(np.median(np.abs(e_c[tr_r]))), "val_median_abs": float(np.median(np.abs(e_c[va_r])))},
                         "i_alone": {"test": summarize(e_i[te_r], boot), "all_rays": summarize(e_i)},
                         "ii_alone": {"test": summarize(e_ii[te_r], boot), "all_rays": summarize(e_ii)},
                         "binding_test": binding(ln_i, ln_ii, te_r), "binding_all": binding(ln_i, ln_ii)}
        # error by binding condition (test rays)
        bi = ln_i[te_r] > ln_ii[te_r]
        out["random"]["composite_test_by_binding"] = {"i_binds": summarize(e_c[te_r][bi]) if bi.any() else None,
                                                      "ii_binds_or_tie": summarize(e_c[te_r][~bi]) if (~bi).any() else None}
        allerr = {"composite": e_c[te_r], "(i) alone": e_i[te_r], "(ii) alone": e_ii[te_r], **REF}
        pairs = [("composite", "(i) alone"), ("composite", "(ii) alone"), ("composite", "Gate 3c: (c) W(T)"), ("composite", "Gate 3b: M1")]
        out["random"]["paired"] = [{"a": a_, "b": b_, **paired_diff(allerr[a_], allerr[b_], boot)} for a_, b_ in pairs if a_ in allerr and b_ in allerr]
        # noise family (sigma = 0)
        GF = np.load(g3c / "gain_family.npz")
        assert np.allclose(GF["s"], fam_grid)
        kf = sig3c.index(float(I2["sigma_family"]))
        lnf_ii = gain_ln(GF["max_trP"][:, :, kf], fam_grid)
        lnf_i = S_grid_ln(fam_dirs, fam_grid)
        lnf_c = np.maximum(lnf_i, lnf_ii)
        lsc = np.log(np.where(np.isfinite(fam_sc), fam_sc, 1.0))
        ef = {"composite": lnf_c - lsc, "i_alone": lnf_i - lsc, "ii_alone": lnf_ii - lsc}
        er3cf = e3c["e_fam_rev"]
        okf = use & np.isfinite(er3cf) & np.isfinite(ef["ii_alone"])
        out["check_ii_vs_3c_family_c_star_0"] = {"max_abs_diff": float(np.max(np.abs(er3cf[okf] - ef["ii_alone"][okf]))),
                                                 "n_nonfinite_mismatch": int(np.sum(np.isfinite(er3cf[use]) != np.isfinite(ef["ii_alone"][use])))}
        out["family"] = {k_: group_block(v, use, fam_eps, fam_ang, fam_draw, trend_pair) for k_, v in ef.items()}
        no90 = use & (fam_ang != 90.0)
        out["family_without_90_(descriptive)"] = {k_: {"all": summarize(v[no90]), "eps_trend_median": float(qinf(trend_pairs(v, fam_ang, fam_draw, fam_eps, no90, *trend_pair), 0.5))}
                                                  for k_, v in ef.items()}
        out["family_binding"] = binding(lnf_i[use], lnf_ii[use])
        out["family_binding_by_eps"] = {f"{e_:g}": binding(lnf_i[use & (fam_eps == e_)], lnf_ii[use & (fam_eps == e_)]) for e_ in sorted(set(fam_eps[use].tolist()))}
        atomic_save_npz(run / "item2_errors.npz", ln_i=ln_i, ln_ii=ln_ii, e_c=e_c, e_i=e_i, e_ii=e_ii, split=split, rays=rays,
                        fam_ln_i=lnf_i, fam_ln_ii=lnf_ii, fam_e_c=ef["composite"], fam_e_i=ef["i_alone"], fam_e_ii=ef["ii_alone"],
                        fam_use=use, fam_eps=fam_eps, fam_ang=fam_ang, fam_draw=fam_draw, fam_sc=fam_sc)
        Cc = I2["criterion"]
        mr = out["random"]["composite"]["test"]["median_abs"]
        mf = out["family"]["composite"]["all"]["median_abs"]
        tr_ = out["family"]["composite"]["eps_trend"]["median"]
        oks = [mr <= float(Cc["random_median"]), mf <= float(Cc["family_median"]), tr_ is not None and abs(tr_) <= float(Cc["eps_trend_abs"])]
        names = [f"random test median {mr:.4f} (<= {Cc['random_median']})", f"family median {mf:.4f} (<= {Cc['family_median']})",
                 f"|eps trend| {abs(tr_) if tr_ is not None else float('nan'):.4f} (<= {Cc['eps_trend_abs']})"]
        verdict = ("holds: " + "; ".join(names)) if all(oks) else ("does not hold: fails " + "; ".join(n for n, o in zip(names, oks) if not o)
                                                                    + " | passes " + ("; ".join(n for n, o in zip(names, oks) if o) or "none"))
        out["criterion"] = {"random_test_median": mr, "family_median": mf, "eps_trend": tr_, "parts_ok": oks, "verdict": verdict,
                            "ci_random": out["random"]["composite"]["test"]["median_abs_ci"]}
        print("    ITEM 2 CRITERION:", verdict, flush=True)
        print("    random test: composite {:.4f}, (i) {:.4f}, (ii) {:.4f}; binding (i) {:.3f} (ii) {:.3f}".format(
            mr, out["random"]["i_alone"]["test"]["median_abs"], out["random"]["ii_alone"]["test"]["median_abs"],
            out["random"]["binding_test"]["frac_i_binds"], out["random"]["binding_test"]["frac_ii_binds"]), flush=True)
        print("    family signed by eps (composite):", {k_: round(v["signed_median"], 4) for k_, v in out["family"]["composite"]["by_eps"].items()}, flush=True)
        return out

    timed("item2", st_item2)

    # ------------------------------------------------------------ item 3: transition family ---------------------------
    I3 = cfg["item3"]
    F2c = cfg2["noise_family"]
    rngf = np.random.default_rng(derive_seed(int(cfg2["seed"]), "family"))
    r_by = {}
    for ang in F2c["angles_deg"]:                                              # Gate 2 RNG stream, (c7) nested r
        for k in range(int(F2c["draws_per_eps"])):
            r = np.zeros(9)
            r[3:8] = rngf.normal(size=5)
            r[3:8] /= np.linalg.norm(r[3:8])
            r_by[(float(ang), k)] = r

    def direction(ang, eps, k):
        t = math.radians(ang)
        base = np.zeros(9)
        base[2], base[1] = math.cos(t), math.sin(t)
        return base.copy() if eps == 0 else (base + eps * r_by[(float(ang), k)]) / math.sqrt(1 + eps ** 2)

    t_rays = []
    for ang in list(I3["angles_deg"]) + [float(I3["separate_angle_deg"])]:
        for eps in I3["eps"]:
            for k in range(int(I3["draws"])):
                if args.frac < 1 and not (k == 0 and float(eps) in (float(I3["eps"][0]), float(I3["eps"][-1]))):
                    continue
                t_rays.append({"angle": float(ang), "eps": float(eps), "draw": k, "d": direction(float(ang), float(eps), k)})
    D3 = np.stack([r_["d"] for r_ in t_rays])
    t_ang = np.array([r_["angle"] for r_ in t_rays])
    t_eps = np.array([r_["eps"] for r_ in t_rays])
    t_draw = np.array([r_["draw"] for r_ in t_rays])
    Dg2 = cfg2["deterministic"]
    m0 = MFE(Re=cfg2["system"]["Re"], Lx_over_pi=cfg2["system"]["Lx_over_pi"], Lz_over_pi=cfg2["system"]["Lz_over_pi"])
    rhs = lambda a: rhs_paper(a, m0.Re, m0.Lx, m0.Lz)  # noqa: E731
    det_grid = np.sqrt(np.geomspace(r_A * (1 + 1e-9), E_B * (1 - 1e-9), int(Dg2["dense_grid"]["n_points"])))

    def fc(dirs_, desc):
        return edge.first_crossing(rhs, dirs_, det_grid, int(Dg2["n_bisect"]), float(Dg2["dt"]), float(Dg2["T_det"]), E_B, r_A, "rk4",
                                   device=device, chunk=1 << 20, progress=progress, desc=desc)

    def st_item3_det():
        out = {}
        # check 1: the Gate 2 family directions are reproduced from the RNG stream
        rebuilt = np.stack([direction(f["angle"], f["eps"], f["draw"]) for f in fam_meta])
        out["check_gate2_directions_max_abs_diff"] = float(np.max(np.abs(rebuilt - fam_dirs)))
        # check 2: the det routine reproduces Gate 2's s_c on the eps = 1e-2 rays
        sel = np.where(fam_eps == float(I3["check_gate2_eps"]))[0]
        rc_ = fc(fam_dirs[sel], "check: Gate 2 eps=1e-2 rays")
        a_, b_ = rc_["s_c"], fam_sc[sel]
        okk = np.isfinite(a_) & np.isfinite(b_)
        out["check_gate2_s_c"] = {"n": int(len(sel)), "max_rel_diff": float(np.max(np.abs(a_[okk] / b_[okk] - 1))) if okk.any() else None,
                                  "n_nonfinite_mismatch": int(np.sum(np.isfinite(a_) != np.isfinite(b_)))}
        r = fc(D3, "transition family dense scan")
        atomic_save_npz(run / "item3_det.npz", d=D3, s_c=r["s_c"], no_crossing=r["no_crossing"], below_grid=r["below_grid"],
                        n_switches=r["n_switches"], angle=t_ang, eps=t_eps, draw=t_draw)
        out["n_rays"] = int(len(D3))
        out["n_unresolved_s_c"] = int(np.sum(~np.isfinite(r["s_c"])))
        out["rays"] = [{"angle": float(t_ang[i]), "eps": float(t_eps[i]), "draw": int(t_draw[i]),
                        "s_c": float(r["s_c"][i]) if np.isfinite(r["s_c"][i]) else None, "n_switches": int(r["n_switches"][i]),
                        "no_crossing": bool(r["no_crossing"][i]), "below_grid": bool(r["below_grid"][i])} for i in range(len(D3))]
        print(f"    direction check {out['check_gate2_directions_max_abs_diff']:.2e}; s_c check {out['check_gate2_s_c']}", flush=True)
        return out

    timed("item3_det", st_item3_det)
    t_grid = np.sqrt(np.geomspace(r_A * (1 + 1e-9), E_B * (1 - 1e-9), int(I3["grid_points"])))
    sig3 = [float(s) for s in I3["gain"]["sigmas"]]

    def st_item3_gain():
        Xa = LAMINAR[None, None, :] + t_grid[None, :, None] * D3[:, None, :]
        Xm = np.array(Xa, copy=True)
        Xm[..., 2] = u_th(Xa, fb.W_inf, kappa_b)                       # Gate 3c merged state (kappa from 3c (b), frozen)
        r = gain_versions(m, Xm.reshape(-1, 9), sig3, float(I3["gain"]["T_g"]), float(I3["gain"]["dt"]), device, chunk=250_000)
        shp = (len(D3), len(t_grid))
        atomic_save_npz(run / "item3_gain.npz", max_trP=r["max_trP"].reshape(*shp, -1), t_at_max=r["t_at_max"].reshape(*shp, -1), s=t_grid,
                        u_th=Xm[..., 2], a3=Xa[..., 2])
        return {"n_rays": int(len(D3)), "grid_points": int(len(t_grid)), "states": int(len(D3) * len(t_grid))}

    timed("item3_gain", st_item3_gain)

    def st_item3():
        Z = np.load(run / "item3_det.npz")
        sc3 = Z["s_c"]
        GZ = np.load(run / "item3_gain.npz")
        out = {"S_star": S_star, "c_star_0": c0}
        lsc = np.log(np.where(np.isfinite(sc3), sc3, 1.0))
        ln_i = S_grid_ln(D3, t_grid)
        res_s = {}
        for ks, sg in enumerate(sig3):
            ln_ii = gain_ln(GZ["max_trP"][:, :, ks], t_grid)
            res_s[sg] = (ln_ii, np.maximum(ln_i, ln_ii))
        ln_ii, ln_c = res_s[float(I3["gain"]["sigma_used"])]
        E3 = {"composite": ln_c - lsc, "i_alone": ln_i - lsc, "ii_alone": ln_ii - lsc}
        ok = np.isfinite(sc3)
        main_ = ok & (t_ang != float(I3["separate_angle_deg"]))
        sep = ok & (t_ang == float(I3["separate_angle_deg"]))
        out["n_main"], out["n_main_unresolved"] = int(np.sum(t_ang != float(I3["separate_angle_deg"]))), int(np.sum(~ok & (t_ang != float(I3["separate_angle_deg"]))))
        out["main"] = {k_: group_block(v, main_, t_eps, t_ang, t_draw) for k_, v in E3.items()}
        out["angle_90"] = {k_: group_block(v, sep, t_eps, t_ang, t_draw) for k_, v in E3.items()} if sep.any() else None
        out["binding_main"] = binding(ln_i[main_], ln_ii[main_])
        out["binding_main_by_eps"] = {f"{e_:g}": binding(ln_i[main_ & (t_eps == e_)], ln_ii[main_ & (t_eps == e_)]) for e_ in sorted(set(t_eps[main_].tolist()))}
        ss = [s for s in sig3 if s != float(I3["gain"]["sigma_used"])]
        out["composite_other_sigma"] = {f"sigma={s:g}": summarize(res_s[s][1][main_] - lsc[main_]) for s in ss}
        out["s_c_by_eps_median"] = {f"{e_:g}": float(np.median(sc3[main_ & (t_eps == e_)])) for e_ in sorted(set(t_eps[main_].tolist()))}
        atomic_save_npz(run / "item3_errors.npz", ln_i=ln_i, ln_ii=ln_ii, e_c=E3["composite"], e_i=E3["i_alone"], e_ii=E3["ii_alone"],
                        s_c=sc3, angle=t_ang, eps=t_eps, draw=t_draw, main=main_, sep=sep)
        Cc = I3["criterion"]
        mm_ = out["main"]["composite"]["all"]["median_abs"]
        over = [k_ for k_, v in out["main"]["composite"]["by_eps"].items() if v["median_abs"] > float(Cc["holds_median"])]
        verdict = (f"holds: pooled median {mm_:.4f} <= {Cc['holds_median']}" if mm_ <= float(Cc["holds_median"]) else
                   f"does not hold: pooled median {mm_:.4f} > {Cc['holds_median']}") + f"; eps with median > {Cc['holds_median']}: {over or 'none'}"
        out["criterion"] = {"pooled_median": mm_, "eps_over": over, "verdict": verdict}
        print("    ITEM 3 CRITERION:", verdict, flush=True)
        print("    per eps (composite median |ln|):", {k_: round(v["median_abs"], 4) for k_, v in out["main"]["composite"]["by_eps"].items()}, flush=True)
        print("    per eps (composite signed):", {k_: round(v["signed_median"], 4) for k_, v in out["main"]["composite"]["by_eps"].items()}, flush=True)
        return out

    timed("item3", st_item3)
    print("done")


if __name__ == "__main__":
    main()
