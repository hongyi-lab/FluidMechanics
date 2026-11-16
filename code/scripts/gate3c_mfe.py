#!/usr/bin/env python
"""Gate 3c for the MFE model: is the edge criterion a state threshold |u_th| = c* or seed x gain = threshold?
(code/configs/gate3c_mfe.yaml; criteria fixed there before the run). Existing data only; no new MC.

Stages (cached in results.json; the gain grid is chunked and resumable):
  theory       item 0: quasi-linear structure, A_y, F, d3, W_inf, W(T); comparison with the Gate 3b M1 form; direct
               ODE check of a3(T) e^{d3 T} = a3(0) + y0^T W(T) y0 at small amplitude
  benchmark    (--frac < 1 only) full-size timing of the analytic kappa search and of the grid rule
  item1        (a) W_inf, (b) kappa W_inf, (c) W(T), (d) 100 random symmetric controls and F; test metric; CRITERION
  gain_checks  item 2 checks: kappa = 0 reproduces the Gate 3 gain grid; pure-a3 ray identical to Gate 3
  gain_grid    item 2: merged gain on the random-ray grid (chunks of 1000 rays), sigma = 0 and 3e-8
  gain_family  merged gain on the 56 family rays
  item2        forward / reverse, family per eps, eps trend, left peak; CRITERION

    python code/scripts/gate3c_mfe.py --config code/configs/gate3c_mfe.yaml --run-dir <dir> [--frac 0.01]
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.integrate import quad_vec
from scipy.linalg import expm
from tqdm.auto import tqdm

from committor import mechanism as mech  # noqa: E402
from committor.feedback import Feedback, analytic_cstar, ray_pq, structure_check, u_th  # noqa: E402
from committor.gain import gain_versions  # noqa: E402
from committor.systems.mfe import LAMINAR, MFE, rhs_paper  # noqa: E402
from committor.threshold import RayCoordinate, fit_cstar, paired_diff, qinf, summarize  # noqa: E402
from committor.utils import (Status, atomic_save_npz, atomic_write_json, derive_seed, git_info, read_json,  # noqa: E402
                             stable_hash, tree_hash)

REPO = Path(__file__).resolve().parents[2]
SIM_CODE = ["feedback.py", "threshold.py", "gain.py", "utils.py", "mechanism.py", "systems/mfe.py"]
MODES = [f"a{i}" for i in range(1, 10)]
Y = slice(3, 8)


def cosF(A, B):
    return float(np.sum(A * B) / (np.linalg.norm(A) * np.linalg.norm(B)))


def kappa_grid(c):
    pos = 10.0 ** np.linspace(float(c["log10_min"]), float(c["log10_max"]), int(c["n_per_sign"]))
    return np.concatenate([-pos[::-1], [0.0], pos])


def refine(grid, k, n):
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    if lo * hi > 0:
        return np.sign(lo) * np.geomspace(abs(lo), abs(hi), n)
    return np.linspace(lo, hi, n)


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
    g2, g3, g3b = (REPO / cfg["data"][k] for k in ("gate2_run", "gate3_run", "gate3b_run"))
    R2, R3, R3b = (json.loads((d / "results.json").read_text()) for d in (g2, g3, g3b))
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
    comp = dirs ** 2 / (dirs ** 2).sum(1, keepdims=True)
    f3, fy = comp[:, 2], comp[:, 3:8].sum(1)
    res.setdefault("meta", {"n_rays": int(N), "split_counts": [int(len(tr_r)), int(len(va_r)), int(len(te_r))],
                            "grid_points": int(G), "grid_range": [s_lo, s_hi], "data": cfg["data"],
                            "criteria": {"item1": cfg["item1"]["criterion"], "item2": cfg["item2"]["criterion"]}})
    save()
    m = MFE(Re=400.0)
    fb = Feedback(m)
    normW = float(np.linalg.norm(fb.W_inf))

    # references on the test rays (Gate 3, Gate 3b), aligned with te_r
    te_ids = rays[te_r]
    G3T, G3bT = np.load(g3 / "test_errors.npz"), np.load(g3b / "test_errors.npz")

    def ref_from(Z, name):
        names = list(Z["names"])
        pos = {int(r): i for i, r in enumerate(Z["test_rays"])}
        if name not in names or not all(int(r) in pos for r in te_ids):
            return None
        return Z["E"][names.index(name)][[pos[int(r)] for r in te_ids]]

    REF = {"Gate 3: E": ref_from(G3T, "E"), "Gate 3: |a3|": ref_from(G3T, "|a3|"),
           "Gate 3: triple (1-a1, a2, a3)": ref_from(G3T, "triple (1-a1, a2, a3)"),
           "Gate 3: MLP (9D, polar input) = ceiling": ref_from(G3T, "MLP (9D, polar input)"),
           "Gate 3: gain (i) (refit c*)": ref_from(G3T, "gain (i)"),
           "Gate 3b: S1 (strict single quadric)": ref_from(G3bT, "S1: single quadric threshold (k = 1 strict)"),
           "Gate 3b: M1": ref_from(G3bT, "M1"), "Gate 3b: M2": ref_from(G3bT, "M2")}
    zf_g3_all = np.load(g3 / "zero_fit_generic_err_i.npy")[rays]                 # Gate 3 gain (i), frozen c*_0, sigma 3e-8
    REF["Gate 3: gain (i) (zero-fit c*_0)"] = zf_g3_all[te_r]
    REF = {k: v for k, v in REF.items() if v is not None}

    # ------------------------------------------------------------ item 0: theory checks ------------------------------
    def st_theory():
        out = {"structure_max_abs_coefficients": structure_check(m), "d3": fb.d3,
               "d3_formula_(beta^2+gamma^2)/Re": float(((np.pi / 2) ** 2 + (2 * np.pi / m.Lz) ** 2) / m.Re),
               "F": fb.F.tolist(), "A_y": fb.Ay.tolist(),
               "eig_A_y": [[float(z.real), float(z.imag)] for z in np.linalg.eigvals(fb.Ay)],
               "eig_B_real": sorted(float(z.real) for z in fb.eig_B), "W_inf": fb.W_inf.tolist(), "W_inf_frobenius": normW,
               "eig_W_inf": np.linalg.eigvalsh(fb.W_inf).tolist(), "F_frobenius": float(np.linalg.norm(fb.F))}
        yt = np.random.default_rng(0).standard_normal(5)
        a_t = LAMINAR.copy()
        a_t[Y] = yt
        out["check_yFy_vs_rhs_a3"] = [float(yt @ fb.F @ yt), float(rhs_paper(a_t, m.Re, m.Lx, m.Lz)[2])]
        Wq, _ = quad_vec(lambda s: np.exp(fb.d3 * s) * expm(fb.Ay.T * s) @ fb.F @ expm(fb.Ay * s), 0, 45.0, epsabs=1e-13, epsrel=1e-12)
        out["W45_closed_form_minus_quadrature_maxabs"] = float(np.abs(fb.W(45.0) - Wq).max())
        # Gate 3b M1: y-y block of its quadratic form, normalised by its a3 coefficient
        tag = R3b["ladder"]["M1"]["chosen"]
        m1, _ = mech.load(g3b / f"model_{tag}.pt", np.array(R3b["training_set"]["x_scale"]), "cpu")
        Qm, bm = m1.physical()
        Qm, bm = Qm[0], bm[0]
        b3 = float(bm[2])
        Wm = Qm[3:8, 3:8] / b3
        Ts = np.concatenate([[0.0], np.geomspace(0.1, 1e4, 401)])
        cos_T = np.array([cosF(fb.W(t), Wm) if t > 0 else np.nan for t in Ts])
        kap_T = np.array([np.sum(fb.W(t) * Wm) / np.sum(fb.W(t) ** 2) if t > 0 else np.nan for t in Ts])
        nW_T = np.array([np.linalg.norm(fb.W(t)) for t in Ts])
        k = int(np.nanargmax(cos_T))
        hi = Ts[np.nan_to_num(cos_T) >= 0.99]
        # T at which |W(T)|_F = |W_M1|_F (monotone in T here)
        nWm = float(np.linalg.norm(Wm))
        j = int(np.argmax(nW_T >= nWm)) if (nW_T >= nWm).any() else -1
        T_match = float(np.interp(nWm, nW_T[j - 1:j + 1], Ts[j - 1:j + 1])) if j > 0 else None
        Xe = Xedge
        u_full = (np.einsum("nj,jk,nk->n", Xe, Qm, Xe) + Xe @ bm) / b3
        u_part = Xe[:, 2] + np.einsum("nj,jk,nk->n", Xe[:, Y], Wm, Xe[:, Y])
        out["M1"] = {"chosen": tag, "b3": b3, "b_over_b3": (bm / b3).tolist(), "W_M1": Wm.tolist(), "W_M1_frobenius": nWm,
                     "cos_W45": cosF(fb.W(45.0), Wm), "cos_W_inf": cosF(fb.W_inf, Wm), "cos_F": cosF(fb.F, Wm),
                     "cos_max": float(cos_T[k]), "T_at_cos_max": float(Ts[k]),
                     "T_range_cos_ge_0.99": [float(hi.min()), float(hi.max())] if len(hi) else None,
                     "scale_kappa_of_W_M1_on_W_inf": float(np.sum(fb.W_inf * Wm) / np.sum(fb.W_inf ** 2)),
                     "scale_kappa_of_W_M1_on_W45": float(np.sum(fb.W(45.0) * Wm) / np.sum(fb.W(45.0) ** 2)),
                     "T_where_norm_W_T_equals_norm_W_M1": T_match,
                     "edge_median_rel_size_of_other_M1_terms": float(np.median(np.abs(u_full - u_part) / np.abs(u_full))),
                     "edge_corr_u_M1_vs_a3_plus_yWy": float(np.corrcoef(u_full, u_part)[0, 1])}
        # direct ODE check of the first-order formula
        def rk4(a, dt, n):
            f = lambda z: rhs_paper(z, m.Re, m.Lx, m.Lz)  # noqa: E731
            for _ in range(n):
                k1 = f(a)
                k2 = f(a + 0.5 * dt * k1)
                k3 = f(a + 0.5 * dt * k2)
                k4 = f(a + dt * k3)
                a = a + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            return a
        rng = np.random.default_rng(derive_seed(seed, "ode_check"))
        rows = []
        for trial in range(5):
            y0 = rng.standard_normal(5)
            y0 /= np.linalg.norm(y0)
            for eps in (1e-5, 1e-4, 1e-3):
                for TT in (45.0, 200.0):
                    a0 = LAMINAR.copy()
                    a0[Y] = eps * y0
                    aT = rk4(a0, 0.05, int(round(TT / 0.05)))
                    rows.append({"trial": trial, "eps": eps, "T": TT, "ode": float(aT[2] * np.exp(fb.d3 * TT) / eps ** 2),
                                 "formula": float(y0 @ fb.W(TT) @ y0)})
        for r_ in rows:
            r_["rel_diff"] = abs(r_["ode"] - r_["formula"]) / max(abs(r_["formula"]), 1e-300)
        out["ode_check"] = rows
        out["ode_check_max_rel_diff_by_eps"] = {f"{e:g}": max(r_["rel_diff"] for r_ in rows if r_["eps"] == e) for e in (1e-5, 1e-4, 1e-3)}
        atomic_save_npz(run / "theory.npz", W_inf=fb.W_inf, F=fb.F, A_y=fb.Ay, W_M1=Wm, Ts=Ts, cos_T=cos_T, kappa_T=kap_T, normW_T=nW_T,
                        W45=fb.W(45.0))
        return out

    timed("theory", st_theory)

    # ------------------------------------------------------------ item 1 helpers ------------------------------------
    I1 = cfg["item1"]
    tr_ln = T(ln_sc[tr_r])
    p_all = dirs[:, 2].copy()
    p_tr = T(p_all[tr_r])

    def q_of(W):
        return np.einsum("ni,ij,nj->n", dirs[:, Y], W, dirs[:, Y])

    def search_kappa(W0, desc):
        q0 = T(q_of(W0)[tr_r])
        grid = kappa_grid(I1["kappa_grid"])
        meds, cs = np.empty(len(grid)), np.empty(len(grid))
        for i, k_ in enumerate(tqdm(grid, desc=desc, disable=not progress, leave=False)):
            meds[i], cs[i] = analytic_cstar(p_tr, k_ * q0, tr_ln, s_lo, s_hi, NL, NR)
        k = int(np.argmin(meds))
        fine = refine(grid, k, int(I1["kappa_grid"]["refine"]))
        mf, cf = np.empty(len(fine)), np.empty(len(fine))
        for i, k_ in enumerate(fine):
            mf[i], cf[i] = analytic_cstar(p_tr, k_ * q0, tr_ln, s_lo, s_hi, NL, NR)
        j = int(np.argmin(mf))
        if mf[j] <= meds[k]:
            kb, mb, cb = float(fine[j]), float(mf[j]), float(cf[j])
        else:
            kb, mb, cb = float(grid[k]), float(meds[k]), float(cs[k])
        return {"kappa": kb, "train_median_analytic": mb, "c_star_analytic": cb, "grid": grid, "grid_medians": meds}

    def search_T(desc):
        Tc = I1["T_grid"]
        grid = np.concatenate([[0.0], np.geomspace(float(Tc["min"]), float(Tc["max"]), int(Tc["n"]))])

        def med_T(t):
            return analytic_cstar(p_tr, T(q_of(fb.W(t))[tr_r]), tr_ln, s_lo, s_hi, NL, NR)
        meds, cs = np.empty(len(grid)), np.empty(len(grid))
        for i, t in enumerate(tqdm(grid, desc=desc, disable=not progress, leave=False)):
            meds[i], cs[i] = med_T(t)
        k = int(np.argmin(meds))
        fine = refine(grid, k, int(Tc["refine"]))
        mf, cf = np.empty(len(fine)), np.empty(len(fine))
        for i, t in enumerate(fine):
            mf[i], cf[i] = med_T(t)
        j = int(np.argmin(mf))
        if mf[j] <= meds[k]:
            tb, mb, cb = float(fine[j]), float(mf[j]), float(cf[j])
        else:
            tb, mb, cb = float(grid[k]), float(meds[k]), float(cs[k])
        return {"T": tb, "train_median_analytic": mb, "c_star_analytic": cb, "grid": grid, "grid_medians": meds}

    def eval_grid(q, with_boot=True):
        """|p s + q s^2| tabulated on the ray grid; Gate 3 grid rule for c*; test errors."""
        C = np.abs(p_all[:, None] * s_grid[None, :] + q[:, None] * s_grid[None, :] ** 2)
        rc = RayCoordinate(C, s_grid, device)
        fit = fit_cstar(rc, tr_r, ln_sc[tr_r], NL, NR)
        thr = [fit["side"] * fit["c_star"]]
        ln_v, _ = rc.ln_s_pred(thr, fit["side"], va_r)
        ln_p, below = rc.ln_s_pred(thr, fit["side"], te_r)
        e = ln_p[:, 0].cpu().numpy() - ln_sc[te_r]
        row = {"side": fit["side"], "c_star": fit["c_star"], "train_median_abs": fit["train_median_abs"],
               "val_median_abs": float(np.median(np.abs(ln_v[:, 0].cpu().numpy() - ln_sc[va_r]))),
               "frac_below_grid_test": float(below[:, 0].float().mean().item()),
               "test": summarize(e, boot if with_boot else None)}
        del rc
        return row, e

    # ------------------------------------------------------------ benchmark (time estimate, --frac only) -------------
    def st_benchmark():
        trf = np.where(RY["split"] == 0)[0]
        pf, lf = T(RY["dirs"][trf, 2]), T(np.log(SC["s_c"][trf]))
        qf = T(np.einsum("ni,ij,nj->n", RY["dirs"][trf][:, Y], fb.W_inf, RY["dirs"][trf][:, Y]))
        analytic_cstar(pf, qf, lf, s_lo, s_hi, NL, NR)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for k_ in np.geomspace(0.1, 10, 20):
            analytic_cstar(pf, k_ * qf, lf, s_lo, s_hi, NL, NR)
        torch.cuda.synchronize()
        t_an = (time.perf_counter() - t0) / 20
        Cf = np.abs(RY["dirs"][:, 2][:, None] * s_grid[None, :] + np.einsum("ni,ij,nj->n", RY["dirs"][:, Y], fb.W_inf, RY["dirs"][:, Y])[:, None] * s_grid[None, :] ** 2)
        t0 = time.perf_counter()
        rc = RayCoordinate(Cf, s_grid, device)
        fit_cstar(rc, trf, np.log(SC["s_c"][trf]), NL, NR)
        torch.cuda.synchronize()
        t_grid = time.perf_counter() - t0
        del rc, Cf
        n_k = 2 * int(I1["kappa_grid"]["n_per_sign"]) + 1 + int(I1["kappa_grid"]["refine"])
        n_T = int(I1["T_grid"]["n"]) + 1 + int(I1["T_grid"]["refine"])
        n_mat = 1 + 1 + int(I1["n_random_controls"])
        est = (n_mat * n_k + n_T) * t_an + (n_mat + 3) * t_grid
        return {"seconds_per_analytic_cstar_full": t_an, "seconds_grid_rule_full": t_grid,
                "n_analytic_evaluations": n_mat * n_k + n_T, "estimate_item1_hours": est / 3600}

    if args.frac < 1:
        timed("benchmark", st_benchmark)

    # ------------------------------------------------------------ item 1 ---------------------------------------------
    def st_item1():
        out, errs, curves = {}, {}, {}
        # (a) W_inf, c* only
        row, e = eval_grid(q_of(fb.W_inf))
        out["a"] = {"label": "(a) W = W_inf, fit c* only", **row}
        errs["a"] = e
        # (b) kappa W_inf
        sb = search_kappa(fb.W_inf, "kappa W_inf")
        row, e = eval_grid(sb["kappa"] * q_of(fb.W_inf))
        out["b"] = {"label": "(b) W = kappa W_inf, fit kappa and c*", "kappa": sb["kappa"],
                    "train_median_analytic": sb["train_median_analytic"], "c_star_analytic": sb["c_star_analytic"], **row}
        errs["b"] = e
        curves["b_grid"], curves["b_meds"] = sb["grid"], sb["grid_medians"]
        # (c) W(T)
        sc_ = search_T("W(T)")
        row, e = eval_grid(q_of(fb.W(sc_["T"])))
        out["c"] = {"label": "(c) W = W(T), fit T and c*", "T": sc_["T"], "train_median_analytic": sc_["train_median_analytic"],
                    "c_star_analytic": sc_["c_star_analytic"], **row}
        errs["c"] = e
        curves["c_grid"], curves["c_meds"] = sc_["grid"], sc_["grid_medians"]
        # check: kappa = 0 is |a3| (must reproduce Gate 3's |a3| test errors)
        row0, e0 = eval_grid(np.zeros(N), with_boot=False)
        out["check_kappa0"] = {"test_median_abs": row0["test"]["median_abs"],
                               "max_abs_diff_vs_gate3_abs_a3": (float(np.max(np.abs(np.where(np.isfinite(e0), e0, 0) - np.where(np.isfinite(REF["Gate 3: |a3|"]), REF["Gate 3: |a3|"], 0))))
                                                                if "Gate 3: |a3|" in REF else None)}
        # (d) F (scaled to |W_inf|) and random symmetric controls
        Fs = fb.F * normW / np.linalg.norm(fb.F)
        sf = search_kappa(Fs, "kappa F")
        row, e = eval_grid(sf["kappa"] * q_of(Fs))
        out["d_F"] = {"label": "(d) W = kappa F (instantaneous coefficient)", "kappa_scaled": sf["kappa"],
                      "kappa_in_F_units": sf["kappa"] * normW / float(np.linalg.norm(fb.F)),
                      "train_median_analytic": sf["train_median_analytic"], **row}
        errs["d_F"] = e
        curves["F_grid"], curves["F_meds"] = sf["grid"], sf["grid_medians"]
        rnd = []
        rnd_meds_curve = []
        E_rnd = np.empty((int(I1["n_random_controls"]), len(te_r)))
        for i in tqdm(range(int(I1["n_random_controls"])), desc="random controls", disable=not progress):
            rg = np.random.default_rng(derive_seed(seed, "random_W", i))
            A_ = rg.standard_normal((5, 5))
            R_ = 0.5 * (A_ + A_.T)
            R_ *= normW / np.linalg.norm(R_)
            sr = search_kappa(R_, f"random {i}")
            row, e = eval_grid(sr["kappa"] * q_of(R_), with_boot=False)
            rnd.append({"i": i, "kappa": sr["kappa"], "train_median_analytic": sr["train_median_analytic"],
                        "cos_to_W_inf": cosF(R_, fb.W_inf), "c_star": row["c_star"], "side": row["side"],
                        "train_median_abs": row["train_median_abs"], "test_median_abs": row["test"]["median_abs"],
                        "test_q90_abs": row["test"]["q90_abs"]})
            E_rnd[i] = e
            rnd_meds_curve.append(sr["grid_medians"])
        tm = np.array([r_["test_median_abs"] for r_ in rnd])
        out["d_random"] = {"label": "(d) 100 random symmetric W, |W|_F = |W_inf|_F, fitted like (b)", "controls": rnd,
                           "test_median_quantiles_0_5_50_95_100": [float(x) for x in np.quantile(tm, [0, 0.05, 0.5, 0.95, 1.0])],
                           "frac_test_median_le_b": float(np.mean(tm <= out["b"]["test"]["median_abs"])),
                           "frac_test_median_le_0.05": float(np.mean(tm <= 0.05)),
                           "max_abs_cos_to_W_inf": float(np.max(np.abs([r_["cos_to_W_inf"] for r_ in rnd])))}
        curves["rnd_meds"] = np.array(rnd_meds_curve)
        atomic_save_npz(run / "item1_curves.npz", **curves)
        np.savez(run / "item1_test_errors.npz", names=np.array(list(errs)), E=np.stack([errs[k] for k in errs]), E_random=E_rnd,
                 test_rays=te_ids)
        # references and paired differences (test rays)
        out["references"] = {k: summarize(v, boot) for k, v in REF.items()}
        allerr = {"(a)": errs["a"], "(b)": errs["b"], "(c)": errs["c"], "(d) F": errs["d_F"], **REF}
        pairs = [("(b)", "Gate 3b: M1"), ("(c)", "Gate 3b: M1"), ("(b)", "Gate 3: |a3|"), ("(c)", "Gate 3: |a3|"),
                 ("(a)", "(b)"), ("(c)", "(b)"), ("(b)", "(d) F"), ("(b)", "Gate 3b: S1 (strict single quadric)"),
                 ("(b)", "Gate 3: MLP (9D, polar input) = ceiling"), ("(b)", "Gate 3: triple (1-a1, a2, a3)")]
        out["paired"] = [{"a": a_, "b": b_, **paired_diff(allerr[a_], allerr[b_], boot)} for a_, b_ in pairs
                         if a_ in allerr and b_ in allerr]
        Cc = I1["criterion"]
        mb, mc = out["b"]["test"]["median_abs"], out["c"]["test"]["median_abs"]
        if min(mb, mc) <= float(Cc["holds_median"]):
            verdict = f"theory quantity holds: min test median of (b), (c) = {min(mb, mc):.4f} <= {Cc['holds_median']}"
        elif mb >= float(Cc["fails_median"]) and mc >= float(Cc["fails_median"]):
            verdict = f"theory quantity does not hold: (b) {mb:.4f} and (c) {mc:.4f} both >= {Cc['fails_median']}"
        else:
            verdict = f"intermediate: (b) {mb:.4f}, (c) {mc:.4f} (neither <= {Cc['holds_median']} nor both >= {Cc['fails_median']})"
        out["criterion"] = {"b": mb, "c": mc, "verdict": verdict,
                            "ci": {"b": out["b"]["test"]["median_abs_ci"], "c": out["c"]["test"]["median_abs_ci"]}}
        print("    ITEM 1 CRITERION:", verdict, flush=True)
        for k in ("a", "b", "c", "d_F"):
            r_ = out[k]
            print(f"    {r_['label']:48s} test {r_['test']['median_abs']:.4f} q90 {r_['test']['q90_abs']:.4f}", flush=True)
        print("    random controls test median quantiles", out["d_random"]["test_median_quantiles_0_5_50_95_100"], flush=True)
        return out

    timed("item1", st_item1)
    kappa_b = float(res["item1"]["b"]["kappa"])

    # ------------------------------------------------------------ item 2: merged gain -------------------------------
    I2 = cfg["item2"]
    sig_list = [float(s) for s in I2["sigmas"]]
    k_rand, k_fam = sig_list.index(float(I2["sigma_random"])), sig_list.index(float(I2["sigma_family"]))
    T_g, dt_g = float(I2["T_g"]), float(I2["dt"])
    W2 = fb.W_inf

    def merged_states(Xa, kappa):
        """a -> a with a3 replaced by u_th = a3 + kappa y^T W_inf y (y and the rest unchanged)."""
        Xm = np.array(Xa, dtype=np.float64, copy=True)
        Xm[..., 2] = u_th(Xa, W2, kappa)
        return Xm

    def st_gain_checks():
        out = {}
        # kappa = 0: identical to the Gate 3 gain grid
        nchk = min(int(I2["check_rays_kappa0"]), N)
        rid = rays[:nchk]
        Xa = LAMINAR[None, None, :] + s_grid[None, :, None] * dirs[:nchk, None, :]
        r = gain_versions(m, merged_states(Xa, 0.0).reshape(-1, 9), sig_list, T_g, dt_g, device, chunk=1_000_000)
        mine = r["max_trP"].reshape(nchk, G, -1)
        sig3 = [float(s) for s in R3["meta"]["sigma_list"]]
        ref = np.stack([np.load(g3 / f"gain_grid_{int(ri) // 1000:03d}.npz")["max_trP"][int(ri) % 1000] for ri in rid])
        d = {}
        for k, sg in enumerate(sig_list):
            a_, b_ = np.log(mine[:, :, k]), np.log(ref[:, :, sig3.index(sg)])
            ok = np.isfinite(a_) & np.isfinite(b_)
            d[f"sigma={sg:g}"] = {"max_abs_diff_ln_max_trP": float(np.max(np.abs(a_[ok] - b_[ok]))), "n_nonfinite_mismatch": int(np.sum(np.isfinite(a_) != np.isfinite(b_)))}
        out["kappa0_vs_gate3_grid"] = {"n_rays": nchk, **d}
        # pure-a3 ray: y0 = 0 so u_th = a3; the merged candidate is gain (i) itself
        GA = np.load(g3 / "gain_a3.npz")
        Xa3 = LAMINAR[None, :] + GA["s"][:, None] * np.eye(9)[2][None, :]
        Xm3 = merged_states(Xa3, kappa_b)
        ra = gain_versions(m, Xm3, [1.0], T_g, dt_g, device)
        out["pure_a3"] = {"max_abs_u_th_minus_a3": float(np.max(np.abs(Xm3[:, 2] - Xa3[:, 2]))),
                          "max_abs_diff_ln_g_vs_gate3": float(np.max(np.abs(np.log(ra["max_trP"][:, 0]) - GA["ln_g_i"]))),
                          "c_star_0_gate3": float(R3["gain_a3"]["i"]["zero_fit_c_star"])}
        return out

    timed("gain_checks", st_gain_checks)

    rpc = int(I2["rows_per_chunk"])
    chunks = list(range(0, N, rpc))

    def st_gain_grid():
        times = []
        for ci, r0 in enumerate(tqdm(chunks, desc="merged gain grid", disable=not progress)):
            f = run / f"gain_grid_{ci:03d}.npz"
            if f.exists():
                continue
            t0 = time.perf_counter()
            rr = np.arange(r0, min(r0 + rpc, N))
            Xa = LAMINAR[None, None, :] + s_grid[None, :, None] * dirs[rr][:, None, :]
            Xm = merged_states(Xa, kappa_b)
            r = gain_versions(m, Xm.reshape(-1, 9), sig_list, T_g, dt_g, device, chunk=1_000_000)
            shp = (len(rr), G)
            atomic_save_npz(f, rows=rr, max_trP=r["max_trP"].reshape(*shp, -1), t_at_max=r["t_at_max"][:, k_rand].reshape(shp).astype(np.float32),
                            u_th=Xm[..., 2])
            times.append(time.perf_counter() - t0)
            print(f"    merged gain chunk {ci + 1}/{len(chunks)}: {times[-1]:.0f} s ({len(rr) * G / times[-1]:.0f} states/s)", flush=True)
        return {"n_chunks": len(chunks), "states": int(N * G), "chunk_seconds": times}

    timed("gain_grid", st_gain_grid)

    FD = np.load(g2 / "family_det.npz")
    fam_meta = R2["family_det"]["rays"]
    fam_dirs, fam_sc = FD["d"], FD["s_c"]
    fam_eps = np.array([f["eps"] for f in fam_meta])
    fam_ang = np.array([f["angle"] for f in fam_meta])
    fam_draw = np.array([f["draw"] for f in fam_meta])
    fam_grid = np.sqrt(np.geomspace(5e-7 * (1 + 1e-9), 0.29 * (1 - 1e-9), int(I2["family"]["grid_points"])))
    use = np.isin(fam_eps, [float(e) for e in I2["family"]["eps_used"]]) & np.isfinite(fam_sc)

    def st_gain_family():
        Xa = LAMINAR[None, None, :] + fam_grid[None, :, None] * fam_dirs[:, None, :]
        Xm = merged_states(Xa, kappa_b)
        r = gain_versions(m, Xm.reshape(-1, 9), sig_list, T_g, dt_g, device, chunk=250_000)
        shp = (len(fam_dirs), len(fam_grid))
        atomic_save_npz(run / "gain_family.npz", max_trP=r["max_trP"].reshape(*shp, -1), t_at_max=r["t_at_max"].reshape(*shp, -1), s=fam_grid,
                        u_th=Xm[..., 2], a3=Xa[..., 2])
        rel = np.abs(Xm[..., 2] - Xa[..., 2]) / np.maximum(np.abs(Xa[..., 2]), 1e-300)
        at_sc = {}
        for e_ in sorted(set(fam_eps[use].tolist())):
            sel = np.where(use & (fam_eps == e_))[0]
            j = np.clip(np.searchsorted(fam_grid, fam_sc[sel]), 0, len(fam_grid) - 1)
            at_sc[f"{e_:g}"] = float(np.median(rel[sel, j]))
        return {"n_rays": int(len(fam_dirs)), "grid_points": int(len(fam_grid)),
                "median_rel_change_of_a3_at_s_c_by_eps": at_sc}

    timed("gain_family", st_gain_family)

    # ------------------------------------------------------------ item 2 evaluation ----------------------------------
    def st_item2():
        parts = [np.load(run / f"gain_grid_{ci:03d}.npz") for ci in range(len(chunks))]
        MT = np.concatenate([p["max_trP"] for p in parts])
        TM = np.concatenate([p["t_at_max"] for p in parts])
        del parts
        out = {"kappa_b": kappa_b, "sigma_random": sig_list[k_rand], "sigma_family": sig_list[k_fam]}
        with np.errstate(divide="ignore"):
            C_r = {k: np.log(MT[:, :, k]) for k in range(len(sig_list))}
        out["frac_nonfinite_grid"] = {f"sigma={sg:g}": float(np.mean(~np.isfinite(C_r[k]))) for k, sg in enumerate(sig_list)}
        allr = np.arange(N)
        E = {}
        for k, sg in enumerate(sig_list):
            rc = RayCoordinate(C_r[k], s_grid, device)
            fwd = fit_cstar(rc, tr_r, ln_sc[tr_r], NL, NR)
            ln_f, bl_f = rc.ln_s_pred([fwd["side"] * fwd["c_star"]], fwd["side"])
            ln_r, bl_r = rc.ln_s_pred([float(I2["c_star_0"])], 1)
            E[("fwd", k)] = ln_f[:, 0].cpu().numpy() - ln_sc
            E[("rev", k)] = ln_r[:, 0].cpu().numpy() - ln_sc
            edge_c = rc.values_at(ln_sc)
            if k == k_rand:
                out["forward"] = {"c_star": fwd["c_star"], "side": fwd["side"], "train_median_abs": fwd["train_median_abs"],
                                  "val_median_abs": float(np.median(np.abs(E[("fwd", k)][va_r]))),
                                  "test": summarize(E[("fwd", k)][te_r], boot),
                                  "frac_below_grid_test": float(bl_f[te_r, 0].float().mean().item()),
                                  "c_star_minus_c_star_0": fwd["c_star"] - float(I2["c_star_0"])}
                out["reverse"] = {"c_star_0": float(I2["c_star_0"]), "side": 1, "test": summarize(E[("rev", k)][te_r], boot),
                                  "all_rays": summarize(E[("rev", k)]), "frac_below_grid_all": float(bl_r[:, 0].float().mean().item())}
                out["edge_c_quantiles_5_50_95"] = [float(x) for x in np.quantile(edge_c[np.isfinite(edge_c)], [0.05, 0.5, 0.95])]
                edge_keep = edge_c
            else:
                out[f"sigma={sg:g}"] = {"forward_c_star": fwd["c_star"], "forward_test_median_abs": float(np.median(np.abs(E[("fwd", k)][te_r]))),
                                        "reverse_test_median_abs": float(np.median(np.abs(E[("rev", k)][te_r])))}
            del rc
        kr = k_rand
        ko = [k for k in range(len(sig_list)) if k != kr]
        for k in ko:
            for d_ in ("fwd", "rev"):
                a_, b_ = E[(d_, k)], E[(d_, kr)]
                ok = np.isfinite(a_) & np.isfinite(b_)
                out[f"sigma={sig_list[k]:g}"][f"{d_}_max_abs_per_ray_diff_vs_primary"] = float(np.max(np.abs(a_[ok] - b_[ok]))) if ok.any() else None
        e_fwd, e_rev = E[("fwd", kr)], E[("rev", kr)]
        # family (eps > 0), forward c* (criterion) and c*_0 (descriptive)
        GF = np.load(run / "gain_family.npz")
        with np.errstate(divide="ignore"):
            C_f = np.log(GF["max_trP"][:, :, k_fam])
        rcf = RayCoordinate(C_f, fam_grid, device)

        def fam_block(cstar, side):
            ln_p, below = rcf.ln_s_pred([side * cstar], side)
            e = ln_p[:, 0].cpu().numpy() - np.log(np.where(np.isfinite(fam_sc), fam_sc, 1.0))
            e[~use] = np.nan
            rec = {"all": summarize(e[use]), "by_eps": {}}
            for ep_ in sorted(set(fam_eps[use].tolist())):
                sel = use & (fam_eps == ep_)
                s_ = summarize(e[sel])
                s_["max_abs"] = float(np.max(np.abs(e[sel])))
                rec["by_eps"][f"{ep_:g}"] = s_
            rec["by_angle"] = {f"{a:g}": float(np.median(np.abs(e[use & (fam_ang == a)]))) for a in np.unique(fam_ang[use])}
            e_lo, e_hi = (float(x) for x in I2["family"]["trend_pair"])
            diffs = []
            for a in np.unique(fam_ang[use]):
                for dr in np.unique(fam_draw[use]):
                    i1 = np.where(use & (fam_ang == a) & (fam_draw == dr) & (fam_eps == e_lo))[0]
                    i2 = np.where(use & (fam_ang == a) & (fam_draw == dr) & (fam_eps == e_hi))[0]
                    if len(i1) == 1 and len(i2) == 1:
                        x1, x2 = e[i1[0]], e[i2[0]]
                        diffs.append(0.0 if (np.isinf(x1) and x1 == x2) else x1 - x2)
            diffs = np.array(diffs)
            sm = [rec["by_eps"][k_]["signed_median"] for k_ in rec["by_eps"]]
            rec["eps_trend"] = {"definition": f"median over (angle, draw) pairs of e(eps={e_lo:g}) - e(eps={e_hi:g})",
                                "n_pairs": int(len(diffs)), "median": float(qinf(diffs, 0.5)) if len(diffs) else None,
                                "pairs": diffs.tolist(), "range_of_signed_medians_by_eps": float(max(sm) - min(sm))}
            rec["per_ray"] = [{"ray": int(i), "angle": float(fam_ang[i]), "eps": float(fam_eps[i]), "draw": int(fam_draw[i]),
                               "ln_err": float(e[i]), "below_grid": bool(below[i, 0].item())} for i in np.where(use)[0]]
            return rec, e

        out["family_forward"], e_fam_fwd = fam_block(out["forward"]["c_star"], out["forward"]["side"])
        out["family_with_c_star_0_(descriptive)"], e_fam_rev = fam_block(float(I2["c_star_0"]), 1)
        # Gate 3b family references (same rays, same deterministic s_c): M1 and gain (i)@0
        F3b = R3b["family"]
        out["family_references_gate3b"] = {"M1": {k_: v["signed_median"] for k_, v in F3b["models"]["M1"]["by_eps"].items()},
                                           "gain (i)@0 (Gate 3 refit c*)": {k_: v["signed_median"] for k_, v in F3b["references_gate3"]["gain (i)@0"]["by_eps"].items()}}
        # left peak (Gate 3b definition)
        ev3 = np.load(g3 / "gain_i_at_edge.npy")[rays]
        P = np.isfinite(ev3) & (ev3 < float(I2["left_peak_cut"]))
        Pt = P[te_r]
        lp = {"n": int(P.sum()), "n_test": int(Pt.sum()),
              "forward_all": summarize(e_fwd[P]) if P.any() else None, "reverse_all": summarize(e_rev[P]) if P.any() else None,
              "gate3_gain_i_zero_fit_all": summarize(zf_g3_all[P]) if P.any() else None,
              "merged_c_at_edge_quantiles_5_50_95": [float(x) for x in np.quantile(edge_keep[P][np.isfinite(edge_keep[P])], [0.05, 0.5, 0.95])] if P.any() else None}
        if Pt.any():
            lp["forward_test"] = summarize(e_fwd[te_r][Pt])
            lp["reverse_test"] = summarize(e_rev[te_r][Pt])
            if "Gate 3: gain (i) (refit c*)" in REF:
                lp["gate3_gain_i_refit_test"] = summarize(REF["Gate 3: gain (i) (refit c*)"][Pt])
        out["left_peak"] = lp
        # error vs the a3 energy fraction (descriptive), forward on test rays and reverse on all rays
        nb = 20

        def binned(xv, e):
            edges = np.quantile(xv, np.linspace(0, 1, nb + 1))
            bi = np.clip(np.searchsorted(edges, xv, side="right") - 1, 0, nb - 1)
            return [{"x_median": float(np.median(xv[bi == b_])), "n": int((bi == b_).sum()),
                     "median_abs": float(np.median(np.abs(e[bi == b_]))), "signed_median": float(np.median(e[bi == b_]))}
                    for b_ in range(nb) if (bi == b_).any()]
        out["vs_a3_fraction"] = {"forward_test": binned(f3[te_r], e_fwd[te_r]), "reverse_all": binned(f3, e_rev),
                                 "gate3_gain_i_zero_fit_all": binned(f3, zf_g3_all)}
        if "Gate 3: gain (i) (refit c*)" in REF:
            out["vs_a3_fraction"]["gate3_gain_i_refit_test"] = binned(f3[te_r], REF["Gate 3: gain (i) (refit c*)"])
        # paired differences on the test rays
        e1 = np.load(run / "item1_test_errors.npz")
        n1 = list(e1["names"])
        allerr = {"merged forward": e_fwd[te_r], "merged reverse": e_rev[te_r], "item 1 (b)": e1["E"][n1.index("b")], **REF}
        pairs = [("merged forward", "Gate 3: gain (i) (refit c*)"), ("merged reverse", "Gate 3: gain (i) (zero-fit c*_0)"),
                 ("merged forward", "item 1 (b)"), ("merged forward", "Gate 3b: M1"), ("merged reverse", "merged forward"),
                 ("merged forward", "Gate 3: |a3|"), ("merged forward", "Gate 3: MLP (9D, polar input) = ceiling")]
        out["paired"] = [{"a": a_, "b": b_, **paired_diff(allerr[a_], allerr[b_], boot)} for a_, b_ in pairs if a_ in allerr and b_ in allerr]
        atomic_save_npz(run / "item2_errors.npz", e_fwd=e_fwd, e_rev=e_rev, f3=f3, fy=fy, split=split, rays=rays, left_peak=P,
                        e_fam_fwd=e_fam_fwd, e_fam_rev=e_fam_rev, fam_use=use, fam_eps=fam_eps, fam_ang=fam_ang, fam_draw=fam_draw,
                        edge_c=edge_keep, zf_g3=zf_g3_all, t_edge_merged=TM[np.arange(N), np.clip(np.searchsorted(s_grid, s_c), 0, G - 1)])
        # criterion
        Cc = I2["criterion"]
        mf, mr = out["forward"]["test"]["median_abs"], out["reverse"]["test"]["median_abs"]
        tr_ = out["family_forward"]["eps_trend"]["median"]
        flat = tr_ is not None and abs(tr_) <= float(I2["family"]["trend_flat_max"])
        if mf <= float(Cc["holds_median"]) and mr <= float(Cc["holds_median"]) and flat:
            verdict = f"two families unified: forward {mf:.4f}, reverse {mr:.4f} <= {Cc['holds_median']}, eps trend {tr_:+.4f}"
        elif mf >= float(Cc["fails_median"]) or mr >= float(Cc["fails_median"]):
            verdict = f"does not hold: forward {mf:.4f}, reverse {mr:.4f} (>= {Cc['fails_median']} in at least one direction), eps trend {tr_:+.4f}"
        else:
            verdict = (f"intermediate: forward {mf:.4f}, reverse {mr:.4f}, eps trend {tr_:+.4f} "
                       f"(flat if |trend| <= {I2['family']['trend_flat_max']}: {flat})")
        out["criterion"] = {"forward_test_median": mf, "reverse_test_median": mr, "eps_trend": tr_, "eps_flat": bool(flat), "verdict": verdict,
                            "ci": {"forward": out["forward"]["test"]["median_abs_ci"], "reverse": out["reverse"]["test"]["median_abs_ci"]}}
        print("    ITEM 2 CRITERION:", verdict, flush=True)
        print("    family forward by eps:", {k_: round(v["signed_median"], 4) for k_, v in out["family_forward"]["by_eps"].items()}, flush=True)
        return out

    timed("item2", st_item2)
    print("done")


if __name__ == "__main__":
    main()
