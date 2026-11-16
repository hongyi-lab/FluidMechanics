#!/usr/bin/env python
"""Gate 2 pre-study for the MFE model (no full labelling).

Stages (each cached in the run directory; re-running resumes):
  heun_dt      Heun step from the deterministic-accuracy rule
  histograms   a1 / E occupation of 1e4 deterministic trajectories -> E_B (valley rule)
  edge         deterministic first-crossing energy E_c along 4096 random rays (+ refinement,
               + linear optimal-growth direction) -> r_A rule; deterministic check on the sphere E = r_A
  lyapunov     stationary covariance of the linearised SDE at the laminar state -> E_ss(sigma), tails
  spontaneous  fraction of paths from the laminar state reaching B within T_max, per sigma
  sigma        two noise levels by the rule in the config
  a_boundary   committor just outside A (E = 1.1 r_A), and the same states with A non-absorbing
  rays         q profiles along 6 rays for both sigma (coarse -> fine grid), and the dt/2 check

    python code/scripts/gate2_prestudy.py --config code/configs/gate2_prestudy.yaml \
        --run-dir experiments/20261115_gate2_prestudy_v1
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import beta as beta_dist

torch._dynamo.config.cache_size_limit = 128

from committor import edge  # noqa: E402
from committor import gate1 as G  # noqa: E402
from committor.sde import counts_from_outcomes, simulate_absorbing  # noqa: E402
from committor.systems.mfe import LAMINAR, MFE, rhs_paper  # noqa: E402
from committor.utils import (Status, atomic_save_npz, atomic_write_json, derive_seed, git_info, load_npz,  # noqa: E402
                             read_json, stable_hash, torch_generator, tree_hash)

REPO = Path(__file__).resolve().parents[2]
SIM_CODE = ["sde.py", "edge.py", "gate1.py", "utils.py", "systems/base.py", "systems/mfe.py"]


def cp_interval(k, n, level=0.95):
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a, k + 1, n - k))
    return lo, hi


def iso_quantiles(s, q, levels=(0.05, 0.5, 0.95)):
    """Non-decreasing least-squares fit of q(s) (PAVA), then the first s where the fit reaches each level
    (linear interpolation)."""
    from sklearn.isotonic import IsotonicRegression
    f = IsotonicRegression(y_min=0, y_max=1).fit(s, q).predict(s)
    out = []
    for L in levels:
        k = np.argmax(f >= L)
        if f[k] < L:
            out.append(float("nan"))
        elif k == 0:
            out.append(float(s[0]))
        else:
            out.append(float(s[k - 1] + (L - f[k - 1]) / (f[k] - f[k - 1]) * (s[k] - s[k - 1])))
    return out, f


def run_states(m, X, M, dt, T_max, seed, device, progress, desc):
    """M absorbing paths per state (M scalar or per-state array); returns n_B, n_A, n_timeout per state
    and the absorption time of every path."""
    Ms = np.broadcast_to(np.asarray(M, dtype=int), (len(X),))
    X0 = torch.as_tensor(np.repeat(X, Ms, axis=0), device=device)
    r = simulate_absorbing(m, X0, dt, int(round(T_max / dt)), "heun", torch_generator(seed, device),
                           compact_every=200, progress=progress, desc=desc, bridge=True, compile_step=True)
    idx = np.repeat(np.arange(len(X)), Ms)
    nB = np.bincount(idx, weights=(r["outcome"] == 1), minlength=len(X)).astype(int)
    nA = np.bincount(idx, weights=(r["outcome"] == 0), minlength=len(X)).astype(int)
    nT = np.bincount(idx, weights=(r["outcome"] == -1), minlength=len(X)).astype(int)
    return {"n_B": nB, "n_A": nA, "n_T": nT, "t_abs": r["steps"] * dt, "outcome": r["outcome"], "path_state": idx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--stop-after", default=None)
    ap.add_argument("--allow-code-change", action="store_true",
                    help="resume although the simulation code hash changed (only for changes that cannot affect cached results)")
    args = ap.parse_args()
    progress = not args.no_progress
    cfg = yaml.safe_load(Path(args.config).read_text())
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    device = torch.device(cfg["device"])
    src = REPO / "code" / "src" / "committor"
    ident = {"config_hash": stable_hash(cfg), "sim_code_hash": tree_hash([src / f for f in SIM_CODE])}
    old = read_json(run_dir / "run_identity.json")
    if old is None:
        atomic_write_json(run_dir / "run_identity.json", ident)
    elif old["config_hash"] != ident["config_hash"]:
        sys.exit(f"{run_dir} was started with config {old['config_hash']}, now {ident['config_hash']}; use a new run dir")
    elif old["sim_code_hash"] != ident["sim_code_hash"] and not args.allow_code_change:
        sys.exit(f"{run_dir} was started with {old}, now {ident}; use a new run dir or --allow-code-change")
    if not (run_dir / "config_used.yaml").exists():
        (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    man = read_json(run_dir / "manifest.json", {"invocations": []})
    man["invocations"].append({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv, "git": git_info(REPO),
                               "sim_code_hash": ident["sim_code_hash"],
                               "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})
    atomic_write_json(run_dir / "manifest.json", man)
    status = Status(run_dir / "status.json")
    seed = int(cfg["seed"])
    sysc = cfg["system"]
    m0 = MFE(Re=sysc["Re"], Lx_over_pi=sysc["Lx_over_pi"], Lz_over_pi=sysc["Lz_over_pi"])
    rhs = lambda a: rhs_paper(a, m0.Re, m0.Lx, m0.Lz)
    res = read_json(run_dir / "results.json", {}) or {}

    def save():
        atomic_write_json(run_dir / "results.json", res)

    def stage(name):
        print(f"--- {name}", flush=True)
        return name not in res

    # ---------------- Heun step ----------------
    if stage("heun_dt"):
        t0 = time.perf_counter()
        h = cfg["heun_dt"]
        X = torch.as_tensor(G.sphere_ics(h["n_states"], 0.3, np.random.default_rng(derive_seed(seed, "heun_dt"))),
                            device=device)
        f = torch.compile(lambda a: edge._advance(rhs, a, 0.01, "rk4"), dynamic=False, fullgraph=True)
        for _ in range(int(round(h["spinup_time"] / 0.01))):
            X = f(X)

        def integ(dt, scheme):
            Y = X.clone()
            g = torch.compile(lambda a: edge._advance(rhs, a, dt, scheme), dynamic=False, fullgraph=True)
            for _ in range(int(round(h["horizon"] / dt))):
                Y = g(Y)
            return Y.cpu().numpy()
        ref = integ(h["reference_rk4_dt"], "rk4")
        errs = {str(dt): float(np.abs(integ(dt, "heun") - ref).max()) for dt in h["candidates"]}
        ok = [dt for dt in h["candidates"] if errs[str(dt)] <= h["tolerance"]]
        res["heun_dt"] = {"errors": errs, "chosen": max(ok) if ok else None, "seconds": time.perf_counter() - t0}
        save()
    dt = float(res["heun_dt"]["chosen"])
    print("Heun dt", dt, res["heun_dt"]["errors"])

    # ---------------- histograms -> E_B ----------------
    hc = cfg["histograms"]
    if stage("histograms"):
        t0 = time.perf_counter()
        A0 = torch.as_tensor(G.sphere_ics(hc["n_ic"], 0.3, np.random.default_rng(derive_seed(20261116, "lifetime_ics"))),
                             device=device)
        step = torch.compile(lambda a: (edge._advance(rhs, a, 0.25, "rk4"),), dynamic=False, fullgraph=True)
        runner = edge._Runner(step, [A0.clone()], 4, True)
        hE = torch.zeros(hc["bins"], dtype=torch.float64, device=device)
        ha = torch.zeros(hc["bins"], dtype=torch.float64, device=device)
        for _ in range(int(hc["t_max"])):
            runner.advance()
            a = runner.buf[0]
            hE += torch.histc(edge.energy(a), hc["bins"], *hc["E_range"])
            ha += torch.histc(a[:, 0], hc["bins"], *hc["a1_range"])
        hE, ha = hE.cpu().numpy(), ha.cpu().numpy()
        eE = np.linspace(*hc["E_range"], hc["bins"] + 1)
        ea = np.linspace(*hc["a1_range"], hc["bins"] + 1)
        cE, ca = 0.5 * (eE[1:] + eE[:-1]), 0.5 * (ea[1:] + ea[:-1])
        k = hc["smooth_bins"]
        sm = lambda h: np.convolve(h, np.ones(k) / k, mode="same")
        sE, sa = sm(hE), sm(ha)
        iL = int(np.argmax(np.where(cE < hc["laminar_mode_E_max"], sE, -1)))
        iT = int(np.argmax(np.where(cE > hc["turbulent_mode_E_min"], sE, -1)))
        iV = iL + int(np.argmin(sE[iL:iT + 1]))
        jL = int(np.argmax(np.where(ca > 1 - hc["laminar_mode_E_max"], sa, -1)))
        jT = int(np.argmax(np.where(ca < 0.8, sa, -1)))
        jV = jT + int(np.argmin(sa[jT:jL + 1]))
        E_B = float(f"{cE[iV]:.2g}")
        atomic_save_npz(run_dir / "histograms.npz", hE=hE, ha=ha, eE=eE, ea=ea)
        res["histograms"] = {"E_laminar_mode": float(cE[iL]), "E_turbulent_mode": float(cE[iT]), "E_valley": float(cE[iV]),
                             "a1_turbulent_mode": float(ca[jT]), "a1_laminar_mode": float(ca[jL]), "a1_valley": float(ca[jV]),
                             "E_B": E_B, "seconds": time.perf_counter() - t0}
        save()
    E_B = float(res["histograms"]["E_B"])
    print("histograms", res["histograms"])

    # ---------------- deterministic edge scan -> r_A ----------------
    ec = cfg["edge_scan"]
    Egrid = np.geomspace(ec["E_grid"][0], E_B * (1 + 1e-9), int(ec["E_grid"][2]))
    sgrid = np.sqrt(Egrid)
    fc = lambda dirs, desc: edge.first_crossing(rhs, dirs, sgrid, int(ec["n_bisect"]), 0.25, float(ec["T"]), E_B,
                                                float(ec["r_det"]), "rk4", device=device, chunk=65536,
                                                progress=progress, desc=desc)
    if stage("edge"):
        t0 = time.perf_counter()
        rng = np.random.default_rng(derive_seed(seed, "edge_dirs"))
        D = edge.random_directions(int(ec["n_dir"]), rng)
        r = fc(D, "edge scan")
        Ec = r["s_c"] ** 2
        og = edge.optimal_growth(m0, np.linspace(1, 600, 600))
        ro = fc(og["direction"][None], "optimal dir")
        # refinement around the smallest E_c
        rf = cfg["edge_scan"]["refine"]
        best_d, best_E = D.copy(), Ec.copy()
        hist_min = [float(Ec.min())]
        for rd in range(int(rf["rounds"])):
            idx = np.argsort(best_E)[: int(rf["n_best"])]
            P = []
            for i in idx:
                v = best_d[i][None] + rf["scale"] * rng.normal(size=(int(rf["n_perturb"]), 9))
                P.append(v / np.linalg.norm(v, axis=1, keepdims=True))
            P = np.concatenate(P)
            rp = fc(P, f"refine {rd + 1}")
            best_d = np.concatenate([best_d, P])
            best_E = np.concatenate([best_E, rp["s_c"] ** 2])
            hist_min.append(float(best_E.min()))
        idx = np.argsort(best_E)[: int(rf["n_best"])]
        worst_dirs = best_d[idx]
        Ec_min = float(min(best_E.min(), (ro["s_c"] ** 2)[0]))
        k = math.floor(math.log10(Ec_min / 10))
        cands = [c * 10 ** kk for kk in (k - 1, k, k + 1) for c in (1, 2, 5)]
        r_A = float(max(c for c in cands if c <= Ec_min / 10))
        # deterministic check on the sphere E = r_A
        Dc = np.concatenate([edge.random_directions(int(ec["r_A_check"]["n_random"]), rng), worst_dirs,
                             og["direction"][None]])
        Xc = LAMINAR + math.sqrt(r_A) * Dc
        cc = edge.classify(rhs, torch.as_tensor(Xc, device=device), 0.25, float(ec["T"]), E_B, float(ec["r_det"]),
                           "rk4", chunk=65536)
        atomic_save_npz(run_dir / "edge.npz", dirs=D, E_c=Ec, out_hi=r["out_hi"], E_max_below=r["E_max_below"],
                        t_hit_below=r["t_hit_below"], worst_dirs=worst_dirs, worst_E_c=best_E[idx],
                        opt_dir=og["direction"], G=og["G"], refine_dirs=best_d, refine_E_c=best_E)
        res["edge"] = {
            "n_dir": int(len(D)), "E_c_quantiles_random": dict(zip(["min", "q01", "q05", "q10", "q25", "q50", "q75", "q90", "q99", "max"],
                                                                 np.quantile(Ec, [0, .01, .05, .1, .25, .5, .75, .9, .99, 1]).tolist())),
            "E_c_max_over_min_random": float(Ec.max() / Ec.min()),
            "no_crossing_random": int(r["no_crossing"].sum()),
            "undecided_at_upper_bracket": int((r["out_hi"] == -1).sum()),
            "E_max_below_quantiles": np.quantile(r["E_max_below"], [0, 0.5, 0.99, 1]).tolist(),
            "optimal_growth": {"G_max": og["G_max"], "t_opt": og["t_opt"], "direction": og["direction"].tolist(),
                               "E_c": float((ro["s_c"] ** 2)[0])},
            "refine_min_history": hist_min, "E_c_min": Ec_min,
            "worst_direction": worst_dirs[0].tolist(), "worst_direction_energy_fractions": (worst_dirs[0] ** 2).tolist(),
            "r_A": r_A,
            "r_A_check": {"n_states": int(len(Xc)), "n_reach_B": int((cc["out"] == 1).sum()),
                          "n_undecided": int((cc["out"] == -1).sum()), "max_E_along": float(cc["E_max"].max()),
                          "t_hit_A_max": float(cc["t_hit"][cc["out"] == 0].max())},
            "seconds": time.perf_counter() - t0}
        save()
    r_A = float(res["edge"]["r_A"])
    print("edge", {k: v for k, v in res["edge"].items() if k not in ("worst_direction", "optimal_growth")})
    if args.stop_after == "edge":
        return

    # ---------------- Lyapunov ----------------
    if stage("lyapunov"):
        S1, resid = edge.lyapunov_laminar(m0)
        lam_ = np.linalg.eigvalsh(S1)
        rng = np.random.default_rng(derive_seed(seed, "lyap_tail"))
        nT = int(cfg["lyapunov"]["tail_samples"])
        chi = rng.chisquare(1, size=(nT, 9)) @ lam_                         # E / sigma^2 under the linear law
        rows = {}
        for s in cfg["sigma_grid"]:
            Ess = s ** 2 * float(np.trace(S1))
            thr = r_A / s ** 2
            p = float((chi > thr).mean())
            # dominant-eigenvalue bound: P(E > r_A) >= P(chi2_1 > r_A / (lam_max sigma^2))
            from scipy.stats import chi2
            rows[str(s)] = {"E_ss": Ess, "r_A_over_E_ss": r_A / Ess, "P_E_gt_r_A_mc": p,
                            "P_E_gt_r_A_dominant_mode": float(chi2.sf(thr / lam_[-1], 1))}
        res["lyapunov"] = {"trace_S1": float(np.trace(S1)), "eig_S1": lam_.tolist(), "residual": resid,
                           "diag_S1": np.diag(S1).tolist(), "per_sigma": rows, "tail_samples": nT}
        save()
    print("lyapunov tr S1", res["lyapunov"]["trace_S1"])

    T_max = float(cfg["T_max"])

    # ---------------- spontaneous transitions ----------------
    sp = cfg["spontaneous"]
    if stage("spontaneous"):
        out = {}
        for s in sp["sigmas"]:
            t0 = time.perf_counter()
            m = MFE(Re=m0.Re, Lx_over_pi=m0.Lx_over_pi, Lz_over_pi=m0.Lz_over_pi, sigma=s, r_A=None, E_B=E_B)
            X0 = torch.as_tensor(LAMINAR, device=device).repeat(int(sp["n_paths"]), 1)
            rec = int(round(sp["record_every_time"] / dt))
            r = simulate_absorbing(m, X0, dt, int(round(T_max / dt)), "heun", torch_generator(derive_seed(seed, "spont", s), device),
                                   compact_every=200, record_every=rec, progress=progress, desc=f"spont {s:g}", bridge=True,
                                   compile_step=True)
            k = int((r["outcome"] == 1).sum())
            n = len(r["outcome"])
            tr = r["rec_step"] * dt >= sp["burn_in"]
            Erec = ((r["rec_X"][tr, 0] - 1) ** 2 + (r["rec_X"][tr, 1:] ** 2).sum(1)) if tr.any() else np.zeros(0)
            out[str(s)] = {"n": n, "n_reach_B": k, "fraction": k / n, "cp95": cp_interval(k, n),
                           "t_reach_B_quantiles": np.quantile(r["steps"][r["outcome"] == 1] * dt, [0, .5, 1]).tolist() if k else None,
                           "mean_E_mc": float(Erec.mean()) if len(Erec) else None,
                           "mean_E_mc_se": float(Erec.reshape(-1).std() / math.sqrt(max(1, len(Erec) / 50))) if len(Erec) else None,
                           "E_ss_lyapunov": s ** 2 * res["lyapunov"]["trace_S1"], "seconds": time.perf_counter() - t0}
            print(s, out[str(s)], flush=True)
        res["spontaneous"] = out
        save()

    # ---------------- sigma choice ----------------
    if stage("sigma"):
        sims = sorted(float(s) for s in res["spontaneous"])
        def spont_ok(s):
            ref = [x for x in sims if x >= s]
            key = str(min(ref)) if ref else str(max(sims))
            if not ref:
                return False
            # find the stored key for that float
            kk = [k for k in res["spontaneous"] if float(k) == float(key)][0]
            return res["spontaneous"][kk]["cp95"][1] <= 1e-3
        ok = [s for s in cfg["sigma_grid"] if res["lyapunov"]["per_sigma"][str(s)]["E_ss"] <= r_A / 10 and spont_ok(s)]
        s_hi = max(ok)
        s_lo = s_hi / 10
        res["sigma"] = {"eligible": ok, "sigma_hi": s_hi, "sigma_lo": float(f"{s_lo:.3g}")}
        save()
    s_hi, s_lo = float(res["sigma"]["sigma_hi"]), float(res["sigma"]["sigma_lo"])
    print("sigma", res["sigma"])
    if args.stop_after == "sigma":
        return

    E = load_npz(run_dir / "edge.npz")
    sigmas = {"sigma_lo": s_lo, "sigma_hi": s_hi}

    # ---------------- A boundary ----------------
    ab = cfg["A_boundary"]
    if stage("a_boundary"):
        rng = np.random.default_rng(derive_seed(seed, "a_boundary"))
        Dr = edge.random_directions(int(ab["n_random"]), rng)
        Dw = np.concatenate([E["worst_dirs"], E["opt_dir"][None]])
        rad = math.sqrt(ab["E_factor"] * r_A)
        out = {}
        for name, s in sigmas.items():
            for absorbA in (True, False):
                t0 = time.perf_counter()
                m = MFE(Re=m0.Re, Lx_over_pi=m0.Lx_over_pi, Lz_over_pi=m0.Lz_over_pi, sigma=s,
                        r_A=r_A if absorbA else None, E_B=E_B)
                Xall = LAMINAR + rad * np.concatenate([Dr, Dw])
                Mall = np.r_[np.full(len(Dr), int(ab["M_random"])), np.full(len(Dw), int(ab["M_worst"]))]
                ra = run_states(m, Xall, Mall, dt, T_max, derive_seed(seed, "ab", name, absorbA), device, progress,
                                f"A-bdry {name} {'absA' if absorbA else 'noA'}")
                rr = {k: ra[k][:len(Dr)] for k in ("n_B", "n_T")}
                rw = {k: ra[k][len(Dr):] for k in ("n_B", "n_T")}
                kr, nr = int(rr["n_B"].sum()), int(len(Dr) * ab["M_random"])
                kw, nw = int(rw["n_B"].sum()), int(len(Dw) * ab["M_worst"])
                out[f"{name}_{'A_absorbing' if absorbA else 'A_not_absorbing'}"] = {
                    "sigma": s, "random": {"n_paths": nr, "n_B": kr, "q": kr / nr, "cp95": cp_interval(kr, nr),
                                           "timeouts": int(rr["n_T"].sum())},
                    "worst": {"n_paths": nw, "n_B": kw, "q": kw / nw, "cp95": cp_interval(kw, nw),
                              "timeouts": int(rw["n_T"].sum()), "max_q_per_direction": float((rw["n_B"] / ab["M_worst"]).max())},
                    "seconds": time.perf_counter() - t0}
                print(name, absorbA, out[f"{name}_{'A_absorbing' if absorbA else 'A_not_absorbing'}"], flush=True)
        res["a_boundary"] = out
        save()

    # ---------------- ray profiles ----------------
    # Coarse log grid around the deterministic edge, then up to `max_zoom` uniform fine grids, each spanning
    # [last point with q <= 0.02, first point with q >= 0.98] of the previous grid, until a ray has >= 4 fine
    # points with 0.02 < q < 0.98. Every simulation result is cached before analysis.
    rc = cfg["rays"]
    max_zoom = int(rc.get("max_zoom", 3))
    lo_q, hi_q = rc["fine_bracket_q"]
    if stage("rays"):
        rng = np.random.default_rng(derive_seed(seed, "rays"))
        Ec = E["E_c"]
        order = np.argsort(Ec)
        pick = lambda q: int(order[int(q * (len(Ec) - 1))])
        dirs = {"refined_min": E["worst_dirs"][0], "random_q10": E["dirs"][pick(0.1)], "random_q50": E["dirs"][pick(0.5)],
                "random_q90": E["dirs"][pick(0.9)], "optimal_growth": E["opt_dir"],
                "random_extra": edge.random_directions(1, rng)[0]}
        names = list(dirs)
        Dm = np.stack([dirs[k] for k in names])
        nR = len(names)
        det_path = run_dir / "rays_det.npz"
        if not det_path.exists():
            det = {}
            for lab, (sch, h) in {"rk4_0.25": ("rk4", 0.25), "heun_dt": ("heun", dt), "heun_dt2": ("heun", dt / 2)}.items():
                rr = edge.first_crossing(rhs, Dm, sgrid, 40, h, float(ec["T"]), E_B, float(ec["r_det"]), sch,
                                         device=device, progress=False)
                det[lab] = rr["s_c"]
            atomic_save_npz(det_path, dirs=Dm, **det)
        det = load_npz(det_path)
        out = {"names": names, "s_c_det": {k: det[k].tolist() for k in ("rk4_0.25", "heun_dt", "heun_dt2")},
               "zoom_rule": f"up to {max_zoom} fine levels of {rc['n_fine']} points, M = {rc['M_fine']}", "profiles": {}}

        def cached_run(path, m, X, M, h, sd, desc):
            if path.exists():
                return load_npz(path)
            r = run_states(m, X, M, h, T_max, sd, device, progress, desc)
            keep = {k: r[k] for k in ("n_B", "n_A", "n_T", "t_abs", "outcome")}
            atomic_save_npz(path, X=X, **keep)
            return keep

        def bracket(S, q):
            j_hi = int(np.argmax(q >= hi_q)) if (q >= hi_q).any() else len(S) - 1
            below = np.where(q[:j_hi] <= lo_q)[0]
            j_lo = int(below[-1]) if len(below) else 0
            return S[j_lo], S[j_hi]

        prof_arrays = {}
        for sname, sv in sigmas.items():
            m = MFE(Re=m0.Re, Lx_over_pi=m0.Lx_over_pi, Lz_over_pi=m0.Lz_over_pi, sigma=sv, r_A=r_A, E_B=E_B)
            sc = det["heun_dt"]
            u = np.asarray(rc["coarse_u"])
            Sc = sc[:, None] * (1 + u[None, :])
            Xc = (LAMINAR[None, None] + Sc[:, :, None] * Dm[:, None, :]).reshape(-1, 9)
            rcz = cached_run(run_dir / f"rays_coarse_{sname}.npz", m, Xc, int(rc["M_coarse"]), dt,
                             derive_seed(seed, "coarse", sname), f"coarse {sname}")
            qc = (rcz["n_B"] / rc["M_coarse"]).reshape(nR, -1)
            S_cur = Sc.copy()
            q_cur = qc.copy()
            final_S = [None] * nR
            final_q = [None] * nR
            levels_used = [0] * nR
            need = list(range(nR))
            for lev in range(1, max_zoom + 1):
                if not need:
                    break
                Sl = np.array([np.linspace(*bracket(S_cur[i], q_cur[i]), int(rc["n_fine"])) for i in need])
                Xl = (LAMINAR[None, None] + Sl[:, :, None] * Dm[need][:, None, :]).reshape(-1, 9)
                rl = cached_run(run_dir / f"rays_fine_{sname}_L{lev}.npz", m, Xl, int(rc["M_fine"]), dt,
                                derive_seed(seed, "fine", sname, lev), f"fine {sname} L{lev}")
                ql = (rl["n_B"] / rc["M_fine"]).reshape(len(need), -1)
                still = []
                for j, i in enumerate(need):
                    S_cur_i, q_cur_i = Sl[j], ql[j]
                    final_S[i], final_q[i], levels_used[i] = S_cur_i, q_cur_i, lev
                    if ((q_cur_i > lo_q) & (q_cur_i < hi_q)).sum() < 4:
                        still.append(i)
                # carry the zoomed grids forward
                S_cur = np.array([final_S[i] if final_S[i] is not None else S_cur[i] for i in range(nR)], dtype=object)
                q_cur = np.array([final_q[i] if final_q[i] is not None else q_cur[i] for i in range(nR)], dtype=object)
                need = still
            final_S = np.array([np.asarray(x, dtype=float) for x in final_S])
            final_q = np.array([np.asarray(x, dtype=float) for x in final_q])
            Xh = (LAMINAR[None, None] + final_S[:, :, None] * Dm[:, None, :]).reshape(-1, 9)
            rh = cached_run(run_dir / f"rays_dt2_{sname}.npz", m, Xh, int(rc["M_fine"]), dt / 2,
                            derive_seed(seed, "fine_dt2", sname), f"fine {sname} dt/2")
            qh = (rh["n_B"] / rc["M_fine"]).reshape(nR, -1)
            rows = {}
            all_t, n_paths_fine, n_to = [], 0, 0
            for lev in range(1, max_zoom + 1):
                p_ = run_dir / f"rays_fine_{sname}_L{lev}.npz"
                if p_.exists():
                    z = load_npz(p_)
                    all_t.append(z["t_abs"][z["outcome"] >= 0])
                    n_paths_fine += len(z["outcome"])
                    n_to += int((z["outcome"] == -1).sum())
            for i, nm in enumerate(names):
                (s05, s50, s95), _ = iso_quantiles(final_S[i], final_q[i])
                (h05, h50, h95), _ = iso_quantiles(final_S[i], qh[i])
                spacing = float(final_S[i][1] - final_S[i][0])
                width = s95 - s05
                resolved = bool(np.isfinite(width) and width > 0 and ((final_q[i] > lo_q) & (final_q[i] < hi_q)).sum() >= 4)
                wd = width if (np.isfinite(width) and width > 0) else (spacing if spacing > 0 else float("nan"))
                d = np.diff(final_q[i])
                rows[nm] = {"s_c_det_rk4": float(det["rk4_0.25"][i]), "s_c_det_heun_dt": float(det["heun_dt"][i]),
                            "s_c_det_heun_dt2": float(det["heun_dt2"][i]), "zoom_levels": int(levels_used[i]),
                            "grid_spacing": spacing, "resolved": resolved,
                            "transition_located": bool(spacing > 0),
                            "s05": s05, "s50": s50, "s95": s95, "width_s": width,
                            "width_rel": width / s50 if s50 else float("nan"),
                            "E50": s50 ** 2, "width_E_rel": (s95 ** 2 - s05 ** 2) / s50 ** 2 if s50 else float("nan"),
                            "s50_minus_det_over_width": (s50 - det["heun_dt"][i]) / wd,
                            "dt_half": {"s05": h05, "s50": h50, "s95": h95, "s50_shift_over_width": (h50 - s50) / wd},
                            "det_shift_heun_vs_rk4_over_width": (det["heun_dt"][i] - det["rk4_0.25"][i]) / wd,
                            "det_shift_heun_dt_vs_dt2_over_width": (det["heun_dt"][i] - det["heun_dt2"][i]) / wd,
                            "n_points_strictly_between": int(((final_q[i] > lo_q) & (final_q[i] < hi_q)).sum()),
                            "max_drop_between_neighbours": float(max(0.0, -d.min())),
                            "coarse_q": qc[i].tolist()}
            tq = np.concatenate(all_t) if all_t else np.zeros(1)
            out["profiles"][sname] = {"sigma": sv, "rays": rows,
                                      "timeout_fraction": n_to / max(n_paths_fine, 1),
                                      "absorption_time_quantiles": dict(zip(["q50", "q90", "q99", "q999", "max"],
                                                                            np.quantile(tq, [.5, .9, .99, .999, 1]).tolist()))}
            prof_arrays[sname] = {"S_f": final_S, "q_dt": final_q, "q_dt2": qh, "S_c": Sc, "q_c": qc}
            print(sname, {nm: (f"{v['width_rel']:.2e}", v["resolved"], round(v["dt_half"]["s50_shift_over_width"], 3))
                          for nm, v in rows.items()}, out["profiles"][sname]["timeout_fraction"], flush=True)
        atomic_save_npz(run_dir / "rays.npz", dirs=Dm, **{f"{k}_{a}": v for k, d in prof_arrays.items() for a, v in d.items()})
        res["rays"] = out
        save()
    print("done")


if __name__ == "__main__":
    main()
