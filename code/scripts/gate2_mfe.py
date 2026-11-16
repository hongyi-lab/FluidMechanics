#!/usr/bin/env python
"""Gate 2 for the MFE model (design of 2026-11-15, code/configs/gate2_mfe.yaml).

Stages (cached; re-running resumes):
  rays          generic random directions and the 70/15/15 ray split
  scan          dense deterministic first-crossing scan -> s_c, E_c, fate switches (H1 scan)
  scan_refine   resolution trend: non-monotone fraction and s_c on a subset at finer grids
  offsets       deterministic labels of the states s_c (1 + u) on every ray
  sensitivity   s_c and offset labels under dt/2, the Heun SDE integrator, E_B x0.8/1.2, r_A x0.8/1.2 (subset)
  family_det    noise family: directions, deterministic first crossing per ray
  family_mc     noise family: MC committor profiles (coarse -> zoom) per sigma; isotonic / lapse-probit / probit fits
  family_sens   M x2, dt/2, r_A x0.8/1.2, E_B x0.8/1.2 on the eps = 0 rays at sigma = 3e-8
  gain_pred     g(s) on the pure-a3 ray and the user's prediction sigma^2 g(s50) = const
  timing        full-scale estimate (only with --frac < 1)

    python code/scripts/gate2_mfe.py --config code/configs/gate2_mfe.yaml --run-dir <dir> [--frac 0.01]
"""
import argparse
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

torch._dynamo.config.cache_size_limit = 256

from committor import edge  # noqa: E402
from committor.fits import iso_with_se, probit_lapse_fit, probit_with_se  # noqa: E402
from committor.gain import noise_gain  # noqa: E402
from committor.sde import simulate_absorbing  # noqa: E402
from committor.systems.mfe import LAMINAR, MFE, rhs_paper  # noqa: E402
from committor.utils import (Status, atomic_save_npz, atomic_write_json, derive_seed, git_info, load_npz,  # noqa: E402
                             read_json, stable_hash, torch_generator, tree_hash)

REPO = Path(__file__).resolve().parents[2]
SIM_CODE = ["sde.py", "edge.py", "gain.py", "fits.py", "utils.py", "systems/base.py", "systems/mfe.py"]


def switches_on(grid_out, step):
    notA = grid_out[:, ::step] != 0
    return np.abs(np.diff(notA.astype(np.int8), axis=1)).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--allow-code-change", action="store_true")
    ap.add_argument("--stop-after", default=None)
    args = ap.parse_args()
    progress = not args.no_progress
    cfg = yaml.safe_load(Path(args.config).read_text())
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    device = torch.device(cfg["device"])
    src = REPO / "code" / "src" / "committor"
    ident = {"config_hash": stable_hash(cfg), "frac": args.frac, "sim_code_hash": tree_hash([src / f for f in SIM_CODE])}
    old = read_json(run / "run_identity.json")
    if old is None:
        atomic_write_json(run / "run_identity.json", ident)
    elif old["config_hash"] != ident["config_hash"] or old["frac"] != ident["frac"]:
        sys.exit(f"{run} was started with {old}, now {ident}; use a new run dir")
    elif old["sim_code_hash"] != ident["sim_code_hash"] and not args.allow_code_change:
        sys.exit(f"simulation code changed since {run} was started; new run dir or --allow-code-change")
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
    S = cfg["system"]
    r_A, E_B = float(cfg["sets"]["r_A"]), float(cfg["sets"]["E_B"])
    m0 = MFE(Re=S["Re"], Lx_over_pi=S["Lx_over_pi"], Lz_over_pi=S["Lz_over_pi"])
    rhs = lambda a: rhs_paper(a, m0.Re, m0.Lx, m0.Lz)
    D = cfg["deterministic"]
    dt_det, T_det = float(D["dt"]), float(D["T_det"])
    frac = args.frac

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

    def grid_for(n, E_B_=E_B, r_A_=r_A):
        return np.sqrt(np.geomspace(r_A_ * (1 + 1e-9), E_B_ * (1 - 1e-9), int(n)))

    s_grid = grid_for(D["dense_grid"]["n_points"])
    fc = lambda dirs, desc, grid=s_grid, E_B_=E_B, r_A_=r_A, dt_=dt_det, scheme="rk4": edge.first_crossing(
        rhs, dirs, grid, int(D["n_bisect"]), dt_, T_det, E_B_, r_A_, scheme, device=device, chunk=1 << 20,
        progress=progress, desc=desc)

    # ---------------- generic rays ----------------
    def st_rays():
        n = max(10, int(round(D["n_rays"] * frac)))
        rng = np.random.default_rng(derive_seed(seed, "generic_dirs"))
        dirs = edge.random_directions(n, rng)
        u = np.random.default_rng(derive_seed(seed, "split")).random(n)
        c = np.cumsum(D["split"])
        split = np.where(u < c[0], 0, np.where(u < c[1], 1, 2)).astype(np.int8)
        atomic_save_npz(run / "rays.npz", dirs=dirs, split=split)
        return {"n_rays": n, "split_counts": np.bincount(split, minlength=3).tolist()}
    timed("rays", st_rays)
    dirs = load_npz(run / "rays.npz")["dirs"]

    # ---------------- dense scan (H1) ----------------
    def st_scan():
        r = fc(dirs, "dense scan")
        atomic_save_npz(run / "scan.npz", s_c=r["s_c"], lo=r["lo"], hi=r["hi"], out_hi=r["out_hi"],
                        no_crossing=r["no_crossing"], below_grid=r["below_grid"], first_index=r["first_index"],
                        n_switches=r["n_switches"], E_max_below=r["E_max_below"], grid_out=r["grid_out"],
                        t_hit_below=r["t_hit_below"], t_hit_above=r["t_hit_above"], s_grid=s_grid)
        Ec = r["s_c"] ** 2
        fin = np.isfinite(Ec)
        order = np.argsort(np.where(fin, Ec, np.inf))[:int(cfg["h1_ray_scan"]["report"]["smallest_E_c_rays"])]
        sw = r["n_switches"]
        G = len(s_grid)
        return {"n_rays": int(len(dirs)), "grid_points": int(G), "grid_spacing_rel_s": float(s_grid[1] / s_grid[0] - 1),
                "E_c_quantiles": dict(zip(["q0", "q01", "q05", "q25", "q50", "q75", "q95", "q99", "q100"],
                                          np.quantile(Ec[fin], cfg["h1_ray_scan"]["report"]["E_c_quantiles"]).tolist())),
                "E_c_max_over_min": float(Ec[fin].max() / Ec[fin].min()),
                "fraction_no_transition_below_cap": float(r["no_crossing"].mean()), "cap_E": E_B,
                "fraction_below_grid": float(r["below_grid"].mean()),
                "fraction_non_monotone_fate": float((sw >= 2).mean()),
                "non_monotone_by_grid_points": {str(G // st): float((switches_on(r["grid_out"], st) >= 2).mean())
                                                for st in (20, 4, 2, 1)},
                "switch_count_hist": np.bincount(sw).tolist(),
                "fraction_undecided_grid_states": float((r["grid_out"] == -1).mean()),
                "fraction_upper_bracket_undecided": float((r["out_hi"][fin] == -1).mean()),
                "slowest_decision_time": float(max(r["t_hit_below"].max(), r["t_hit_above"].max())),
                "E_max_below_quantiles": np.quantile(r["E_max_below"][fin], [0, 0.5, 0.99, 1]).tolist(),
                "smallest_E_c_rays": [{"ray": int(i), "E_c": float(Ec[i]), "d": dirs[i].tolist(),
                                       "energy_fractions": (dirs[i] ** 2).tolist()} for i in order]}
    timed("scan", st_scan)
    print({k: v for k, v in res["scan"].items() if k != "smallest_E_c_rays"})
    SC = load_npz(run / "scan.npz")
    s_c = SC["s_c"]

    # ---------------- resolution trend ----------------
    def st_refine():
        nr = D["nonmonotone_refine"]
        n = max(5, int(round(nr["n_rays"] * frac)))
        rs = np.random.default_rng(derive_seed(seed, "refine_rays")).choice(len(dirs), n, replace=False)
        out = {"n_rays": int(n), "base_grid": int(len(s_grid)),
               "base_non_monotone": float((SC["n_switches"][rs] >= 2).mean())}
        for G in nr["grids"]:
            r = fc(dirs[rs], f"refine {G}", grid=grid_for(G))
            dl = np.abs(np.log(r["s_c"] / s_c[rs]))
            out[str(G)] = {"non_monotone": float((r["n_switches"] >= 2).mean()),
                           "mean_switches": float(r["n_switches"].mean()),
                           "abs_dln_s_c_median": float(np.nanmedian(dl)), "abs_dln_s_c_max": float(np.nanmax(dl)),
                           "n_s_c_changed_over_1e-6": int((dl > 1e-6).sum())}
            print("   ", G, out[str(G)], flush=True)
        return out
    timed("scan_refine", st_refine)
    if args.stop_after == "scan":
        return
    U = np.asarray(D["offsets_u"], dtype=float)

    # ---------------- offset states, deterministic labels ----------------
    def st_offsets():
        idx = np.where(np.isfinite(s_c))[0]
        Sx = s_c[idx, None] * (1 + U[None, :])
        X = (LAMINAR[None, None] + Sx[:, :, None] * dirs[idx][:, None, :]).reshape(-1, 9)
        r = edge.classify(rhs, torch.as_tensor(X, device=device), dt_det, T_det, E_B, r_A, "rk4", chunk=1 << 20)
        lab = r["out"].reshape(len(idx), len(U))
        atomic_save_npz(run / "offsets.npz", ray=idx, u=U, X=X.reshape(len(idx), len(U), 9), label=lab,
                        t_hit=r["t_hit"].reshape(len(idx), len(U)))
        return {"n_states": int(lab.size), "n_rays": int(len(idx)),
                "fraction_B_per_u": dict(zip([f"{x:g}" for x in U], np.round((lab == 1).mean(0), 4).tolist())),
                "fraction_undecided": float((lab == -1).mean())}
    timed("offsets", st_offsets)

    # ---------------- sensitivity of deterministic labels ----------------
    def st_sens():
        sv = D["sensitivity"]
        n = max(5, int(round(sv["n_rays"] * frac)))
        rs = np.random.default_rng(derive_seed(seed, "sens_rays")).choice(np.where(np.isfinite(s_c))[0], n, replace=False)
        OF = load_npz(run / "offsets.npz")
        pos = {r: k for k, r in enumerate(OF["ray"])}
        Xo = np.stack([OF["X"][pos[r]] for r in rs]).reshape(-1, 9)
        base = np.stack([OF["label"][pos[r]] for r in rs]).reshape(-1)
        out = {"n_rays": int(n)}
        for name, v in sv["variants"].items():
            t0 = time.perf_counter()
            E_B_ = E_B * v.get("E_B_scale", 1.0)
            r_A_ = r_A * v.get("r_A_scale", 1.0)
            dt_ = v.get("dt", dt_det)
            scheme = v.get("scheme", "rk4")
            s_new = np.full(n, np.nan)
            n_fallback = 0
            # local first-crossing grid over [0.9, 1.1] s_c (all rays in one batch through first_crossing on
            # ray-specific grids: done by scaling each ray's direction so that the common grid u in [0.9, 1.1] maps to s)
            ug = np.linspace(0.9, 1.1, int(sv["local_grid"]))
            dirs_scaled = dirs[rs] * s_c[rs, None]                      # a = lam + u * (s_c d)
            rr = edge.first_crossing(rhs, dirs_scaled, ug, int(D["n_bisect"]), dt_, T_det, E_B_, r_A_, scheme,
                                     device=device, chunk=1 << 20, progress=False)
            ok = np.isfinite(rr["s_c"])                                  # first local point A and a crossing inside
            s_new[ok] = rr["s_c"][ok] * s_c[rs][ok]
            if (~ok).any():
                n_fallback = int((~ok).sum())
                rf = fc(dirs[rs[~ok]], f"sens fallback {name}", grid=grid_for(len(s_grid), E_B_, r_A_), E_B_=E_B_,
                        r_A_=r_A_, dt_=dt_, scheme=scheme)
                s_new[~ok] = rf["s_c"]
            dl = np.abs(np.log(s_new / s_c[rs]))
            lo_ = edge.classify(rhs, torch.as_tensor(Xo, device=device), dt_, T_det, E_B_, r_A_, scheme, chunk=1 << 20)["out"]
            out[name] = {"abs_dln_s_c": {"median": float(np.nanmedian(dl)), "q90": float(np.nanquantile(dl, 0.9)),
                                         "max": float(np.nanmax(dl))},
                         "n_fallback_dense": n_fallback,
                         "offset_label_changes": int((lo_ != base).sum()), "offset_states": int(base.size),
                         "label_change_by_u": dict(zip([f"{x:g}" for x in U], (lo_ != base).reshape(n, -1).mean(0).round(4).tolist())),
                         "seconds": time.perf_counter() - t0}
            print("   ", name, out[name]["abs_dln_s_c"], out[name]["offset_label_changes"], round(out[name]["seconds"], 1), flush=True)
        return out
    timed("sensitivity", st_sens)
    if args.stop_after == "deterministic":
        return

    # ---------------- noise family ----------------
    F = cfg["noise_family"]
    P = F["profile"]
    sde = F["sde"]
    fam_rays = []
    rngf = np.random.default_rng(derive_seed(seed, "family"))
    for ang in F["angles_deg"]:
        t = math.radians(ang)
        base = np.zeros(9)
        base[2], base[1] = math.cos(t), math.sin(t)
        rs_dir = []
        for k in range(int(F["draws_per_eps"])):                       # (c7) nested: one r per (angle, draw)
            r = np.zeros(9)
            r[3:8] = rngf.normal(size=5)
            r[3:8] /= np.linalg.norm(r[3:8])
            rs_dir.append(r)
        for eps in F["eps"]:
            for k in range(1 if eps == 0 else int(F["draws_per_eps"])):
                v = base.copy() if eps == 0 else (base + eps * rs_dir[k]) / math.sqrt(1 + eps ** 2)
                fam_rays.append({"angle": float(ang), "eps": float(eps), "draw": k, "d": v})
    FD = np.stack([f["d"] for f in fam_rays])
    pairs = [(si, ri) for si in range(len(F["sigmas"])) for ri in range(len(fam_rays))]
    if frac < 1:
        want = cfg["timing"]["frac_family_pairs"]
        pairs = [(si, ri) for (si, ri) in pairs for w in want
                 if F["sigmas"][si] == w["sigma"] and fam_rays[ri]["angle"] == w["angle"]
                 and fam_rays[ri]["eps"] == w["eps"] and fam_rays[ri]["draw"] == w["draw"]]

    def st_family_det():
        r = fc(FD, "family dense scan")
        atomic_save_npz(run / "family_det.npz", d=FD, s_c=r["s_c"], n_switches=r["n_switches"], no_crossing=r["no_crossing"],
                        E_max_below=r["E_max_below"])
        return {"n_rays": len(fam_rays), "rays": [{"angle": f["angle"], "eps": f["eps"], "draw": f["draw"],
                                                    "E_c_det": float(r["s_c"][i] ** 2) if np.isfinite(r["s_c"][i]) else None,
                                                    "n_switches": int(r["n_switches"][i])} for i, f in enumerate(fam_rays)]}
    timed("family_det", st_family_det)

    def run_states(m, X, M, dt, tag):
        path = run / f"{tag}.npz"
        Ms = np.broadcast_to(np.asarray(M, dtype=int), (len(X),))
        if path.exists():
            z = load_npz(path)
            if z["X"].shape == X.shape and np.array_equal(z["X"], X) and np.array_equal(z["M"], Ms):
                return z
            raise RuntimeError(f"cache {path} does not match the requested batch")
        X0 = torch.as_tensor(np.repeat(X, Ms, axis=0), device=device)
        t0 = time.perf_counter()
        r = simulate_absorbing(m, X0, dt, int(round(sde["T_max"] / dt)), sde["scheme"],
                               torch_generator(derive_seed(seed, tag), device), compact_every=200, progress=progress,
                               desc=tag, bridge=bool(sde["boundary_bridge"]), compile_step=True)
        idx = np.repeat(np.arange(len(X)), Ms)
        z = {"X": X, "M": Ms, "n_B": np.bincount(idx, weights=(r["outcome"] == 1), minlength=len(X)).astype(int),
             "n_T": np.bincount(idx, weights=(r["outcome"] == -1), minlength=len(X)).astype(int),
             "path_steps": np.int64(r["steps"].sum()), "max_steps": np.int64(r["steps"].max()), "n_paths": np.int64(len(idx)),
             "seconds": np.float64(time.perf_counter() - t0), "t_abs_q": np.quantile(r["steps"] * dt, [0.5, 0.99, 1.0])}
        atomic_save_npz(path, **z)
        return z

    lo_q, hi_q = P["bracket_q"]

    def bracket(Sv, q):
        j_hi = int(np.argmax(q >= hi_q)) if (q >= hi_q).any() else None
        if j_hi is None:
            return None
        below = np.where(q[:j_hi] <= lo_q)[0]
        j_lo = int(below[-1]) if len(below) else 0
        return Sv[j_lo], Sv[j_hi]

    def interior(q):
        return int(((q > lo_q) & (q < hi_q)).sum())

    def fits(Sv, k, Mv, seed_):
        out = {}
        iso = iso_with_se(Sv, k, Mv, seed=seed_)
        out.update({f"iso_{a}": b for a, b in iso.items()})
        s50l, wl, lam = probit_lapse_fit(Sv, k, Mv)
        out.update({"lapse_s50": s50l, "lapse_width": wl, "lapse": lam})
        pr = probit_with_se(Sv, k, Mv, n_boot=100, seed=seed_)
        out.update({f"probit_{a}": b for a, b in pr.items()})
        out["s50"], out["s50_se"], out["width"] = iso["s50"], iso["s50_se"], iso["width"]   # primary (c8)
        return out

    def st_family_mc():
        out = {"pairs": len(pairs), "profiles": []}
        coarse_s = np.sqrt(np.geomspace(P["coarse_E"][0], P["coarse_E"][1] * (1 - 1e-9), int(P["coarse_E"][2])))
        for si, sig in enumerate(F["sigmas"]):
            rays_here = [ri for (s_, ri) in pairs if s_ == si]
            if not rays_here:
                continue
            m = MFE(Re=m0.Re, Lx_over_pi=m0.Lx_over_pi, Lz_over_pi=m0.Lz_over_pi, sigma=sig, r_A=r_A, E_B=E_B)
            Xc = (LAMINAR[None, None] + coarse_s[None, :, None] * FD[rays_here][:, None, :]).reshape(-1, 9)
            zc = run_states(m, Xc, int(P["M_coarse"]), float(sde["dt"]), f"fam_coarse_s{si}")
            kc = zc["n_B"].reshape(len(rays_here), -1)
            grids = {ri: (coarse_s.copy(), kc[j], int(P["M_coarse"]), 0) for j, ri in enumerate(rays_here)}
            Mfine = {ri: (int(P["M_fine_a3"]) if (fam_rays[ri]["angle"] == 0 and fam_rays[ri]["eps"] == 0) else int(P["M_fine"]))
                     for ri in rays_here}
            need = [ri for ri in rays_here if interior(grids[ri][1] / grids[ri][2]) < int(P["min_interior_points"])]
            for lev in range(1, int(P["max_zoom"]) + 1):
                todo = [ri for ri in need if bracket(grids[ri][0], grids[ri][1] / grids[ri][2]) is not None]
                if not todo:
                    break
                Sl = np.array([np.linspace(*bracket(grids[ri][0], grids[ri][1] / grids[ri][2]), int(P["n_fine"])) for ri in todo])
                Xl = (LAMINAR[None, None] + Sl[:, :, None] * FD[todo][:, None, :]).reshape(-1, 9)
                Ml = np.repeat([Mfine[ri] for ri in todo], int(P["n_fine"]))
                zl = run_states(m, Xl, Ml, float(sde["dt"]), f"fam_fine_s{si}_L{lev}")
                kl = zl["n_B"].reshape(len(todo), -1)
                for j, ri in enumerate(todo):
                    grids[ri] = (Sl[j], kl[j], Mfine[ri], lev)
                need = [ri for ri in todo if interior(grids[ri][1] / grids[ri][2]) < int(P["min_interior_points"])]
            for ri in rays_here:
                Sv, kv, Mv, lev = grids[ri]
                q = kv / Mv
                f = fam_rays[ri]
                row = {"sigma": sig, "ray": ri, "angle": f["angle"], "eps": f["eps"], "draw": f["draw"], "zoom_levels": lev,
                       "M": Mv, "interior_points": interior(q), "q_min": float(q.min()), "q_max": float(q.max()),
                       "transition_found": bool((q <= lo_q).any() and (q >= hi_q).any())}
                row["resolved"] = row["transition_found"] and row["interior_points"] >= int(P["min_interior_points"])
                if row["transition_found"]:
                    row.update(fits(Sv, kv, Mv, ri))
                    row["E50"] = row["s50"] ** 2
                    row["width_rel"] = row["width"] / row["s50"]
                out["profiles"].append(row)
                atomic_save_npz(run / f"fam_profile_s{si}_r{ri}.npz", S=Sv, k=kv, M=np.int64(Mv))
                print("    profile", {k_: row.get(k_) for k_ in ("sigma", "angle", "eps", "draw", "s50", "s50_se", "lapse_s50",
                                                                "probit_s50", "width_rel", "zoom_levels", "interior_points", "resolved")}, flush=True)
        work = sorted(p for p in run.glob("fam_*_s*.npz") if "profile" not in p.name)
        zs = [load_npz(p) for p in work]
        out["paths"] = int(sum(int(z["n_paths"]) for z in zs))
        out["path_steps"] = int(sum(int(z["path_steps"]) for z in zs))
        out["timeouts"] = int(sum(int(z["n_T"].sum()) for z in zs))
        out["batches"] = [{"name": p.name, "n_paths": int(z["n_paths"]), "max_time": float(z["t_abs_q"][-1]),
                           "seconds": float(z["seconds"])} for p, z in zip(work, zs)]
        return out
    timed("family_mc", st_family_mc)

    # ---------------- noise family sensitivity ----------------
    def st_family_sens():
        sig = 3.0e-8
        si = F["sigmas"].index(sig)
        cand = [p for p in res["family_mc"]["profiles"] if p["sigma"] == sig and p["eps"] == 0 and p.get("resolved")]
        out = {"n_profiles": len(cand), "coverage": "eps = 0 rays at sigma = 3e-8 only"}
        if not cand:
            out["note"] = "no resolved eps = 0 profile at sigma 3e-8"
            return out
        sv = F["sensitivity"]
        Xs, Mb, seg = [], [], []
        for p in cand:
            z = load_npz(run / f"fam_profile_s{si}_r{p['ray']}.npz")
            Xs.append(LAMINAR[None] + z["S"][:, None] * FD[p["ray"]][None])
            Mb.append(np.full(len(z["S"]), int(z["M"])))
            seg.append((p["ray"], z["S"], int(z["M"]), z["k"]))
        Xall, Mall = np.concatenate(Xs), np.concatenate(Mb)
        mk = lambda **kw: MFE(Re=m0.Re, Lx_over_pi=m0.Lx_over_pi, Lz_over_pi=m0.Lz_over_pi, sigma=sig,
                              r_A=kw.get("r_A", r_A), E_B=kw.get("E_B", E_B))
        variants = {"M_double": (mk(), 2 * Mall, float(sde["dt"])), "dt_half": (mk(), Mall, float(sv["dt_half"]["dt"]))}
        for fct in sv["r_A"]:
            variants[f"r_A_{fct}"] = (mk(r_A=r_A * fct), Mall, float(sde["dt"]))
        for fct in sv["E_B"]:
            variants[f"E_B_{fct}"] = (mk(E_B=E_B * fct), Mall, float(sde["dt"]))
        for name, (m, Mv, dtv) in variants.items():
            z = run_states(m, Xall, Mv, dtv, f"fam_sens_{name}")
            rows, off = [], 0
            for (ri, Sv, Mb_, kb) in seg:
                n_ = len(Sv)
                kv = z["n_B"][off:off + n_]
                Mvv = Mv[off:off + n_]
                off += n_
                a = iso_with_se(Sv, kb, Mb_, n_boot=200, seed=ri)
                b = iso_with_se(Sv, kv, Mvv, n_boot=200, seed=ri + 1)
                se = math.hypot(a["s50_se"], b["s50_se"])
                rows.append({"ray": ri, "ds50_rel": b["s50"] / a["s50"] - 1, "z": (b["s50"] - a["s50"]) / se if se > 0 else None,
                             "dwidth_rel": b["width"] / a["width"] - 1 if a["width"] else None})
            out[name] = {"profiles": rows, "max_abs_ds50_rel": float(max(abs(r_["ds50_rel"]) for r_ in rows)),
                         "timeouts": int(z["n_T"].sum())}
            print("   ", name, out[name]["max_abs_ds50_rel"], flush=True)
        return out
    timed("family_sens", st_family_sens)

    # ---------------- user's noise-gain prediction on the pure-a3 ray ----------------
    def st_gain():
        G = cfg["noise_gain_prediction"]["gain"]
        s = np.linspace(2.5e-3, 7.0e-3, 451)
        X = np.repeat(LAMINAR[None], len(s), 0)
        X[:, 2] = s
        g = noise_gain(m0, X, 1.0, T_g=float(G["T_g"]), dt=float(G["dt"]), device=device)
        atomic_save_npz(run / "gain_a3.npz", s=s, g=g["max_trP"], t_at_max=g["t_at_max"])
        lg = np.log(g["max_trP"])
        dlg = np.gradient(lg, s)
        lg_at = lambda x: float(np.interp(x, s, lg))
        slope_at = lambda x: float(np.interp(x, s, dlg))
        ginv = lambda lnK: float(np.interp(lnK, lg, s))
        prof = sorted([p for p in res["family_mc"]["profiles"] if p["angle"] == 0.0 and p["eps"] == 0 and p.get("resolved")],
                      key=lambda p: p["sigma"])
        out = {"g_range": [float(g["max_trP"].min()), float(g["max_trP"].max())],
               "t_at_max_range": [float(g["t_at_max"].min()), float(g["t_at_max"].max())],
               "monotone_in_s": bool(np.all(np.diff(lg) > 0)), "user": cfg["noise_gain_prediction"]["user_values_s50"],
               "user_ln_sigma2_g": {str(k): math.log(float(k) ** 2) + lg_at(v) for k, v in cfg["noise_gain_prediction"]["user_values_s50"].items()},
               "by_estimator": {}}
        for est, key, sekey in (("isotonic", "iso_s50", "iso_s50_se"), ("lapse_probit", "lapse_s50", None), ("probit", "probit_s50", "probit_s50_se")):
            rows = [{"sigma": p["sigma"], "s50": p[key], "s50_se": (p[sekey] if sekey else p["iso_s50_se"]),
                     "ln_sigma2_g": math.log(p["sigma"] ** 2) + lg_at(p[key])} for p in prof if np.isfinite(p.get(key, np.nan))]
            for r_ in rows:
                r_["ln_sigma2_g_se"] = slope_at(r_["s50"]) * r_["s50_se"]
            e = {"rows": rows}
            if len(rows) >= 2:
                mid = rows[len(rows) // 2]
                e["prediction_from_middle"] = []
                for r_ in rows:
                    if r_ is mid:
                        continue
                    sp = ginv(mid["ln_sigma2_g"] - math.log(r_["sigma"] ** 2))
                    # propagated SE of the relative error: from the middle s50 (through g) and from the target s50
                    se_pred = mid["s50_se"] * slope_at(mid["s50"]) / slope_at(sp)
                    rel = sp / r_["s50"] - 1
                    e["prediction_from_middle"].append({"sigma": r_["sigma"], "s50_mc": r_["s50"], "s50_pred": sp, "rel_err": rel,
                                                        "rel_err_se": math.hypot(se_pred / r_["s50"], r_["s50_se"] / r_["s50"]),
                                                        "null_no_sigma_dependence_rel_err": mid["s50"] / r_["s50"] - 1})
                e["p_eff"] = []
                for a, b in zip(rows[:-1], rows[1:]):
                    dl = math.log(b["sigma"] / a["sigma"])
                    p_ = (lg_at(a["s50"]) - lg_at(b["s50"])) / dl                   # sigma^p g(s50) = const
                    se_ = math.hypot(a["ln_sigma2_g_se"], b["ln_sigma2_g_se"]) / dl
                    e["p_eff"].append({"sigmas": [a["sigma"], b["sigma"]], "p_eff": p_, "se": se_})
            out["by_estimator"][est] = e
        return out
    timed("gain_pred", st_gain)
    print("gain", {k: v for k, v in res["gain_pred"].items() if k not in ("user",)})

    if frac < 1 and "timing" not in res:
        res["timing"] = estimate(res, cfg, len(fam_rays), rhs, device, run)
        save()
        print("timing", res["timing"])
    print("done")


# ------------------------------------------------------------------------------------------------
# full-scale estimate
# ------------------------------------------------------------------------------------------------

def _bench_det(rhs, n, device, dt=0.25, scheme="rk4", T=100.0):
    """Seconds per step of edge.classify at batch n, after compiling and capturing at that exact shape."""
    rng = np.random.default_rng(0)
    X = torch.as_tensor(LAMINAR + 3e-3 * edge.random_directions(n, rng), device=device)
    edge.classify(rhs, X, dt, 5.0, 0.29, 1e-30, scheme)          # compile + capture at this shape
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    edge.classify(rhs, X, dt, T, 0.29, 1e-30, scheme)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    edge.classify(rhs, X, dt, 2 * T, 0.29, 1e-30, scheme)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return ((t2 - t1) - (t1 - t0)) / (T / dt)                     # difference removes the per-call capture cost


def _bench_sde(n, device, steps=1500):
    m = MFE(sigma=3e-8, r_A=None, E_B=0.29)
    X = torch.as_tensor(LAMINAR, device=device).repeat(n, 1)
    simulate_absorbing(m, X, 0.01, 30, "heun", torch_generator(0, device), progress=False, bridge=True, compile_step=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    simulate_absorbing(m, X, 0.01, steps, "heun", torch_generator(0, device), progress=False, bridge=True, compile_step=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps


def estimate(res, cfg, n_fam_rays, rhs, device, run):
    """Full-scale wall time: per-step costs benchmarked at the full batch sizes (compiled at those shapes), times
    the number of steps a batch runs (set by its slowest row / path). Reported as a range: lower = slowest times
    observed in this run, upper = T_det / T_max and the maximum number of zoom levels."""
    D, F = cfg["deterministic"], cfg["noise_family"]
    P, sde = F["profile"], F["sde"]
    n_full, G = D["n_rays"], D["dense_grid"]["n_points"]
    nb = int(D["n_bisect"])
    t_obs = min(max(res["scan"]["slowest_decision_time"], 1000.0), D["T_det"])
    est = {"bench": {}}
    def c(n, dt=0.25, scheme="rk4"):
        key = f"det_{scheme}_{dt}_{n}"
        if key not in est["bench"]:
            est["bench"][key] = _bench_det(rhs, n, device, dt, scheme)
        return est["bench"][key]
    chunk = 1 << 20

    def det_cost(rows, t, dt=0.25, scheme="rk4"):
        k = math.ceil(rows / chunk)
        last = rows - (k - 1) * chunk
        return ((k - 1) * c(chunk, dt, scheme) + c(max(last, 2), dt, scheme)) * t / dt

    for tag, t in (("low", t_obs), ("high", float(D["T_det"]))):
        scan = det_cost(n_full * G, t) + nb * det_cost(n_full, t)
        nr = D["nonmonotone_refine"]
        refine = sum(det_cost(nr["n_rays"] * g, t) + nb * det_cost(nr["n_rays"], t) for g in nr["grids"])
        offs = det_cost(n_full * len(D["offsets_u"]), t)
        sens = 0.0
        ns = D["sensitivity"]["n_rays"]
        for v in D["sensitivity"]["variants"].values():
            dt_, sch = v.get("dt", 0.25), v.get("scheme", "rk4")
            sens += det_cost(ns * D["sensitivity"]["local_grid"], t, dt_, sch) + nb * det_cost(ns, t, dt_, sch) \
                + det_cost(ns * len(D["offsets_u"]), t, dt_, sch)
        est[f"deterministic_seconds_{tag}"] = {"scan": scan, "scan_refine": refine, "offsets": offs, "sensitivity": sens}
    # family MC
    fm = res["family_mc"]
    t_mc_obs = max([b["max_time"] for b in fm["batches"]] + [1500.0])
    n_sig = len(F["sigmas"])
    n_coarse = n_fam_rays * int(P["coarse_E"][2]) * int(P["M_coarse"])
    n_zoom = n_fam_rays * int(P["n_fine"]) * int(P["M_fine"])
    n_sens = 8 * int(P["n_fine"]) * int(P["M_fine_a3"])
    cs = {}
    for n in (n_coarse, n_zoom, 2000, n_sens):
        if n not in cs:
            cs[n] = _bench_sde(n, device)
            est["bench"][f"sde_{n}"] = cs[n]
    obs_levels = max([p.get("zoom_levels", 1) for p in fm["profiles"]] + [1])
    dt = sde["dt"]
    for tag, tmax, lev in (("low", t_mc_obs, obs_levels), ("high", sde["T_max"], int(P["max_zoom"]))):
        bulk, tail = 1500.0 / dt, max(tmax - 1500.0, 0.0) / dt
        batch = lambda n: bulk * cs[n] + tail * cs[2000]
        mc = n_sig * (batch(n_coarse) + lev * batch(n_zoom))
        fs = 6 * batch(n_sens) + (bulk * 2 * cs[n_sens] + 2 * tail * cs[2000])        # 6 variants at dt + dt/2
        est[f"family_seconds_{tag}"] = {"family_mc": mc, "family_sens": fs, "family_det": res["family_det"]["seconds"],
                                        "gain_pred": res["gain_pred"]["seconds"]}
    for tag in ("low", "high"):
        est[f"total_hours_{tag}"] = (sum(est[f"deterministic_seconds_{tag}"].values()) + sum(est[f"family_seconds_{tag}"].values())) / 3600
    est["observed"] = {"slowest_decision_time": t_obs, "slowest_mc_path_time": t_mc_obs, "zoom_levels": obs_levels}
    est["this_run_seconds"] = {k: res[k]["seconds"] for k in res if isinstance(res[k], dict) and "seconds" in res[k]}
    return est


if __name__ == "__main__":
    main()
