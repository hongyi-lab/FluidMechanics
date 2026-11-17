#!/usr/bin/env python
"""Gate 4 (continued, 2026-11-17): revised item 5 (5a, 5b, 5c; post-hoc) and 4b (pilot A/B/T_det with the Re ladder, resolution
ladder), following gate4_dns.yaml (history 07:40, e15-e20). Stops only when a preset ladder is used up.

    python code/scripts/gate4b_dns.py --config code/configs/gate4_dns.yaml --cfg5 code/configs/gate5_dns.yaml --run-dir <dir>

Resumable: finished steps are kept in results.json; pilot chunks and s_c searches are checkpointed in the run dir.
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

from committor.dns import Grid, find_sc_det, merge_counts, run_det, step_nl_c, _kmax_t
from committor.dns_rays import build_rays, ladder_rays
from committor.utils import atomic_save_npz, atomic_write_json, derive_seed, git_info, read_json

REPO = Path(__file__).resolve().parents[2]
SHARED = ["mirror y about 1 (u,-v,w)(x,2-y,z)", "shift-reflect (u,v,-w)(x+Lx/2,y,-z)", "shift-rotate (-u,-v,w)(-x,-y,z+Lz/2)"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def make_grid(cfg, N, Re):
    S = cfg["system"]
    g = Grid(*N, Re=Re, Lx=S["Lx_over_pi"] * math.pi, Lz=S["Lz_over_pi"] * math.pi, Ly=S["Ly_ext"], device=cfg["device"])
    g.set_enforced(SHARED)
    return g


def nkey(N):
    return "x".join(str(int(n)) for n in N)


def timing_per_case_step(g, dt, n_cases=32, n_steps=40):
    """measured wall time per case-step of the full nonlinear step (with the per-step Courant number)."""
    rng = np.random.default_rng(0)
    d = rng.standard_normal((n_cases, 9))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    u = g.clean(g.from_coeffs(0.05 * d))
    E1, E2 = g.lap_factor(dt), g.lap_factor(0.5 * dt)
    km = _kmax_t(g)
    for _ in range(3):
        u, _c = step_nl_c(g, u, dt, E1, E2, km)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_steps):
        u, _c = step_nl_c(g, u, dt, E1, E2, km)
    torch.cuda.synchronize()
    return (time.time() - t0) / (n_steps * n_cases)


# ------------------------------------------------------------------------------------------------------------ 5a
def check_5a(cfg, g, dt):
    C = cfg["revised_5"]["5a"]
    rng = np.random.default_rng(derive_seed(int(cfg["seed"]), "dt"))                 # the 4a.5 directions (e19)
    dd = rng.standard_normal((int(C["n_dirs"]), 9))
    dd /= np.linalg.norm(dd, axis=1, keepdims=True)
    u0 = g.clean(g.from_coeffs(math.sqrt(float(C["E0"])) * dd))
    T = float(C["t_max"])
    out = {"dt": dt, "dt_half": dt / 2, "E0": float(C["E0"]), "directions": dd.tolist()}
    E = {}
    for h in (dt, dt / 2):
        r = run_det(g, u0, h, T, None, None, absorb=False, courant_limit=float(cfg["revised_5"]["5b"]["courant_limit"]), max_levels=0)
        E[h] = r["E"]
        out[f"courant_max_dt{h:g}"] = float(np.nanmax(r["courant_max"]))
        out[f"n_nan_dt{h:g}"] = int(np.sum(~np.isfinite(r["E"])))
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(E[dt] / E[dt / 2] - 1)
    n_nan = int(np.sum(~np.isfinite(rel)))
    per = [float(np.max(rel[k])) if np.isfinite(rel[k]).all() else None for k in range(len(dd))]
    out["max_rel_dE_per_direction"] = per
    out["n_nan_samples"] = n_nan
    out["max_rel_dE"] = None if n_nan else float(np.max(rel))
    out["pass"] = bool(n_nan == 0 and np.max(rel) < float(C["tol"]))
    out["E_range"] = [float(np.nanmin(E[dt])), float(np.nanmax(E[dt]))]
    return out


def item1_stable(cfg, g, dt):
    """descriptive (not a criterion; 4a is not re-run): the 4a item-1 measures on runs at a stable dt, NaN counted explicitly.
    The 4a.5 directions at E(0) = 1e-6 and 1e-3, t <= 200: symmetry residual max|g u' - u'| before re-symmetrisation after every
    step, max|div u| and E after every 1.0."""
    from committor.dns import lawson_rk4
    rng = np.random.default_rng(derive_seed(int(cfg["seed"]), "dt"))
    dd = rng.standard_normal((4, 9))
    dd /= np.linalg.norm(dd, axis=1, keepdims=True)
    out = {"dt": dt}
    for E0 in (1e-6, 1e-3):
        u = g.clean(g.from_coeffs(math.sqrt(E0) * dd))
        E1, E2 = g.lap_factor(dt), g.lap_factor(0.5 * dt)
        km = _kmax_t(g)
        sm, dm, cm = [], [], []
        per = int(round(1.0 / dt))
        for k in range(1, int(round(200.0 / dt)) + 1):
            box = {}

            def rhs(s_):
                from committor.dns import nl_full_c
                if "mx" not in box:
                    n_, box["mx"] = nl_full_c(g, s_[0])
                    return (n_,)
                return (g.nl_full(s_[0]),)
            (un,) = lawson_rk4(g, (u,), rhs, dt, E1, E2)
            cm.append((dt * (box["mx"] * km).sum(1)).cpu().numpy())
            un = un * g.mask
            sm.append(g.sym_residual(un).cpu().numpy())
            u = g.symmetrize(un)
            if k % per == 0:
                dm.append(g.max_div(u).cpu().numpy())
        sm, dm, cm = np.array(sm), np.array(dm), np.array(cm)
        out[f"E0={E0:g}"] = {"sym_residual_max": float(np.nanmax(sm)), "sym_residual_n_nan": int(np.isnan(sm).sum()),
                             "div_max": float(np.nanmax(dm)), "div_n_nan": int(np.isnan(dm).sum()),
                             "courant_max_every_step": float(np.nanmax(cm)), "courant_n_nan": int(np.isnan(cm).sum()),
                             "E_final": g.energy(u).cpu().numpy().tolist()}
    return out


# ------------------------------------------------------------------------------------------------------------ pilot
def pilot(cfg, run, Re, log_):
    P = cfg["pilot"]
    PR = cfg["pilot_runtime"]
    dt = float(cfg["_dt"])
    N = cfg["grids"][0]
    g = make_grid(cfg, N, Re)
    seed = int(cfg["seed"])
    T = float(P["T_pilot"])
    climit = float(cfg["revised_5"]["5b"]["courant_limit"])
    lv = int(cfg["revised_5"]["5b"]["max_rerun_levels"])
    E0s = np.geomspace(float(P["E_range"][0]), float(P["E_range"][1]), int(P["n_amp"]))
    n_amp = len(E0s)
    tag = f"Re{Re:g}"
    st_path = run / f"pilot_{tag}_state.json"
    st = read_json(st_path, {}) or {}
    # 1% timing (e4, e18): one chunk of trajectories for 1% of T_pilot
    if "per_case_step_s" not in st:
        t_c = timing_per_case_step(g, dt, n_cases=int(PR["chunk_dirs"]) * n_amp)
        per_dir = n_amp * (T / dt) * t_c
        n_dirs = int(max(int(P["n_dirs_min"]), min(64, math.floor(30 * 60 / per_dir))))
        st.update({"per_case_step_s": t_c, "time_per_direction_s_no_early_stop": per_dir, "n_dirs_rule": n_dirs,
                   "estimate_min_no_early_stop": n_dirs * per_dir / 60})
        atomic_write_json(st_path, st)
        log_(f"  pilot {tag}: 1% timing {t_c * 1e3:.3f} ms/case-step; {per_dir:.0f} s per direction; n_dirs {n_dirs}; "
             f"estimate {n_dirs * per_dir / 60:.1f} min (no early-stop saving)")
    n_dirs = int(st["n_dirs_rule"])
    rng = np.random.default_rng(derive_seed(seed, "pilot", Re))
    dirs = rng.standard_normal((n_dirs, 9))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    cd = int(PR["chunk_dirs"])
    chunks = [list(range(i, min(n_dirs, i + cd))) for i in range(0, n_dirs, cd)]
    st.setdefault("chunks_done", [])
    st.setdefault("chunk_seconds", [])
    cap = float(PR["cap_min"]) * 60
    for ci, ch in enumerate(chunks):
        f = run / f"pilot_{tag}_chunk{ci}.npz"
        if ci in st["chunks_done"] and f.exists():
            continue
        el = float(np.sum(st["chunk_seconds"]))
        mean_c = float(np.mean(st["chunk_seconds"])) if st["chunk_seconds"] else (len(ch) * st["time_per_direction_s_no_early_stop"])
        if el + mean_c > cap:
            st["dropped_chunks"] = list(range(ci, len(chunks)))
            log_(f"  pilot {tag}: cap guard: elapsed {el / 60:.1f} min + chunk {mean_c / 60:.1f} min > {cap / 60:.0f} min; chunks {ci}.. dropped")
            break
        t0 = time.time()
        A = (np.sqrt(E0s)[None, :, None] * dirs[ch][:, None, :]).reshape(-1, 9)
        u0 = g.clean(g.from_coeffs(A))
        r = run_det(g, u0, dt, T, None, None, absorb=False, E_dead=float(PR["E_dead"]), courant_limit=climit, max_levels=lv,
                    progress=True, desc=f"pilot {tag} chunk {ci}", log=log_)
        atomic_save_npz(f, E=r["E"], fate=r["fate"], t_hit=r["t_hit"], courant_max=r["courant_max"], level=r["level"],
                        nan_final=r["nan_final"], dirs_idx=np.repeat(ch, n_amp), amp_idx=np.tile(np.arange(n_amp), len(ch)),
                        counts=np.array(json.dumps(r["counts"])))
        st["chunks_done"].append(ci)
        st["chunk_seconds"].append(time.time() - t0)
        atomic_write_json(st_path, st)
        log_(f"  pilot {tag} chunk {ci}: {time.time() - t0:.0f} s; re-runs dt/2 {r['counts']['n_rerun_dt_half']}, dt/4 "
             f"{r['counts']['n_rerun_dt_quarter']}; NaN at level 0 {r['counts']['nan_per_level'][0]}")
    done = sorted(st["chunks_done"])
    Z = [np.load(run / f"pilot_{tag}_chunk{ci}.npz") for ci in done]
    E = np.concatenate([z["E"] for z in Z])
    fate = np.concatenate([z["fate"] for z in Z])
    t_hit = np.concatenate([z["t_hit"] for z in Z])
    nanf = np.concatenate([z["nan_final"] for z in Z])
    di = np.concatenate([z["dirs_idx"] for z in Z])
    ai = np.concatenate([z["amp_idx"] for z in Z])
    counts = {}
    for z in Z:
        counts = merge_counts(counts, json.loads(str(z["counts"])))
    used_dirs = sorted(set(di.tolist()))
    out = {"Re": Re, "grid": N, "dt": dt, "n_dirs_rule": n_dirs, "n_dirs_used": len(used_dirs), "dropped_chunks": st.get("dropped_chunks", []),
           "pilot_minutes": float(np.sum(st["chunk_seconds"])) / 60, "timing": {k: st[k] for k in ("per_case_step_s", "time_per_direction_s_no_early_stop",
                                                                                                 "estimate_min_no_early_stop")},
           "E0": E0s.tolist(), "rerun_counts": counts}
    # histogram (e9): all recorded samples; samples after E_dead stop -> laminar bin; NaN samples excluded and counted
    H = P["histogram"]
    n_rec = E.shape[1]
    dead = fate == 2
    n_dead_samples = 0
    for j in np.where(dead)[0]:
        k = int(round(t_hit[j]))
        n_dead_samples += n_rec - 1 - k
    vals = E[np.isfinite(E)]                                                   # +-inf / NaN never enter the histogram
    n_nan_samples = int(np.sum(~np.isfinite(E))) - n_dead_samples        # NaN-final tails included (excluded from the histogram)
    edges = np.linspace(float(H["E_range"][0]), float(H["E_range"][1]), int(H["bins"]) + 1)
    cnt, _ = np.histogram(vals, bins=edges)
    n_out_range = int(np.sum((vals < edges[0]) | (vals > edges[-1])))
    cnt = cnt.astype(np.float64)
    cnt[0] += n_dead_samples
    total = cnt.sum()
    bw = edges[1] - edges[0]
    dens = cnt / (total * bw)
    sm = np.convolve(dens, np.ones(int(H["smooth_bins"])) / int(H["smooth_bins"]), mode="same")
    cen = 0.5 * (edges[:-1] + edges[1:])
    lam_i = int(np.argmax(np.where(cen < float(H["laminar_mode_E_max"]), sm, -np.inf)))
    start = float(H["laminar_mode_E_max"])
    tur_i = int(np.argmax(np.where((cen > start) & (cen > cen[lam_i]), sm, -np.inf)))
    val_i = lam_i + int(np.argmin(sm[lam_i:tur_i + 1]))                       # as the Gate 2 pre-study (endpoints included)
    E_val = float(cen[val_i])
    frac_above = float((cnt[val_i + 1:].sum()) / total)
    ratio = 0.0 if sm[tur_i] <= 0 else (float(sm[tur_i] / sm[val_i]) if sm[val_i] > 0 else float("inf"))
    clear = bool(ratio >= float(H["clear_ratio"]) and frac_above >= float(H["turbulent_mass_min"]))
    E_B = float(f"{E_val:.2g}")
    out["histogram"] = {"laminar_mode_E": float(cen[lam_i]), "turbulent_mode_E": float(cen[tur_i]), "valley_E": E_val,
                        "turbulent_over_valley": ratio, "frac_samples_above_valley": frac_above, "clear_valley": clear, "E_B": E_B,
                        "n_samples": float(total), "n_dead_samples_in_laminar_bin": int(n_dead_samples), "n_nan_samples_excluded": int(n_nan_samples),
                        "n_samples_out_of_range": n_out_range}
    atomic_save_npz(run / f"pilot_{tag}_hist.npz", edges=edges, counts=cnt, smooth=sm)
    if not clear:
        out["decision"] = "no clear valley"
        return out
    # E_c per direction, r_A (e10)
    reachB = np.array([np.nanmax(E[j]) >= E_B if np.isfinite(E[j]).any() else False for j in range(len(E))]) | nanf
    Ec = {}
    for d_ in used_dirs:
        m = (di == d_) & reachB
        Ec[d_] = float(E0s[ai[m].min()]) if m.any() else None
    fin = [v for v in Ec.values() if v is not None]
    out["E_c_per_direction"] = {str(k): v for k, v in Ec.items()}
    out["n_dirs_never_B"] = int(sum(v is None for v in Ec.values()))
    if not fin:
        out["decision"] = "no pilot trajectory reaches B"
        return out
    minEc = min(fin)
    cands = sorted([m_ * 10.0 ** k for k in range(-12, 1) for m_ in P["r_A_candidates_mantissa"] if m_ * 10.0 ** k <= minEc / 10], reverse=True)
    # rebound check
    rngb = np.random.default_rng(derive_seed(seed, "rebound", Re))
    extra = rngb.standard_normal((int(P["n_rebound_random"]), 9))
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    rb_dirs = np.concatenate([dirs[used_dirs], extra])
    reb = []
    r_A = None
    for c_ in cands:
        f = run / f"pilot_{tag}_rebound_{c_:g}.npz"
        if f.exists():
            zr = np.load(f)
            fr, cr = zr["fate"], json.loads(str(zr["counts"]))
        else:
            u0 = g.clean(g.from_coeffs(math.sqrt(c_) * rb_dirs))
            rr = run_det(g, u0, dt, T, None, E_B, absorb=True, E_dead=float(PR["E_dead"]), courant_limit=climit, max_levels=lv,
                         progress=True, desc=f"rebound {c_:g}", log=log_)
            fr, cr = rr["fate"], rr["counts"]
            atomic_save_npz(f, fate=fr, t_hit=rr["t_hit"], counts=np.array(json.dumps(cr)))
        nB = int(np.sum(fr == 1))
        reb.append({"r_A": c_, "n_runs": int(len(fr)), "n_reach_B": nB, "n_undecided_at_T_not_dead": int(np.sum(fr == -1)), "rerun_counts": cr})
        log_(f"  pilot {tag}: rebound check r_A = {c_:g}: {nB} of {len(fr)} reach B")
        if nB == 0:
            r_A = c_
            break
    out["min_E_c"] = minEc
    out["rebound_checks"] = reb
    out["r_A"] = r_A
    if r_A is None:
        out["decision"] = "no r_A candidate with zero rebound"
        return out
    # T_det (e10): decision time of each pilot trajectory from its recorded E
    tdec = np.full(len(E), np.inf)
    for j in range(len(E)):
        e = E[j]
        hit = np.where((e < r_A) | (e >= E_B))[0]
        if len(hit):
            tdec[j] = float(hit[0])
        elif nanf[j]:
            tdec[j] = float(t_hit[j])
    steps = np.arange(float(P["T_det_step"]), T + 1e-9, float(P["T_det_step"]))
    fr_und = [float(np.mean(tdec > s_)) for s_ in steps]
    ok = [s_ for s_, f_ in zip(steps, fr_und) if f_ < float(P["undecided_max"])]
    T_det = float(ok[0]) if ok else T
    out["undecided_fraction_by_T"] = {f"{s_:g}": f_ for s_, f_ in zip(steps, fr_und)}
    out["T_det"] = T_det
    out["T_det_undecided_fraction"] = float(np.mean(tdec > T_det))
    out["decision"] = "ok"
    return out


# ------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cfg5", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sc-grid", default=None, help="sc-only mode: e.g. 64x32x32")
    ap.add_argument("--sc-dt", default=None)
    args = ap.parse_args()
    sc_only = args.sc_grid is not None
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg5 = yaml.safe_load(Path(args.cfg5).read_text())
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    if not (run / "config_used.yaml").exists():
        (run / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    man_path = run / ("manifest.json" if not sc_only else f"manifest_sc_{args.sc_grid}_dt{args.sc_dt}.json")
    man = read_json(man_path, {"invocations": []})
    man["invocations"].append({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv, "git": git_info(REPO),
                               "python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})
    atomic_write_json(man_path, man)
    res = read_json(run / "results.json", {}) or {}

    def save():
        if sc_only:                                    # concurrent sc-only processes never write results.json
            return
        atomic_write_json(run / "results.json", res)

    R5 = cfg["revised_5"]
    ladder_dt = [float(x) for x in R5["dt_ladder"]]
    climit = float(R5["5b"]["courant_limit"])
    lv = int(R5["5b"]["max_rerun_levels"])
    grids = cfg["grids"]
    res.setdefault("log", [])

    def note(msg):
        log(msg)
        res["log"].append(time.strftime("%H:%M:%S ") + msg)
        save()

    # ---- 5a at 48x24x24 on the dt ladder (only until the first rung passes; that rung is pinned as base_dt for the pilot
    #      and the resolution ladder, e16; later rungs are handled in the 5c/5a rung loop below)
    di = int(res["base_dt_index"]) if "base_dt_index" in res else 0
    while True:
        if di >= len(ladder_dt):
            res["stopped"] = "5a: dt ladder used up"
            save()
            note("STOP: 5a dt ladder used up")
            return
        dt = ladder_dt[di]
        key = f"5a_{nkey(grids[0])}_dt{dt:g}"
        if key not in res:
            note(f"--- {key}")
            res[key] = check_5a(cfg, make_grid(cfg, grids[0], float(cfg["system"]["Re"])), dt)
            save()
            note(f"{key}: max rel dE {res[key]['max_rel_dE']} (tol {R5['5a']['tol']}), NaN {res[key]['n_nan_samples']}, pass {res[key]['pass']}")
        if res[key]["pass"]:
            break
        di += 1
        res["dt_index"] = di
        save()
    res.setdefault("base_dt_index", di)
    res.setdefault("dt_index", di)
    save()
    cfg["_dt"] = ladder_dt[int(res["base_dt_index"])]
    if "item1_stable_dt" not in res:
        res["item1_stable_dt"] = item1_stable(cfg, make_grid(cfg, grids[0], float(cfg["system"]["Re"])), cfg["_dt"])
        save()
        note(f"item-1 measures on stable runs (descriptive): {json.dumps({k: v for k, v in res['item1_stable_dt'].items() if k != 'dt'})}")
    # ---- pilot with the Re ladder
    Res = [float(cfg["system"]["Re"])] + [float(x) for x in cfg["re_fallback"]]
    chosen = None
    for Re in Res:
        key = f"pilot_Re{Re:g}_dt{cfg['_dt']:g}"
        if key not in res:
            note(f"--- {key}")
            res[key] = pilot(cfg, run, Re, note)
            save()
            h = res[key]["histogram"]
            note(f"{key}: decision {res[key]['decision']}; valley {h['valley_E']:.3g} (turb/valley {h['turbulent_over_valley']:.2f}, "
                 f"above {h['frac_samples_above_valley']:.3f}), E_B {h['E_B']}, r_A {res[key].get('r_A')}, T_det {res[key].get('T_det')}; "
                 f"{res[key]['pilot_minutes']:.1f} min, dirs {res[key]['n_dirs_used']}")
        if res[key]["decision"] == "ok":
            chosen = res[key]
            break
    if chosen is None:
        res["stopped"] = "pilot: Re ladder used up"
        save()
        note("STOP: pilot Re ladder used up")
        return
    Re, r_A, E_B, T_det = chosen["Re"], chosen["r_A"], chosen["histogram"]["E_B"], chosen["T_det"]
    if cfg.get("T_det_override") is not None:                                 # user decision ~10:55 (post-hoc)
        res["T_det_pilot_rule"] = T_det
        T_det = float(cfg["T_det_override"])
    res["sets"] = {"Re": Re, "r_A": r_A, "E_B": E_B, "T_det": T_det}
    save()
    # ---- resolution ladder at the current dt (e12, e20)
    rays = build_rays(cfg5, REPO)
    LD, names = ladder_rays(rays)
    SC = cfg["s_c"]

    def sc_at(N, dt_, tag):
        key = f"sc_{nkey(N)}_dt{dt_:g}"
        if key in res:
            return np.array([np.nan if v is None else v for v in res[key]["s_c"]])
        fj = run / f"{key}.json"                                                # written by a concurrent --sc-grid process
        if fj.exists():
            res[key] = json.loads(fj.read_text())
            save()
            note(f"    {key} read from {fj.name} (concurrent process): {[None if v is None else round(v, 6) for v in res[key]['s_c']]}")
            return np.array([np.nan if v is None else v for v in res[key]["s_c"]])
        g = make_grid(cfg, N, Re)
        note(f"--- s_c on the 6 check rays at {nkey(N)}, dt {dt_:g} ({tag})")
        t_c = timing_per_case_step(g, dt_, n_cases=12)
        ub = len(LD) * (int(SC["n_coarse"]) + 12) * (T_det / dt_) * t_c / 60
        note(f"    1% timing: {t_c * 1e3:.3f} ms/case-step (batch 12); upper bound {ub:.0f} min (every run to T_det)")
        r = find_sc_det(g, LD, r_A, E_B, T_det, dt_, int(SC["n_coarse"]), float(SC["rel_tol"]), courant_limit=climit, max_levels=lv,
                        ckpt=run / f"ckpt_{key}.npz", log=note)
        res[key] = {"s_c": [None if not np.isfinite(v) else float(v) for v in r["s_c"]], "names": names, "no_crossing": r["no_crossing"].tolist(),
                    "below_grid": r["below_grid"].tolist(), "n_switches": r["n_switches"].tolist(), "n_runs": r["n_runs"],
                    "n_undecided_runs": r["n_undecided_runs"], "minutes": r["seconds"] / 60, "rerun_counts": r["counts"],
                    "per_case_step_ms": t_c * 1e3, "T_det": T_det, "out_hi_fate": [int(x) for x in r["out_hi"]],
                    "n_rays_edge_set_by_undecided": int(np.sum(r["out_hi"] == -1)), "rule": r["rule"],
                    "flag_undecided": [bool(x) for x in r["flag_undecided"]], "n_rays_flagged_undecided": int(np.sum(r["flag_undecided"]))}
        save()
        if sc_only:
            atomic_write_json(run / f"{key}.json", res[key])
        note(f"    s_c {nkey(N)} dt {dt_:g}: {[None if v is None else round(v, 6) for v in res[key]['s_c']]} ({r['seconds'] / 60:.1f} min)")
        return r["s_c"]

    def status(key, i):
        r = res[key]
        if r.get("flag_undecided", [False] * 6)[i]:
            return "undecided"                                                  # never counted as agreement (+inf)
        return "no_crossing" if r["no_crossing"][i] else ("below_grid" if r["below_grid"][i] else "ok")

    if sc_only:
        sc_at([int(x) for x in args.sc_grid.split("x")], float(args.sc_dt), "concurrent")
        note(f"sc-only {args.sc_grid} dt {args.sc_dt}: done")
        return

    def med_dln(a, b, ka=None, kb=None):
        """e20 (amended before any real run): both resolved -> |d ln s_c|; unresolved at both with the same status -> 0
        (agreement); otherwise +inf."""
        d = np.abs(np.log(b) - np.log(a))
        for i in range(len(d)):
            if not np.isfinite(d[i]):
                same = ka is not None and status(ka, i) == status(kb, i) and status(ka, i) not in ("ok", "undecided")
                d[i] = 0.0 if same else np.inf
        return float(np.median(d)), d

    tol_res = float(cfg["convergence"]["tol"])
    dt = cfg["_dt"]
    chosen_N = None
    cap_lad = float(cfg["convergence"]["time_cap_min"])
    for k in range(len(grids) - 1):
        spent = sum(res[k_]["minutes"] for k_ in res if k_.startswith("sc_") and k_.endswith(f"_dt{dt:g}") and isinstance(res[k_], dict) and "minutes" in res[k_])
        if spent > cap_lad:
            note(f"    ladder time so far {spent:.0f} min > cap {cap_lad:.0f} min: reported, the ladder continues (e22)")
        a, b = sc_at(grids[k], dt, "ladder"), sc_at(grids[k + 1], dt, "ladder")
        m, d = med_dln(a, b, f"sc_{nkey(grids[k])}_dt{dt:g}", f"sc_{nkey(grids[k + 1])}_dt{dt:g}")
        key = f"ladder_{nkey(grids[k])}_vs_{nkey(grids[k + 1])}_dt{dt:g}"
        res[key] = {"median_abs_dln_sc": m, "per_ray": [None if not np.isfinite(x) else float(x) for x in d], "tol": tol_res, "pass": bool(m <= tol_res)}
        save()
        note(f"{key}: median |d ln s_c| {m:.4f} (tol {tol_res}) -> {'use ' + nkey(grids[k]) if m <= tol_res else 'next rung'}")
        if m <= tol_res:
            chosen_N = grids[k]
            break
    if chosen_N is None:
        res["stopped"] = "resolution ladder used up"
        save()
        note("STOP: resolution ladder used up")
        return
    # ---- 5c and 5a at the current dt rung (e15, e16): 5c at the chosen resolution, 5a at 48x24x24 and at the chosen
    #      resolution (if finer), at the chosen Re; any failure -> next rung; stop only when the ladder is used up.
    di = int(res.get("dt_index", res["base_dt_index"]))
    while True:
        if di >= len(ladder_dt):
            res["stopped"] = "5a/5c: dt ladder used up"
            save()
            note("STOP: dt ladder used up (5a/5c)")
            return
        dt = ladder_dt[di]
        cfg["_dt"] = dt
        key = f"5c_{nkey(chosen_N)}_dt{dt:g}"
        a, b = sc_at(chosen_N, dt, "5c"), sc_at(chosen_N, dt / 2, "5c dt/2")
        m, d = med_dln(a, b, f"sc_{nkey(chosen_N)}_dt{dt:g}", f"sc_{nkey(chosen_N)}_dt{dt / 2:g}")
        res[key] = {"median_abs_dln_sc": m, "per_ray": [None if not np.isfinite(x) else float(x) for x in d], "tol": float(R5["5c"]["tol"]),
                    "pass": bool(m <= float(R5["5c"]["tol"]))}
        save()
        note(f"{key}: median |d ln s_c| {m:.4f} (tol {R5['5c']['tol']}) -> {'pass' if res[key]['pass'] else 'next dt rung'}")
        ok = res[key]["pass"]
        if ok:
            for N5 in [grids[0]] + ([chosen_N] if nkey(chosen_N) != nkey(grids[0]) else []):
                k5 = f"5a_{nkey(N5)}_dt{dt:g}" + ("" if Re == float(cfg["system"]["Re"]) else f"_Re{Re:g}")
                if k5 not in res:
                    res[k5] = check_5a(cfg, make_grid(cfg, N5, Re), dt)
                    save()
                    note(f"{k5}: max rel dE {res[k5]['max_rel_dE']}, pass {res[k5]['pass']}")
                ok = ok and res[k5]["pass"]
        if ok:
            break
        di += 1
        res["dt_index"] = di
        save()
    if cfg.get("zero_undecided_check"):
        ks = [f"sc_{nkey(chosen_N)}_dt{cfg['_dt']:g}", f"sc_{nkey(chosen_N)}_dt{cfg['_dt'] / 2:g}"]
        und = {k_: int(res[k_]["n_undecided_runs"]) for k_ in ks}
        res["zero_undecided_check"] = {"searches": und, "pass": bool(all(v == 0 for v in und.values()))}
        save()
        note(f"zero-undecided check at the chosen resolution: {und} -> {'pass' if res['zero_undecided_check']['pass'] else 'FAIL'}")
        if not res["zero_undecided_check"]["pass"]:
            res["stopped"] = "zero-undecided check failed at T_det = 8000 (no further T_det rung)"
            save()
            note("STOP: zero-undecided check failed")
            return
    res["choice"] = {"Re": Re, "grid": chosen_N, "dt": cfg["_dt"], "r_A": r_A, "E_B": E_B, "T_det": T_det}
    save()
    atomic_write_json(run / "choice.json", res["choice"])
    note(f"4b DONE: {res['choice']}")


if __name__ == "__main__":
    main()
