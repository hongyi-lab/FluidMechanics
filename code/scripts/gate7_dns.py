#!/usr/bin/env python
"""Gate 7 (DNS): is the fast channel a numerical artefact or a real route? Config: code/configs/gate7_dns.yaml (criteria fixed
before any run). Stages (each resumable, independent stages may run as separate processes, h14):

    python code/scripts/gate7_dns.py --config code/configs/gate7_dns.yaml --run-dir <dir> --stage timing
    ... --stage truth --level 96 --rays W#3,W#7,W#9,R#22,R#15,W#11
    ... --stage truth --level 128 --rays W#1
    ... --stage theory --level 64|96|128
    ... --stage diag --level 64|96|128 --rays ...
    ... --stage scan --level 64|96
    ... --stage analysis
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

from committor.dns import Grid, _kmax_t, step_nl_c
from committor.dns_local import PARTS, find_sc_local, run_diag
from committor.dns_rays import build_rays
from committor.dns_theory import A_of, run_delta_q
from committor.dns import run_det
from committor.threshold import RayCoordinate
from committor.utils import atomic_save_npz, atomic_write_json, git_info, read_json

REPO = Path(__file__).resolve().parents[2]
SHARED = ["mirror y about 1 (u,-v,w)(x,2-y,z)", "shift-reflect (u,v,-w)(x+Lx/2,y,-z)", "shift-rotate (-u,-v,w)(-x,-y,z+Lz/2)"]
LADDER_NAMES = ["F eps=1e-06 draw 0", "F eps=0.01 draw 0", "F eps=1 draw 0", "R #1", "R #13", "W #1"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def make_grid(cfg4, N, Re):
    S = cfg4["system"]
    g = Grid(*N, Re=Re, Lx=S["Lx_over_pi"] * math.pi, Lz=S["Lz_over_pi"] * math.pi, Ly=S["Ly_ext"], device=cfg4["device"])
    g.set_enforced(SHARED)
    return g


class Ctx:
    def __init__(self, cfg):
        self.cfg = cfg
        D = cfg["data"]
        self.cfg5 = yaml.safe_load((REPO / D["gate5_config"]).read_text())
        self.cfg4 = yaml.safe_load((REPO / D["gate4_config"]).read_text())
        self.g5 = REPO / D["gate5_run"]
        self.g4b = REPO / D["gate4b_run"]
        ch = json.loads((self.g5 / "gate4b_choice_used.json").read_text())
        self.Re, self.r_A, self.E_B, self.T_det = float(ch["Re"]), float(ch["r_A"]), float(ch["E_B"]), float(ch["T_det"])
        assert ch["grid"] == cfg["levels"]["64"]["grid"] and float(ch["dt"]) == cfg["levels"]["64"]["dt"]
        R5 = self.cfg4["revised_5"]["5b"]
        self.climit, self.lv = float(R5["courant_limit"]), int(R5["max_rerun_levels"])
        rays = build_rays(self.cfg5, REPO)
        self.R_d, self.W_d = rays["R"][0], rays["W"][0]
        self.nF = len(rays["F"][0])
        # Gate 5 truth (64)
        TF = np.load(self.g5 / "truth_RW.npz")
        self.t5 = {k: TF[k] for k in TF.files}
        # Gate 4b ladder at 96 (W#1, R#13)
        self.lad96 = np.load(self.g4b / "ckpt_sc_96x48x48_dt0.0625.npz")
        assert json.loads((self.g4b / "sc_96x48x48_dt0.0625.json").read_text())["names"] == LADDER_NAMES

    def dir(self, name):
        s, k = name.split("#")
        return (self.R_d if s == "R" else self.W_d)[int(k) - 1]

    def idx5(self, name):
        s, k = name.split("#")
        return int(k) - 1 if s == "R" else len(self.R_d) + int(k) - 1

    def grid(self, level):
        L = self.cfg["levels"][str(level)]
        return make_grid(self.cfg4, L["grid"], self.Re), float(L["dt"])

    def all_rays(self):
        R = self.cfg["rays"]
        return R["decision"] + R["descriptive"] + R["controls"]


def truth_file(run, level, rays):
    return run / f"truth_{level}_{'_'.join(r.replace('#', '') for r in rays)}.json"


def load_truths(run):
    """level -> ray name -> ray record (Gate 7 s_c searches)."""
    out = {}
    for f in sorted(run.glob("truth_*.json")):
        st = json.loads(f.read_text())
        if st.get("phase") != "done":
            continue
        lv = f.name.split("_")[1]
        for r in st["rays"]:
            out.setdefault(lv, {})[r["name"]] = dict(r, file=f.name)
    return out


def sc_at(ctx, run, level, name):
    """(s_c, lo, hi, t_hit_lo, t_hit_hi, source) at a level, or None."""
    level = str(level)
    if level == "64":
        i = ctx.idx5(name)
        t = ctx.t5
        if not np.isfinite(t["s_c"][i]):
            return None
        return float(t["s_c"][i]), float(t["lo"][i]), float(t["hi"][i]), float(t["t_hit_lo"][i]), float(t["t_hit_hi"][i]), "Gate 5 truth (64)"
    if level == "96" and name in ("W#1", "R#13"):
        j = LADDER_NAMES.index(name.replace("#", " #"))
        L = ctx.lad96
        lo, hi = float(L["lo"][j]), float(L["hi"][j])
        return 0.5 * (lo + hi), lo, hi, float(L["t_hit_lo"][j]), float(L["t_hit_hi"][j]), "Gate 4b ladder (96)"
    T = load_truths(run).get(level, {})
    if name not in T or T[name]["s_c"] is None:
        return None
    r = T[name]
    return r["s_c"], r["lo"], r["hi"], r["t_hit_lo"], r["t_hit_hi"], f"Gate 7 ({r['file']})"


def prev_level(level):
    return {"96": "64", "128": "96"}[str(level)]


def per_case_step(g, dt, n, steps=20):
    rng = np.random.default_rng(0)
    d = rng.standard_normal((n, 9))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    u = g.clean(g.from_coeffs(0.05 * d))
    E1, E2 = g.lap_factor(dt), g.lap_factor(0.5 * dt)
    km = _kmax_t(g)
    for _ in range(3):
        u, _c = step_nl_c(g, u, dt, E1, E2, km)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(steps):
        u, _c = step_nl_c(g, u, dt, E1, E2, km)
    torch.cuda.synchronize()
    return (time.time() - t0) / steps, torch.cuda.max_memory_allocated() / 2 ** 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--level", default=None)
    ap.add_argument("--rays", default=None)
    ap.add_argument("--max-batch", type=int, default=64)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    if not (run / "config_used.yaml").exists():
        (run / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    mf = run / f"manifest_{args.stage}{'_' + args.level if args.level else ''}{'_' + args.rays.replace('#', '').replace(',', '_') if args.rays else ''}.json"
    man = read_json(mf, {"invocations": []})
    man["invocations"].append({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv, "git": git_info(REPO),
                               "python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})
    atomic_write_json(mf, man)
    ctx = Ctx(cfg)
    rays = args.rays.split(",") if args.rays else None
    t0 = time.time()
    {"timing": st_timing, "truth": st_truth, "theory": st_theory, "diag": st_diag, "scan": st_scan, "analysis": st_analysis}[args.stage](
        ctx, run, args.level, rays, args)
    log(f"--- {args.stage} {args.level or ''} {args.rays or ''}: {(time.time() - t0) / 60:.1f} min")


# ---------------------------------------------------------------------------------------------------------------- 1% timing
def st_timing(ctx, run, level, rays, args):
    out = {"per_step_s": {}, "peak_GiB": {}}
    for lv in ("64", "96", "128"):
        g, dt = ctx.grid(lv)
        for n in (1, 6, 12, 36):
            s, mem = per_case_step(g, dt, n)
            out["per_step_s"][f"{lv} batch {n}"] = s
            out["peak_GiB"][f"{lv} batch {n}"] = mem
            log(f"    {lv} batch {n}: {s * 1e3:.1f} ms/step ({s / n * 1e3:.2f} ms/case-step), peak {mem:.2f} GiB")
        del g
        torch.cuda.empty_cache()
    # typical round length: the longest near-edge decision time at 64 over the Gate 7 rays (x 1.3); upper bound T_det
    names = ctx.all_rays()
    tl = [ctx.t5["t_hit_lo"][ctx.idx5(n)] for n in names]
    th = [ctx.t5["t_hit_hi"][ctx.idx5(n)] for n in names]
    t_round = 1.3 * float(np.nanmax(tl + th))
    out["t_round_typical"] = t_round
    n_round = 1 + 6                                     # local grid + bisection from a x1.43 bracket to 1% (6 halvings)
    out["n_rounds"] = n_round

    def job_h(lv, n_rays, t_len):
        ps = out["per_step_s"]
        dt = ctx.cfg["levels"][lv]["dt"]
        b_grid = min(36, 6 * n_rays)
        key = min((1, 6, 12, 36), key=lambda b: abs(b - b_grid))
        grid_h = ps[f"{lv} batch {key}"] * (t_len / dt) / 3600          # grid round at full batch for its length (pessimistic)
        keyb = min((1, 6, 12, 36), key=lambda b: abs(b - n_rays))
        bis_h = (n_round - 1) * ps[f"{lv} batch {keyb}"] * (t_len / dt) / 3600
        return grid_h + bis_h
    est = {"96 (6 rays)": job_h("96", 6, t_round), "128 W#1": job_h("128", 1, t_round), "128 (5 rays)": job_h("128", 5, t_round),
           "128 (2 rays: R#22, W#11)": job_h("128", 2, t_round)}
    ub = {"96 (6 rays)": job_h("96", 6, ctx.T_det), "128 W#1": job_h("128", 1, ctx.T_det), "128 (5 rays)": job_h("128", 5, ctx.T_det)}
    out["estimate_h"], out["upper_bound_h"] = est, ub
    # critical path: max(96 then 128 main, 128 W#1); concurrent processes share the GPU (sum of the work as the pessimistic case)
    out["critical_path_full_h"] = max(est["96 (6 rays)"] + est["128 (5 rays)"], est["128 W#1"])
    out["sum_of_work_full_h"] = est["96 (6 rays)"] + est["128 (5 rays)"] + est["128 W#1"]
    out["sum_of_work_reduced_h"] = est["96 (6 rays)"] + est["128 (2 rays: R#22, W#11)"] + est["128 W#1"]
    out["budget_h"] = float(ctx.cfg["budget_hours"])
    out["decision"] = "full 128 set" if out["sum_of_work_full_h"] <= out["budget_h"] else "reduced 128 set (W#1, R#22, W#11)"
    for k, v in est.items():
        log(f"    estimate {k}: {v:.1f} h (upper bound {ub.get(k, float('nan')):.1f} h)")
    log(f"    sum of work: full {out['sum_of_work_full_h']:.1f} h, reduced {out['sum_of_work_reduced_h']:.1f} h; budget {out['budget_h']} h -> {out['decision']}")
    atomic_write_json(run / "timing.json", out)


# ------------------------------------------------------------------------------------------------------------------- 7a
def _skip_flagged(ctx, run, level, rays, stage):
    """review fix (2026-11-18 ~10:40): a ray without s_c at the level needed is skipped and recorded, not asserted, so that one
    flagged ray does not stop the other rays of the chain (h4/h6: flag and report)."""
    ok, skipped = [], []
    for n in rays:
        if sc_at(ctx, run, level, n) is None:
            st = load_truths(run).get(str(level), {}).get(n, {}).get("status", "no s_c / search not finished")
            skipped.append({"ray": n, "level": str(level), "reason": st})
            log(f"    {stage}: skip {n}: no s_c at {level} ({st})")
        else:
            ok.append(n)
    if skipped:
        f = run / f"skipped_{stage}.json"
        atomic_write_json(f, read_json(f, []) + skipped)
    return ok


def st_truth(ctx, run, level, rays, args):
    L = ctx.cfg["local_grid"]
    rays = _skip_flagged(ctx, run, prev_level(level), rays, f"truth_{level}")
    if not rays:
        return
    s_prev = [sc_at(ctx, run, prev_level(level), n)[0] for n in rays]
    g, dt = ctx.grid(level)
    log(f"--- truth at {level} (dt {dt:g}) for {rays}; s_prev {np.round(s_prev, 6).tolist()}")
    find_sc_local(g, np.array([ctx.dir(n) for n in rays]), rays, s_prev, L["factors"], L["extend_up"], L["extend_down"], ctx.r_A, ctx.E_B,
                  ctx.T_det, dt, float(L["rel_width"]), ctx.climit, ctx.lv, args.max_batch, ckpt=truth_file(run, level, rays), log=log,
                  desc=f"s_c {level}", progress=True)


# ------------------------------------------------------------------------------------------------------------------- 7b
def st_theory(ctx, run, level, rays, args):
    names = rays or ctx.all_rays()
    TH = ctx.cfg["theory"]
    f = run / f"theory_{level}.npz"
    if f.exists() and [str(x) for x in np.load(f)["names"]] == list(names) and (str(level) != "64" or (run / "theory_64_vs_gate5.json").exists()):
        log(f"    theory {level}: done earlier")
        return
    g, dt = ctx.grid(level)
    r = run_delta_q(g, np.array([ctx.dir(n) for n in names]), float(TH["T_S"]), dt, every=float(TH["output_every"]), progress=True,
                    desc=f"theory {level}")
    atomic_save_npz(run / f"theory_{level}.npz", aa=r["aa"], ab=r["ab"], bb=r["bb"], t=r["t"], names=np.array(names))
    if str(level) == "64":
        Z = np.load(ctx.g5 / "theory.npz")
        n = r["aa"].shape[1]
        rows = [ctx.nF + ctx.idx5(x) for x in names]
        rel = {k: float(np.max(np.abs(r[k] - Z[k][rows, :n]) / np.maximum(np.abs(Z[k][rows, :n]).max(axis=1, keepdims=True), 1e-300)))
               for k in ("aa", "ab", "bb")}
        atomic_write_json(run / "theory_64_vs_gate5.json", rel)
        log(f"    64 vs Gate 5 theory.npz: max rel diff {rel}")


# ------------------------------------------------------------------------------------------------------------------- 7c
def st_diag(ctx, run, level, rays, args):
    D = ctx.cfg["diag"]
    rays = _skip_flagged(ctx, run, level, rays, f"diag_{level}")
    if not rays:
        return
    tag = "_".join(x.replace("#", "") for x in rays)
    if (run / f"diag_{level}_{tag}.npz").exists():
        log(f"    diag {level} {tag}: done earlier")
        return
    g, dt = ctx.grid(level)
    names, amps, which = [], [], []
    for n in rays:
        p = sc_at(ctx, run, level, n)
        for tag_, s in (("lo", p[1]), ("hi", p[2])):
            names.append(n)
            which.append(tag_)
            amps.append(s * ctx.dir(n))
    log(f"--- diag at {level} for {rays} ({len(amps)} runs to t = {D['T']})")
    r = run_diag(g, np.array(amps), dt, float(D["T"]), float(D["every"]), float(D["high_band_fraction"]), ctx.climit, ctx.lv, log=log)
    atomic_save_npz(run / f"diag_{level}_{tag}.npz", rec=r["rec"], t=r["t"], dt_used=r["dt_used"], courant_max=r["courant_max"],
                    flag_last=r["flag_last"], names=np.array(names), which=np.array(which), s=np.array([np.linalg.norm(a) for a in amps]),
                    parts=np.array(PARTS))
    log(f"    diag {level}: max Courant {np.max(r['courant_max']):.2f}; re-runs {int(np.sum(r['dt_used'] < dt))}")


# ------------------------------------------------------------------------------------------------------- W#1 fine scan
def st_scan(ctx, run, level, rays, args):
    F = ctx.cfg["fine_scan"]
    g, dt = ctx.grid(level)
    s = np.geomspace(float(F["s_min"]), float(F["s_max"]), int(F["n"]))
    out_f = run / f"scan_{level}.json"
    if out_f.exists():
        log(f"    scan {level}: done earlier")
        return
    u0 = g.clean(g.from_coeffs(s[:, None] * ctx.dir(F["ray"])[None, :]))
    r = run_det(g, u0, dt, ctx.T_det, ctx.r_A, ctx.E_B, absorb=True, max_batch=args.max_batch, courant_limit=ctx.climit,
                max_levels=ctx.lv, progress=True, desc=f"scan {level}", log=log)
    notA = r["fate"] != 0
    atomic_write_json(out_f, {"ray": F["ray"], "level": level, "s": s.tolist(), "fate": r["fate"].tolist(), "t_hit": r["t_hit"].tolist(),
                              "dt_used": r["dt_used"].tolist(), "courant_max": r["courant_max"].tolist(), "counts": r["counts"],
                              "n_switches": int(np.abs(np.diff(notA.astype(int))).sum()), "monotone": bool(np.abs(np.diff(notA.astype(int))).sum() == 1)})
    log(f"    scan {level}: fates {''.join({0: 'A', 1: 'B', -1: 'u', 2: 'd'}[int(f)] for f in r['fate'])}")


# ------------------------------------------------------------------------------------------------------------- analysis
def st_analysis(ctx, run, level, rays, args):
    """7a-7c summary and the preset verdict (h12). Review fixes (2026-11-18 ~10:40, before any result existed): expected levels
    per ray (128 only for the effective 128 set); a missing or flagged s_c at an expected level is reported and the ray counts
    for neither verdict (a control with a missing level fails); s_i with no crossing gives err(i) = +inf (Gate 5 convention);
    the first-order ratio p is not formed where p(64) is zero (pure-wave rays: P0 delta = 0)."""
    cfg = ctx.cfg
    C = cfg["criteria"]
    S_star = float(cfg["constants"]["S_star"])
    TH = cfg["theory"]
    sg = np.sqrt(np.geomspace(ctx.r_A * (1 + 1e-9), ctx.E_B * (1 - 1e-9), int(TH["s_grid_points"])))
    names = ctx.all_rays()
    levels = ["64", "96", "128"]
    tm = read_json(run / "timing.json", {})
    set128 = cfg["rays"]["at_128_reduced"] if str(tm.get("decision", "")).startswith("reduced") else cfg["rays"]["at_128"]
    res = {"S_star": S_star, "set_128": set128, "levels": {}, "rays": {}}
    th = {}
    for lv in levels:
        f = run / f"theory_{lv}.npz"
        if f.exists():
            Z = np.load(f)
            th[lv] = {str(n): (Z["aa"][i], Z["ab"][i], Z["bb"][i]) for i, n in enumerate(Z["names"])}
    num_err = {}
    for n in names:
        R = {}
        expected = ["64", "96"] + (["128"] if n in set128 else [])
        for lv in levels:
            p = sc_at(ctx, run, lv, n)
            rr = {"s_c": None, "expected": lv in expected}
            if p is not None:
                rr.update({"s_c": p[0], "lo": p[1], "hi": p[2], "t_hit_lo": p[3], "t_hit_hi": p[4], "source": p[5],
                           "bracket_rel_width": (p[2] - p[1]) / p[1]})
            elif lv in expected:
                T = load_truths(run).get(lv, {})
                rr["status"] = T[n]["status"] if n in T else "missing (search not run or not finished)"
            if lv in th and n in th[lv]:
                aa, ab, bb = th[lv][n]
                S = A_of(aa[None], ab[None], bb[None], sg)[0].max(axis=1)
                rc = RayCoordinate(S[None], sg, "cpu")
                ln_i = float(rc.ln_s_pred([S_star], 1)[0][0, 0])
                rr.update({"s_i": math.exp(ln_i) if np.isfinite(ln_i) else "+inf (no crossing)", "p_first_order": float(np.sqrt(np.maximum(aa, 0)).max()),
                           "r_second_order": float(np.sqrt(np.maximum(bb, 0)).max())})
                if rr["s_c"] is not None:
                    e = ln_i - math.log(rr["s_c"])
                    num_err[(n, lv)] = e
                    rr["err_i"] = e if np.isfinite(e) else "+inf (no crossing)"
                    rr["S_at_s_c"] = float(np.interp(math.log(rr["s_c"]), np.log(sg), S))
            R[lv] = rr
        s64 = R["64"].get("s_c")
        for lv in ("96", "128"):
            if R[lv].get("s_c") and R[prev_level(lv)].get("s_c"):
                R[lv]["dln_vs_prev"] = math.log(R[lv]["s_c"]) - math.log(R[prev_level(lv)]["s_c"])
            if "p_first_order" in R[lv] and "p_first_order" in R["64"]:
                p64, r64 = R["64"]["p_first_order"], R["64"]["r_second_order"]
                R[lv]["ratio_p_vs_64"] = R[lv]["p_first_order"] / p64 if p64 > 1e-8 * max(r64, 1e-300) else "not formed (p(64) ~ 0: pure-wave ray)"
                R[lv]["ratio_r_vs_64"] = R[lv]["r_second_order"] / r64
                if isinstance(R[lv].get("s_i"), float) and isinstance(R["64"].get("s_i"), float):
                    R[lv]["ratio_s_i_vs_64"] = R[lv]["s_i"] / R["64"]["s_i"]
                if s64:
                    aa, ab, bb = th[lv][n]
                    a0, b0, c0 = th["64"][n]
                    R[lv]["ratio_S_at_s_c64_vs_64"] = float(A_of(aa[None], ab[None], bb[None], np.array([s64]))[0].max() /
                                                            A_of(a0[None], b0[None], c0[None], np.array([s64]))[0].max())
        top = expected[-1]
        R["highest"] = top
        R["complete"] = all(R[lv].get("s_c") is not None for lv in expected)
        if R["complete"]:
            R["dln_top2"] = math.log(R[expected[-1]]["s_c"]) - math.log(R[expected[-2]]["s_c"])
        res["rays"][n] = R
    # controls (h12): every adjacent pair of the expected levels, |dln| <= 0.02; a missing level fails
    ctl, ok_ctl = {}, True
    for n in cfg["rays"]["controls"]:
        R = res["rays"][n]
        exp_lv = [lv for lv in ("96", "128") if R[lv]["expected"]]
        d = {lv: R[lv].get("dln_vs_prev") for lv in exp_lv}
        ctl[n] = d
        if any(v is None or abs(v) > float(C["controls_max_abs_dln"]) for v in d.values()):
            ok_ctl = False
    res["controls"] = {"dln": ctl, "pass": ok_ctl}
    # verdict (5 decision rays, each at its highest expected level)
    art, real, incomplete = [], [], []
    for n in cfg["rays"]["decision"]:
        R = res["rays"][n]
        top = R["highest"]
        if not R["complete"] or (n, top) not in num_err:
            incomplete.append(n)
            continue
        e, tt = num_err[(n, top)], R[top]["t_hit_hi"]
        if abs(e) <= float(C["artefact"]["max_abs_err_i"]):
            art.append(n)
        if abs(e) >= float(C["real_route"]["min_abs_err_i"]) and tt is not None and tt < float(C["real_route"]["t_hit_hi_below"]) \
                and abs(R["dln_top2"]) <= float(C["real_route"]["max_abs_dln_top2"]):
            real.append(n)
    rule = lambda: ("numerical artefact" if len(art) >= int(C["artefact"]["min_rays"]) else  # noqa: E731
                    ("real route" if len(real) >= int(C["real_route"]["min_rays"]) else "undetermined"))
    v = rule() if ok_ctl else "not made (controls failed)"
    res["verdict"] = {"artefact_rays": art, "real_route_rays": real, "incomplete_rays": incomplete, "verdict": v,
                      "verdict_if_controls_ignored (descriptive)": rule()}
    for lv in ("64", "96"):
        f = run / f"scan_{lv}.json"
        if f.exists():
            S = json.loads(f.read_text())
            res.setdefault("fine_scan", {})[lv] = {"monotone": S["monotone"], "n_switches": S["n_switches"],
                                                    "fates": "".join({0: "A", 1: "B", -1: "u", 2: "d"}[int(x)] for x in S["fate"])}
    for f in sorted(run.glob("skipped_*.json")):
        res.setdefault("skipped", []).extend(json.loads(f.read_text()))
    tf = run / "theory_64_vs_gate5.json"
    if tf.exists():
        res["theory_64_vs_gate5_max_rel"] = json.loads(tf.read_text())
    atomic_write_json(run / "results.json", res)
    log(f"    controls {ctl} -> {'pass' if ok_ctl else 'FAIL'}; artefact {art}; real route {real}; incomplete {incomplete}; verdict: {v}")
    for n in names:
        R = res["rays"][n]
        log("    " + n + ": " + "; ".join(f"{lv} s_c {R[lv]['s_c']:.5g} t_hi {R[lv].get('t_hit_hi')} err_i {R[lv].get('err_i')}"
                                         for lv in levels if R[lv].get("s_c")))


if __name__ == "__main__":
    main()
