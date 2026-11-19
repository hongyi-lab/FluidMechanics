#!/usr/bin/env python
"""Slow route at another Reynolds number (quick test, descriptive). Config: code/configs/slowre_dns.yaml.

    python code/scripts/slowre_dns.py --config code/configs/slowre_dns.yaml --run-dir <dir> --stage theory
    ... --stage gain                 (s_ii; needs theory)
    ... --stage truth --level 64     (all rays; needs theory for the grid centre)
    ... --stage truth --level 96     (the resolution control rays)
    ... --stage pilot                (E after the first E_B crossing, Re = 400 and the new Re)
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate9_dns as G9  # noqa: E402
from gate6_dns import first_cross  # noqa: E402
from gate7_dns import log  # noqa: E402
from committor.dns import run_det  # noqa: E402
from committor.dns_gate9 import find_sc_grid9  # noqa: E402
from committor.dns_theory import A_of, run_delta_q, run_gain  # noqa: E402
from committor.threshold import RayCoordinate  # noqa: E402
from committor.utils import atomic_save_npz, atomic_write_json, git_info, read_json  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


class CtxRe(G9.Ctx9):
    """Gate 9 context (rays, levels, ladder) with Re and T_det replaced; self.cfg (Gate 7 levels) is left alone."""

    def __init__(self, cs):
        cfg9 = yaml.safe_load((REPO / cs["data"]["gate9_config"]).read_text())
        super().__init__(cfg9, REPO / cs["data"]["gate9_run"])
        self.cs = cs
        self.Re0 = self.Re                                                   # 400 (Gate 4b choice)
        assert self.Re0 == 400.0
        F = cs["fate"]
        assert self.r_A == float(F["r_A"]) and self.E_B == float(F["E_B"])
        self.Re = float(cs["Re"])
        self.T_det = float(F["T_det"])

    def grid_at(self, level, Re):
        keep = self.Re
        self.Re = float(Re)
        try:
            g, dt = self.grid(level)
        finally:
            self.Re = keep
        assert abs(g.nu - 1.0 / float(Re)) < 1e-15
        return g, dt


def s_grid(ctx, n):
    return np.sqrt(np.geomspace(ctx.r_A * (1 + 1e-9), ctx.E_B * (1 - 1e-9), int(n)))


def st_theory(ctx, run, args):
    f = run / "theory_64.npz"
    names = ctx.cs["rays"]
    TH = ctx.cfg9["theory"]
    if not f.exists():
        D = np.array([ctx.dir(n) for n in names])
        g, dt = ctx.grid_at(TH["level"], ctx.Re)
        log(f"    theory {TH['level']} at Re = {g.Re}, nu = {g.nu:.6g}, dt = {dt}")
        r = run_delta_q(g, D, float(TH["T_S"]), dt, every=float(TH["output_every"]), progress=True, desc="theory 64")
        atomic_save_npz(f, aa=r["aa"], ab=r["ab"], bb=r["bb"], t=r["t"], names=np.array(names))
    T1 = np.load(f)
    sg = s_grid(ctx, TH["s_grid_points"])
    A = A_of(T1["aa"], T1["ab"], T1["bb"], sg)                                 # (R, n_s, n_t)
    S = A.max(axis=2)
    S_star = float(ctx.cfg9["constants"]["S_star"])
    ln_i = RayCoordinate(S, sg, "cpu").ln_s_pred([S_star], 1)[0]
    ln_i = np.asarray(ln_i.cpu() if torch.is_tensor(ln_i) else ln_i, dtype=np.float64)[:, 0]
    out = {"Re": ctx.Re, "S_star": S_star, "rays": []}
    for i, n in enumerate(names):
        out["rays"].append({"ray": n, "s_i": G9.fin(math.exp(ln_i[i])), "S_max_on_grid": float(S[i].max())})
    atomic_write_json(run / "s_i.json", out)
    log("    s_i: " + "; ".join(f"{r['ray']} {r['s_i']:.5g}" for r in out["rays"]))


def st_gain(ctx, run, args):
    names = ctx.cs["rays"]
    TH = ctx.cfg9["theory"]
    GN = TH["gain"]
    c_star = float(ctx.cfg9["constants"]["c_star"])
    SI = json.loads((run / "s_i.json").read_text())["rays"]
    assert [r["ray"] for r in SI] == names
    sg = s_grid(ctx, TH["s_grid_points"])
    ln_i = np.array([math.log(r["s_i"]) if r["s_i"] is not None else math.log(sg[-1]) for r in SI])
    K, hw = int(GN["n_amp"]), float(GN["ln_half_width"])
    offs = np.linspace(-hw, hw, K)
    step = offs[1] - offs[0]
    D = np.array([ctx.dir(n) for n in names])
    g, dt = ctx.grid_at(TH["level"], ctx.Re)

    def cgrid(idx, L, tag):
        rr = run_gain(g, D[idx], np.exp(L), float(GN["T_end"]), dt, every=float(GN["output_every"]), progress=True, desc=f"gain {tag}")
        with np.errstate(divide="ignore"):
            return np.log(rr["yy"].max(axis=2)), rr["yy"].argmax(axis=2) * float(GN["output_every"]), rr["courant_M"]
    f2 = run / "gain_64_block1.npz"
    if not f2.exists():
        L1 = ln_i[:, None] + offs[None]
        c1, tm1, cm1 = cgrid(np.arange(len(names)), L1, "block 1")
        atomic_save_npz(f2, lns=L1, c=c1, t_at_max=tm1, courant_M=cm1)
        log("    gain block 1 done")
    B1 = np.load(f2)
    first = [first_cross(B1["lns"][i], B1["c"][i], c_star) for i in range(len(names))]
    ext = [i for i, (_, fl) in enumerate(first) if fl in ("below", "never")]
    f3 = run / "gain_64_block2.npz"
    if ext and not f3.exists():
        L2 = np.array([(B1["lns"][i, 0] - step * np.arange(K, 0, -1)) if first[i][1] == "below" else (B1["lns"][i, -1] + step * np.arange(1, K + 1))
                       for i in ext])
        c2, tm2, cm2 = cgrid(np.array(ext), L2, "block 2")
        atomic_save_npz(f3, idx=np.array(ext), lns=L2, c=c2, t_at_max=tm2, courant_M=cm2)
        log(f"    gain block 2 done ({len(ext)} rays)")
    B2 = np.load(f3) if f3.exists() else None
    out = {"Re": ctx.Re, "c_star": c_star, "rays": []}
    for i, n in enumerate(names):
        L, Cc, TM = B1["lns"][i], B1["c"][i], B1["t_at_max"][i]
        ln_ii, fl = first[i]
        if B2 is not None and i in list(B2["idx"]):
            j = list(B2["idx"]).index(i)
            below = fl == "below"
            L = np.concatenate([B2["lns"][j], L]) if below else np.concatenate([L, B2["lns"][j]])
            Cc = np.concatenate([B2["c"][j], Cc]) if below else np.concatenate([Cc, B2["c"][j]])
            TM = np.concatenate([B2["t_at_max"][j], TM]) if below else np.concatenate([TM, B2["t_at_max"][j]])
            ln_ii, fl2 = first_cross(L, Cc, c_star)
            fl = fl2 + " (after one extension)"
        out["rays"].append({"ray": n, "s_ii": math.exp(ln_ii), "ii_flag": fl, "lns": L.tolist(), "c": Cc.tolist(), "t_at_max": TM.tolist(),
                            "t_at_max_flag": bool(np.max(TM) >= float(ctx.cs["gain_flag_t"]))})
    atomic_write_json(run / "s_ii.json", out)
    log("    s_ii: " + "; ".join(f"{r['ray']} {r['s_ii']:.5g} ({r['ii_flag']})" for r in out["rays"]))


def st_truth(ctx, run, args):
    lv = str(args.level)
    names = ctx.cs["rays"] if lv == "64" else ctx.cs["control_96"]
    if args.rays:                                                            # extra control run into its own checkpoint
        names = args.rays.split(",")
    SI = {r["ray"]: r["s_i"] for r in json.loads((run / "s_i.json").read_text())["rays"]}
    center = [SI[n] for n in names]
    assert all(c is not None for c in center), "s_i missing"
    T = ctx.cs["truth"]
    g, dt = ctx.grid_at(lv, ctx.Re)
    D = np.array([ctx.dir(n) for n in names])
    log(f"--- truth {lv} at Re = {g.Re} (T_det {ctx.T_det}): {names}; centres (s_i) {[round(c, 6) for c in center]}")
    find_sc_grid9(g, D, names, center, T["factors"], T["ext_up"], T["ext_down"], ctx.r_A, ctx.E_B, ctx.T_det, dt, float(T["rel_width"]),
                  ctx.climit, ctx.lv, args.max_batch, ckpt=run / f"truth_{lv}{args.suffix}.json", log=log, desc=f"s_c {lv} Re{int(ctx.Re)}")


def st_pilot(ctx, run, args):
    f = run / "pilot_det.npz"                     # (review fix 16:20) run_det with the ladder; pilot.npz (run_nonlinear) was stopped
    if f.exists():
        log("    pilot_det.npz exists")
        return
    P = ctx.cs["pilot"]
    R9 = {r["ray"]: r for r in json.loads((REPO / ctx.cs["data"]["gate9_run"] / "results.json").read_text())["rows"]}
    names = P["rays"]
    amps = [float(P["factor_of_sc400"]) * R9[n]["s_c"] for n in names]
    D = np.array([a * ctx.dir(n) for a, n in zip(amps, names)])
    out = {"names": np.array(names), "amps": np.array(amps)}
    for Re in (ctx.Re0, ctx.Re):
        g, dt = ctx.grid_at("64", Re)
        u0 = g.clean(g.from_coeffs(D))
        r = run_det(g, u0, dt, float(P["T"]), None, None, record_every=1.0, absorb=False, courant_limit=ctx.climit, max_levels=ctx.lv,
                    progress=True, desc=f"pilot Re{int(Re)}", log=log)
        out[f"E_{int(Re)}"], out[f"courant_{int(Re)}"], out[f"dt_used_{int(Re)}"], out[f"nan_final_{int(Re)}"] = (
            r["E"], r["courant_max"], r["dt_used"], r["nan_final"])
        log(f"    pilot Re {Re}: counts {r['counts']}")
        del g
        torch.cuda.empty_cache()
    atomic_save_npz(f, **out)
    log("    pilot done")


def pilot_summary(ctx, run):
    Z = np.load(run / "pilot_det.npz")
    rows = []
    for Re in (ctx.Re0, ctx.Re):
        E = Z[f"E_{int(Re)}"]
        for j, n in enumerate(Z["names"]):
            e = E[j]
            fin_ = np.isfinite(e)
            base = {"Re": Re, "ray": str(n), "s": float(Z["amps"][j]), "dt_used": float(Z[f"dt_used_{int(Re)}"][j]),
                    "nan_final": bool(Z[f"nan_final_{int(Re)}"][j]), "courant_max": float(Z[f"courant_{int(Re)}"][j]),
                    "n_nonfinite": int((~fin_).sum()), "t_first_nonfinite": (float(np.argmax(~fin_)) if (~fin_).any() else None)}
            hit = np.where(fin_ & (e >= ctx.E_B))[0]
            if len(hit) == 0:
                rows.append(dict(base, t_first_EB=None, E_max=float(np.nanmax(e)) if fin_.any() else None))
                continue
            k = int(hit[0])
            after = e[k:][np.isfinite(e[k:])]
            rows.append(dict(base, t_first_EB=float(k), E_min_after=float(after.min()), E_median_after=float(np.median(after)),
                             frac_below_EB_after=float(np.mean(after < ctx.E_B)), n_finite_after=int(len(after)),
                             E_final=(float(e[-1]) if np.isfinite(e[-1]) else None)))
    return rows


def implied_S(T1, i, s):
    return float(A_of(T1["aa"][i:i + 1], T1["ab"][i:i + 1], T1["bb"][i:i + 1], np.array([s])).max())


def implied_c(lns, c, s):
    lns, c = np.asarray(lns), np.asarray(c)
    o = np.argsort(lns)
    x = math.log(s)
    if x < lns[o][0] or x > lns[o][-1]:
        return None
    return float(np.interp(x, lns[o], c[o]))


def st_analysis(ctx, run, args):
    names = ctx.cs["rays"]
    g9 = REPO / ctx.cs["data"]["gate9_run"]
    P9 = {r["ray"]: r for r in json.loads((g9 / "predictions_9.json").read_text())["rays"]}
    R9 = {r["ray"]: r for r in json.loads((g9 / "results.json").read_text())["rows"]}
    GN9 = json.loads((g9 / "gain_64.json").read_text())
    T9 = np.load(g9 / "theory_64.npz")
    n9 = [str(x) for x in T9["names"]]
    T1 = np.load(run / "theory_64.npz")
    SI = {r["ray"]: r for r in json.loads((run / "s_i.json").read_text())["rays"]}
    SII = {r["ray"]: r for r in json.loads((run / "s_ii.json").read_text())["rays"]}
    TR = {r["name"]: r for r in json.loads((run / "truth_64.json").read_text())["rays"]}
    TR96, C96 = {}, {}
    for f96 in sorted(run.glob("truth_96*.json")):                          # V#1 (driver) and V#2 (added after the review)
        Z96 = json.loads(f96.read_text())
        C96[f96.name] = {"phase": Z96.get("phase"), "counts": Z96.get("counts")}
        TR96.update({r["name"]: r for r in Z96["rays"]})
    S_star, c_star = float(ctx.cfg9["constants"]["S_star"]), float(ctx.cfg9["constants"]["c_star"])
    rows = []
    for i, n in enumerate(names):
        s_i, s_ii = SI[n]["s_i"], SII[n]["s_ii"]
        s_slow = max(s_i, s_ii)
        t = TR[n]
        Lg, Cg, TMg = map(np.asarray, (SII[n]["lns"], SII[n]["c"], SII[n]["t_at_max"]))
        jx = int(np.argmax(Cg >= c_star)) if (Cg >= c_star).any() else len(Cg) - 1
        r = {"ray": n, "s_i": s_i, "s_ii": s_ii, "ii_flag": SII[n]["ii_flag"], "gain_t_at_max_max": float(TMg.max()),
             "gain_t_at_max_upto_cross": float(TMg[:jx + 1].max()), "gain_flag_upto_cross": bool(TMg[:jx + 1].max() >= float(ctx.cs["gain_flag_t"])),
             "binding": "ii" if s_ii > s_i else "i", "s_slow": s_slow, "status": t["status"], "n_switches": t.get("n_switches"),
             "non_monotone": t.get("non_monotone")}
        # Re = 400 (Gate 9: theory and gain at 64, truth at 96)
        p9, r9 = P9[n], R9[n]
        sc4 = r9["s_c"]
        b9 = [b for b in GN9[n]]
        L9 = np.concatenate([np.asarray(b["lns"]) for b in b9])
        C9 = np.concatenate([np.asarray(b["c"]) for b in b9])
        j9 = n9.index(n)
        r.update({"s_slow_400": p9["s_slow"], "s_c_400": sc4, "err_400": math.log(p9["s_slow"] / sc4), "t_hit_hi_400": r9["t_hit_hi"],
                  "binding_400": "ii" if p9["s_ii"] > p9["s_i"] else "i",
                  "S_at_sc_400": implied_S(T9, j9, sc4), "c_at_sc_400": implied_c(L9, C9, sc4)})
        if t.get("s_c") is not None:
            sc = t["s_c"]
            r.update({"s_c": sc, "lo": t["lo"], "hi": t["hi"], "t_hit_hi": t["t_hit_hi"], "h_b": math.log(t["hi"] / t["lo"]) / 2,
                      "err": math.log(s_slow / sc), "err_i": math.log(s_i / sc), "err_ii": math.log(s_ii / sc),
                      "shift": math.log(sc / sc4), "S_at_sc": implied_S(T1, i, sc), "c_at_sc": implied_c(SII[n]["lns"], SII[n]["c"], sc)})
        if n in TR96 and TR96[n].get("s_c") is not None and "s_c" in r:
            s96 = TR96[n]["s_c"]
            r.update({"s_c_96": s96, "dln_96_64": math.log(s96 / r["s_c"]), "t_hit_hi_96": TR96[n]["t_hit_hi"],
                      "err_96": math.log(s_slow / s96), "shift_96": math.log(s96 / sc4)})        # same level as the Re = 400 truth
        rows.append(r)
    ok = [r for r in rows if "err" in r]
    ae = np.abs([r["err"] for r in ok])
    ae4 = np.abs([r["err_400"] for r in ok])
    sh = np.array([r["shift"] for r in ok])
    summ = {"n": len(ok), "abs_err_median": float(np.median(ae)) if len(ok) else None, "abs_err_max": float(np.max(ae)) if len(ok) else None,
            "abs_err_400_median": float(np.median(ae4)) if len(ok) else None, "abs_err_400_max": float(np.max(ae4)) if len(ok) else None,
            "shift_median": float(np.median(sh)) if len(ok) else None, "shift_range": [float(sh.min()), float(sh.max())] if len(ok) else None,
            "n_rule_closer_than_no_change": int(sum(abs(r["err"]) < abs(r["shift"]) for r in ok)),
            "S_at_sc_over_S_star": {r["ray"]: r["S_at_sc"] / S_star for r in ok},
            "S_at_sc_over_S_star_400": {r["ray"]: r["S_at_sc_400"] / S_star for r in ok},
            "c_at_sc_minus_c_star": {r["ray"]: (None if r["c_at_sc"] is None else r["c_at_sc"] - c_star) for r in ok},
            "c_at_sc_minus_c_star_400": {r["ray"]: (None if r["c_at_sc_400"] is None else r["c_at_sc_400"] - c_star) for r in ok},
            "note_levels": "s_c(500) at 64, s_c(400) at 96 (Gate 9); shift_96 / err_96 on the 96 control rays are same-level",
            "not_ok": [(r["ray"], r["status"]) for r in rows if "err" not in r]}
    res = {"Re": ctx.Re, "Re0": ctx.Re0, "T_det": ctx.T_det, "S_star": S_star, "c_star": c_star, "rows": rows, "summary": summ,
           "pilot": pilot_summary(ctx, run) if (run / "pilot_det.npz").exists() else None,
           "truth_64_phase": json.loads((run / "truth_64.json").read_text()).get("phase"),
           "counts_64": json.loads((run / "truth_64.json").read_text()).get("counts"), "truth_96_files": C96}
    atomic_write_json(run / "results.json", res)
    log(f"    analysis: n {summ['n']}, |err| median {summ['abs_err_median']}, max {summ['abs_err_max']}; shift median {summ['shift_median']}; "
        f"rule closer than no-change {summ['n_rule_closer_than_no_change']}/{summ['n']}; not ok {summ['not_ok']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--level", default="64")
    ap.add_argument("--max-batch", type=int, default=96)
    ap.add_argument("--rays", default=None)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    cs = yaml.safe_load(Path(args.config).read_text())
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    if not (run / "config_used.yaml").exists():
        (run / "config_used.yaml").write_text(yaml.safe_dump(cs, sort_keys=False))
    tag = args.stage + (f"_{args.level}{args.suffix}" if args.stage == "truth" else "")
    mf = run / f"manifest_{tag}.json"
    man = read_json(mf, {"invocations": []})
    man["invocations"].append({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv, "git": git_info(REPO),
                               "python": platform.python_version(), "torch": torch.__version__,
                               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    atomic_write_json(mf, man)
    ctx = CtxRe(cs)
    t0 = time.time()
    {"theory": st_theory, "gain": st_gain, "truth": st_truth, "pilot": st_pilot, "analysis": st_analysis}[args.stage](ctx, run, args)
    log(f"--- {tag}: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
