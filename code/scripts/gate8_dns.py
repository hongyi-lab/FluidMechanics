#!/usr/bin/env python
"""Gate 8 (DNS): blind test of the third condition (fast channel = oblique-wave route); 8d records; 8c completes Gate 7.
Config: code/configs/gate8_dns.yaml (criteria fixed before any run). Stages (each resumable, run one at a time, h13):

    python code/scripts/gate8_dns.py --config code/configs/gate8_dns.yaml --run-dir <dir> --stage posthoc
    ... --stage predict          (8a: frozen predictions; committed before 8b)
    ... --stage truth            (8b: s_c at 96 for the blind rays and X#1)
    ... --stage d8               (8d: kz-split theory and DNS, linear wave at s_c(96))
    ... --stage c8_prepare       (8c: seed <run>/8c_gate7_128 for the unchanged Gate 7 script)
    ... --stage analysis
"""
import argparse
import json
import math
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate6_dns import first_cross  # noqa: E402
from gate7_dns import Ctx, load_truths, log, sc_at  # noqa: E402
from committor.dns_gate8 import KZ_BINS, PARTS8, diag_kz, gain_alpha, theory_kz, wave_gain  # noqa: E402
from committor.dns_local import find_sc_local  # noqa: E402
from committor.dns_rays import build_rays  # noqa: E402
from committor.dns_theory import A_of, run_delta_q, run_gain  # noqa: E402
from committor.threshold import RayCoordinate  # noqa: E402
from committor.utils import atomic_save_npz, atomic_write_json, git_info, read_json  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


class Ctx8(Ctx):
    def __init__(self, cfg8):
        self.cfg8 = cfg8
        cfg7 = yaml.safe_load((REPO / cfg8["data"]["gate7_config"]).read_text())
        super().__init__(cfg7)
        self.g7 = REPO / cfg8["data"]["gate7_run"]
        self.g6 = REPO / cfg8["data"]["gate6_run"]
        rays = build_rays(self.cfg5, REPO)
        self.F_d, self.F_meta = rays["F"][0], rays["F"][1]

    def dir(self, name):
        if name == "X#1":
            d = np.zeros(9)
            d[7] = 1.0                                               # (h1) NJP mode 8
            return d
        if name.startswith("F"):
            return self.F_d[int(name.split("#")[1]) - 1]
        return super().dir(name)

    def all52(self):
        return [f"F#{i + 1}" for i in range(self.nF)] + [f"R#{i + 1}" for i in range(len(self.R_d))] + [f"W#{i + 1}" for i in range(len(self.W_d))]

    def fname(self, name):
        if name.startswith("F"):
            m = self.F_meta[int(name.split("#")[1]) - 1]
            return f"F eps={m['eps']:g} draw {m['draw']}"
        return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--max-batch", type=int, default=64)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    if not (run / "config_used.yaml").exists():
        (run / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    mf = run / f"manifest_{args.stage}.json"
    man = read_json(mf, {"invocations": []})
    man["invocations"].append({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv, "git": git_info(REPO),
                               "python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})
    atomic_write_json(mf, man)
    ctx = Ctx8(cfg)
    t0 = time.time()
    {"posthoc": st_posthoc, "predict": st_predict, "truth": st_truth, "d8": st_d8, "c8_prepare": st_c8_prepare, "analysis": st_analysis}[args.stage](
        ctx, run, args)
    log(f"--- {args.stage}: {(time.time() - t0) / 60:.1f} min")


# ------------------------------------------------------------------------------------------------------------- G (h2)
def g_all(ctx, run):
    """G at 64 and 96 for the 52 rays and X#1, and ||Pw delta(10)||^2 at 64; cached in g_all.npz."""
    f = run / "g_all.npz"
    names = ctx.all52() + ["X#1"]
    if f.exists():
        Z = np.load(f)
        if [str(x) for x in Z["names"]] == names:
            return {k: Z[k] for k in Z.files}
    T = float(ctx.cfg8["third_condition"]["T_G"])
    D = np.array([ctx.dir(n) for n in names])
    out = {"names": np.array(names)}
    for lv in ("64", "96"):
        g, dt = ctx.grid(lv)
        r = wave_gain(g, D, T=T, dt=dt, every=1.0)
        out[f"G_{lv}"], out[f"tmax_{lv}"], out[f"series_{lv}"] = r["G"], r["t_at_max"], r["series"]
        log(f"    G at {lv}: W#1 {r['G'][names.index('W#1')]:.4f}, X#1 {r['G'][-1]:.4f}")
        del g
        torch.cuda.empty_cache()
    out["t"] = np.arange(out["series_64"].shape[1]) * 1.0
    out["G10_64"] = out["series_64"][:, 10]
    atomic_save_npz(f, **out)
    return out


def gate56_slow(ctx):
    """s_i, s_ii (and flag) at 64 for the R and W rays from Gates 5/6 (h4), and the energy-only prediction."""
    A5 = np.load(ctx.g5 / "analysis.npz")
    R6 = json.loads((ctx.g6 / "results.json").read_text())["analysis"]["rays_RW"]
    out = {}
    for r in R6:
        n = f"{r['set']}#{r['k']}"
        i = ctx.nF + ctx.idx5(n)
        sc64 = float(A5["sc"][i])
        out[n] = {"s_c_64": sc64, "s_i": math.exp(float(A5["pred_S"][i])), "s_ii": sc64 * math.exp(float(r["ln_err_ii"])),
                  "ii_flag": r["ii_flag"], "s_E": math.exp(float(A5["pred_E"][i]))}
    return out


# --------------------------------------------------------------------------------------------------------- post-hoc
def st_posthoc(ctx, run, args):
    """User's post-hoc observations 1-3 and the 52-ray table (descriptive, h14)."""
    G = g_all(ctx, run)
    names = [str(x) for x in G["names"]]
    ix = {n: i for i, n in enumerate(names)}
    out = {}
    # obs 1: ||Pw delta(10)||^2 at 64 vs a8^2 on the W rays
    W = [f"W#{k}" for k in range(1, 13)]
    a8 = np.array([ctx.dir(n)[7] for n in W])
    g10 = np.array([G["G10_64"][ix[n]] for n in W])
    k_ls = float(np.sum(g10 * a8 ** 2) / np.sum(a8 ** 4))
    rows5 = json.loads((ctx.g5 / "results.json").read_text())["analysis"]["rays"]
    S_edge = {f"{r['set']}#{r['k']}": r.get("S_at_edge") for r in rows5 if r["set"] in ("R", "W")}
    out["obs1"] = {"fit_G10_over_a8sq_lsq": k_ls, "per_ray": [
        {"ray": n, "a8": float(a), "a8_sq": float(a * a), "G10_64": float(v), "ratio_G10_over_a8sq": float(v / (a * a)) if a * a > 1e-12 else None,
         "fit_rel_resid": float((v - k_ls * a * a) / v) if v > 0 else None, "S_at_edge_64": S_edge.get(n),
         "t_hit_hi_64": float(ctx.t5["t_hit_hi"][ctx.idx5(n)])} for n, a, v in zip(W, a8, g10)]}
    log(f"    obs 1: G10 ~ {k_ls:.3f} a8^2 (least squares through 0)")
    # obs 2: early (t <= 20) kx = alpha peak at the highest Gate 7 level
    top = {"W#1": "128", "W#3": "96", "W#7": "96", "W#9": "96", "R#22": "128"}
    D7 = {}
    for f in sorted(ctx.g7.glob("diag_*.npz")):
        Z = np.load(f)
        lv = f.name.split("_")[1]
        P = [str(p) for p in Z["parts"]]
        for i, (n, w) in enumerate(zip(Z["names"], Z["which"])):
            D7[(str(n), lv, str(w))] = (Z["t"], Z["rec"][i], P)
    o2 = {}
    for n, lv in top.items():
        for w in ("lo", "hi"):
            t, rec, P = D7[(n, lv, w)]
            m = t <= 20.0 + 1e-9
            j = int(np.argmax(rec[m, P.index("kx=alpha")]))
            o2[f"{n} {lv} {w}"] = {"max_kx_alpha_t_le_20": float(rec[m, P.index("kx=alpha")][j]), "t_at_max": float(t[m][j])}
    out["obs2"] = o2
    # obs 3: DNS kx = alpha (Gate 7 diag at 64) vs the Gate 6 linear wave ||y||^2 at s_c(64) (mid amplitude, every 10)
    Z6 = np.load(ctx.g6 / "gain_RW.npz")
    K = Z6["c"].shape[1]
    rowsRW = [f"R#{k}" for k in range(13, 25)] + W
    o3 = {}
    for n in ("W#1", "W#3", "W#7", "W#9", "R#22", "R#13", "R#15", "W#11"):
        j = list(Z6["idx"]).index(rowsRW.index(n))
        yy = Z6["yy_sub"][j, K // 2]                                     # t = 0, 10, 20, ...
        rr = {}
        for w in ("lo", "hi"):
            t, rec, P = D7[(n, "64", w)]
            ka = rec[:, P.index("kx=alpha")]
            kw = ka + rec[:, P.index("kx>=2alpha")]
            rr[w] = {f"t={tt}": {"dns_kx_alpha": float(ka[int(tt)]), "dns_kx_ne_0": float(kw[int(tt)]), "lin_y": float(yy[tt // 10]),
                                 "ratio_alpha": float(ka[int(tt)] / yy[tt // 10]), "ratio_kx_ne_0": float(kw[int(tt)] / yy[tt // 10])}
                     for tt in range(0, 130, 10)}
        o3[n] = rr
    out["obs3"] = o3
    # 52-ray table
    tab = []
    TF = np.load(ctx.g5 / "truth_F.npz")
    for n in ctx.all52() + ["X#1"]:
        d = ctx.dir(n)
        if n.startswith("F"):
            th = float(TF["t_hit_hi"][int(n.split("#")[1]) - 1])
        elif n == "X#1":
            th = None
        else:
            th = float(ctx.t5["t_hit_hi"][ctx.idx5(n)])
        i = ix[n]
        tab.append({"ray": ctx.fname(n), "a8": float(d[7]), "a8_sq": float(d[7] ** 2), "G_64": float(G["G_64"][i]), "G_96": float(G["G_96"][i]),
                    "G_96_over_64": float(G["G_96"][i] / G["G_64"][i]), "t_at_max_96": float(G["tmax_96"][i]), "G10_64": float(G["G10_64"][i]),
                    "t_hit_hi_64": th, "fast_64": (None if th is None else bool(th < 600.0))})
    out["table52"] = tab
    atomic_write_json(run / "posthoc.json", out)
    log("    posthoc written: " + "; ".join(f"{r['ray']} a8^2 {r['a8_sq']:.3f} G96 {r['G_96']:.3f}" for r in tab if r["ray"].startswith(("W", "X"))))


# ------------------------------------------------------------------------------------------------------------- 8a
def x1_slow(ctx, run):
    """s_i and s_ii of X#1 at 64 by the Gate 5/6 code paths (h4); cached in x1_gate56.json."""
    f = run / "x1_gate56.json"
    if f.exists():
        return json.loads(f.read_text())
    X = ctx.cfg8["x1_gate56"]
    g, dt = ctx.grid(X["level"])
    d = ctx.dir("X#1")[None]
    TH = ctx.cfg["theory"]
    r = run_delta_q(g, d, float(TH["T_S"]), dt, every=float(TH["output_every"]), progress=True, desc="X#1 theory 64")
    atomic_save_npz(run / "x1_theory_64.npz", aa=r["aa"], ab=r["ab"], bb=r["bb"], t=r["t"])
    sg = np.sqrt(np.geomspace(ctx.r_A * (1 + 1e-9), ctx.E_B * (1 - 1e-9), int(TH["s_grid_points"])))
    S = A_of(r["aa"], r["ab"], r["bb"], sg)[0].max(axis=1)
    S_star = float(ctx.cfg8["constants"]["S_star"])
    ln_i = float(RayCoordinate(S[None], sg, "cpu").ln_s_pred([S_star], 1)[0][0, 0])
    log(f"    X#1: s_i = {math.exp(ln_i) if np.isfinite(ln_i) else 'no crossing'}")
    GN = X["gain"]
    K, hw = int(GN["n_amp"]), float(GN["ln_half_width"])
    c_star = float(ctx.cfg8["constants"]["c_star"])
    center = ln_i if np.isfinite(ln_i) else math.log(sg[-1])
    offs = np.linspace(-hw, hw, K)
    lns = center + offs
    blocks = []

    def cgrid(lns_):
        rr = run_gain(g, d, np.exp(lns_)[None], float(GN["T_end"]), dt, every=float(GN["output_every"]), progress=True, desc="X#1 gain 64")
        with np.errstate(divide="ignore"):
            return np.log(rr["yy"][0].max(axis=1)), rr["yy"][0].argmax(axis=1) * float(GN["output_every"]), float(np.max(rr["courant_M"]))
    c, tm, cmM = cgrid(lns)
    blocks.append({"lns": lns.tolist(), "c": c.tolist(), "t_at_max": tm.tolist(), "courant_M_max": cmM})
    ln_ii, flag = first_cross(lns, c, c_star)
    if flag in ("below", "never"):                                        # (h4) one more block on the clamped side
        step = offs[1] - offs[0]
        lns2 = (lns[0] - step * np.arange(K, 0, -1)) if flag == "below" else (lns[-1] + step * np.arange(1, K + 1))
        c2, tm2, cm2 = cgrid(lns2)
        blocks.append({"lns": lns2.tolist(), "c": c2.tolist(), "t_at_max": tm2.tolist(), "courant_M_max": cm2})
        L = np.concatenate([lns2, lns]) if flag == "below" else np.concatenate([lns, lns2])
        C = np.concatenate([c2, c]) if flag == "below" else np.concatenate([c, c2])
        ln_ii, flag = first_cross(L, C, c_star)
        flag = flag + " (after one extension)"
    out = {"s_i": math.exp(ln_i) if np.isfinite(ln_i) else None, "s_ii": math.exp(ln_ii), "ii_flag": flag, "blocks": blocks,
           "S_max_on_grid": float(S.max())}
    atomic_write_json(f, out)
    return out


def st_predict(ctx, run, args):
    out_f = run / "predictions_8a.json"
    if out_f.exists():
        log("    predictions_8a.json exists (frozen); not recomputed")
        return
    G = g_all(ctx, run)
    names = [str(x) for x in G["names"]]
    lv = ctx.cfg8["third_condition"]["level"]
    TC = ctx.cfg8["third_condition"]
    iW1 = names.index(TC["calib_ray"])
    P_star = float(TC["calib_s_c_128"]) ** 2 * float(G[f"G_{lv}"][iW1])
    slow = gate56_slow(ctx)
    x1 = x1_slow(ctx, run)
    slow["X#1"] = {"s_c_64": None, "s_i": x1["s_i"], "s_ii": x1["s_ii"], "ii_flag": x1["ii_flag"], "s_E": slow["W#1"]["s_E"]}
    rough = {**ctx.cfg8["user_rough"]["fast"], **ctx.cfg8["user_rough"]["slow"]}
    rough_route = {**{k: "fast" for k in ctx.cfg8["user_rough"]["fast"]}, **{k: "slow" for k in ctx.cfg8["user_rough"]["slow"]}}
    rows = []
    for n in ctx.cfg8["rays"]["predict"]:
        i = names.index(n)
        Gv, G64 = float(G[f"G_{lv}"][i]), float(G["G_64"][i])
        s_fast = math.sqrt(P_star / Gv)
        sl = slow[n]
        s_slow = max(sl["s_i"], sl["s_ii"])
        s_pred = min(s_fast, s_slow)
        rows.append({"ray": n, "a8": float(ctx.dir(n)[7]), "G_96": Gv, "G_64": G64, "G_96_over_64": Gv / G64, "s_fast": s_fast,
                     "s_i": sl["s_i"], "s_ii": sl["s_ii"], "ii_flag": sl["ii_flag"], "s_slow": s_slow, "s_pred": s_pred,
                     "route": "fast" if s_fast < s_slow else "slow", "ln_s_fast_over_s_slow": math.log(s_fast / s_slow),
                     "s_E_energy_only": sl["s_E"], "s_c_64_gate5": sl["s_c_64"],
                     "user_rough": rough.get(n), "user_rough_route": rough_route.get(n),
                     "role": ("blind" if n in ctx.cfg8["rays"]["blind"] else ("blind (X#1, reported alone)" if n == "X#1" else
                              ("not blind" if n in ctx.cfg8["rays"]["not_blind"] else "prediction only")))})
    res = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "P_star": P_star, "G_W1_96": float(G[f"G_{lv}"][iW1]),
           "calib_s_c_128": float(TC["calib_s_c_128"]), "S_star": float(ctx.cfg8["constants"]["S_star"]),
           "c_star": float(ctx.cfg8["constants"]["c_star"]), "x1_gate56": {k: v for k, v in x1.items() if k != "blocks"}, "rays": rows}
    atomic_write_json(out_f, res)
    lines = ["| ray | role | a8 | G_96 | G_64 | s_fast | s_i | s_ii | s_slow | s_pred | route | user rough |", "|" + "---|" * 12]
    for r in rows:
        ur = "" if r["user_rough"] is None else f"{r['user_rough']:g} ({r['user_rough_route']})"
        lines.append(f"| {r['ray']} | {r['role']} | {r['a8']:+.3f} | {r['G_96']:.4g} | {r['G_64']:.4g} | {r['s_fast']:.4g} | {r['s_i']:.4g} | "
                     f"{r['s_ii']:.4g} ({r['ii_flag']}) | {r['s_slow']:.4g} | **{r['s_pred']:.4g}** | {r['route']} | {ur} |")
    (run / "predictions_8a.md").write_text(f"# Gate 8a frozen predictions ({res['frozen_at']})\n\nP* = {P_star:.6g} (s_c(W#1, 128)^2 G_96(W#1)); "
                                           f"S* = {res['S_star']:.6g}; c* = {res['c_star']:.6g}.\n\n" + "\n".join(lines) + "\n")
    log(f"    8a frozen: P* {P_star:.5g}; " + "; ".join(f"{r['ray']} {r['s_pred']:.4g} {r['route']}" for r in rows))


# ------------------------------------------------------------------------------------------------------------- 8b
def st_truth(ctx, run, args):
    P = json.loads((run / "predictions_8a.json").read_text())
    names = ctx.cfg8["rays"]["blind"] + ctx.cfg8["rays"]["blind_extra"]
    s_prev = []
    for n in names:
        if n == "X#1":
            s_prev.append(next(r["s_pred"] for r in P["rays"] if r["ray"] == "X#1"))     # (h6)
        else:
            s_prev.append(sc_at(ctx, ctx.g7, "64", n)[0])
    lv = ctx.cfg8["truth"]["level"]
    L = ctx.cfg["local_grid"]
    g, dt = ctx.grid(lv)
    log(f"--- truth at {lv} (dt {dt:g}) for {names}; s_prev {np.round(s_prev, 6).tolist()}")
    find_sc_local(g, np.array([ctx.dir(n) for n in names]), names, s_prev, L["factors"], L["extend_up"], L["extend_down"], ctx.r_A, ctx.E_B,
                  ctx.T_det, dt, float(L["rel_width"]), ctx.climit, ctx.lv, args.max_batch, ckpt=run / f"truth_{lv}_blind.json", log=log,
                  desc=f"s_c {lv}", progress=True)


def sc96(ctx, run, n):
    """(s_c, lo, hi, t_hit_lo, t_hit_hi, source) at 96: Gate 8 truth, else Gate 7 / Gate 4b ladder."""
    T = load_truths(run).get("96", {})
    if n in T:
        r = T[n]
        return (r["s_c"], r["lo"], r["hi"], r["t_hit_lo"], r["t_hit_hi"], f"Gate 8 ({r['file']})") if r["s_c"] is not None else None
    return sc_at(ctx, ctx.g7, "96", n)


# ------------------------------------------------------------------------------------------------------------- 8d
def st_d8(ctx, run, args):
    D = ctx.cfg8["d8"]
    names = [n for n in ctx.cfg8["rays"]["d8"] if sc96(ctx, run, n) is not None]
    skipped = [n for n in ctx.cfg8["rays"]["d8"] if n not in names]
    if skipped:
        log(f"    8d: no s_c(96) for {skipped}; skipped (recorded)")
        atomic_write_json(run / "skipped_d8.json", skipped)
    g, dt = ctx.grid(D["level"])
    dirs = np.array([ctx.dir(n) for n in names])
    sc = np.array([sc96(ctx, run, n)[0] for n in names])
    T, ev = float(D["T"]), float(D["every"])
    f1 = run / "d8_theory_kz.npz"
    if not f1.exists():
        r = theory_kz(g, dirs, T, dt, every=ev, progress=True, desc="8d theory kz 96")
        atomic_save_npz(f1, **{f"{p}_{k}": r[p][k] for p in ("streak", "roll") for k in ("aa", "ab", "bb")}, t=r["t"], names=np.array(names), s_c=sc)
        log("    8d theory kz done")
    f2 = run / "d8_diag_kz.npz"
    if not f2.exists():
        nm, wh, amps = [], [], []
        for n in names:
            p = sc96(ctx, run, n)
            for w, s in (("lo", p[1]), ("hi", p[2])):
                nm.append(n)
                wh.append(w)
                amps.append(s * ctx.dir(n))
        r = diag_kz(g, np.array(amps), dt, T, ev, 2.0 / 3.0, ctx.climit, ctx.lv, log=log)
        atomic_save_npz(f2, rec=r["rec"], t=r["t"], dt_used=r["dt_used"], courant_max=r["courant_max"], flag_last=r["flag_last"],
                        names=np.array(nm), which=np.array(wh), s=np.array([np.linalg.norm(a) for a in amps]), parts=np.array(PARTS8))
        log(f"    8d diag kz done: max Courant {np.max(r['courant_max']):.2f}; re-runs {int(np.sum(r['dt_used'] < dt))}")
    f3 = run / "d8_gain_alpha.npz"
    if not f3.exists():
        r = gain_alpha(g, dirs, sc[:, None], T, dt, every=ev, progress=True, desc="8d linear wave 96")
        atomic_save_npz(f3, yy=r["yy"][:, 0], ya=r["ya"][:, 0], t=r["t"], courant_M=r["courant_M"][:, 0], names=np.array(names), s_c=sc)
        log(f"    8d linear wave done: max Courant(M) {np.max(r['courant_M']):.2f}")


# ------------------------------------------------------------------------------------------------------------- 8c
def st_c8_prepare(ctx, run, args):
    """(h12) seed <run>/8c_gate7_128 for the unchanged Gate 7 script."""
    sub = run / "8c_gate7_128"
    sub.mkdir(exist_ok=True)
    copied = []
    for pat in ("truth_*.json", "theory_*.npz", "scan_*.json", "theory_64_vs_gate5.json"):
        for f in sorted(ctx.g7.glob(pat)):
            if not (sub / f.name).exists():
                shutil.copy2(f, sub / f.name)
                copied.append(f.name)
    tm = json.loads((ctx.g7 / "timing.json").read_text())
    tm["decision_gate7"] = tm["decision"]
    tm["decision"] = "full 128 set (Gate 8c: W#3, W#7, W#9 added at 128; gate8_dns.yaml h12)"
    atomic_write_json(sub / "timing.json", tm)
    atomic_write_json(sub / "seeded_from_gate7.json", {"from": str(ctx.g7.relative_to(REPO)), "files": copied, "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"    8c folder seeded: {len(copied)} files")


# ------------------------------------------------------------------------------------------------------------- analysis
def st_analysis(ctx, run, args):
    cfg8 = ctx.cfg8
    C = cfg8["criteria"]
    P = json.loads((run / "predictions_8a.json").read_text())
    pr = {r["ray"]: r for r in P["rays"]}
    fast_below = float(cfg8["truth"]["route_fast_below"])
    res = {"P_star": P["P_star"], "frozen_at": P["frozen_at"]}

    def row(n, blind):
        p = pr[n]
        t = sc96(ctx, run, n)
        r = {"ray": n, "blind": blind, "s_pred": p["s_pred"], "route_pred": p["route"], "s_fast": p["s_fast"], "s_slow": p["s_slow"],
             "s_E": p["s_E_energy_only"]}
        if t is None:
            st = load_truths(run).get("96", {}).get(n, {}).get("status", "no s_c")
            r.update({"s_c_96": None, "status": st})
            return r
        r.update({"s_c_96": t[0], "lo": t[1], "hi": t[2], "t_hit_lo": t[3], "t_hit_hi": t[4], "source": t[5],
                  "err_3cond": math.log(p["s_pred"] / t[0]), "err_2cond": math.log(p["s_slow"] / t[0]), "err_energy": math.log(p["s_E_energy_only"] / t[0]),
                  "err_fast_alone": math.log(p["s_fast"] / t[0]),
                  "route_true": None if t[4] is None else ("fast" if t[4] < fast_below else "slow")})
        r["route_right"] = None if r["route_true"] is None else r["route_true"] == r["route_pred"]
        return r
    blind = [row(n, True) for n in cfg8["rays"]["blind"]]
    ok = [r for r in blind if r["s_c_96"] is not None]
    med = lambda k, rows: float(np.median([abs(r[k]) for r in rows])) if rows else None  # noqa: E731
    m3 = med("err_3cond", ok)
    v = (None if m3 is None else ("third condition holds" if m3 <= float(C["holds_median"]) else
                                 ("does not hold" if m3 >= float(C["fails_median"]) else "partly")))
    res["blind"] = {"rows": blind, "n_used": len(ok), "n_flagged": len(blind) - len(ok), "median_abs_err_3cond": m3,
                    "median_abs_err_2cond": med("err_2cond", ok), "median_abs_err_energy": med("err_energy", ok),
                    "verdict": v, "route_right": sum(bool(r.get("route_right")) for r in ok), "route_n": sum(r.get("route_right") is not None for r in ok)}
    res["X#1"] = row("X#1", True)
    nb = [row(n, False) for n in cfg8["rays"]["not_blind"]]
    sub = run / "8c_gate7_128"
    R7 = json.loads((sub / "results.json").read_text()) if (sub / "results.json").exists() else json.loads((ctx.g7 / "results.json").read_text())
    for r in nb:
        q = R7["rays"].get(r["ray"], {}).get("128", {})
        if q.get("s_c"):
            r["s_c_128"] = q["s_c"]
            r["err_3cond_vs_128"] = math.log(r["s_pred"] / q["s_c"])
    res["not_blind"] = nb
    res["c8"] = {"source": str((sub / "results.json").resolve().relative_to(REPO))  # review fix 18:2x (relative run dir) if (sub / "results.json").exists() else "not run",
                 "verdict": R7.get("verdict"), "controls": R7.get("controls"),
                 "rays": {n: {lv: {k: R7["rays"][n][lv].get(k) for k in ("s_c", "t_hit_hi", "err_i", "dln_vs_prev")} for lv in ("64", "96", "128")
                              if isinstance(R7["rays"][n].get(lv), dict)} for n in R7["rays"]}}
    # 8d summaries
    f1, f2, f3 = run / "d8_theory_kz.npz", run / "d8_diag_kz.npz", run / "d8_gain_alpha.npz"
    if f1.exists() and f2.exists() and f3.exists():
        Th, Dg, Ga = np.load(f1), np.load(f2), np.load(f3)
        tt = [int(x) for x in cfg8["d8"]["report_times"]]
        PP = [str(p) for p in Dg["parts"]]
        d8 = {}
        for i, n in enumerate([str(x) for x in Th["names"]]):
            s = float(Th["s_c"][i])
            th = {}
            for p in ("streak", "roll"):
                A2 = (s ** 2 * Th[f"{p}_aa"][i] + 2 * s ** 3 * Th[f"{p}_ab"][i] + s ** 4 * Th[f"{p}_bb"][i])    # (3, n_t)
                th[p] = {f"t={t}": dict(zip(KZ_BINS, A2[:, t].tolist())) for t in tt}
                cross = np.where(A2[1] > A2[0])[0]
                th[p]["first_t_2gamma_gt_gamma"] = float(Th["t"][cross[0]]) if len(cross) else None
            dn = {}
            for w in ("lo", "hi"):
                j = [k for k in range(len(Dg["names"])) if str(Dg["names"][k]) == n and str(Dg["which"][k]) == w][0]
                rec = Dg["rec"][j]
                o = {}
                for p in ("streak", "roll"):
                    B = np.stack([rec[:, PP.index(f"{p} {b}")] for b in KZ_BINS])
                    o[p] = {f"t={t}": dict(zip(KZ_BINS, B[:, t].tolist())) for t in tt}
                    cross = np.where(B[1] > B[0])[0]
                    o[p]["first_t_2gamma_gt_gamma"] = float(Dg["t"][cross[0]]) if len(cross) else None
                ka = rec[:, PP.index("kx=alpha")]
                kw = ka + rec[:, PP.index("kx>=2alpha")]
                o["linear_wave_vs_dns"] = {f"t={t}": {"dns_kx_alpha": float(ka[t]), "dns_kx_ne_0": float(kw[t]), "lin_y": float(Ga["yy"][i, t]),
                                                      "lin_y_alpha": float(Ga["ya"][i, t]), "ratio_alpha": float(ka[t] / Ga["ya"][i, t])}
                                           for t in [0, 10, 20, 30, 50, 100, 150, 200, 300, 400] if t < len(ka) and np.isfinite(ka[t])}
                o["dt_used"] = float(Dg["dt_used"][j])
                dn[w] = o
            d8[n] = {"s_c_96": s, "theory": th, "dns": dn, "courant_M_lin": float(Ga["courant_M"][i])}
        res["d8"] = d8
    atomic_write_json(run / "results.json", res)
    b = res["blind"]
    log(f"    8b: n {b['n_used']} (flagged {b['n_flagged']}); median |err| 3-cond {b['median_abs_err_3cond']}, 2-cond {b['median_abs_err_2cond']}, "
        f"energy {b['median_abs_err_energy']} -> {b['verdict']}; routes right {b['route_right']}/{b['route_n']}")
    log(f"    X#1: {res['X#1']}")
    log(f"    8c verdict: {res['c8']['verdict']}")


if __name__ == "__main__":
    main()
