#!/usr/bin/env python
"""Gate 9 (DNS): blind test outside the nine patterns. Config: code/configs/gate9_dns.yaml (criteria fixed before any run).
Stages (each resumable; one process at a time):

    python code/scripts/gate9_dns.py --config code/configs/gate9_dns.yaml --run-dir <dir> --stage posthoc8   (Gate 8 post-hoc check)
    ... --stage rays        (9.1: build G, O, V; checks at 64/96/128; the three 128 rays)
    ... --stage calib       (calibration half-widths, h11)
    ... --stage predict     (9.2: frozen predictions; committed and pushed before 9.3)
    ... --stage truth --cls G|O|M|V   (9.3 at 96)
    ... --stage budget --before V|128 (h16; exit code 3 = cut)
    ... --stage truth128    (9.3 grid uncertainty)
    ... --stage analysis    (9.4)
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
from gate6_dns import first_cross  # noqa: E402
from gate7_dns import load_truths, log, sc_at  # noqa: E402
from gate8_dns import Ctx8, gate56_slow  # noqa: E402
from committor.dns_gate8 import wave_gain  # noqa: E402
from committor.dns_gate9 import (binom_two_sided, boot_median_ci, build_rays, conformal_halfwidth, embed, extend_basis,  # noqa: E402
                                 find_sc_grid9, extract)
from committor.dns_local import find_sc_local  # noqa: E402
from committor.dns_theory import A_of, run_delta_q, run_gain  # noqa: E402
from committor.threshold import RayCoordinate  # noqa: E402
from committor.utils import atomic_save_npz, atomic_write_json, git_info, read_json  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EXTRA = [f"G#{i}" for i in range(1, 11)] + [f"O#{i}" for i in range(1, 7)] + [f"V#{i}" for i in range(1, 7)]
INF = math.inf


def fin(x):
    return float(x) if x is not None and np.isfinite(x) else None


class Ctx9(Ctx8):
    def __init__(self, cfg9, run):
        super().__init__(cfg9)
        self.cfg9 = cfg9
        self.run = run
        self.g8 = REPO / cfg9["data"]["gate8_run"]
        f = run / "rays9.npz"
        self.C = None
        if f.exists():
            Z = np.load(f)
            assert [str(x) for x in Z["names"]] == EXTRA
            self.C = Z["C"]

    def names9(self):
        return EXTRA + [f"M#{i}" for i in range(1, 7)]

    def cls(self, name):
        return name.split("#")[0]

    def grid(self, level):
        g, dt = super().grid(level)
        if self.C is not None:
            extend_basis(g, list(self.C))
        return g, dt

    def dir(self, name):
        n_extra = 0 if self.C is None else len(EXTRA)
        v = np.zeros(9 + n_extra)
        c = self.cls(name)
        if c in ("G", "O", "V"):
            assert self.C is not None
            v[9 + EXTRA.index(name)] = 1.0
        elif c == "M":
            th = math.radians(float(self.cfg9["rays9"]["M"]["theta_deg"][int(name.split("#")[1]) - 1]))
            v[2], v[7] = math.cos(th), math.sin(th)                         # (h5)
        else:
            v[:9] = super().dir(name)
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--cls", default=None)
    ap.add_argument("--before", default=None)
    ap.add_argument("--max-batch", type=int, default=64)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    if not (run / "config_used.yaml").exists():
        (run / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    tag = args.stage + (f"_{args.cls}" if args.cls else "") + (f"_{args.before}" if args.before else "")
    mf = run / f"manifest_{tag}.json"
    man = read_json(mf, {"invocations": []})
    man["invocations"].append({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv, "git": git_info(REPO),
                               "python": platform.python_version(), "torch": torch.__version__,
                               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    atomic_write_json(mf, man)
    ctx = Ctx9(cfg, run)
    t0 = time.time()
    rc = {"posthoc8": st_posthoc8, "rays": st_rays, "calib": st_calib, "predict": st_predict, "truth": st_truth, "budget": st_budget,
          "truth128": st_truth128, "analysis": st_analysis}[args.stage](ctx, run, args)
    log(f"--- {tag}: {(time.time() - t0) / 60:.1f} min")
    if rc:
        sys.exit(int(rc))


# ----------------------------------------------------------------------------------------------------------- truths
def best_truth(ctx, n):
    """(s_c, level, source) at the highest level available (h11, h17)."""
    R8c = json.loads((ctx.g8 / "8c_gate7_128" / "results.json").read_text())["rays"]
    q = R8c.get(n, {}).get("128")
    if isinstance(q, dict) and q.get("s_c"):
        return float(q["s_c"]), "128", "Gate 8c / Gate 7 (128)"
    T8 = {r["name"]: r for r in json.loads((ctx.g8 / "truth_96_blind.json").read_text())["rays"]}
    if n in T8 and T8[n]["s_c"] is not None:
        return float(T8[n]["s_c"]), "96", "Gate 8 (96)"
    if n[0] in "RW":
        t = sc_at(ctx, ctx.g7, "96", n)
        if t is not None:
            return float(t[0]), "96", t[5]
        return float(ctx.t5["s_c"][ctx.idx5(n)]), "64", "Gate 5 (64)"
    if n.startswith("F"):
        A5 = np.load(ctx.g5 / "analysis.npz")
        return float(A5["sc"][int(n.split("#")[1]) - 1]), "64", "Gate 5 (64)"
    raise KeyError(n)


def slow_F(ctx):
    """s_i, s_ii, flag for the F rays from Gates 5/6 (matched by eps and draw)."""
    A5 = np.load(ctx.g5 / "analysis.npz")
    RF = json.loads((ctx.g6 / "results.json").read_text())["analysis"]["rays_F"]
    out = {}
    for k in range(ctx.nF):
        m = ctx.F_meta[k]
        r = [x for x in RF if abs(x["eps"] - m["eps"]) <= 1e-12 * max(1.0, m["eps"]) and x["draw"] == m["draw"]]
        assert len(r) == 1, (k, m)
        sc64 = float(A5["sc"][k])
        out[f"F#{k + 1}"] = {"eps": float(m["eps"]), "draw": int(m["draw"]), "s_c_64": sc64, "s_i": math.exp(float(A5["pred_S"][k])),
                             "s_ii": sc64 * math.exp(float(r[0]["ln_err_ii"])), "ii_flag": r[0]["ii_flag"], "used_for_c_star": bool(r[0]["used_for_c_star"])}
    return out


def P_star(ctx):
    return float(json.loads((REPO / ctx.cfg9["constants"]["P_star_source"]).read_text())["P_star"])


def rule9(G, Ew0, s_slow, Pst, g_min):
    """9.0 rule (+ h9 edge cases)."""
    g = G / Ew0 if Ew0 > 0 else 0.0
    fast = bool(Ew0 > 0 and g >= g_min)
    s_fast = math.sqrt(Pst / G) if fast else INF
    s_pred = min(s_fast, s_slow)
    return {"g": g, "fast_exists": fast, "s_fast": s_fast, "s_pred": s_pred, "route": "fast" if s_fast < s_slow else "slow"}


# ----------------------------------------------------------------------------------------------------------- posthoc8
def st_posthoc8(ctx, run, args):
    """(h17) the user's Gate 8 observations, descriptive."""
    R8 = json.loads((ctx.g8 / "results.json").read_text())
    P8 = {r["ray"]: r for r in json.loads((ctx.g8 / "predictions_8a.json").read_text())["rays"]}
    e8 = [abs(r["err_3cond"]) for r in R8["blind"]["rows"]]
    nb = int(ctx.cfg9["stats"]["n_boot"])
    out = {"blind8": {"abs_err": e8, "median": float(np.median(e8)), "boot95": boot_median_ci(e8, nb, int(ctx.cfg9["seed"]))}}
    rows = []
    for n in [f"R#{k}" for k in range(13, 25)] + [f"W#{k}" for k in range(1, 13)] + ["X#1"]:
        s, lv, src = best_truth(ctx, n)
        sp = P8[n]["s_pred"]
        rows.append({"ray": n, "s_pred_8a": sp, "s_c": s, "level": lv, "source": src, "err": math.log(sp / s), "abs_err": abs(math.log(sp / s)),
                     "le_0p03": bool(sp <= 0.03)})
    a = [r for r in rows if r["le_0p03"]]
    b = [r for r in rows if not r["le_0p03"]]
    out["rays25"] = {"rows": rows, "n_le": len(a), "n_gt": len(b),
                     "le": {"median_abs": float(np.median([r["abs_err"] for r in a])), "max_abs": float(max(r["abs_err"] for r in a))},
                     "gt": {"abs_sorted": sorted(r["abs_err"] for r in b), "signs": [int(np.sign(r["err"])) for r in b]}}
    atomic_write_json(run / "posthoc8.json", out)
    log(f"    8-ray median {out['blind8']['median']:.3f}, bootstrap 95% {out['blind8']['boot95']}")
    log(f"    25 rays: s_pred <= 0.03 n {len(a)}, median |err| {out['rays25']['le']['median_abs']:.4f}, max {out['rays25']['le']['max_abs']:.4f}; "
        f"> 0.03 n {len(b)}: {[round(x, 3) for x in out['rays25']['gt']['abs_sorted']]} signs {out['rays25']['gt']['signs']}")


# ----------------------------------------------------------------------------------------------------------- 9.1 rays
def st_rays(ctx, run, args):
    f = run / "rays9.npz"
    if f.exists():
        log("    rays9.npz exists; not rebuilt")
        return
    spec = ctx.cfg9["rays9"]
    g, _ = Ctx8.grid(ctx, spec["reference_level"])                     # plain grid (nine modes only)
    R = build_rays(g, spec)
    C = np.stack([R[n]["C"] for n in EXTRA])
    meta = {n: {k: v for k, v in R[n].items() if k != "C"} for n in EXTRA}
    # checks at every level (h1): energy, divergence, symmetry, nine-mode fraction of the embedded field
    for lv in ("64", "96", "128"):
        gl, _ = Ctx8.grid(ctx, lv)
        fh = embed(gl, C)
        E = gl.energy(fh).cpu().numpy()
        dv = gl.max_div(fh).cpu().numpy()
        sy = gl.sym_residual(fh).cpu().numpy()
        cl = gl.energy(gl.clean(fh) - fh).cpu().numpy()
        a = gl.coeffs(fh).cpu().numpy()
        for i, n in enumerate(EXTRA):
            meta[n][f"check_{lv}"] = {"E": float(E[i]), "max_div": float(dv[i]), "sym_residual": float(sy[i]), "clean_change": float(cl[i]),
                                      "nine_mode_fraction": float(np.sum(a[i] ** 2))}
        del gl, fh
        torch.cuda.empty_cache()
    # the three 128 rays (h13), with fallbacks in |k|^2 order
    order = {c: sorted([n for n in EXTRA if n.startswith(c)], key=lambda n: -meta[n]["k2_mean"]) for c in ("G", "V")}
    sel = {"O": ["O#6"] + sorted([n for n in EXTRA if n.startswith("O") and n != "O#6"], key=lambda n: -meta[n]["k2_mean"]), "G": order["G"], "V": order["V"]}
    atomic_save_npz(f, C=C, names=np.array(EXTRA))
    atomic_write_json(run / "rays9.json", {"rays": meta, "select128_order": sel, "built_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    for n in EXTRA:
        m = meta[n]
        log(f"    {n}: seed {m['seed']} (tries {len(m.get('tries', [1]))}), nine-mode {m['nine_mode_fraction']:.4f}, |k|^2 {m['k2_mean']:.3f}, "
            f"E64/96/128 {m['check_64']['E']:.15f}/{m['check_96']['E']:.15f}/{m['check_128']['E']:.15f}, div {m['max_div']:.1e}, "
            f"sym {m['sym_residual']:.1e}, outside band {m['energy_outside_band']:.1e}")
    log(f"    128 rays (h13): O#6, {sel['G'][0]}, {sel['V'][0]}")


# ----------------------------------------------------------------------------------------------------------- calibration
def st_calib(ctx, run, args):
    Pst = P_star(ctx)
    gmin = float(ctx.cfg9["rule"]["g_min"])
    rmax = float(ctx.cfg9["rule"]["range_max"])
    Z = np.load(ctx.g8 / "g_all.npz")
    gnames = [str(x) for x in Z["names"]]
    slowRW = gate56_slow(ctx)
    sF = slow_F(ctx)
    x1 = json.loads((ctx.g8 / "x1_gate56.json").read_text())
    CB = ctx.cfg9["calibration"]
    Fset = [n for n, v in sF.items() if not any(abs(v["eps"] - e) <= 1e-12 for e in CB["slow"]["F_exclude_eps"])]
    sets = {"slow": Fset + list(CB["slow"]["R"]), "fast": list(CB["fast"])}
    out = {"P_star": Pst, "sets": {}}
    for route, names in sets.items():
        rows = []
        for n in names:
            i = gnames.index(n)
            G96, Ew0 = float(Z["G_96"][i]), float(Z["series_96"][i, 0])
            if n.startswith("F"):
                sl = sF[n]
            elif n == "X#1":
                sl = {"s_i": x1["s_i"], "s_ii": x1["s_ii"], "ii_flag": x1["ii_flag"]}
            else:
                sl = slowRW[n]
            s_slow = max(sl["s_i"], sl["s_ii"])
            r9 = rule9(G96, Ew0, s_slow, Pst, gmin)
            s, lv, src = best_truth(ctx, n)
            rows.append({"ray": ctx.fname(n) if n.startswith("F") else n, "key": n, "G96": G96, "Ew0": Ew0, "g": r9["g"], "s_fast": fin(r9["s_fast"]),
                         "s_slow": s_slow, "s_pred": r9["s_pred"], "route9": r9["route"], "in_range": bool(r9["s_pred"] <= rmax),
                         "s_c": s, "level": lv, "source": src, "abs_err": abs(math.log(s / r9["s_pred"])),
                         "matches_set": bool(r9["route"] == route and r9["s_pred"] <= rmax)})
        ae = [r["abs_err"] for r in rows]
        h = {f"h{int(round(100 * p))}": conformal_halfwidth(ae, p) for p in ctx.cfg9["rule"]["p_levels"]}
        out["sets"][route] = {"n": len(rows), "rows": rows, **h, "sorted_abs_err": sorted(ae),
                              "mismatches": [r["ray"] for r in rows if not r["matches_set"]], "user": CB["user_values"][route]}
        log(f"    calib {route}: n {len(rows)}, h50 {h['h50']:.4f}, h80 {h['h80']:.4f} (user {CB['user_values'][route]}); "
            f"mismatches {out['sets'][route]['mismatches']}")
    atomic_write_json(run / "calib.json", out)


# ----------------------------------------------------------------------------------------------------------- 9.2 predict
def budget_start(run):
    f = run / "budget.json"
    B = read_json(f, None)
    if B is None:
        B = {"start": time.strftime("%Y-%m-%d %H:%M:%S"), "start_epoch": time.time(), "decisions": []}
        atomic_write_json(f, B)
    return B


def st_predict(ctx, run, args):
    out_f = run / "predictions_9.json"
    if out_f.exists():
        log("    predictions_9.json exists (frozen); not recomputed")
        return
    budget_start(run)
    cal = json.loads((run / "calib.json").read_text())
    C9 = ctx.cfg9
    TH = C9["theory"]
    names = ctx.names9()
    D = np.array([ctx.dir(n) for n in names])
    S_star, c_star = float(C9["constants"]["S_star"]), float(C9["constants"]["c_star"])
    Pst = P_star(ctx)
    g, dt = ctx.grid(TH["level"])
    f1 = run / "theory_64.npz"
    if not f1.exists():
        r = run_delta_q(g, D, float(TH["T_S"]), dt, every=float(TH["output_every"]), progress=True, desc="9.2 theory 64")
        atomic_save_npz(f1, aa=r["aa"], ab=r["ab"], bb=r["bb"], t=r["t"], names=np.array(names))
        log("    theory 64 done")
    T1 = np.load(f1)
    sg = np.sqrt(np.geomspace(ctx.r_A * (1 + 1e-9), ctx.E_B * (1 - 1e-9), int(TH["s_grid_points"])))
    S = A_of(T1["aa"], T1["ab"], T1["bb"], sg).max(axis=2)                       # (R, n_s)
    ln_i = RayCoordinate(S, sg, "cpu").ln_s_pred([S_star], 1)[0]
    ln_i = np.asarray(ln_i.cpu() if torch.is_tensor(ln_i) else ln_i, dtype=np.float64)[:, 0]
    GN = TH["gain"]
    K, hw = int(GN["n_amp"]), float(GN["ln_half_width"])
    offs = np.linspace(-hw, hw, K)
    step = offs[1] - offs[0]
    center = np.where(np.isfinite(ln_i), ln_i, math.log(sg[-1]))

    def cgrid(idx, L, tag):
        rr = run_gain(g, D[idx], np.exp(L), float(GN["T_end"]), dt, every=float(GN["output_every"]), progress=True, desc=f"9.2 gain 64 {tag}")
        with np.errstate(divide="ignore"):
            return np.log(rr["yy"].max(axis=2)), rr["yy"].argmax(axis=2) * float(GN["output_every"]), rr["courant_M"]
    f2 = run / "gain_64_block1.npz"
    if not f2.exists():
        L1 = center[:, None] + offs[None]
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
    ii = {}
    gain_rec = {}
    B2 = np.load(f3) if f3.exists() else None
    for i, n in enumerate(names):
        L, Cc = B1["lns"][i], B1["c"][i]
        blocks = [{"lns": L.tolist(), "c": Cc.tolist(), "t_at_max": B1["t_at_max"][i].tolist(), "courant_M_max": float(np.max(B1["courant_M"][i]))}]
        ln_ii, fl = first[i]
        if B2 is not None and i in list(B2["idx"]):
            j = list(B2["idx"]).index(i)
            L2, C2 = B2["lns"][j], B2["c"][j]
            blocks.append({"lns": L2.tolist(), "c": C2.tolist(), "t_at_max": B2["t_at_max"][j].tolist(), "courant_M_max": float(np.max(B2["courant_M"][j]))})
            LL = np.concatenate([L2, L]) if fl == "below" else np.concatenate([L, L2])
            CC = np.concatenate([C2, Cc]) if fl == "below" else np.concatenate([Cc, C2])
            ln_ii, fl2 = first_cross(LL, CC, c_star)
            fl = fl2 + " (after one extension)"
        ii[n] = (math.exp(ln_ii), fl)
        gain_rec[n] = blocks
    atomic_write_json(run / "gain_64.json", gain_rec)
    # wave gain at 64 and 96 (h8)
    f4 = run / "wave_gain.npz"
    if not f4.exists():
        out = {}
        for lv in ("64", "96"):
            gl, dtl = ctx.grid(lv)
            r = wave_gain(gl, D, T=float(C9["rule"]["T_G"]), dt=dtl, every=1.0)
            out[f"G_{lv}"], out[f"tmax_{lv}"], out[f"series_{lv}"] = r["G"], r["t_at_max"], r["series"]
            del gl
            torch.cuda.empty_cache()
        atomic_save_npz(f4, names=np.array(names), **out)
    WG = np.load(f4)
    a9 = g.coeffs(g.clean(g.from_coeffs(D))).cpu().numpy()[:, :9]
    R9 = json.loads((run / "rays9.json").read_text())["rays"]
    rmax, bnd, gmin = float(C9["rule"]["range_max"]), float(C9["rule"]["out_of_range_bound"]), float(C9["rule"]["g_min"])
    sE = math.exp(float(np.load(ctx.g5 / "analysis.npz")["pred_E"][ctx.nF]))          # constant (h10)
    rows = []
    for i, n in enumerate(names):
        s_i = math.exp(ln_i[i]) if np.isfinite(ln_i[i]) else INF
        s_ii, iif = ii[n]
        s_slow = max(s_i, s_ii)
        G96, G64, Ew0 = float(WG["G_96"][i]), float(WG["G_64"][i]), float(WG["series_96"][i, 0])
        r9 = rule9(G96, Ew0, s_slow, Pst, gmin)
        sp = r9["s_pred"]
        inr = bool(np.isfinite(sp) and sp <= rmax)
        row = {"ray": n, "cls": ctx.cls(n), "nine_mode_fraction": float(np.sum(a9[i] ** 2)), "s_i": fin(s_i), "s_i_flag": "ok" if np.isfinite(s_i) else "no crossing",
               "S_max_on_grid": float(S[i].max()), "s_ii": s_ii, "ii_flag": iif, "s_slow": fin(s_slow), "G_64": G64, "G_96": G96, "G_96_over_64": G96 / G64,
               "t_at_max_96": float(WG["tmax_96"][i]), "E_w0": Ew0, "g": r9["g"], "fast_exists": r9["fast_exists"], "s_fast": fin(r9["s_fast"]),
               "s_pred": fin(sp), "route": r9["route"], "in_range": inr, "no_prediction": bool(not np.isfinite(sp)),
               "baselines": {"gate8a": fin(min(math.sqrt(Pst / G96), s_slow)), "two_cond": fin(s_slow), "energy": sE}}
        if inr:
            cs = cal["sets"][r9["route"]]
            row["intervals"] = {f"p{int(round(100 * p))}": [sp * math.exp(-cs[f"h{int(round(100 * p))}"]), sp * math.exp(cs[f"h{int(round(100 * p))}"])]
                                for p in C9["rule"]["p_levels"]}
        else:
            row["lower_bound"] = fin(bnd * sp)
        if n in R9:
            m = R9[n]
            row.update({"seed": m["seed"], "k2_mean": m["k2_mean"], **({"k": m["k"]} if "k" in m else {}), **({"l": m["l"]} if "l" in m else {})})
        else:
            row["theta_deg"] = float(C9["rays9"]["M"]["theta_deg"][int(n.split("#")[1]) - 1])
        rows.append(row)
    res = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "P_star": Pst, "S_star": S_star, "c_star": c_star, "s_E": sE,
           "calibration": {k: {"n": v["n"], "h50": v["h50"], "h80": v["h80"]} for k, v in cal["sets"].items()}, "rays": rows}
    atomic_write_json(out_f, res)
    f = lambda x: "—" if x is None else f"{x:.4g}"  # noqa: E731
    lines = ["| ray | 9-mode frac | s_i | s_ii | s_slow | G_96 (G96/G64) | E_w(0) | g | s_fast | **s_pred** | route | range | 50% / 80% interval or bound | 8a rule | 2-cond |",
             "|" + "---|" * 15]
    for r in rows:
        iv = (f"[{r['intervals']['p50'][0]:.4g}, {r['intervals']['p50'][1]:.4g}] / [{r['intervals']['p80'][0]:.4g}, {r['intervals']['p80'][1]:.4g}]"
              if r["in_range"] else (f"s_c >= {r['lower_bound']:.4g}" if r.get("lower_bound") else "no prediction"))
        lines.append(f"| {r['ray']} | {r['nine_mode_fraction']:.3f} | {f(r['s_i'])} | {f(r['s_ii'])} ({r['ii_flag']}) | {f(r['s_slow'])} | {r['G_96']:.4g} "
                     f"({r['G_96_over_64']:.6f}) | {r['E_w0']:.3g} | {r['g']:.3g} | {f(r['s_fast'])} | **{f(r['s_pred'])}** | {r['route']} | "
                     f"{'in' if r['in_range'] else 'out'} | {iv} | {f(r['baselines']['gate8a'])} | {f(r['baselines']['two_cond'])} |")
    (run / "predictions_9.md").write_text(f"# Gate 9.2 frozen predictions ({res['frozen_at']})\n\nP* = {Pst:.6g}; S* = {S_star:.6g}; c* = {c_star:.6g}; "
                                          f"energy-only baseline {sE:.6g} (all rays). Calibration half-widths: "
                                          + "; ".join(f"{k} n {v['n']} h50 {v['h50']:.4f} h80 {v['h80']:.4f}" for k, v in res["calibration"].items())
                                          + ".\n\n" + "\n".join(lines) + "\n")
    log("    9.2 frozen: " + "; ".join(f"{r['ray']} {f(r['s_pred'])} {r['route']} {'in' if r['in_range'] else 'out'}" for r in rows))


# ----------------------------------------------------------------------------------------------------------- 9.3 truth
def st_truth(ctx, run, args):
    assert args.cls in ("G", "O", "M", "V")
    assert (run / "PUSHED_9.txt").exists(), "9.2 not pushed"
    P = json.loads((run / "predictions_9.json").read_text())
    rows = [r for r in P["rays"] if r["cls"] == args.cls]
    names = [r["ray"] for r in rows]
    T = ctx.cfg9["truth"]
    center = [r["s_pred"] if r["s_pred"] is not None else float(T["no_prediction_center"]) for r in rows]
    g, dt = ctx.grid(T["level"])
    D = np.array([ctx.dir(n) for n in names])
    log(f"--- truth 96 {args.cls}: {names}; centres {[round(c, 6) for c in center]}")
    find_sc_grid9(g, D, names, center, T["factors"], T["ext_up"], T["ext_down"], ctx.r_A, ctx.E_B, ctx.T_det, dt, float(T["rel_width"]),
                  ctx.climit, ctx.lv, args.max_batch, ckpt=run / f"truth_96_{args.cls}.json", log=log, desc=f"s_c 96 {args.cls}")


def st_budget(ctx, run, args):
    """(h16) exit code 3 = cut."""
    B = json.loads((run / "budget.json").read_text())
    BG = ctx.cfg9["budget"]
    el = (time.time() - B["start_epoch"]) / 3600.0
    e128 = float(BG["est_128_hours"])
    if args.before == "V":
        done = [json.loads((run / f"truth_96_{c}.json").read_text()) for c in ("G", "O", "M") if (run / f"truth_96_{c}.json").exists()]
        sec = sum(d.get("seconds", 0.0) for d in done)
        nr = sum(len(d["rays"]) for d in done)
        eV = 6 * sec / max(nr, 1) / 3600.0
        cut = el + eV + e128 > float(BG["hours"])
        dec = {"before": "V", "at": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_h": el, "est_V_h": eV, "est_128_h": e128, "cut": cut}
    else:
        cut = el + e128 > float(BG["hours"])
        dec = {"before": "128", "at": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_h": el, "est_128_h": e128, "cut": cut}
    B["decisions"].append(dec)
    atomic_write_json(run / "budget.json", B)
    log(f"    budget: {dec}")
    return 3 if cut else 0


def truths96(run):
    out = {}
    for c in ("G", "O", "M", "V"):
        f = run / f"truth_96_{c}.json"
        if f.exists():
            st = json.loads(f.read_text())
            if st.get("phase") == "done":
                for r in st["rays"]:
                    out[r["name"]] = r
    return out


def st_truth128(ctx, run, args):
    T96 = truths96(run)
    sel = json.loads((run / "rays9.json").read_text())["select128_order"]
    names = []
    for c in ("O", "G", "V"):
        cand = [n for n in sel[c] if n in T96 and T96[n].get("s_c") is not None]
        if cand:
            names.append(cand[0])
        else:
            log(f"    128: no {c} ray with s_c(96) (cut or flagged); skipped")
    T = ctx.cfg9["truth128"]
    g, dt = ctx.grid(T["level"])
    D = np.array([ctx.dir(n) for n in names])
    s_prev = [T96[n]["s_c"] for n in names]
    log(f"--- truth 128: {names}; s_prev {[round(s, 6) for s in s_prev]}")
    find_sc_local(g, D, names, s_prev, T["factors"], T["ext_up"], T["ext_down"], ctx.r_A, ctx.E_B, ctx.T_det, dt, float(ctx.cfg9["truth"]["rel_width"]),
                  ctx.climit, ctx.lv, args.max_batch, ckpt=run / "truth_128.json", log=log, desc="s_c 128")


# ----------------------------------------------------------------------------------------------------------- 9.4 analysis
def stats_block(err, nb, seed):
    e = np.asarray(err, dtype=np.float64)
    if len(e) == 0:
        return {"n": 0}
    a = np.abs(e)
    return {"n": int(len(e)), "median": float(np.median(e)), "q25": float(np.quantile(e, 0.25)), "q75": float(np.quantile(e, 0.75)),
            "IQR": float(np.quantile(e, 0.75) - np.quantile(e, 0.25)), "abs_q90": float(np.quantile(a, 0.9)), "abs_max": float(a.max()),
            "abs_median": float(np.median(a)), "abs_median_boot95": boot_median_ci(a, nb, seed)}


def st_analysis(ctx, run, args):
    C9 = ctx.cfg9
    P = json.loads((run / "predictions_9.json").read_text())
    T96 = truths96(run)
    T128 = {}
    f128 = run / "truth_128.json"
    if f128.exists() and json.loads(f128.read_text()).get("phase") == "done":
        T128 = {r["name"]: r for r in json.loads(f128.read_text())["rays"]}
    nb, seed = int(C9["stats"]["n_boot"]), int(C9["seed"])
    fast_below = float(C9["truth"]["route_fast_below"])
    # grid differences per class (h14)
    dgrid = {}
    for n, r in T128.items():
        if r.get("s_c") is not None and T96.get(n, {}).get("s_c") is not None:
            dgrid[ctx.cls(n)] = {"ray": n, "dln": math.log(r["s_c"] / T96[n]["s_c"]), "s_c_128": r["s_c"], "s_c_96": T96[n]["s_c"],
                                 "t_hit_hi_128": r.get("t_hit_hi")}
    G9 = C9["truth128"]["grid_9mode"]
    rows = []
    for p in P["rays"]:
        n = p["ray"]
        t = T96.get(n)
        r = {"ray": n, "cls": p["cls"], "s_pred": p["s_pred"], "route_pred": p["route"], "in_range": p["in_range"], "pred": p}
        if t is None:
            r["status"] = "not run"
        else:
            r.update({"status": t["status"], "fates": t.get("fates"), "non_monotone": t.get("non_monotone"), "grid_status": t.get("grid_status"),
                      "n_switches": t.get("n_switches")})
        r["no_prediction"] = p["s_pred"] is None
        if t is not None and t.get("s_c") is not None:
            sc = t["s_c"]
            hb = math.log(t["hi"] / t["lo"]) / 2
            if p["cls"] in dgrid:
                dg, dsrc = abs(dgrid[p["cls"]]["dln"]), f"128 ray {dgrid[p['cls']]['ray']}"
            else:
                dg, dsrc = float(G9[p["route"]]), f"nine-mode 96->128 ({p['route']})"
            r.update({"s_c": sc, "lo": t["lo"], "hi": t["hi"], "t_hit_lo": t["t_hit_lo"], "t_hit_hi": t["t_hit_hi"], "h_b": hb, "d_grid": dg, "d_grid_source": dsrc,
                      "u": math.sqrt(hb ** 2 + dg ** 2),
                      "route_true": None if t["t_hit_hi"] is None else ("fast" if t["t_hit_hi"] < fast_below else "slow"),
                      "s_c_128": T128.get(n, {}).get("s_c")})
        elif t is not None and str(t["status"]).startswith("not bracketed"):                         # (h18) censored truth
            c0 = t["s_center"]
            if t["status"].endswith("(all A)"):
                r["censored"] = {"side": ">", "bound": c0 * max(C9["truth"]["ext_up"])}
            else:
                r["censored"] = {"side": "<", "bound": c0 * min(C9["truth"]["ext_down"])}
            if p["s_pred"] is not None:
                r["censored"]["ln_err_bound"] = math.log(p["s_pred"] / r["censored"]["bound"])
                if p["in_range"]:
                    r["cover"] = {k: False for k in p["intervals"]}                                    # every half-width < ln 8
                elif p.get("lower_bound") is not None:
                    r["bound_ok"] = r["censored"]["side"] == ">"                                          # > 8 s_pred passes; < 0.177 s_pred fails
        if "s_c" in r and p["s_pred"] is not None:
            sc = r["s_c"]
            r.update({"err": math.log(p["s_pred"] / sc), "err_base": {k: (math.log(v / sc) if v is not None else None) for k, v in p["baselines"].items()}})
            if p["in_range"]:
                r["cover"] = {k: bool(v[0] <= sc <= v[1]) for k, v in p["intervals"].items()}
            else:
                r["bound_ok"] = (bool(sc >= p["lower_bound"]) if p.get("lower_bound") is not None else None)
            r["M_xy"] = ({"x": sc / p["s_slow"] if p["s_slow"] else None, "y": sc / p["s_fast"] if p["s_fast"] else None} if p["cls"] == "M" else None)
        rows.append(r)
    ok = [r for r in rows if "err" in r]
    cens = [r for r in rows if "censored" in r]
    groups = {}
    for gname in ("G", "O", "V", "M", "all"):
        for rng in ("in", "out"):
            inG = lambda r: (gname == "all" or r["cls"] == gname) and (r["in_range"] == (rng == "in")) and not r["no_prediction"]  # noqa: E731
            sel = [r for r in ok if inG(r)]
            csel = [r for r in cens if inG(r)]
            blk = {"rays": [r["ray"] for r in sel], "rule": stats_block([r["err"] for r in sel], nb, seed),
                   "censored": [{"ray": r["ray"], **r["censored"]} for r in csel]}
            for b in ("gate8a", "two_cond", "energy"):
                pairs = [(r["err"], r["err_base"][b]) for r in sel if r["err_base"][b] is not None]
                blk[b] = stats_block([x for _, x in pairs], nb, seed)
                blk[b]["n_no_prediction"] = len(sel) - len(pairs)
                d = [abs(e) - abs(x) for e, x in pairs]
                blk[f"paired_vs_{b}"] = {"median": float(np.median(d)) if d else None, "boot95": boot_median_ci(d, nb, seed) if d else None,
                                         "n_rule_better": int(sum(x < 0 for x in d)), "n": len(d)}
            if rng == "in":
                cv = sel + [r for r in csel if "cover" in r]
                for pk, pv in (("p50", 0.5), ("p80", 0.8)):
                    k = sum(r["cover"][pk] for r in cv)
                    blk[f"coverage_{pk}"] = {"k": int(k), "n": len(cv), "rate": (k / len(cv)) if cv else None, "binom_p": binom_two_sided(k, len(cv), pv),
                                             "n_censored_in_n": len(cv) - len(sel)}
            else:
                bb = [r["bound_ok"] for r in sel + csel if r.get("bound_ok") is not None]
                blk["bound"] = {"k": int(sum(bb)), "n": len(bb)}
            groups[f"{gname}/{rng}"] = blk
    routes = {}
    for a in ("fast", "slow"):
        for b in ("fast", "slow"):
            routes[f"pred {a} / true {b}"] = sum(1 for r in ok if r["route_pred"] == a and r.get("route_true") == b)
    res = {"rows": rows, "groups": groups, "routes": routes, "grid_128": dgrid, "budget": read_json(run / "budget.json", None),
           "flagged": [{"ray": r["ray"], "status": r["status"]} for r in rows if r["status"] not in ("ok",)],
           "no_prediction": [{"ray": r["ray"], "s_c": r.get("s_c")} for r in rows if r["no_prediction"]],
           "non_monotone": [r["ray"] for r in rows if r.get("non_monotone")],
           "t_hit_hi": {r["ray"]: r.get("t_hit_hi") for r in ok}}
    atomic_write_json(run / "results.json", res)
    A = groups["all/in"]
    log(f"    9.4: rays with s_c {len(ok)} / {len(rows)}; all/in n {A['rule']['n']}, |err| median {A['rule'].get('abs_median')}, "
        f"coverage 50% {A.get('coverage_p50')}, 80% {A.get('coverage_p80')}; routes {routes}; flagged {res['flagged']}")


if __name__ == "__main__":
    main()
