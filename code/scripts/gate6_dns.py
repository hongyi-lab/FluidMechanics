#!/usr/bin/env python
"""Gate 6 (DNS): the second gate (seed x gain about the second-order mean flow) and the composite criterion max(s_i, s_ii).
Config: code/configs/gate6_dns.yaml (criteria fixed before any run); rays, s_c and S* from the Gate 5 run.

    python code/scripts/gate6_dns.py --config code/configs/gate6_dns.yaml --cfg5 code/configs/gate5_dns.yaml \
        --gate4 code/configs/gate4_dns.yaml --gate5-run <dir> --run-dir <dir> --stage gain_F|gain_RW|analysis|all
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

from committor.dns import Grid
from committor.dns_rays import build_rays
from committor.dns_theory import run_gain
from committor.threshold import summarize
from committor.utils import atomic_save_npz, atomic_write_json, git_info, read_json

REPO = Path(__file__).resolve().parents[2]
SHARED = ["mirror y about 1 (u,-v,w)(x,2-y,z)", "shift-reflect (u,v,-w)(x+Lx/2,y,-z)", "shift-rotate (-u,-v,w)(-x,-y,z+Lz/2)"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fmt(v):
    """JSON-safe signed error: finite -> float, +-inf -> explicit string, NaN -> None."""
    v = float(v)
    if np.isnan(v):
        return None
    return v if np.isfinite(v) else ("+inf (no crossing)" if v > 0 else "-inf")


def first_cross(lns, c, cstar):
    """first crossing of c >= cstar along increasing ln s (linear interpolation in (ln s, c)); clamps (g2)."""
    above = c >= cstar
    if above[0]:
        return lns[0], "below"
    if not above.any():
        return lns[-1], "never"
    j = int(np.argmax(above))
    w = (cstar - c[j - 1]) / (c[j] - c[j - 1])
    return lns[j - 1] + w * (lns[j] - lns[j - 1]), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cfg5", required=True)
    ap.add_argument("--gate4", required=True)
    ap.add_argument("--gate5-run", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stage", default="all")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg5 = yaml.safe_load(Path(args.cfg5).read_text())
    cfg4 = yaml.safe_load(Path(args.gate4).read_text())
    g5 = Path(args.gate5_run)
    ch = json.loads((g5 / "gate4b_choice_used.json").read_text())
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    if not (run / "config_used.yaml").exists():
        (run / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    man = read_json(run / "manifest.json", {"invocations": []})
    man["invocations"].append({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv, "git": git_info(REPO),
                               "python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})
    atomic_write_json(run / "manifest.json", man)
    res = read_json(run / "results.json", {}) or {}

    def save():
        atomic_write_json(run / "results.json", res)

    Re, N, dt = float(ch["Re"]), ch["grid"], float(ch["dt"])
    S = cfg4["system"]
    g = Grid(*N, Re=Re, Lx=S["Lx_over_pi"] * math.pi, Lz=S["Lz_over_pi"] * math.pi, Ly=S["Ly_ext"], device=cfg4["device"])
    g.set_enforced(SHARED)
    rays = build_rays(cfg5, REPO)
    F_d, F_meta, _ = rays["F"]
    R_d, R_meta, _ = rays["R"]
    W_d, W_meta, _ = rays["W"]
    TF, TRW = np.load(g5 / "truth_F.npz"), np.load(g5 / "truth_RW.npz")
    scF = TF["s_c"]
    scR, scW = TRW["s_c"][:len(R_d)], TRW["s_c"][len(R_d):]
    GN = cfg["gain"]
    K = int(GN["n_amp"])
    offs = np.linspace(-float(GN["ln_half_width"]), float(GN["ln_half_width"]), K)
    n_cal = int(cfg5["rays"]["R"]["n_cal"])
    stages = ["gain_F", "analysis_F", "gain_RW", "analysis"] if args.stage == "all" else [args.stage]   # g5: F first (6.1/6.2), then 6.3

    def gains(dirs, sc, tag):
        ok = np.isfinite(sc)
        idx = np.where(ok)[0]
        amps = np.exp(np.log(sc[idx])[:, None] + offs[None, :])
        t0 = time.time()
        _ = run_gain(g, dirs[idx[:2]], amps[:2], 10.0, dt, every=float(GN["output_every"]))
        est = (time.time() - t0) / 10.0 * float(GN["T_end"]) * len(idx) / 2
        log(f"    1% timing: estimate {est / 60:.1f} min for {len(idx)} rays x {K} amplitudes (batches of 2 rays in the timing)")
        r = run_gain(g, dirs[idx], amps, float(GN["T_end"]), dt, every=float(GN["output_every"]), max_rays=16, progress=True)
        yy = r["yy"]
        with np.errstate(divide="ignore"):
            c = np.log(yy.max(axis=2))
        tmax = yy.argmax(axis=2) * float(GN["output_every"])
        atomic_save_npz(run / f"gain_{tag}.npz", idx=idx, amps=amps, c=c, t_at_max=tmax, yy_sub=yy[:, :, ::10], courant_M=r["courant_M"])
        return {"n_rays": int(len(idx)), "n_skipped_unresolved": int(np.sum(~ok)), "estimate_min": est / 60,
                "courant_M_max": float(np.max(r["courant_M"])) if len(idx) else None,
                "n_max_at_T_end": int(np.sum(tmax >= float(GN["T_end"]) - 1e-9))}

    def st_gain_F():
        return gains(F_d, scF, "F")

    def st_gain_RW():
        d = np.concatenate([R_d[n_cal:], W_d])
        sc = np.concatenate([scR[n_cal:], scW])
        return gains(d, sc, "RW")

    def st_analysis():
        A5 = np.load(g5 / "analysis.npz")
        ln_i_all = A5["pred_S"]                                    # Gate 5 first gate (S* from R first 12), all 52 rays
        nF, nR = len(F_d), len(R_d)
        eps = np.array([m["eps"] for m in F_meta])
        draw = np.array([m["draw"] for m in F_meta])
        ZF = np.load(run / "gain_F.npz")
        cF = np.full((nF, K), np.nan)
        cF[ZF["idx"]] = ZF["c"]
        tF = np.full((nF, K), np.nan)
        tF[ZF["idx"]] = ZF["t_at_max"]
        mid = K // 2
        ref = np.where(eps == 1e-6)[0]
        cstar = float(np.mean(cF[ref, mid]))
        cstar_ok = bool(np.isfinite(cstar))                                    # review fix: NaN c* -> not evaluable
        out = {"c_star": cstar, "c_star_ok": cstar_ok, "c_star_rays": [F_meta[i] for i in ref], "c_at_edge_ref": cF[ref, mid].tolist()}
        lns_grid = np.log(scF)[:, None] + offs[None, :]

        def evaluate(cmat, sc, ln_i, lns):
            n = len(sc)
            ln_ii = np.full(n, np.nan)
            flags = []
            for i in range(n):
                if not cstar_ok or not np.isfinite(sc[i]) or not np.isfinite(cmat[i]).all():
                    flags.append("unresolved")
                    continue
                ln_ii[i], f = first_cross(lns[i], cmat[i], cstar)
                flags.append(f)
            ln_c = np.maximum(ln_i, ln_ii)
            lsc = np.log(sc)
            return ln_ii, ln_c, lsc, flags

        def block(ln_i, ln_ii, ln_c, lsc, flags, sel):
            sel = np.asarray(sel)
            sel = sel[np.isfinite(lsc[sel]) & np.isfinite(ln_ii[sel])]
            if len(sel) == 0:
                return {"n": 0, "composite": None, "i_alone": None, "ii_alone": None}
            e_c, e_i, e_ii = (ln_c - lsc)[sel], (ln_i - lsc)[sel], (ln_ii - lsc)[sel]
            bi = ln_i[sel] > ln_ii[sel]
            bii = ln_ii[sel] > ln_i[sel]
            return {"n": int(len(sel)), "composite": summarize(e_c), "i_alone": summarize(e_i), "ii_alone": summarize(e_ii),
                    "frac_i_binds": float(np.mean(bi)) if len(sel) else None, "frac_ii_binds": float(np.mean(bii)) if len(sel) else None,
                    "frac_ties": float(np.mean(~bi & ~bii)) if len(sel) else None,
                    "ii_clamped_below": int(sum(flags[i] == "below" for i in sel)), "ii_clamped_never": int(sum(flags[i] == "never" for i in sel))}

        def verdict(m):
            if m is None:
                return "not evaluable (no resolved ray)"
            C = cfg["criteria"]["composite"]
            return "holds" if m <= float(C["holds_median"]) else ("does not hold" if m >= float(C["fails_median"]) else "partly holds")

        ln_iF = ln_i_all[:nF]
        ln_iiF, ln_cF, lscF, flF = evaluate(cF, scF, ln_iF, lns_grid)
        others = np.array([i for i in range(nF) if i not in set(ref.tolist())])
        b61 = block(ln_iF, ln_iiF, ln_cF, lscF, flF, others)
        b61["signed_composite_by_eps"] = {f"{e:g}": float(np.median((ln_cF - lscF)[others[(eps[others] == e) & np.isfinite(ln_iiF[others])]]))
                                          for e in sorted(set(eps[others].tolist())) if np.any((eps[others] == e) & np.isfinite(ln_iiF[others]))}
        med = lambda b: b["composite"]["median_abs"] if b["composite"] else None  # noqa: E731
        b61["verdict"] = verdict(med(b61))
        out["6.1"] = b61
        log(f"    6.1: n {b61['n']}, composite median {med(b61)} -> {b61['verdict']}; (i) {b61['i_alone'] and b61['i_alone']['median_abs']}, "
            f"(ii) {b61['ii_alone'] and b61['ii_alone']['median_abs']}")
        # 6.2 (g3)
        h = offs[1] - offs[0]
        sl = {}
        for e in (1e-6, 1e-4, 1e-2):
            for k in (0, 1):
                i = np.where((eps == e) & (draw == k))[0][0]
                sl[(e, k)] = float((cF[i, mid + 1] - cF[i, mid - 1]) / (2 * h))
        pred = -2.0 / float(np.mean(list(sl.values())))
        meas_d = []
        for k in (0, 1):
            i6, i2 = np.where((eps == 1e-6) & (draw == k))[0][0], np.where((eps == 1e-2) & (draw == k))[0][0]
            meas_d.append(float((np.log(scF[i2]) - np.log(scF[i6])) / math.log(1e4)))
        meas = float(np.mean(meas_d))
        ratio = pred / meas
        lo, hi = (float(x) for x in cfg["criteria"]["slope_ratio"])
        per_draw = {}
        for k in (0, 1):
            pk = -2.0 / float(np.mean([sl[(e, k)] for e in (1e-6, 1e-4, 1e-2)]))
            per_draw[k] = {"predicted": pk, "measured": meas_d[k], "ratio": pk / meas_d[k]}
        v62 = ("not evaluable (unresolved)" if not np.isfinite(ratio) else ("holds" if lo <= ratio <= hi else "does not hold"))
        out["6.2"] = {"dc_dlns_at_edge": {f"eps={e:g} draw {k}": v for (e, k), v in sl.items()}, "predicted_slope": pred, "measured_slope": meas,
                      "ratio": ratio, "range": [lo, hi], "verdict": v62, "per_draw": per_draw}
        log(f"    6.2: predicted {pred:.4f}, measured {meas:.4f}, ratio {ratio:.3f} -> {out['6.2']['verdict']}")
        out["n_max_at_T_end_F"] = int(np.sum(tF >= float(GN["T_end"]) - 1e-9))
        # 6.3
        if (run / "gain_RW.npz").exists() and res.get("gain_RW", {}).get("done"):
            Z = np.load(run / "gain_RW.npz")
            sc = np.concatenate([scR[n_cal:], scW])
            n = len(sc)
            cm = np.full((n, K), np.nan)
            cm[Z["idx"]] = Z["c"]
            ln_i = np.concatenate([ln_i_all[nF + n_cal:nF + nR], ln_i_all[nF + nR:]])
            lns = np.log(sc)[:, None] + offs[None, :]
            ln_ii, ln_c, lsc, fl = evaluate(cm, sc, ln_i, lns)
            nRt = nR - n_cal
            b = {"all": block(ln_i, ln_ii, ln_c, lsc, fl, np.arange(n)), "R": block(ln_i, ln_ii, ln_c, lsc, fl, np.arange(nRt)),
                 "W": block(ln_i, ln_ii, ln_c, lsc, fl, np.arange(nRt, n))}
            b["verdict"] = verdict(med(b["all"]))
            out["6.3"] = b
            log(f"    6.3: composite median {med(b['all'])} -> {b['verdict']} (R {med(b['R'])}, W {med(b['W'])})")
            rowsRW = [{"set": (R_meta[n_cal:] + W_meta)[i]["set"], "k": (R_meta[n_cal:] + W_meta)[i]["k"], "c_at_edge": float(cm[i, mid]) if np.isfinite(cm[i, mid]) else None,
                       "ln_err_composite": fmt(ln_c[i] - lsc[i]),
                       "ln_err_i": fmt(ln_i[i] - lsc[i]),
                       "ln_err_ii": fmt(ln_ii[i] - lsc[i]), "ii_flag": fl[i]} for i in range(n)]
            out["rays_RW"] = rowsRW
        out["rays_F"] = [{**F_meta[i], "c_at_edge": float(cF[i, mid]) if np.isfinite(cF[i, mid]) else None,
                          "ln_err_composite": fmt(ln_cF[i] - lscF[i]),
                          "ln_err_i": fmt(ln_iF[i] - lscF[i]),
                          "ln_err_ii": fmt(ln_iiF[i] - lscF[i]), "ii_flag": flF[i],
                          "used_for_c_star": bool(i in set(ref.tolist()))} for i in range(nF)]
        return out

    fns = {"gain_F": st_gain_F, "gain_RW": st_gain_RW, "analysis": st_analysis, "analysis_F": st_analysis}
    for s in stages:
        if s in res and res[s].get("done"):
            log(f"--- {s}: done earlier")
            continue
        log(f"--- {s}")
        t0 = time.time()
        out = fns[s]()
        out["seconds"] = time.time() - t0
        out["done"] = True
        res[s] = out
        save()
    log("done")


if __name__ == "__main__":
    main()
