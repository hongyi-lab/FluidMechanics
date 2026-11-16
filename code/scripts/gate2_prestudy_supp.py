#!/usr/bin/env python
"""Gate 2 pre-study, supplementary checks S1 (a3 ray, wide scan) and S2 (is B committed?).

    python code/scripts/gate2_prestudy_supp.py --config code/configs/gate2_prestudy_supp.yaml \
        --run-dir experiments/20261115_gate2_prestudy_supp_v1
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

from committor.sde import simulate_absorbing  # noqa: E402
from committor.systems.mfe import LAMINAR, MFE  # noqa: E402
from committor.utils import (atomic_save_npz, atomic_write_json, derive_seed, git_info, load_npz, read_json,  # noqa: E402
                             stable_hash, torch_generator, tree_hash)

REPO = Path(__file__).resolve().parents[2]


def cp(k, n):
    lo = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(0.975, k + 1, n - k))
    return lo, hi


def iso_q(s, q, levels=(0.05, 0.5, 0.95)):
    from sklearn.isotonic import IsotonicRegression
    f = IsotonicRegression(y_min=0, y_max=1).fit(s, q).predict(s)
    out = []
    for L in levels:
        k = int(np.argmax(f >= L))
        if f[k] < L:
            out.append(float("nan"))
        elif k == 0:
            out.append(float(s[0]))
        else:
            out.append(float(s[k - 1] + (L - f[k - 1]) / (f[k] - f[k - 1]) * (s[k] - s[k - 1])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()
    progress = not args.no_progress
    cfg = yaml.safe_load(Path(args.config).read_text())
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg["device"])
    src = REPO / "code" / "src" / "committor"
    ident = {"config_hash": stable_hash(cfg),
             "sim_code_hash": tree_hash([src / f for f in ("sde.py", "utils.py", "systems/base.py", "systems/mfe.py")])}
    old = read_json(run_dir / "run_identity.json")
    if old is None:
        atomic_write_json(run_dir / "run_identity.json", ident)
    elif old != ident:
        sys.exit(f"{run_dir} was started with {old}, now {ident}; use a new run dir")
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    man = read_json(run_dir / "manifest.json", {"invocations": []})
    man["invocations"].append({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv, "git": git_info(REPO)})
    atomic_write_json(run_dir / "manifest.json", man)

    pre = REPO / cfg["prestudy_run"]
    P = read_json(pre / "results.json")
    E_B, r_A = P["histograms"]["E_B"], P["edge"]["r_A"]
    dt, T_max = P["heun_dt"]["chosen"], 6000.0
    sig = {"sigma_lo": P["sigma"]["sigma_lo"], "sigma_hi": P["sigma"]["sigma_hi"]}
    det = load_npz(pre / "rays_det.npz")
    names = P.get("rays", {}).get("names") or ["refined_min", "random_q10", "random_q50", "random_q90",
                                                "optimal_growth", "random_extra"]
    Dm = det["dirs"]
    seed = int(cfg["seed"])
    res = read_json(run_dir / "results.json", {}) or {}

    def run(m, X, M, tag, ret_final=False, A_absorbing=True):
        path = run_dir / f"{tag}.npz"
        if path.exists():
            return load_npz(path)
        Ms = np.broadcast_to(np.asarray(M, dtype=int), (len(X),))
        X0 = torch.as_tensor(np.repeat(X, Ms, axis=0), device=device)
        r = simulate_absorbing(m, X0, dt, int(round(T_max / dt)), "heun", torch_generator(derive_seed(seed, tag), device),
                               compact_every=200, progress=progress, desc=tag, bridge=True, compile_step=True,
                               return_final=ret_final)
        idx = np.repeat(np.arange(len(X)), Ms)
        out = {"n_B": np.bincount(idx, weights=(r["outcome"] == 1), minlength=len(X)).astype(int),
               "n_T": np.bincount(idx, weights=(r["outcome"] == -1), minlength=len(X)).astype(int),
               "outcome": r["outcome"], "t_abs": r["steps"] * dt, "path_state": idx, "X": X}
        if ret_final:
            out["final_X"] = r["final_X"]
        atomic_save_npz(path, **out)
        return out

    # ---------------- S1: a3 ray ----------------
    c1 = cfg["S1_a3_ray"]
    ia3 = names.index("optimal_growth")
    d3 = Dm[ia3]
    sc3 = float(det["heun_dt"][ia3])
    if "S1" not in res:
        out = {"direction_energy_fractions": (d3 ** 2).tolist(), "s_c_det": sc3}
        for sname, sv in sig.items():
            m = MFE(sigma=sv, r_A=r_A, E_B=E_B)
            S = sc3 * np.asarray(c1["s_over_sc_grid"])
            r0 = run(m, LAMINAR + S[:, None] * d3[None], int(c1["M_coarse"]), f"S1_coarse_{sname}")
            q = r0["n_B"] / c1["M_coarse"]
            S_cur, q_cur, lev_used = S, q, 0
            for lev in range(1, int(c1["max_zoom"]) + 1):
                if ((q_cur > 0.02) & (q_cur < 0.98)).sum() >= 4:
                    break
                j_hi = int(np.argmax(q_cur >= 0.98)) if (q_cur >= 0.98).any() else len(S_cur) - 1
                below = np.where(q_cur[:j_hi] <= 0.02)[0]
                j_lo = int(below[-1]) if len(below) else 0
                S_new = np.linspace(S_cur[j_lo], S_cur[j_hi], int(c1["n_fine"]))
                rl = run(m, LAMINAR + S_new[:, None] * d3[None], int(c1["M_fine"]), f"S1_fine_{sname}_L{lev}")
                S_cur, q_cur, lev_used = S_new, rl["n_B"] / c1["M_fine"], lev
            s05, s50, s95 = iso_q(S_cur, q_cur)
            out[sname] = {"sigma": sv, "coarse_s_over_sc": c1["s_over_sc_grid"], "coarse_q": q.tolist(),
                          "zoom_levels": lev_used, "final_s": S_cur.tolist(), "final_q": q_cur.tolist(),
                          "s05": s05, "s50": s50, "s95": s95, "E50": s50 ** 2, "s50_over_s_c_det": s50 / sc3,
                          "width_rel": (s95 - s05) / s50}
            print("S1", sname, {k: out[sname][k] for k in ("E50", "s50_over_s_c_det", "width_rel", "zoom_levels")}, flush=True)
        res["S1"] = out
        atomic_write_json(run_dir / "results.json", res)

    # ---------------- S2: B commitment ----------------
    c2 = cfg["S2_B_commitment"]
    if "S2" not in res:
        sv = sig[c2["sigma"]]
        m = MFE(sigma=sv, r_A=r_A, E_B=E_B)
        prof = P["rays"]["profiles"][c2["sigma"]]["rays"]
        starts = [LAMINAR + prof[nm]["s50"] * Dm[i] for i, nm in enumerate(names) if np.isfinite(prof[nm]["s50"])]
        labels = [nm for nm in names if np.isfinite(prof[nm]["s50"])]
        s50_a3 = res["S1"][c2["sigma"]]["s50"]
        if np.isfinite(s50_a3):
            starts.append(LAMINAR + s50_a3 * d3)
            labels.append("a3_ray_S1")
        starts = np.array(starts)
        r = run(m, starts, int(c2["M"]), "S2_entry", ret_final=True)
        hitB = r["outcome"] == 1
        XB = r["final_X"][hitB]
        src_lab = np.array(labels)[r["path_state"][hitB]]
        # free evolution after entering B (no absorption), record E at the end and whether E < 1e-3 was reached
        mf = MFE(sigma=sv, r_A=float(c2["decayed"]), E_B=None)          # absorbing only in "decayed"
        X0 = torch.as_tensor(XB, device=device)
        rf = simulate_absorbing(mf, X0, dt, int(round(c2["follow_time"] / dt)), "heun",
                                torch_generator(derive_seed(seed, "S2_follow"), device), compact_every=200,
                                progress=progress, desc="S2 follow", bridge=True, compile_step=True)
        decayed = rf["outcome"] == 0
        Ex = (XB[:, 3:8] ** 2).sum(1)
        Eentry = (XB[:, 0] - 1) ** 2 + (XB[:, 1:] ** 2).sum(1)
        per = {}
        for lab in labels:
            k = src_lab == lab
            if k.sum():
                per[lab] = {"n_entries": int(k.sum()), "fraction_decayed_within_follow": float(decayed[k].mean()),
                            "cp95": cp(int(decayed[k].sum()), int(k.sum())),
                            "median_x_dependent_energy_at_entry": float(np.median(Ex[k]))}
        thr = 1e-3 * Eentry
        lowx = Ex < thr
        res["S2"] = {"sigma": sv, "n_paths": int(len(r["outcome"])), "n_entries": int(hitB.sum()),
                     "fraction_decayed_within_follow": float(decayed.mean()),
                     "cp95": cp(int(decayed.sum()), int(len(decayed))),
                     "reference_turbulence": 1 - math.exp(-c2["follow_time"] / 10500.0),
                     "entries_with_x_dependent_energy_below_1e-3_of_E": int(lowx.sum()),
                     "decayed_fraction_low_x": float(decayed[lowx].mean()) if lowx.any() else None,
                     "decayed_fraction_high_x": float(decayed[~lowx].mean()) if (~lowx).any() else None,
                     "decay_time_quantiles": np.quantile(rf["steps"][decayed] * dt, [0, .5, 1]).tolist() if decayed.any() else None,
                     "per_start": per}
        atomic_save_npz(run_dir / "S2_follow.npz", XB=XB, decayed=decayed, t_decay=rf["steps"] * dt, src=src_lab.astype(str))
        atomic_write_json(run_dir / "results.json", res)
        print("S2", {k: v for k, v in res["S2"].items() if k != "per_start"}, flush=True)
        for k, v in res["S2"]["per_start"].items():
            print("   ", k, v)
    print("done")


if __name__ == "__main__":
    main()
