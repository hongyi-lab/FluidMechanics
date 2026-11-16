#!/usr/bin/env python
"""Gate 3 for the MFE model: coordinate ranking by threshold prediction (code/configs/gate3_mfe.yaml).

Uses only Gate 2 data (10 000 rays, 200 000 offset states, the noise-family s50); no new MC.

Stages (cached; re-running resumes; gain_grid is chunked on disk):
  gain_a3       both gain versions on the pure-a3 ray: p_eff of each version (user item 2), zero-fit calibration
  gain_family   both versions along the 56 family rays (4000-point grid), sigma = 0, 3e-9, 3e-8, 3e-7
  gain_grid     both versions on the 1000-point dense grid of every generic ray (chunks of rays)
  gain_offsets  both versions at the 20 offset states of every ray
  classifiers   logistic candidates and the MLP ceiling, trained on train rays (val: l2 / early stopping)
  threshold     c* on train rays, test metric with bootstrap, accuracy vs |u|, incremental, gap to ceiling, controls
  zero_fit      user item 3: c*_0 from the pure-a3 ray only; (a) family s50, (b) generic s_c; refit controls
  family        same metric on the family (s50 per sigma, c* leave-one-angle-out), all candidates, descriptive

    python code/scripts/gate3_mfe.py --config code/configs/gate3_mfe.yaml --run-dir <dir> [--frac 0.01]
"""
import argparse
import json
import platform
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import yaml

torch._dynamo.config.cache_size_limit = 256

from committor.gain import gain_versions  # noqa: E402
from committor.ranking import fit_logistic, fit_mlp, poly_features  # noqa: E402
from committor.systems.mfe import LAMINAR, MFE  # noqa: E402
from committor.threshold import RayCoordinate, abs_err, fit_cstar, median0, paired_diff, qinf, summarize  # noqa: E402
from committor.utils import (Status, atomic_save_npz, atomic_write_json, derive_seed, git_info, read_json,  # noqa: E402
                             stable_hash, tree_hash)

REPO = Path(__file__).resolve().parents[2]
SIM_CODE = ["gain.py", "threshold.py", "ranking.py", "utils.py", "systems/mfe.py"]
SIG3 = [3e-9, 3e-8, 3e-7]


# ---------------------------------------------------------------- features -------------------------------------------
def feat(X, name):
    """X (..., 9) torch or numpy."""
    if name == "E":
        return (X[..., 0] - 1.0) ** 2 + (X[..., 1:] ** 2).sum(-1)
    if name == "one_minus_a1":
        return 1.0 - X[..., 0]
    if name.startswith("abs_a"):
        return abs(X[..., int(name[5:]) - 1])
    if name.startswith("a"):
        return X[..., int(name[1:]) - 1]
    raise KeyError(name)


def fm(X, names):
    return np.stack([np.asarray(feat(X, n)) for n in names], axis=-1)


class Logistic:
    def __init__(self, base, degree, extra, info):
        self.base, self.degree, self.extra = base, degree, extra
        self.w, self.b = info["coef_standardised"], info["intercept"]
        self.mu, self.sd = info["feature_mean"], info["feature_std"]

    def phi(self, X, extra_cols):
        F = fm(X, self.base) if self.base != "all" else np.asarray(X)
        P, _ = poly_features(F.reshape(-1, F.shape[-1]), self.degree)
        if self.extra:
            P = np.concatenate([P, np.stack([e.reshape(-1) for e in extra_cols], axis=1)], axis=1)
        return P

    def score(self, X, extra_cols=()):
        shp = X.shape[:-1]
        out = np.empty(int(np.prod(shp)))
        Xf = X.reshape(-1, 9)
        ef = [e.reshape(-1) for e in extra_cols]
        for i in range(0, len(Xf), 2_000_000):
            P = self.phi(Xf[i:i + 2_000_000], [e[i:i + 2_000_000] for e in ef])
            out[i:i + len(P)] = ((P - self.mu) / self.sd) @ self.w + self.b
        return out.reshape(shp)


def polar(X):
    """bijective re-parametrisation of the 9D state: (unit direction d, ln s) with x = lam + s d."""
    D = np.asarray(X).reshape(-1, 9) - LAMINAR[None, :]
    s = np.linalg.norm(D, axis=1)
    return np.concatenate([D / s[:, None], np.log(s)[:, None]], axis=1)


class MLPScore:
    def __init__(self, info, device, transform=None):
        self.net, self.mu, self.sd, self.device = info["net"], info["feature_mean"], info["feature_std"], device
        self.transform = transform

    @torch.no_grad()
    def score(self, X):
        shp = X.shape[:-1]
        Xf = X.reshape(-1, 9)
        out = np.empty(len(Xf))
        for i in range(0, len(Xf), 2_000_000):
            xb = Xf[i:i + 2_000_000] if self.transform is None else self.transform(Xf[i:i + 2_000_000])
            Z = torch.as_tensor((xb - self.mu) / self.sd, dtype=torch.float64, device=self.device)
            out[i:i + len(Z)] = self.net(Z).squeeze(1).cpu().numpy()
        return out.reshape(shp)


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
    ident = {"config_hash": stable_hash(cfg), "frac": args.frac, "sim_code_hash": tree_hash([src / f for f in SIM_CODE] + [Path(__file__).resolve()])}
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
    m = MFE(Re=400.0)
    Gc = cfg["gain"]
    sig_list = [float(s) for s in Gc["sigmas"]]
    k_prim = sig_list.index(float(Gc["sigma_generic_primary"]))
    T_g, dt_g = float(Gc["T_g"]), float(Gc["dt"])

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

    # ------------------------------------------------------------ Gate 2 data ----------------------------------------
    g2 = REPO / cfg["data"]["gate2_run"]
    R2 = json.loads((g2 / "results.json").read_text())
    RY, SC, OF = np.load(g2 / "rays.npz"), np.load(g2 / "scan.npz"), np.load(g2 / "offsets.npz")
    rng_sub = np.random.default_rng(derive_seed(seed, "frac"))
    N_all = len(RY["dirs"])
    rays = np.arange(N_all) if args.frac >= 1 else np.sort(rng_sub.choice(N_all, max(30, int(round(args.frac * N_all))), replace=False))
    dirs, split = RY["dirs"][rays], RY["split"][rays]
    s_c, grid_out, s_grid = SC["s_c"][rays], SC["grid_out"][rays], SC["s_grid"]
    U = OF["u"]
    Xo, Lo = OF["X"][rays], OF["label"][rays].astype(np.int64)
    N, G = len(rays), len(s_grid)
    ln_sc = np.log(s_c)
    tr_r, va_r, te_r = (np.where(split == k)[0] for k in (0, 1, 2))
    Xg = LAMINAR[None, None, :] + s_grid[None, :, None] * dirs[:, None, :]            # (N, G, 9)
    dense_idx = np.round(np.linspace(0, G - 1, int(cfg["data"]["dense_states_per_ray"]))).astype(int)
    boot = np.random.default_rng(derive_seed(seed, "boot")).integers(0, len(te_r), (int(cfg["threshold_prediction"]["bootstrap"]), len(te_r)))
    res.setdefault("meta", {"n_rays": int(N), "split_counts": [int(len(tr_r)), int(len(va_r)), int(len(te_r))],
                            "grid_points": int(G), "sigma_list": sig_list, "sigma_generic_primary": sig_list[k_prim],
                            "dense_states_per_ray": int(len(dense_idx)), "gate2_run": cfg["data"]["gate2_run"]})
    save()

    # ------------------------------------------------------------ gain on the pure-a3 ray (item 2) -------------------
    a3g = Gc["a3_grid"]
    s_a3 = np.linspace(float(a3g[0]), float(a3g[1]), int(a3g[2]))

    def st_gain_a3():
        X = LAMINAR[None, :] + s_a3[:, None] * np.eye(9)[2][None, :]
        r = gain_versions(m, X, [1.0], T_g, dt_g, device)                    # y0 = 0: tr P_sigma = sigma^2 tr Q
        ln_gi, ln_gii = np.log(r["max_trP"][:, 0]), np.log(r["trQ_a2pk"])
        atomic_save_npz(run / "gain_a3.npz", s=s_a3, ln_g_i=ln_gi, ln_g_ii=ln_gii, t_at_max=r["t_at_max"][:, 0],
                        t_a2pk=r["t_a2pk"])
        out = {"monotone_i": bool(np.all(np.diff(ln_gi) > 0)), "monotone_ii": bool(np.all(np.diff(ln_gii) > 0)),
               "t_at_max_range": [float(r["t_at_max"].min()), float(r["t_at_max"].max())],
               "t_a2pk_range": [float(r["t_a2pk"].min()), float(r["t_a2pk"].max())],
               "check_vs_gate2_ln_g_i_maxabs": float(np.max(np.abs(ln_gi - np.log(np.load(g2 / "gain_a3.npz")["g"]))))}
        rows = {est: R2["gain_pred"]["by_estimator"][est]["rows"] for est in ("isotonic", "lapse_probit")}
        for ver, lng in (("i", ln_gi), ("ii", ln_gii)):
            slope = np.gradient(lng, s_a3)
            g_at = lambda s: float(np.interp(s, s_a3, lng))
            d_at = lambda s: float(np.interp(s, s_a3, slope))
            vres = {}
            for est, rr in rows.items():
                s50 = np.array([r_["s50"] for r_ in rr])
                se = np.array([r_["s50_se"] for r_ in rr])
                c = np.array([np.log(sg ** 2) + g_at(s) for sg, s in zip(SIG3, s50)])
                c_se = np.array([abs(d_at(s)) * e for s, e in zip(s50, se)])
                peff = []
                for a, b in ((0, 1), (1, 2)):
                    p = (g_at(s50[a]) - g_at(s50[b])) / np.log(SIG3[b] / SIG3[a])
                    pse = np.hypot(c_se[a], c_se[b]) / np.log(SIG3[b] / SIG3[a])
                    peff.append({"sigmas": [SIG3[a], SIG3[b]], "p_eff": float(p), "se": float(pse)})
                pred = []
                for i in (0, 2):
                    target = c[1] - np.log(SIG3[i] ** 2)
                    j = int(np.argmax(lng >= target)) if (lng >= target).any() else -1     # first crossing (lng need not be monotone)
                    below = j == 0
                    if j < 0:                                                   # never reached on the a3 grid
                        sp = float("inf")
                    elif j == 0:                                                # already reached at s_a3[0]: extrapolate linearly
                        sp = float(s_a3[0] + (target - lng[0]) / (lng[1] - lng[0]) * (s_a3[1] - s_a3[0]))
                    else:
                        sp = float(s_a3[j - 1] + (target - lng[j - 1]) / (lng[j] - lng[j - 1]) * (s_a3[j] - s_a3[j - 1]))
                    e_se = np.hypot(c_se[1], c_se[i]) / abs(d_at(s50[i])) / s50[i]
                    pred.append({"sigma": SIG3[i], "s50_mc": float(s50[i]), "s50_pred": sp, "rel_err": sp / s50[i] - 1,
                                 "rel_err_se": float(e_se), "below_a3_grid_extrapolated": bool(below),
                                 "rel_err_bound_if_below": float(s_a3[0] / s50[i] - 1) if below else None})
                vres[est] = {"c_at_s50": c.tolist(), "c_se": c_se.tolist(), "p_eff": peff, "prediction_from_middle": pred}
            iso_c = np.array(vres["isotonic"]["c_at_s50"])
            vres["zero_fit_c_star"] = float(iso_c.mean())
            vres["zero_fit_c_spread"] = [float(iso_c.min()), float(iso_c.max())]
            vres["dlng_dlns_at_s50_mid"] = float(d_at(rows["isotonic"][1]["s50"]) * rows["isotonic"][1]["s50"])
            vres["isotonic"]["se_note"] = "Gate 2 isotonic s50_se"
            vres["lapse_probit"]["se_note"] = "Gate 2 provides no lapse-specific s50_se: the isotonic s50_se is used as a proxy"
            out[ver] = vres
        # sensitivity to the RK4 step (review: version (ii) samples t* on the dt grid; no interpolation of t*)
        dt_f = dt_g / 8.0
        rf = gain_versions(m, X, [1.0], T_g, dt_f, device)
        lf = {"i": np.log(rf["max_trP"][:, 0]), "ii": np.log(rf["trQ_a2pk"])}
        sens = {"dt_fine": dt_f}
        iso = np.array([r_["s50"] for r_ in rows["isotonic"]])
        for ver, lng in (("i", ln_gi), ("ii", ln_gii)):
            g_f = lambda s: float(np.interp(s, s_a3, lf[ver]))
            sens[ver] = {"max_abs_diff_ln_g": float(np.max(np.abs(lf[ver] - lng))),
                         "monotone_fine": bool(np.all(np.diff(lf[ver]) > 0)),
                         "n_nonmonotone_steps_dt": int(np.sum(np.diff(lng) <= 0)),
                         "n_nonmonotone_steps_fine": int(np.sum(np.diff(lf[ver]) <= 0)),
                         "p_eff_isotonic_fine": [float((g_f(iso[a]) - g_f(iso[b])) / np.log(SIG3[b] / SIG3[a])) for a, b in ((0, 1), (1, 2))],
                         "c_at_s50_isotonic_fine": [float(np.log(sg ** 2) + g_f(s)) for sg, s in zip(SIG3, iso)]}
            sens[ver]["zero_fit_c_star_fine"] = float(np.mean(sens[ver]["c_at_s50_isotonic_fine"]))
        sens["t_a2pk_max_abs_diff"] = float(np.max(np.abs(rf["t_a2pk"] - r["t_a2pk"])))
        out["dt_sensitivity"] = sens
        return out

    timed("gain_a3", st_gain_a3)
    cstar0 = {v: res["gain_a3"][v]["zero_fit_c_star"] for v in ("i", "ii")}

    # ------------------------------------------------------------ gain along the family rays --------------------------
    FD = np.load(g2 / "family_det.npz")
    fam_dirs = FD["d"]
    fam_grid = np.sqrt(np.geomspace(5e-7 * (1 + 1e-9), 0.29 * (1 - 1e-9), int(Gc["family_grid_points"])))
    profiles = [p for p in R2["family_mc"]["profiles"] if p.get("resolved") and "s50" in p]

    def st_gain_family():
        X = (LAMINAR[None, None, :] + fam_grid[None, :, None] * fam_dirs[:, None, :]).reshape(-1, 9)
        r = gain_versions(m, X, sig_list, T_g, dt_g, device, chunk=250_000)
        shp = (len(fam_dirs), len(fam_grid))
        atomic_save_npz(run / "gain_family.npz", max_trP=r["max_trP"].reshape(*shp, -1), v2_a2pk=r["v2_a2pk"].reshape(shp),
                        trQ_a2pk=r["trQ_a2pk"].reshape(shp), t_a2pk=r["t_a2pk"].reshape(shp),
                        t_at_max=r["t_at_max"].reshape(*shp, -1), s=fam_grid)
        return {"n_rays": int(len(fam_dirs)), "grid_points": int(len(fam_grid)), "n_resolved_profiles": len(profiles)}

    timed("gain_family", st_gain_family)
    GF = np.load(run / "gain_family.npz")

    def fam_gain(ver, sigma):
        k = sig_list.index(sigma)
        if ver == "i":
            return np.log(GF["max_trP"][:, :, k])
        return np.log(GF["v2_a2pk"] + sigma ** 2 * GF["trQ_a2pk"])

    # ------------------------------------------------------------ gain on the generic dense grid ----------------------
    rows_per_chunk = 1000

    def st_gain_grid():
        chunks = list(range(0, N, rows_per_chunk))
        times = []
        for ci, r0 in enumerate(chunks):
            f = run / f"gain_grid_{ci:03d}.npz"
            if f.exists():
                continue
            t0 = time.perf_counter()
            rr = np.arange(r0, min(r0 + rows_per_chunk, N))
            r = gain_versions(m, Xg[rr].reshape(-1, 9), sig_list, T_g, dt_g, device, chunk=1_000_000)
            shp = (len(rr), G)
            atomic_save_npz(f, rows=rr, max_trP=r["max_trP"].reshape(*shp, -1), v2_a2pk=r["v2_a2pk"].reshape(shp),
                            trQ_a2pk=r["trQ_a2pk"].reshape(shp), t_a2pk=r["t_a2pk"].reshape(shp).astype(np.float32),
                            t_at_max=r["t_at_max"][:, k_prim].reshape(shp).astype(np.float32))
            times.append(time.perf_counter() - t0)
            print(f"    gain_grid chunk {ci + 1}/{len(chunks)}: {times[-1]:.0f} s", flush=True)
        return {"n_chunks": len(chunks), "states": int(N * G), "chunk_seconds": times}

    timed("gain_grid", st_gain_grid)
    parts = [np.load(run / f"gain_grid_{ci:03d}.npz") for ci in range(len(range(0, N, rows_per_chunk)))]
    GG = {k: np.concatenate([p[k] for p in parts]) for k in ("max_trP", "v2_a2pk", "trQ_a2pk", "t_a2pk", "t_at_max")}
    del parts

    def grid_gain(ver, sigma):
        k = sig_list.index(sigma)
        if ver == "i":
            return np.log(GG["max_trP"][:, :, k])
        return np.log(GG["v2_a2pk"] + sigma ** 2 * GG["trQ_a2pk"])

    def st_gain_offsets():
        r = gain_versions(m, Xo.reshape(-1, 9), sig_list, T_g, dt_g, device, chunk=500_000)
        shp = Xo.shape[:2]
        atomic_save_npz(run / "gain_offsets.npz", max_trP=r["max_trP"].reshape(*shp, -1), v2_a2pk=r["v2_a2pk"].reshape(shp),
                        trQ_a2pk=r["trQ_a2pk"].reshape(shp))
        return {"states": int(np.prod(shp))}

    timed("gain_offsets", st_gain_offsets)
    GO = np.load(run / "gain_offsets.npz")

    def off_gain(ver, sigma):
        k = sig_list.index(sigma)
        if ver == "i":
            return np.log(GO["max_trP"][:, :, k])
        return np.log(GO["v2_a2pk"] + sigma ** 2 * GO["trQ_a2pk"])

    sp = sig_list[k_prim]
    gain_cols_grid = {"gain_i": grid_gain("i", sp), "gain_ii": grid_gain("ii", sp)}
    gain_cols_off = {"gain_i": off_gain("i", sp), "gain_ii": off_gain("ii", sp)}

    # ------------------------------------------------------------ classifiers ----------------------------------------
    TPc = cfg["threshold_prediction"]
    fit_rows = np.concatenate([tr_r, va_r])
    Xtr = np.concatenate([Xo[fit_rows].reshape(-1, 9), Xg[fit_rows][:, dense_idx].reshape(-1, 9)])
    ytr = np.concatenate([Lo[fit_rows].reshape(-1), grid_out[fit_rows][:, dense_idx].reshape(-1)]).astype(np.float64)
    sp_tr = np.concatenate([np.repeat(split[fit_rows], Xo.shape[1]), np.repeat(split[fit_rows], len(dense_idx))]).astype(np.int8)
    gain_tr = {k: np.concatenate([gain_cols_off[k][fit_rows].reshape(-1), gain_cols_grid[k][fit_rows][:, dense_idx].reshape(-1)])
               for k in gain_cols_grid}
    keep = ytr >= 0
    Mtr = np.ones_like(ytr)

    log_specs = []
    base3 = ["one_minus_a1", "a2", "a3"]
    for pair in combinations(base3, 2):
        log_specs.append((f"pair ({pair[0]}, {pair[1]})", list(pair), 2, []))
    log_specs.append(("triple (1-a1, a2, a3)", base3, 2, []))
    log_specs.append(("linear logistic (9D)", "all", 1, []))
    log_specs.append(("triple + log gain (i)", base3, 2, ["gain_i"]))
    log_specs.append(("triple + log gain (ii)", base3, 2, ["gain_ii"]))
    models = {}

    def st_classifiers():
        out = {"n_train_states": int((sp_tr == 0).sum()), "n_val_states": int((sp_tr == 1).sum()),
               "train_fraction_B": float(ytr[sp_tr == 0].mean())}
        for name, base, deg, extra in log_specs:
            mdl = Logistic(base, deg, extra, {"coef_standardised": None, "intercept": None, "feature_mean": None, "feature_std": None})
            Phi = mdl.phi(Xtr, [gain_tr[e] for e in extra])
            z, info = fit_logistic(Phi[keep], ytr[keep], Mtr[keep], sp_tr[keep], TPc["logistic_l2_grid"], device)
            np.savez(run / f"model_{name.replace(' ', '_').replace('/', '')}.npz", w=info["coef_standardised"], b=info["intercept"],
                     mu=info["feature_mean"], sd=info["feature_std"])
            out[name] = {"l2": info["l2"], "val_nll": info["val_nll"],
                         "train_acc": float(((z[sp_tr[keep] == 0] > 0) == (ytr[keep][sp_tr[keep] == 0] > 0.5)).mean())}
            print("   ", name, out[name], flush=True)
        z, info = fit_mlp(Xtr[keep], ytr[keep], Mtr[keep], sp_tr[keep], TPc["mlp"], derive_seed(seed, "mlp"), device, progress)
        torch.save({"state": info["net"].state_dict(), "mu": info["feature_mean"], "sd": info["feature_std"]}, run / "model_mlp.pt")
        out["MLP (9D)"] = {"val_nll": info["val_nll"], "best_seed": info["best_seed"], "best_epoch": info["best_epoch"],
                           "train_acc": float(((z[sp_tr[keep] == 0] > 0) == (ytr[keep][sp_tr[keep] == 0] > 0.5)).mean())}
        print("    MLP", out["MLP (9D)"], flush=True)
        Fp = polar(Xtr[keep])
        z, info = fit_mlp(Fp, ytr[keep], Mtr[keep], sp_tr[keep], TPc["mlp"], derive_seed(seed, "mlp_polar"), device, progress)
        torch.save({"state": info["net"].state_dict(), "mu": info["feature_mean"], "sd": info["feature_std"]}, run / "model_mlp_polar.pt")
        out["MLP (9D, polar input)"] = {"val_nll": info["val_nll"], "best_seed": info["best_seed"], "best_epoch": info["best_epoch"],
                                        "train_acc": float(((z[sp_tr[keep] == 0] > 0) == (ytr[keep][sp_tr[keep] == 0] > 0.5)).mean())}
        print("    MLP polar", out["MLP (9D, polar input)"], flush=True)
        return out

    timed("classifiers", st_classifiers)
    for name, base, deg, extra in log_specs:
        z = np.load(run / f"model_{name.replace(' ', '_').replace('/', '')}.npz")
        models[name] = Logistic(base, deg, extra, {"coef_standardised": z["w"], "intercept": float(z["b"]),
                                                    "feature_mean": z["mu"], "feature_std": z["sd"]})
    from committor.ranking import _make_mlp
    ck = torch.load(run / "model_mlp.pt", weights_only=False)
    net = _make_mlp(9, TPc["mlp"]["hidden"]).to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    mlp = MLPScore({"net": net, "feature_mean": ck["mu"], "feature_std": ck["sd"]}, device)
    ck = torch.load(run / "model_mlp_polar.pt", weights_only=False)
    net_p = _make_mlp(10, TPc["mlp"]["hidden"]).to(device)
    net_p.load_state_dict(ck["state"])
    net_p.eval()
    mlp_p = MLPScore({"net": net_p, "feature_mean": ck["mu"], "feature_std": ck["sd"]}, device, transform=polar)

    # ------------------------------------------------------------ candidates as (grid, offsets, family) callables -----
    cands = [("E", "baseline", lambda X: feat(X, "E"))]
    for i in range(1, 10):
        cands.append((f"a{i}", "single mode", lambda X, i=i: feat(X, f"a{i}")))
    for i in range(1, 10):
        cands.append((f"|a{i}|", "single |mode|", lambda X, i=i: feat(X, f"abs_a{i}")))
    for name, base, deg, extra in log_specs:
        if extra:
            continue
        cands.append((name, "logistic", lambda X, name=name: models[name].score(X)))
    cands.append(("MLP (9D)", "ceiling", lambda X: mlp.score(X)))
    cands.append(("MLP (9D, polar input)", "ceiling", lambda X: mlp_p.score(X)))

    def coord_generic(name):
        """(grid (N, G), offsets (N, 20)) of a candidate on the generic rays."""
        for nm, kind, fn in cands:
            if nm == name:
                return fn(Xg), fn(Xo)
        if name.startswith("gain (") and name.endswith(")"):           # gain (i) / gain (ii), sigma = primary
            ver = name[6:-1]
            return grid_gain(ver, sp), off_gain(ver, sp)
        if name.startswith("gain ") and "@" in name:                    # gain (i)@sigma variants
            ver, sg = name[6:name.index(")")], float(name.split("@")[1])
            return grid_gain(ver, sg), off_gain(ver, sg)
        if name.startswith("triple + log gain"):
            k = "gain_i" if "(i)" in name else "gain_ii"
            return (models[name].score(Xg, [gain_cols_grid[k]]), models[name].score(Xo, [gain_cols_off[k]]))
        raise KeyError(name)

    all_names = [c[0] for c in cands] + ["gain (i)", "gain (ii)", "triple + log gain (i)", "triple + log gain (ii)"]
    sig_variants = [f"gain ({v})@{s:g}" for v in ("i", "ii") for s in sig_list if s != sp]

    # ------------------------------------------------------------ threshold metric ------------------------------------
    pos_u = U > 0

    def st_threshold():
        out = {"candidates": {}, "irreducible": {}}
        A_above = (Lo[te_r][:, pos_u] == 0)
        out["irreducible"] = {"by_u": {f"{u:g}": float(A_above[:, k].mean()) for k, u in enumerate(U[pos_u])},
                              "overall_u_pos": float(A_above.mean()),
                              "all_rays_by_u": {f"{u:g}": float((Lo[:, pos_u][:, k] == 0).mean()) for k, u in enumerate(U[pos_u])},
                              "frac_B_below_edge_all_rays": float((Lo[:, ~pos_u] == 1).mean())}
        errs = {}
        for name in all_names + sig_variants:
            C, Co = coord_generic(name)
            rc = RayCoordinate(C, s_grid, device)
            fit = fit_cstar(rc, tr_r, ln_sc[tr_r])
            ln_p, below = rc.ln_s_pred([fit["side"] * fit["c_star"]], fit["side"], te_r)
            e = (ln_p[:, 0].cpu().numpy() - ln_sc[te_r])
            errs[name] = e
            row = {"side": fit["side"], "c_star": fit["c_star"], "train_median_abs": fit["train_median_abs"],
                   "test": summarize(e, boot), "frac_below_grid": float(below[:, 0].float().mean().item())}
            pred = (fit["side"] * Co[te_r] >= fit["side"] * fit["c_star"])
            lab = Lo[te_r] == 1
            acc = (pred == lab).mean(0)
            row["accuracy_by_u"] = {f"{u:g}": float(a) for u, a in zip(U, acc)}
            absu = np.unique(np.abs(U))
            row["accuracy_by_abs_u"] = {f"{au:g}": float((pred == lab)[:, np.isclose(np.abs(U), au)].mean()) for au in absu}
            row["accuracy_overall"] = float((pred == lab).mean())
            out["candidates"][name] = row
            print(f"    {name:28s} side {fit['side']:+d}  train med {fit['train_median_abs']:.4f}  test med "
                  f"{row['test']['median_abs']:.4f} q90 {row['test']['q90_abs']:.4f}  acc {row['accuracy_overall']:.4f}", flush=True)
            del rc
            torch.cuda.empty_cache()
        np.savez(run / "test_errors.npz", names=np.array(list(errs)), E=np.stack([errs[k] for k in errs]), test_rays=rays[te_r])
        # incremental (better member / best pair chosen on the TRAINING objective)
        trm = {k: out["candidates"][k]["train_median_abs"] for k in errs}
        inc = []
        names3 = {"one_minus_a1": "1-a1"}
        single = {"one_minus_a1": "a1", "a2": "a2", "a3": "a3"}
        pairs = [n for n in errs if n.startswith("pair (")]
        for pn in pairs:
            mem = pn[6:-1].split(", ")
            cand_m = []
            for x in mem:
                cand_m += [single[x], "|" + single[x] + "|"]
            better = min(cand_m, key=lambda k: trm[k])
            inc.append({"a": pn, "b": better, **paired_diff(errs[pn], errs[better], boot)})
        best_pair = min(pairs, key=lambda k: trm[k])
        ceil = min(("MLP (9D)", "MLP (9D, polar input)"), key=lambda k: trm[k])
        out["ceiling"] = {"name": ceil, "rule": "the MLP with the lower training median |ln| (c* objective)"}
        for a_, b_ in (("triple (1-a1, a2, a3)", best_pair), (ceil, "triple (1-a1, a2, a3)"),
                       (ceil, "linear logistic (9D)"), ("MLP (9D, polar input)", "MLP (9D)"), ("triple + log gain (i)", "triple (1-a1, a2, a3)"),
                       ("triple + log gain (ii)", "triple (1-a1, a2, a3)"), ("gain (i)", "E"), ("gain (ii)", "E"),
                       ("gain (i)", "gain (ii)"), ("gain (i)", "|a3|"), ("triple (1-a1, a2, a3)", "E")):
            inc.append({"a": a_, "b": b_, **paired_diff(errs[a_], errs[b_], boot)})
        out["incremental"] = inc
        out["gap_to_ceiling"] = {k: paired_diff(errs[k], errs[ceil], boot) for k in all_names if k != ceil}
        # controls: c* fitted on permuted training s_c (coordinate fixed), random projections
        rng = np.random.default_rng(derive_seed(seed, "controls"))
        shuf = {}
        for name in ("E", "a3", "|a3|", "gain (i)", "gain (ii)", "triple (1-a1, a2, a3)", "MLP (9D)", "MLP (9D, polar input)"):
            C, _ = coord_generic(name)
            rc = RayCoordinate(C, s_grid, device)
            meds = []
            for _ in range(int(cfg["controls"]["shuffled_labels"])):
                perm = rng.permutation(ln_sc[tr_r])
                fit = fit_cstar(rc, tr_r, perm)
                ln_p, _ = rc.ln_s_pred([fit["side"] * fit["c_star"]], fit["side"], te_r)
                meds.append(float(np.median(np.abs(ln_p[:, 0].cpu().numpy() - ln_sc[te_r]))))
            shuf[name] = {"median_abs_quantiles_5_50_95": [float(x) for x in qinf(meds, [0.05, 0.5, 0.95])],
                          "real": out["candidates"][name]["test"]["median_abs"]}
            del rc
        out["control_shuffled_s_c"] = shuf
        W = rng.normal(size=(int(cfg["controls"]["n_random_projections"]), 9))
        W /= np.linalg.norm(W, axis=1, keepdims=True)
        rp = {"signed": [], "abs": []}
        for w in W:
            for kind in ("signed", "abs"):
                proj = dirs @ w
                C = s_grid[None, :] * (np.abs(proj) if kind == "abs" else proj)[:, None]
                rc = RayCoordinate(C, s_grid, device)
                fit = fit_cstar(rc, tr_r, ln_sc[tr_r], n_levels=400, n_refine=200)
                ln_p, below = rc.ln_s_pred([fit["side"] * fit["c_star"]], fit["side"], te_r)
                e = ln_p[:, 0].cpu().numpy() - ln_sc[te_r]
                rp[kind].append([float(np.median(np.abs(e))), float(np.mean(np.isposinf(e))), float(below[:, 0].float().mean().item())])
        out["control_random_projections"] = {}
        for kind in ("signed", "abs"):
            a_ = np.array(rp[kind])
            out["control_random_projections"][kind] = {
                "median_abs_quantiles_0_5_50_95_100": [float(x) for x in qinf(a_[:, 0], [0, 0.05, 0.5, 0.95, 1])],
                "frac_no_crossing_median": float(np.median(a_[:, 1])), "frac_below_grid_median": float(np.median(a_[:, 2]))}
        out["control_random_projections"]["note"] = ("signed: c = w.(x - lam), as configured; it has one sign along each ray, so about half "
                                                     "the rays can never cross (a trivially weak null). abs: c = |w.(x - lam)|, monotone in s "
                                                     "on every ray, the null comparable to |a_i| (added after review)")
        # edge values of the gain coordinates on all rays (for the zero-fit picture)
        out["gain_at_edge"] = {}
        for ver in ("i", "ii"):
            rc = RayCoordinate(grid_gain(ver, sp), s_grid, device)
            ev = rc.values_at(ln_sc)
            out["gain_at_edge"][ver] = {"quantiles_0_5_25_50_75_95_100": np.quantile(ev[np.isfinite(ev)], [0, .05, .25, .5, .75, .95, 1]).tolist(),
                                        "c_star_zero_fit": cstar0[ver]}
            np.save(run / f"gain_{ver}_at_edge.npy", ev)
            del rc
        return out

    timed("threshold", st_threshold)

    # ------------------------------------------------------------ zero-fit test (item 3) ------------------------------
    prof_s50 = np.array([p["s50"] for p in profiles])
    prof_ray = np.array([p["ray"] for p in profiles])
    prof_sig = np.array([p["sigma"] for p in profiles])
    prof_ang = np.array([p["angle"] for p in profiles])
    prof_eps = np.array([p["eps"] for p in profiles])
    is_cal = (prof_ang == 0.0) & (prof_eps == 0.0) & (np.array([p["draw"] for p in profiles]) == 0)

    def fam_rcs(Cfun):
        """RayCoordinate per sigma (shared object when the coordinate does not depend on sigma)."""
        out, cache = {}, {}
        for sg in SIG3:
            C = Cfun(sg)
            key = id(C)
            if key not in cache:
                cache[key] = (C, RayCoordinate(C, fam_grid, device))   # keep C alive so that its id is not reused
            out[sg] = cache[key][1]
        return out

    def fam_err(rcs, cstar, side, with_below=False):
        """ln(s_pred/s50) of every resolved profile for one threshold (and the below-grid mask)."""
        e = np.full(len(profiles), np.nan)
        bl = np.zeros(len(profiles), dtype=bool)
        for sg in SIG3:
            idx = np.where(prof_sig == sg)[0]
            ln_p, below = rcs[sg].ln_s_pred([side * cstar], side, prof_ray[idx])
            e[idx] = ln_p[:, 0].cpu().numpy() - np.log(prof_s50[idx])
            bl[idx] = below[:, 0].cpu().numpy()
        return (e, bl) if with_below else e

    def fam_summary(e, bl=None):
        def S(msk):
            s_ = summarize(e[msk])
            if bl is not None:
                s_["frac_below_grid"] = float(bl[msk].mean())
            return s_
        allm = np.ones(len(e), dtype=bool)
        out = {"all": S(allm), "excluding_calibration": S(~is_cal), "excluding_angle_90": S(prof_ang != 90.0)}
        for sg in SIG3:
            out[f"sigma={sg:g}"] = S(prof_sig == sg)
        for ep in (0.0, 1e-6, 1e-4, 1e-2):
            out[f"eps={ep:g}"] = S(prof_eps == ep)
        for ang in np.unique(prof_ang):
            out[f"angle={ang:g}"] = S(prof_ang == ang)
        out["per_profile"] = [{"sigma": float(a), "angle": float(b), "eps": float(c), "draw": int(p["draw"]), "ln_err": float(x),
                               "below_grid": (bool(bl[k]) if bl is not None else None)}
                              for k, (a, b, c, p, x) in enumerate(zip(prof_sig, prof_ang, prof_eps, profiles, e))]
        return out

    def fam_fit(rcs, rows, n_levels=801):
        """single c* (both sides tried) minimising the median |ln(s_pred/s50)| over the given profiles."""
        best = None
        for side in (1, -1):
            vals, groups = [], []
            for sg in SIG3:
                idx = rows[prof_sig[rows] == sg]
                if len(idx):
                    vals.append(side * rcs[sg].values_at(np.log(prof_s50[idx]), prof_ray[idx]))
                    groups.append((sg, idx))
            v = np.concatenate(vals)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            thr = np.unique(np.quantile(v, np.linspace(0, 1, n_levels)))
            Ef = lambda th: torch.cat([abs_err(rcs[sg].ln_s_pred(th, side, prof_ray[idx])[0], np.log(prof_s50[idx])) for sg, idx in groups])
            med = median0(Ef(thr)).cpu().numpy()
            k = int(np.argmin(med))
            fine = np.unique(np.concatenate([np.linspace(thr[max(k - 1, 0)], thr[min(k + 1, len(thr) - 1)], 400), [thr[k]]]))
            medf = median0(Ef(fine)).cpu().numpy()
            kf = int(np.argmin(medf))
            if best is None or medf[kf] < best[0] - 1e-15:
                best = (float(medf[kf]), side, side * float(fine[kf]))
        return {"c_star": best[2], "side": best[1], "fit_median_abs": best[0]}

    def st_zero_fit():
        out = {"c_star_zero_fit": cstar0, "calibration": {v: res["gain_a3"][v]["isotonic"]["c_at_s50"] for v in ("i", "ii")}}
        for ver in ("i", "ii"):
            rcs = fam_rcs(lambda sg, ver=ver: fam_gain(ver, sg))
            vo = {}
            e0, b0 = fam_err(rcs, cstar0[ver], 1, with_below=True)
            vo["family_zero_fit"] = fam_summary(e0, b0)
            ins = fam_fit(rcs, np.arange(len(profiles)))
            vo["family_refit_in_sample"] = {**ins, "errors": fam_summary(*fam_err(rcs, ins["c_star"], ins["side"], with_below=True))}
            el = np.full(len(profiles), np.nan)
            bll = np.zeros(len(profiles), dtype=bool)
            lo_fits = {}
            for ang in np.unique(prof_ang):
                trn = np.where(prof_ang != ang)[0]
                f_ = fam_fit(rcs, trn)
                lo_fits[f"{ang:g}"] = {"c_star": f_["c_star"], "side": f_["side"]}
                e_, b_ = fam_err(rcs, f_["c_star"], f_["side"], with_below=True)
                el[prof_ang == ang] = e_[prof_ang == ang]
                bll[prof_ang == ang] = b_[prof_ang == ang]
            vo["family_refit_leave_one_angle_out"] = {"c_star_by_left_out_angle": lo_fits, "errors": fam_summary(el, bll)}
            # (b) generic rays
            gb = {}
            for sg in sig_list:
                rc = RayCoordinate(grid_gain(ver, sg), s_grid, device)
                ln_p, below = rc.ln_s_pred([cstar0[ver]], 1)
                e = ln_p[:, 0].cpu().numpy() - ln_sc
                fb = float(below[:, 0].float().mean().item())
                ln_x, _ = rc.ln_s_pred([cstar0[ver]], 1, extrapolate_below=True)
                ex = ln_x[:, 0].cpu().numpy() - ln_sc
                gb[f"sigma={sg:g}"] = {"all_rays": summarize(e), "test_rays": summarize(e[te_r], boot),
                                       "frac_below_grid": fb,
                                       "median_abs_is_grid_bound": fb > 0,
                                       "all_rays_below_grid_extrapolated": summarize(ex),
                                       "note": ("the metric is defined on the Gate 2 grid: where the first grid point already "
                                                "satisfies the rule s_pred = s_1, so |ln| there is a lower bound on the error of the "
                                                "frozen rule; 'all_rays_below_grid_extrapolated' extends c linearly in ln s below s_1 "
                                                "(descriptive)")}
                if sg == sp:
                    np.save(run / f"zero_fit_generic_err_{ver}.npy", e)
                del rc
            vo["generic_zero_fit"] = gb
            vo["generic_refit_control"] = res["threshold"]["candidates"][f"gain ({ver})"]
            out[ver] = vo
            np.save(run / f"zero_fit_family_err_{ver}.npy", e0)
        # version (ii): states whose a2^2 maximum on [0, T_g] is at t = 0 (then c(ii) = ln|y0|^2, no noise term)
        fam_t0 = GF["t_a2pk"] == 0
        prof_t0 = np.array([fam_t0[r_].any() for r_ in prof_ray])
        jj = np.clip(np.searchsorted(s_grid, np.exp(ln_sc)), 0, G - 1)
        t_at_edge = GG["t_a2pk"][np.arange(N), jj]
        out["ii_t_star_zero"] = {"family_rays_all_points": [int(r_) for r_ in np.where(fam_t0.all(1))[0]],
                                 "family_frac_grid_points": float(fam_t0.mean()),
                                 "family_profiles_on_such_rays": int(prof_t0.sum()),
                                 "family_profiles_on_such_rays_angles": sorted({float(a_) for a_ in prof_ang[prof_t0]}),
                                 "generic_frac_grid_points": float((GG["t_a2pk"] == 0).mean()),
                                 "generic_frac_rays_t0_at_edge_gridpoint": float((t_at_edge == 0).mean())}
        return out

    timed("zero_fit", st_zero_fit)

    # ------------------------------------------------------------ family metric (all candidates, descriptive) --------
    Xf = LAMINAR[None, None, :] + fam_grid[None, :, None] * fam_dirs[:, None, :]

    def st_family():
        out = {}
        for name in all_names:
            if name.startswith("gain ("):
                ver = name[6:-1]
                Cf = lambda sg, ver=ver: fam_gain(ver, sg)
            elif name.startswith("triple + log gain"):
                ver = "i" if "(i)" in name else "ii"
                Cf = lambda sg, ver=ver, name=name: models[name].score(Xf, [fam_gain(ver, sg)])
            else:
                fn = [c for c in cands if c[0] == name][0][2]
                Cfix = fn(Xf)
                Cf = lambda sg, Cfix=Cfix: Cfix
            rcs = fam_rcs(Cf)
            el = np.full(len(profiles), np.nan)
            for sg in SIG3:
                for ang in np.unique(prof_ang):
                    trn = np.where((prof_ang != ang) & (prof_sig == sg))[0]
                    tst = (prof_ang == ang) & (prof_sig == sg)
                    if not tst.any():
                        continue
                    f_ = fam_fit(rcs, trn)
                    e_ = fam_err(rcs, f_["c_star"], f_["side"])
                    el[tst] = e_[tst]
            s_ = fam_summary(el)
            out[name] = {k: v for k, v in s_.items() if k != "per_profile"}
            print(f"    family {name:28s} median {out[name]['all']['median_abs']:.4f}", flush=True)
        return out

    timed("family", st_family)
    print("done")


if __name__ == "__main__":
    main()
