#!/usr/bin/env python
"""Gate 3b for the MFE model: can the edge be described by a few quadratic mechanism quantities?
(code/configs/gate3b_mfe.yaml; criteria fixed there before the run). Gate 2 data only; no new MC.

Stages (cached in results.json; model training resumable from checkpoints):
  describe       item 1: states at s_c of the random rays in the mechanism planes (descriptive numbers + figure data)
  benchmark      (--frac < 1 only) seconds per epoch of each model type on the FULL training set (time estimate)
  quad_logistic  (h5b) convex quadratic logistic in 9D: the strict single-quadric threshold
  ladder         M_1..M_4 (random restarts + nested warm start), physics k = 2 (restarts), additive BCE variant
  additive       (h6) additive physics version u1/alpha3^2 + u2/alphay^2 = 1 by a direct theta search
  evaluate       c* on train rays, test metric with bootstrap, restart diagnostics, paired differences, CRITERION
  structure      eigenvalues and principal axes of every Q_i (raw and canonical basis), k = 2 plane data
  family         frozen models (and Gate 3 references) on the eps > 0 noise-family rays vs the deterministic s_c
  gain_failure   item 3: gain (i) error vs the y energy fraction, the left peak of Gate 3 fig3 (b1), triple terms

    python code/scripts/gate3b_mfe.py --config code/configs/gate3b_mfe.yaml --run-dir <dir> [--frac 0.01]
"""
import argparse
import importlib.util
import json
import platform
import sys
import time
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

from committor import mechanism as mech  # noqa: E402
from committor.ranking import _make_mlp, fit_logistic, poly_features  # noqa: E402
from committor.systems.mfe import LAMINAR  # noqa: E402
from committor.threshold import RayCoordinate, fit_cstar, paired_diff, qinf, summarize  # noqa: E402
from committor.utils import (Status, atomic_save_npz, atomic_write_json, derive_seed, git_info, read_json,  # noqa: E402
                             stable_hash, tree_hash)

REPO = Path(__file__).resolve().parents[2]
SIM_CODE = ["mechanism.py", "threshold.py", "ranking.py", "utils.py", "systems/mfe.py"]
MODES = [f"a{i}" for i in range(1, 10)]


def load_gate3_module():
    spec = importlib.util.spec_from_file_location("gate3_mfe", REPO / "code" / "scripts" / "gate3_mfe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- model wrappers ------------------------------------
class QuadLogistic:
    """c = x^T Q x + b^T x + const (convex BCE fit on the 54 degree-2 monomials of x)."""

    def __init__(self, Q, b, const):
        self.Q, self.bv, self.const = np.asarray(Q), np.asarray(b), float(const)
        self.k = 1

    def score(self, X, chunk=2_000_000):
        shp = X.shape[:-1]
        Xf = np.asarray(X).reshape(-1, 9)
        out = np.empty(len(Xf))
        for i in range(0, len(Xf), chunk):
            x = Xf[i:i + chunk]
            out[i:i + chunk] = ((x @ self.Q) * x).sum(1) + x @ self.bv + self.const
        return out.reshape(shp)

    def physical(self):
        return self.Q[None], self.bv[None]


def coef_to_quadratic(info):
    """standardised logistic coefficients on poly_features(x, 2) -> (Q, b, const) acting on x."""
    w, mu, sd, b0 = info["coef_standardised"], info["feature_mean"], info["feature_std"], info["intercept"]
    coef = w / sd
    const = b0 - float(np.sum(w * mu / sd))
    b = coef[:9].copy()
    Q = np.zeros((9, 9))
    for n, (i, j) in enumerate(combinations_with_replacement(range(9), 2)):
        c = coef[9 + n]
        if i == j:
            Q[i, i] = c
        else:
            Q[i, j] = Q[j, i] = 0.5 * c
    return Q, b, const


class LinearU:
    """c = w1 u1 + w2 u2 (+ const) with u = (a3^2, |y|^2): the additive physics version."""

    def __init__(self, w, const=0.0):
        self.w, self.const, self.k = np.asarray(w, dtype=np.float64), float(const), 2

    def score(self, X):
        X = np.asarray(X)
        return self.w[0] * X[..., 2] ** 2 + self.w[1] * (X[..., 3:8] ** 2).sum(-1) + self.const

    def physical(self):
        Q = np.zeros((2, 9, 9))
        Q[0, 2, 2] = 1.0
        for i in range(3, 8):
            Q[1, i, i] = 1.0
        return Q, np.zeros((2, 9))


class TorchModel:
    def __init__(self, model):
        self.m, self.k = model, model.k

    def score(self, X):
        return mech.score(self.m, X)

    def physical(self):
        return self.m.physical()


def eig_report(Q, b, Xb, top=4):
    """Eigen-structure of x^T Q x + b^T x (physical units) on boundary states Xb (numpy (n, 9))."""
    Qs = 0.5 * (Q + Q.T)
    lam, V = np.linalg.eigh(Qs)
    order = np.argsort(-np.abs(lam))
    lam, V = lam[order], V[:, order]
    scale = np.max(np.abs(lam)) if np.max(np.abs(lam)) > 0 else 1.0
    tol = 1e-3 * scale
    quad = np.einsum("nj,jl,nl->n", Xb, Qs, Xb)
    lin = Xb @ b
    axes = []
    for i in range(len(lam)):
        comp = V[:, i] ** 2
        axes.append({"eigenvalue": float(lam[i]), "eigenvalue_rel": float(lam[i] / scale),
                     "composition": {MODES[j]: float(comp[j]) for j in range(9)},
                     "main_modes": [f"{MODES[j]} {comp[j]:.2f}" for j in np.argsort(-comp) if comp[j] >= 0.05],
                     "vector": V[:, i].tolist()})
    # contribution of each eigen-direction to the quadratic part at the boundary states (|lam_i (v_i.x)^2| share)
    proj = (Xb @ V) ** 2 * lam[None, :]
    share = np.abs(proj) / np.maximum(np.abs(proj).sum(1, keepdims=True), 1e-300)
    for i in range(len(lam)):
        axes[i]["median_share_at_boundary"] = float(np.median(share[:, i]))
    nb_ = float(np.linalg.norm(b))
    bcomp = (np.asarray(b) ** 2 / nb_ ** 2) if nb_ > 0 else np.zeros(9)
    return {"n_positive": int(np.sum(lam > tol)), "n_negative": int(np.sum(lam < -tol)), "n_near_zero": int(np.sum(np.abs(lam) <= tol)),
            "definite": bool(np.all(lam > tol) or np.all(lam < -tol)),
            "b_norm_phys": nb_,
            "b_composition": {MODES[j]: float(bcomp[j]) for j in range(9)},
            "b_main_modes": [f"{MODES[j]} {bcomp[j]:.2f}" for j in np.argsort(-bcomp) if bcomp[j] >= 0.05],
            "median_linear_share_at_boundary": float(np.median(np.abs(lin) / np.maximum(np.abs(lin) + np.abs(quad), 1e-300))),
            "frac_boundary_u_negative": float(np.mean(quad + lin < 0)),
            "median_ratio_linear_to_quadratic_at_boundary": float(np.median(np.abs(lin) / np.maximum(np.abs(quad), 1e-300))),
            "axes_top": axes[:top], "axes_all": axes}


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
    ident = {"config_hash": stable_hash(cfg), "frac": args.frac,
             "sim_code_hash": tree_hash([src / f for f in SIM_CODE] + [Path(__file__).resolve()])}
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
    seed3 = int(cfg["data"]["gate3_seed"])

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

    # ------------------------------------------------------------ Gate 2 / Gate 3 data -------------------------------
    g2, g3 = REPO / cfg["data"]["gate2_run"], REPO / cfg["data"]["gate3_run"]
    R2, R3 = json.loads((g2 / "results.json").read_text()), json.loads((g3 / "results.json").read_text())
    RY, SC, OF = np.load(g2 / "rays.npz"), np.load(g2 / "scan.npz"), np.load(g2 / "offsets.npz")
    rng_sub = np.random.default_rng(derive_seed(seed3, "frac"))              # the Gate 3 ray subset in --frac runs
    N_all = len(RY["dirs"])
    rays = np.arange(N_all) if args.frac >= 1 else np.sort(rng_sub.choice(N_all, max(30, int(round(args.frac * N_all))), replace=False))
    dirs, split = RY["dirs"][rays], RY["split"][rays]
    s_c, grid_out, s_grid = SC["s_c"][rays], SC["grid_out"][rays], SC["s_grid"]
    Xo_a, Lo = OF["X"][rays], OF["label"][rays].astype(np.int64)
    N, G = len(rays), len(s_grid)
    ln_sc = np.log(s_c)
    tr_r, va_r, te_r = (np.where(split == k)[0] for k in (0, 1, 2))
    Xg = s_grid[None, :, None] * dirs[:, None, :]                                     # x = a - lam on the dense grid (N, G, 9)
    Xo = Xo_a - LAMINAR[None, None, :]
    dense_idx = np.round(np.linspace(0, G - 1, int(cfg["data"]["dense_states_per_ray"]))).astype(int)
    TPc = cfg["threshold_prediction"]
    boot = np.random.default_rng(derive_seed(seed3, "boot")).integers(0, len(te_r), (int(TPc["bootstrap"]), len(te_r)))
    Xedge = s_c[:, None] * dirs                                                        # x at s_c
    res.setdefault("meta", {"n_rays": int(N), "split_counts": [int(len(tr_r)), int(len(va_r)), int(len(te_r))],
                            "grid_points": int(G), "gate2_run": cfg["data"]["gate2_run"], "gate3_run": cfg["data"]["gate3_run"],
                            "criterion": cfg["criterion"]})
    save()

    # ------------------------------------------------------------ item 1: description --------------------------------
    def st_describe():
        q = {"|a3|": np.abs(Xedge[:, 2]), "|a2|": np.abs(Xedge[:, 1]), "|y|": np.linalg.norm(Xedge[:, 3:8], axis=1),
             "1-a1": -Xedge[:, 0], "a9": Xedge[:, 8]}
        atomic_save_npz(run / "mech_edge.npz", abs_a3=q["|a3|"], abs_a2=q["|a2|"], abs_y=q["|y|"], one_minus_a1=q["1-a1"],
                        a9=q["a9"], split=split, rays=rays, s_c=s_c)
        nb = int(cfg["descriptive"]["n_bins"])
        out = {"n_rays": int(N), "quantiles_5_50_95": {k: [float(x) for x in np.quantile(v, [0.05, 0.5, 0.95])] for k, v in q.items()},
               "frac_1_minus_a1_positive": float(np.mean(q["1-a1"] > 0)), "planes": {}}
        planes = [("|a3|", "|y|", "|a2|"), ("|a2|", "|y|", "|a3|"), ("|a3|", "|a2|", "|y|")]
        for p, o, miss in planes:
            rec = {}
            for xa, ya in ((p, o), (o, p)):
                lx, ly = np.log(q[xa]), np.log(q[ya])
                ok = np.isfinite(lx) & np.isfinite(ly)
                edges = np.quantile(lx[ok], np.linspace(0, 1, nb + 1))
                bi = np.clip(np.searchsorted(edges, lx, side="right") - 1, 0, nb - 1)
                resid = np.full(N, np.nan)
                iqr_bins, med_bins, ctr = [], [], []
                for b_ in range(nb):
                    sel = ok & (bi == b_)
                    if sel.sum() < 5:
                        continue
                    mb = np.median(ly[sel])
                    resid[sel] = ly[sel] - mb
                    iqr_bins.append(float(np.subtract(*np.quantile(ly[sel], [0.75, 0.25]))))
                    med_bins.append(float(mb))
                    ctr.append(float(np.median(lx[sel])))
                iqr_all = float(np.subtract(*np.quantile(ly[ok], [0.75, 0.25])))
                rr = np.isfinite(resid)
                corr = {}
                for cn, cv in (("1-a1", q["1-a1"]), ("|1-a1|", np.abs(q["1-a1"])), ("a9", q["a9"]), ("|a9|", np.abs(q["a9"])),
                               (f"ln {miss}", np.log(q[miss]))):
                    cc = np.isfinite(cv) & rr
                    corr[cn] = float(spearmanr(resid[cc], cv[cc]).statistic)
                rec[f"ln {ya} given ln {xa}"] = {
                    "overall_IQR": iqr_all, "median_within_bin_IQR": float(np.median(iqr_bins)),
                    "ratio_within_to_overall": float(np.median(iqr_bins) / iqr_all) if iqr_all > 0 else None,
                    "median_abs_resid": float(np.median(np.abs(resid[rr]))),
                    "spearman_resid_with": corr, "bin_centres": ctr, "bin_medians": med_bins, "bin_IQR": iqr_bins}
            out["planes"][f"({p}, {o})"] = rec
        return out

    timed("describe", st_describe)

    # ------------------------------------------------------------ training set (as Gate 3) ---------------------------
    fit_rows = np.concatenate([tr_r, va_r])
    Xtr = np.concatenate([Xo[fit_rows].reshape(-1, 9), Xg[fit_rows][:, dense_idx].reshape(-1, 9)])
    ytr = np.concatenate([Lo[fit_rows].reshape(-1), grid_out[fit_rows][:, dense_idx].reshape(-1)]).astype(np.float64)
    sp_tr = np.concatenate([np.repeat(split[fit_rows], Xo.shape[1]), np.repeat(split[fit_rows], len(dense_idx))]).astype(np.int8)
    keep = ytr >= 0
    x_scale = np.sqrt(np.mean(Xtr[keep & (sp_tr == 0)] ** 2, axis=0))
    trm, vam = keep & (sp_tr == 0), keep & (sp_tr == 1)
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=device)  # noqa: E731
    Xtr_t, ytr_t, Xva_t, yva_t = T(Xtr[trm]), T(ytr[trm]), T(Xtr[vam]), T(ytr[vam])
    LC = cfg["ladder"]
    hidden = list(LC["g_hidden"])
    res.setdefault("training_set", {"n_train_states": int(trm.sum()), "n_val_states": int(vam.sum()),
                                    "n_undecided_dropped": int((~keep).sum()), "train_fraction_B": float(ytr[trm].mean()),
                                    "x_scale": x_scale.tolist()})
    save()

    # ------------------------------------------------------------ benchmark (time estimate, --frac only) -------------
    def st_benchmark():
        RYf, SCf, OFf = RY["dirs"], SC, OF
        spf = RY["split"]
        fr = np.where(spf <= 1)[0]
        Xb = np.concatenate([(OFf["X"][fr] - LAMINAR[None, None, :]).reshape(-1, 9),
                             (s_grid[None, dense_idx, None] * RYf[fr][:, None, :]).reshape(-1, 9)])
        yb = np.concatenate([OFf["label"][fr].reshape(-1), SCf["grid_out"][fr][:, dense_idx].reshape(-1)]).astype(np.float64)
        spb = np.concatenate([np.repeat(spf[fr], 20), np.repeat(spf[fr], len(dense_idx))])
        kb = yb >= 0
        Xt_, yt_ = T(Xb[kb & (spb == 0)]), T(yb[kb & (spb == 0)])
        Xv_, yv_ = T(Xb[kb & (spb == 1)]), T(yb[kb & (spb == 1)])
        out = {"n_train_states_full": int(len(yt_))}
        nep = int(cfg["timing"]["benchmark_epochs"])
        for name, k, kind in [("M1", 1, "free"), ("M2", 2, "free"), ("M3", 3, "free"), ("M4", 4, "free"), ("phys2", 2, "phys2")]:
            mdl = mech.QuadMech(k, hidden, x_scale, kind).to(device)
            gen = torch.Generator().manual_seed(1)
            mdl.reset_g(gen)
            mdl.init_random(gen, float(LC["init_scale_Q"]), Xt_, float(LC["tau_init_rel"]))
            opt = torch.optim.Adam(mdl.parameters(), lr=float(LC["lr"]))
            for rep in range(2):                                   # first pass: warm-up
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(nep):
                    opt.zero_grad()
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(mdl(Xt_), yt_)
                    loss.backward()
                    opt.step()
                    with torch.no_grad():
                        torch.nn.functional.binary_cross_entropy_with_logits(mdl(Xv_), yv_).item()
                torch.cuda.synchronize()
                dt_ep = (time.perf_counter() - t0) / nep
            out[name] = {"seconds_per_epoch_full": dt_ep}
            print(f"    {name}: {dt_ep * 1e3:.1f} ms/epoch at full size", flush=True)
        # grid evaluation + c* fit at full size (one model)
        Xgf = s_grid[None, :, None] * RYf[:, None, :]
        t0 = time.perf_counter()
        C = mech.score(mdl, Xgf)
        rc = RayCoordinate(C, s_grid, device)
        trf = np.where(spf == 0)[0]
        fit_cstar(rc, trf, np.log(SCf["s_c"][trf]))
        torch.cuda.synchronize()
        out["seconds_grid_eval_and_cstar_full"] = time.perf_counter() - t0
        del rc, C, Xgf
        n_runs = {"M1": 2 * int(LC["n_random_restarts"]) + 4, "M2": int(LC["n_random_restarts"]) + 2, "M3": int(LC["n_random_restarts"]) + 1,
                  "M4": int(LC["n_random_restarts"]) + 1, "phys2": int(cfg["physics_k2"]["n_random_restarts"])}
        n_eval = sum(n_runs.values()) + 4
        for ep_assumed in (2000, int(LC["max_epochs"])):
            tt = sum(n_runs[k] * out[k]["seconds_per_epoch_full"] * ep_assumed for k in n_runs)
            out[f"estimate_training_hours_if_{ep_assumed}_epochs"] = tt / 3600
        out["estimate_eval_hours"] = n_eval * out["seconds_grid_eval_and_cstar_full"] / 3600
        torch.cuda.empty_cache()
        return out

    if args.frac < 1:
        timed("benchmark", st_benchmark)

    # ------------------------------------------------------------ (h5b) quadratic logistic -----------------------------
    def st_quad_logistic():
        Phi, names = poly_features(Xtr[keep], 2)
        z, info = fit_logistic(Phi, ytr[keep], np.ones(int(keep.sum())), sp_tr[keep], TPc["logistic_l2_grid"], device)
        Q, b, const = coef_to_quadratic(info)
        atomic_save_npz(run / "model_quad_logistic.npz", Q=Q, b=b, const=np.array(const))
        return {"l2": info["l2"], "val_nll": info["val_nll"], "n_features": int(Phi.shape[1])}

    timed("quad_logistic", st_quad_logistic)
    zq = np.load(run / "model_quad_logistic.npz")
    quad_log = QuadLogistic(zq["Q"], zq["b"], float(zq["const"]))

    # ------------------------------------------------------------ ladder training ------------------------------------
    def fpath(tag):
        return run / f"model_{tag}.pt", run / f"ckpt_{tag}.pt"

    def train_one(tag, model):
        fin, ck = fpath(tag)
        info = mech.train(model, Xtr_t, ytr_t, Xva_t, yva_t, LC, ck, fin, desc=tag, progress=progress)
        print(f"    {tag}: val {info['val_nll']:.5f} train {info['train_nll']:.5f} best ep {info['best_epoch']} "
              f"run {info['epochs_run']}", flush=True)
        return {k: v for k, v in info.items() if k != "history_every_50"}

    def st_ladder():
        out = {}
        # physics k = 2 first: it is also a warm start of M2 (phys2 is a special case of M2; added after the 1% run)
        runs = {}
        for r in range(int(cfg["physics_k2"]["n_random_restarts"])):
            tag = f"phys2_r{r}"
            mdl = mech.QuadMech(2, hidden, x_scale, "phys2").to(device)
            gen = torch.Generator().manual_seed(derive_seed(seed, "phys2", r))
            mdl.reset_g(gen)
            mdl.init_random(gen, 0.0, Xtr_t, float(LC["tau_init_rel"]))
            runs[tag] = train_one(tag, mdl)
        out["phys2"] = {"restarts": runs, "chosen": min(runs, key=lambda t: runs[t]["val_nll"]), "rule": "lowest val NLL (h2)"}
        e3 = np.zeros((9, 9))
        e3[2, 2] = 1.0
        # (h5b) strict single-quadric threshold: k = 1 with g affine in z = asinh(u/tau)/5 (monotone: {c >= c*} = {u1 >= const})
        runs = {}
        for r in range(int(LC["n_random_restarts"])):
            tag = f"S1_r{r}"
            mdl = mech.QuadMech(1, [], x_scale, "free").to(device)
            gen = torch.Generator().manual_seed(derive_seed(seed, "strict", r))
            mdl.reset_g(gen)
            mdl.init_random(gen, float(LC["init_scale_Q"]), Xtr_t, float(LC["tau_init_rel"]))
            runs[tag] = train_one(tag, mdl)
        for tag, (Qw, bw, src) in (("S1_warm", (quad_log.Q, quad_log.bv, "convex quadratic logistic")),
                                   ("S1_warm_phys", (e3, np.zeros(9), "u1 = a3^2"))):
            mdl = mech.QuadMech(1, [], x_scale, "free").to(device)
            gen = torch.Generator().manual_seed(derive_seed(seed, "strict", tag))
            mdl.reset_g(gen)
            mdl.set_quadratic(0, Qw, bw)
            mdl.set_tau(Xtr_t, float(LC["tau_init_rel"]), [0])
            runs[tag] = train_one(tag, mdl)
            runs[tag]["warm_from"] = src
        out["S1"] = {"restarts": runs, "chosen": min(runs, key=lambda t: runs[t]["val_nll"]), "rule": "lowest val NLL (h2)"}
        prev = None
        for k in [int(x) for x in LC["k"]]:
            runs = {}
            for r in range(int(LC["n_random_restarts"])):
                tag = f"M{k}_r{r}"
                mdl = mech.QuadMech(k, hidden, x_scale, "free").to(device)
                gen = torch.Generator().manual_seed(derive_seed(seed, "ladder", k, r))
                mdl.reset_g(gen)
                mdl.init_random(gen, float(LC["init_scale_Q"]), Xtr_t, float(LC["tau_init_rel"]))
                runs[tag] = train_one(tag, mdl)
            if LC["nested_warm_start"]:
                tag = f"M{k}_warm"
                mdl = mech.QuadMech(k, hidden, x_scale, "free").to(device)
                gen = torch.Generator().manual_seed(derive_seed(seed, "ladder", k, "warm"))
                mdl.reset_g(gen)
                if k == 1:
                    mdl.set_quadratic(0, quad_log.Q, quad_log.bv)
                    mdl.set_tau(Xtr_t, float(LC["tau_init_rel"]), [0])
                    src = "quadratic logistic"
                else:
                    mdl.init_random(gen, float(LC["init_scale_Q"]), Xtr_t, float(LC["tau_init_rel"]), slots=[k - 1])
                    pm, _ = mech.load(fpath(prev)[0], x_scale, device)
                    mdl.warm_from(pm)
                    with torch.no_grad():
                        d0 = float((mdl(Xva_t) - pm(Xva_t)).abs().max().item())
                    src = prev
                    print(f"    warm start {tag} from {prev}: max |c - c_prev| at start = {d0:.2e}", flush=True)
                runs[tag] = train_one(tag, mdl)
                runs[tag]["warm_from"] = src
            if LC["physics_warm_start"] and k in (1, 2):          # added after the 1% run (config history)
                tag = f"M{k}_warm_phys"
                mdl = mech.QuadMech(k, hidden, x_scale, "free").to(device)
                gen = torch.Generator().manual_seed(derive_seed(seed, "ladder", k, "warm_phys"))
                mdl.reset_g(gen)
                if k == 1:
                    mdl.set_quadratic(0, e3, np.zeros(9))
                    mdl.set_tau(Xtr_t, float(LC["tau_init_rel"]), [0])
                    src = "u1 = a3^2, g random"
                else:
                    pm, _ = mech.load(fpath(out["phys2"]["chosen"])[0], x_scale, device)
                    mdl.copy_from_phys2(pm)
                    with torch.no_grad():
                        d0 = float((mdl(Xva_t) - pm(Xva_t)).abs().max().item())
                    src = out["phys2"]["chosen"] + " (exact copy)"
                    print(f"    warm start {tag} from {src}: max |c - c_phys2| at start = {d0:.2e}", flush=True)
                runs[tag] = train_one(tag, mdl)
                runs[tag]["warm_from"] = src
            chosen = min(runs, key=lambda t: runs[t]["val_nll"])
            out[f"M{k}"] = {"restarts": runs, "chosen": chosen, "rule": "lowest val NLL (h2)"}
            prev = chosen
            print(f"    M{k}: chosen {chosen}", flush=True)
        # additive BCE variant: linear logistic on (u1, u2), signs free
        U2 = np.stack([Xtr[keep][:, 2] ** 2, (Xtr[keep][:, 3:8] ** 2).sum(1)], axis=1)
        z, info = fit_logistic(U2, ytr[keep], np.ones(int(keep.sum())), sp_tr[keep], TPc["logistic_l2_grid"], device)
        w = info["coef_standardised"] / info["feature_std"]
        const = info["intercept"] - float(np.sum(info["coef_standardised"] * info["feature_mean"] / info["feature_std"]))
        atomic_save_npz(run / "model_phys2_additive_bce.npz", w=w, const=np.array(const))
        out["phys2_additive_bce"] = {"l2": info["l2"], "val_nll": info["val_nll"], "w": w.tolist(), "const": const}
        return out

    timed("ladder", st_ladder)
    L = res["ladder"]

    # ------------------------------------------------------------ (h6) additive version by direct theta search -------
    q3, qy = dirs[:, 2] ** 2, (dirs[:, 3:8] ** 2).sum(1)

    def st_additive():
        S1 = float(np.median(q3[tr_r] * s_c[tr_r] ** 2))
        S2 = float(np.median(qy[tr_r] * s_c[tr_r] ** 2))
        th = np.linspace(0.0, np.pi / 2, int(cfg["physics_k2"]["additive_theta_grid"]))
        m_t = torch.as_tensor(ln_sc[tr_r], device=device)
        q3t, qyt = torch.as_tensor(q3[tr_r], device=device), torch.as_tensor(qy[tr_r], device=device)
        meds, ts = np.empty(len(th)), np.empty(len(th))
        for i, t_ in enumerate(th):
            wq = np.cos(t_) / S1 * q3t + np.sin(t_) / S2 * qyt
            m = m_t + 0.5 * torch.log(wq)                       # ln s_pred - ln s_c = 0.5 ln c* - m
            mf = m[torch.isfinite(m)]
            cand = torch.quantile(mf, torch.linspace(0, 1, 1000, device=device, dtype=torch.float64))
            d = torch.abs(cand[:, None] - m[None, :])
            d = torch.where(torch.isnan(d), torch.full_like(d, float("inf")), d)
            med = torch.sort(d, dim=1).values
            n = d.shape[1]
            med = 0.5 * (med[:, (n - 1) // 2] + med[:, n // 2])
            j = int(torch.argmin(med))
            lo, hi = cand[max(j - 1, 0)], cand[min(j + 1, len(cand) - 1)]
            fine = torch.linspace(float(lo), float(hi), 1000, device=device, dtype=torch.float64)
            d = torch.abs(fine[:, None] - m[None, :])
            d = torch.where(torch.isnan(d), torch.full_like(d, float("inf")), d)
            s_ = torch.sort(d, dim=1).values
            mf2 = 0.5 * (s_[:, (n - 1) // 2] + s_[:, n // 2])
            jf = int(torch.argmin(mf2))
            meds[i], ts[i] = float(mf2[jf]), float(fine[jf])
        ib = int(np.argmin(meds))
        w = np.array([np.cos(th[ib]) / S1, np.sin(th[ib]) / S2])
        atomic_save_npz(run / "model_phys2_additive_theta.npz", w=w, theta=np.array(th[ib]), theta_grid=th, train_median_analytic=meds)
        return {"theta_best": float(th[ib]), "theta_best_deg": float(np.degrees(th[ib])), "train_median_analytic": float(meds[ib]),
                "train_median_at_theta_0_(pure_a3)": float(meds[0]), "train_median_at_theta_90_(pure_y)": float(meds[-1]),
                "w": w.tolist(), "S1_S2": [S1, S2],
                "note": "analytic search: c = w.u is s^2 (w.q) along a ray, so s_pred = sqrt(c*/(w.q)); the final c* and metric "
                        "use the standard grid rule (stage evaluate)"}

    timed("additive", st_additive)
    za = np.load(run / "model_phys2_additive_theta.npz")
    add_theta = LinearU(za["w"])
    zb = np.load(run / "model_phys2_additive_bce.npz")
    add_bce = LinearU(zb["w"], float(zb["const"]))

    def tmodel(tag):
        m_, _ = mech.load(fpath(tag)[0], x_scale, device)
        return TorchModel(m_)

    S1N = "S1: single quadric threshold (k = 1 strict)"
    MODELS = {"S1: single quadric threshold (k = 1 strict)": tmodel(L["S1"]["chosen"]),
              "quadratic logistic (9D, convex BCE; diagnostic)": quad_log}
    for k in [int(x) for x in LC["k"]]:
        MODELS[f"M{k}"] = tmodel(L[f"M{k}"]["chosen"])
    MODELS["phys2: g(a3^2, |y|^2)"] = tmodel(L["phys2"]["chosen"])
    MODELS["phys2 additive (theta search)"] = add_theta
    MODELS["phys2 additive (BCE, signs free)"] = add_bce

    # ------------------------------------------------------------ evaluation -----------------------------------------
    G3T = np.load(g3 / "test_errors.npz")
    g3_names = list(G3T["names"])
    g3_pos = {int(r): i for i, r in enumerate(G3T["test_rays"])}
    te_ids = rays[te_r]
    have_g3 = all(int(r) in g3_pos for r in te_ids)
    REFS = ["E", "|a3|", "triple (1-a1, a2, a3)", "MLP (9D, polar input)", "gain (i)"]
    ref_err = {n: G3T["E"][g3_names.index(n)][[g3_pos[int(r)] for r in te_ids]] for n in REFS} if have_g3 else {}

    def eval_coord(model):
        C = model.score(Xg)
        rc = RayCoordinate(C, s_grid, device)
        fit = fit_cstar(rc, tr_r, ln_sc[tr_r])
        thr = [fit["side"] * fit["c_star"]]
        out = {"side": fit["side"], "c_star": fit["c_star"], "train_median_abs": fit["train_median_abs"]}
        ln_v, _ = rc.ln_s_pred(thr, fit["side"], va_r)
        out["val_median_abs"] = float(np.median(np.abs(ln_v[:, 0].cpu().numpy() - ln_sc[va_r])))
        ln_p, below = rc.ln_s_pred(thr, fit["side"], te_r)
        e = ln_p[:, 0].cpu().numpy() - ln_sc[te_r]
        out["frac_below_grid_test"] = float(below[:, 0].float().mean().item())
        del rc
        torch.cuda.empty_cache()
        return out, e

    def st_evaluate():
        out = {"models": {}, "restarts": {}, "paired": [], "references_gate3": {}}
        errs = {}
        for name, mdl in MODELS.items():
            row, e = eval_coord(mdl)
            row["test"] = summarize(e, boot)
            errs[name] = e
            out["models"][name] = row
            print(f"    {name:40s} train {row['train_median_abs']:.4f} val {row['val_median_abs']:.4f} test {row['test']['median_abs']:.4f} "
                  f"q90 {row['test']['q90_abs']:.4f}", flush=True)
        for fam in ["S1"] + [f"M{k}" for k in LC["k"]] + ["phys2"]:
            for tag in L[fam]["restarts"]:
                row, e = eval_coord(tmodel(tag))
                row["test_median_abs"] = float(np.median(np.abs(e)))
                row["val_nll"] = L[fam]["restarts"][tag]["val_nll"]
                row["chosen"] = tag == L[fam]["chosen"]
                out["restarts"][tag] = row
        for n in REFS:
            if n in ref_err:
                out["references_gate3"][n] = summarize(ref_err[n], boot)
        allerr = dict(errs)
        allerr.update({f"Gate 3: {n}": v for n, v in ref_err.items()})
        np.savez(run / "test_errors.npz", names=np.array(list(allerr)), E=np.stack([allerr[k] for k in allerr]), test_rays=te_ids)
        pairs = [("M1", "S1: single quadric threshold (k = 1 strict)"),
                 ("S1: single quadric threshold (k = 1 strict)", "quadratic logistic (9D, convex BCE; diagnostic)"),
                 ("phys2 additive (theta search)", "S1: single quadric threshold (k = 1 strict)")]
        ks = [int(x) for x in LC["k"]]
        pairs += [(f"M{k}", f"M{k - 1}") for k in ks[1:]]
        pairs += [("phys2: g(a3^2, |y|^2)", "M2"), ("phys2 additive (theta search)", "phys2: g(a3^2, |y|^2)"),
                  ("phys2 additive (BCE, signs free)", "phys2 additive (theta search)")]
        if have_g3:
            pairs += [(f"M{k}", "Gate 3: MLP (9D, polar input)") for k in ks]
            pairs += [(f"M{k}", "Gate 3: triple (1-a1, a2, a3)") for k in ks]
            pairs += [("phys2: g(a3^2, |y|^2)", "Gate 3: |a3|"), ("phys2 additive (theta search)", "Gate 3: |a3|"),
                      ("S1: single quadric threshold (k = 1 strict)", "Gate 3: triple (1-a1, a2, a3)")]
        for a_, b_ in pairs:
            out["paired"].append({"a": a_, "b": b_, **paired_diff(allerr[a_], allerr[b_], boot)})
        # criterion (fixed in the config before the run)
        Cc = cfg["criterion"]
        med = {f"M{k}": out["models"][f"M{k}"]["test"]["median_abs"] for k in ks}
        passing = [k for k in ks if med[f"M{k}"] <= float(Cc["pass_median"])]
        if passing:
            verdict = f"smallest k with test median <= {Cc['pass_median']}: k = {min(passing)}"
        elif med[f"M{max(ks)}"] >= float(Cc["fail_median_at_k4"]):
            verdict = f"k = {max(ks)} median >= {Cc['fail_median_at_k4']}: quadratic mechanism quantities are not enough"
        else:
            verdict = (f"intermediate: no k reaches {Cc['pass_median']}; k = {max(ks)} median {med[f'M{max(ks)}']:.4f} "
                       f"is below {Cc['fail_median_at_k4']}")
        # nesting check (config history, review): a chosen M_k must not have a higher val NLL than a model it contains
        vn = {f"M{k}": L[f"M{k}"]["restarts"][L[f"M{k}"]["chosen"]]["val_nll"] for k in ks}
        vn["phys2"] = L["phys2"]["restarts"][L["phys2"]["chosen"]]["val_nll"]
        vn["S1"] = L["S1"]["restarts"][L["S1"]["chosen"]]["val_nll"]
        subs = {1: ["S1"], 2: ["M1", "phys2"]}
        viol = []
        for k in ks:
            for sname in subs.get(k, [f"M{k - 1}"]):
                if vn[f"M{k}"] > vn[sname] + 1e-12:
                    viol.append(f"M{k} val NLL {vn[f'M{k}']:.5f} > {sname} {vn[sname]:.5f}")
        if viol:
            verdict += " | NESTING VIOLATED (optimisation, not the class): " + "; ".join(viol)
        out["criterion"] = {"medians": med, "smallest_k_pass": (min(passing) if passing else None), "verdict": verdict,
                            "ci": {f"M{k}": out["models"][f"M{k}"]["test"]["median_abs_ci"] for k in ks},
                            "nesting_val_nll": vn, "nesting_violations": viol}
        print("    CRITERION:", verdict, flush=True)
        return out

    timed("evaluate", st_evaluate)

    # ------------------------------------------------------------ structure of the quadratic forms --------------------
    Xb_tr = Xedge[tr_r]

    def canon(model):
        """canonical basis (h7) for a torch model: standardise u over the train boundary states, rotate to eig(J)."""
        m_ = model.m
        Ub = torch.as_tensor(Xb_tr, device=device)
        with torch.no_grad():
            u = m_.u(Ub)
        sd = u.std(0)
        uu = u.clone().requires_grad_(True)
        c = m_.from_u(uu)
        grad, = torch.autograd.grad(c.sum(), uu)
        gs = (grad * sd[None, :]).detach()
        J = (gs.T @ gs / len(gs)).cpu().numpy()
        lamJ, V = np.linalg.eigh(J)
        o = np.argsort(-lamJ)
        lamJ, V = lamJ[o], V[:, o]
        Q, b = model.physical()
        sdn = sd.cpu().numpy()
        Qc = np.einsum("ij,ikl->jkl", V / sdn[:, None], Q)
        bc = (V / sdn[:, None]).T @ b
        W = (u.detach() / sd[None, :]).cpu().numpy() @ V                     # canonical coordinates at the boundary states
        cb = c.detach().cpu().numpy()
        rho = np.array([spearmanr(W[:, j], cb).statistic for j in range(W.shape[1])])
        sign_g = np.where(rho < 0, -1.0, 1.0)                                # orient each canonical u to correlate positively with c
        slope = (gs.cpu().numpy() @ V) * sign_g[None, :]
        Qc, bc = Qc * sign_g[:, None, None], bc * sign_g[:, None]
        # parametrisation-free importance: increase of the val NLL when canonical coordinate j is held at its median
        sub = Xva_t[:: max(1, len(Xva_t) // 100_000)]
        ysub = yva_t[:: max(1, len(Xva_t) // 100_000)]
        with torch.no_grad():
            uv = m_.u(sub)
            Wv = (uv / sd[None, :]) @ torch.as_tensor(V, device=device)
            base = torch.nn.functional.binary_cross_entropy_with_logits(m_.from_u(uv), ysub).item()
            imp = []
            for j in range(Wv.shape[1]):
                W2 = Wv.clone()
                W2[:, j] = torch.median(Wv[:, j])
                u2 = (W2 @ torch.as_tensor(V.T, device=device)) * sd[None, :]
                imp.append(torch.nn.functional.binary_cross_entropy_with_logits(m_.from_u(u2), ysub).item() - base)
        return {"J_eigenvalues": lamJ.tolist(), "J_share": (lamJ / lamJ.sum()).tolist(), "rotation": V.tolist(),
                "orientation_spearman_with_c": rho.tolist(), "frac_positive_slope": (slope > 0).mean(0).tolist(),
                "ablation_val_nll_increase": imp, "val_nll_subsample": base,
                "u_sd_at_train_boundary": sdn.tolist(), "tau": torch.exp(m_.log_tau).detach().cpu().tolist()}, Qc, bc

    def st_structure():
        out = {}
        Q, b = quad_log.physical()
        out["quadratic logistic (9D, convex BCE; diagnostic)"] = {"raw": [eig_report(Q[0], b[0], Xb_tr)]}
        Q, b = MODELS[S1N].physical()
        info, Qc, bc = canon(MODELS[S1N])
        out[S1N] = {"raw": [eig_report(Q[0], b[0], Xb_tr)], "canonical": info, "canonical_forms": [eig_report(Qc[0], bc[0], Xb_tr)]}
        for k in [int(x) for x in LC["k"]]:
            mdl = MODELS[f"M{k}"]
            Q, b = mdl.physical()
            rec = {"raw": [eig_report(Q[i], b[i], Xb_tr) for i in range(k)]}
            info, Qc, bc = canon(mdl)
            rec["canonical"] = info
            rec["canonical_forms"] = [eig_report(Qc[j], bc[j], Xb_tr) for j in range(k)]
            # agreement between models: cosine (Frobenius, scaled units) of the leading canonical form with M1's form
            out[f"M{k}"] = rec
            np.savez(run / f"quadratic_forms_M{k}.npz", Q=Q, b=b, Qc=Qc, bc=bc)
        info, _, _ = canon(MODELS["phys2: g(a3^2, |y|^2)"])
        out["phys2: g(a3^2, |y|^2)"] = {"canonical": info}
        # similarity of the leading canonical forms to simple physical forms (scaled units, Frobenius cosine)
        S = x_scale

        def cos(Qa, Qb):
            A_, B_ = Qa * S[:, None] * S[None, :], Qb * S[:, None] * S[None, :]
            return float(np.sum(A_ * B_) / (np.linalg.norm(A_) * np.linalg.norm(B_)))
        simple = {"a3^2": np.diag(np.eye(9)[2]), "a2^2": np.diag(np.eye(9)[1]), "|y|^2": np.diag([0, 0, 0, 1, 1, 1, 1, 1, 0.0]),
                  "a1^2": np.diag(np.eye(9)[0]), "a9^2": np.diag(np.eye(9)[8])}
        out["cosine_to_simple_forms"] = {}
        for k in [int(x) for x in LC["k"]]:
            z = np.load(run / f"quadratic_forms_M{k}.npz")
            out["cosine_to_simple_forms"][f"M{k}"] = [{n: cos(z["Qc"][j], Qs_) for n, Qs_ in simple.items()} for j in range(k)]
        Qq, _ = quad_log.physical()
        out["cosine_to_simple_forms"]["quadratic logistic"] = [{n: cos(Qq[0], Qs_) for n, Qs_ in simple.items()}]
        # k = 2 plane data
        ev = res["evaluate"]["models"]
        Xb_te = torch.as_tensor(Xedge[te_r], device=device)
        for name, fn in (("M2", "plane_M2.npz"), ("phys2: g(a3^2, |y|^2)", "plane_phys2.npz")):
            m_ = MODELS[name].m
            with torch.no_grad():
                ub = m_.u(Xb_te).cpu().numpy()
                tau = torch.exp(m_.log_tau).cpu().numpy()
            if name == "M2":
                zb = np.arcsinh(ub / tau)
                lo, hi = np.quantile(zb, 0.002, axis=0) - 1.0, np.quantile(zb, 0.998, axis=0) + 1.0
                g1, g2_ = np.linspace(lo[0], hi[0], 400), np.linspace(lo[1], hi[1], 400)
                Z1, Z2 = np.meshgrid(g1, g2_, indexing="ij")
                Ugrid = np.stack([np.sinh(Z1) * tau[0], np.sinh(Z2) * tau[1]], axis=-1)
                coords = zb
            else:
                lu = np.log(np.maximum(ub, 1e-300))
                lo, hi = np.quantile(lu, 0.002, axis=0) - 1.0, np.quantile(lu, 0.998, axis=0) + 1.0
                g1, g2_ = np.linspace(lo[0], hi[0], 400), np.linspace(lo[1], hi[1], 400)
                Z1, Z2 = np.meshgrid(g1, g2_, indexing="ij")
                Ugrid = np.stack([np.exp(Z1), np.exp(Z2)], axis=-1)
                coords = lu
            with torch.no_grad():
                Cgrid = m_.from_u(torch.as_tensor(Ugrid.reshape(-1, 2), device=device)).cpu().numpy().reshape(400, 400)
            r_ = ev[name]
            extra = {}
            if name != "M2":
                ra, rb = ev["phys2 additive (theta search)"], ev["phys2 additive (BCE, signs free)"]
                extra = {"add_theta_w": add_theta.w, "add_theta_cstar": np.array(ra["c_star"]),
                         "add_bce_w": add_bce.w, "add_bce_const": np.array(add_bce.const), "add_bce_cstar": np.array(rb["c_star"])}
            atomic_save_npz(run / fn, g1=g1, g2=g2_, C=Cgrid, c_star=np.array(r_["c_star"]), side=np.array(r_["side"]),
                            boundary=coords, tau=tau, **extra)
        ra = ev["phys2 additive (theta search)"]
        w = np.asarray(add_theta.w)
        out["additive_alphas"] = {"alpha3": float(np.sqrt(ra["c_star"] / w[0])) if w[0] > 0 else None,
                                  "alphay": float(np.sqrt(ra["c_star"] / w[1])) if w[1] > 0 else None,
                                  "theta_deg": res["additive"]["theta_best_deg"]}
        return out

    timed("structure", st_structure)

    # ------------------------------------------------------------ noise family (eps > 0), frozen models --------------
    FD = np.load(g2 / "family_det.npz")
    fam_meta = R2["family_det"]["rays"]
    fam_dirs, fam_sc = FD["d"], FD["s_c"]
    fam_eps = np.array([f["eps"] for f in fam_meta])
    fam_ang = np.array([f["angle"] for f in fam_meta])
    use = np.isin(fam_eps, [float(e) for e in cfg["family"]["eps_used"]]) & np.isfinite(fam_sc)
    fam_grid = np.sqrt(np.geomspace(5e-7 * (1 + 1e-9), 0.29 * (1 - 1e-9), int(cfg["family"]["grid_points"])))
    Xf = fam_grid[None, :, None] * fam_dirs[:, None, :]                              # x

    def st_family():
        g3m = load_gate3_module()
        Xfa = Xf + LAMINAR[None, None, :]
        C3 = R3["threshold"]["candidates"]
        refs = {}
        refs["E"] = (g3m.feat(Xfa, "E"), C3["E"])
        refs["|a3|"] = (np.abs(Xfa[..., 2]), C3["|a3|"])
        zt = np.load(g3 / "model_triple_(1-a1,_a2,_a3).npz")
        tri = g3m.Logistic(["one_minus_a1", "a2", "a3"], 2, [], {"coef_standardised": zt["w"], "intercept": float(zt["b"]),
                                                                  "feature_mean": zt["mu"], "feature_std": zt["sd"]})
        refs["triple (1-a1, a2, a3)"] = (tri.score(Xfa), C3["triple (1-a1, a2, a3)"])
        ck = torch.load(g3 / "model_mlp_polar.pt", weights_only=False)
        net = _make_mlp(10, R3_hidden).to(device)
        net.load_state_dict(ck["state"])
        net.eval()
        mp = g3m.MLPScore({"net": net, "feature_mean": ck["mu"], "feature_std": ck["sd"]}, device, transform=g3m.polar)
        refs["MLP (9D, polar input)"] = (mp.score(Xfa), C3["MLP (9D, polar input)"])
        GF = np.load(g3 / "gain_family.npz")
        assert np.allclose(GF["s"], fam_grid)
        with np.errstate(divide="ignore"):                       # y0 = 0 on the eps = 0 rays (not used): ln 0 = -inf
            refs["gain (i)@0"] = (np.log(GF["max_trP"][:, :, 0]), C3["gain (i)@0"])
        out = {"n_rays_used": int(use.sum()), "eps": sorted(set(fam_eps[use].tolist())), "models": {}, "references_gate3": {}}

        def fam_eval(C, cstar, side):
            rc = RayCoordinate(C, fam_grid, device)
            ln_p, below = rc.ln_s_pred([side * cstar], side)
            e = ln_p[:, 0].cpu().numpy() - np.log(np.where(np.isfinite(fam_sc), fam_sc, 1.0))
            e[~use] = np.nan
            rec = {"by_eps": {}, "all": summarize(e[use])}
            for ep_ in sorted(set(fam_eps[use].tolist())):
                sel = use & (fam_eps == ep_)
                s_ = summarize(e[sel])
                s_["max_abs"] = float(np.max(np.abs(e[sel])))
                rec["by_eps"][f"{ep_:g}"] = s_
            rec["by_angle"] = {f"{a:g}": float(np.median(np.abs(e[use & (fam_ang == a)]))) for a in np.unique(fam_ang[use])}
            rec["per_ray"] = [{"ray": int(i), "angle": float(fam_ang[i]), "eps": float(fam_eps[i]), "ln_err": float(e[i]),
                               "below_grid": bool(below[i, 0].item())} for i in np.where(use)[0]]
            return rec

        ev = res["evaluate"]["models"]
        for name, mdl in MODELS.items():
            r_ = ev[name]
            out["models"][name] = fam_eval(mdl.score(Xf), r_["c_star"], r_["side"])
            print(f"    family {name:40s} " + "  ".join(f"eps {k}: {v['median_abs']:.4f}" for k, v in out["models"][name]["by_eps"].items()), flush=True)
        for name, (C, c3) in refs.items():
            out["references_gate3"][name] = fam_eval(C, c3["c_star"], c3["side"])
            print(f"    family ref {name:36s} " + "  ".join(f"eps {k}: {v['median_abs']:.4f}" for k, v in out["references_gate3"][name]["by_eps"].items()), flush=True)
        return out

    R3_hidden = yaml.safe_load((g3 / "config_used.yaml").read_text())["threshold_prediction"]["mlp"]["hidden"]
    timed("family", st_family)

    # ------------------------------------------------------------ item 3: gain (i) failure ---------------------------
    def st_gain_failure():
        GFc = cfg["gain_failure"]
        sig_list = [float(s) for s in R3["meta"]["sigma_list"]]
        kp = sig_list.index(float(GFc["sigma"]))
        fy = (dirs[:, 3:8] ** 2).sum(1) / (dirs ** 2).sum(1)
        f3 = dirs[:, 2] ** 2 / (dirs ** 2).sum(1)
        comp = dirs ** 2 / (dirs ** 2).sum(1, keepdims=True)
        zf_all = np.load(g3 / "zero_fit_generic_err_i.npy")[rays]                  # frozen c*_0, sigma = 3e-8, all rays
        rf_te = ref_err.get("gain (i)")                                            # refitted c*, test rays
        ev_all = np.load(g3 / "gain_i_at_edge.npy")[rays]
        # gain (i) at the edge for every sigma and t at the max (primary sigma), from the Gate 3 gain grid
        rows_per_chunk = 1000
        parts = [np.load(g3 / f"gain_grid_{ci:03d}.npz") for ci in range(int(np.ceil(N_all / rows_per_chunk)))]
        MT = np.concatenate([p["max_trP"] for p in parts])[rays]
        TM = np.concatenate([p["t_at_max"] for p in parts])[rays]
        del parts
        c_sig = {}
        for k, sg in enumerate(sig_list):
            rc = RayCoordinate(np.log(MT[:, :, k]), s_grid, device)
            c_sig[f"{sg:g}"] = rc.values_at(ln_sc)
            del rc
        j_hi = np.clip(np.searchsorted(s_grid, s_c), 1, G - 1)
        t_edge = np.where(s_c - s_grid[j_hi - 1] < s_grid[j_hi] - s_c, TM[np.arange(N), j_hi - 1], TM[np.arange(N), j_hi])
        ln_y0sq = np.log((Xedge[:, 3:8] ** 2).sum(1))
        nb = int(cfg["descriptive"]["n_bins"])

        def binned(xv, e, rows):
            xv, e = xv[rows], e[rows]
            edges = np.quantile(xv, np.linspace(0, 1, nb + 1))
            bi = np.clip(np.searchsorted(edges, xv, side="right") - 1, 0, nb - 1)
            out = []
            for b_ in range(nb):
                sel = bi == b_
                if not sel.any():
                    continue
                ee = e[sel]
                out.append({"x_median": float(np.median(xv[sel])), "n": int(sel.sum()), "median_abs": float(np.median(np.abs(ee))),
                            "q90_abs": float(qinf(np.abs(ee), 0.9)), "signed_median": float(np.median(ee)),
                            "frac_no_crossing": float(np.mean(np.isposinf(ee)))})
            fin = np.isfinite(e)
            return {"bins": out, "spearman_signed_finite": float(spearmanr(xv[fin], e[fin]).statistic),
                    "spearman_abs_inf_as_large": float(spearmanr(xv, np.where(np.isfinite(e), np.abs(e), 1e9)).statistic)}

        allr = np.arange(N)
        out = {"y_energy_fraction_definition": "|d_y|^2 / |d|^2 of the ray direction (= |y0|^2/|x0|^2 at any amplitude)",
               "zero_fit_vs_fy_all_rays": binned(fy, zf_all, allr),
               "zero_fit_vs_f3_all_rays": binned(f3, zf_all, allr)}
        if rf_te is not None:
            e_te = np.full(N, np.nan)
            e_te[te_r] = rf_te
            out["refit_vs_fy_test_rays"] = binned(fy, e_te, te_r)
            out["refit_vs_f3_test_rays"] = binned(f3, e_te, te_r)
        atomic_save_npz(run / "gain_failure.npz", fy=fy, f3=f3, zero_fit_err=zf_all, refit_err_test=(rf_te if rf_te is not None else np.array([])),
                        test_rows=te_r, c_edge=ev_all, c_edge_sigma0=c_sig["0"], t_edge=t_edge, ln_y0sq=ln_y0sq, comp=comp)
        # left peak
        lp = {}
        for cut in [float(GFc["left_peak_cut"])] + [float(c) for c in GFc["left_peak_cut_sensitivity"]]:
            P = np.isfinite(ev_all) & (ev_all < cut)
            rec = {"n": int(P.sum()), "frac": float(P.mean())}
            rec["mean_modal_fraction_peak"] = {MODES[i]: float(comp[P, i].mean()) for i in range(9)}
            rec["mean_modal_fraction_all"] = {MODES[i]: float(comp[:, i].mean()) for i in range(9)}
            rec["f3_quantiles_5_50_95_peak"] = [float(x) for x in np.quantile(f3[P], [0.05, 0.5, 0.95])]
            rec["f3_quantiles_5_50_95_all"] = [float(x) for x in np.quantile(f3, [0.05, 0.5, 0.95])]
            rec["fy_quantiles_5_50_95_peak"] = [float(x) for x in np.quantile(fy[P], [0.05, 0.5, 0.95])]
            for th in (1e-3, 1e-2):
                rec[f"frac_peak_with_f3_below_{th:g}"] = float(np.mean(f3[P] < th))
                rec[f"frac_of_rays_with_f3_below_{th:g}_in_peak"] = float(np.mean(P[f3 < th])) if np.any(f3 < th) else None
            rec["s_c_quantiles_5_50_95_peak"] = [float(x) for x in np.quantile(s_c[P], [0.05, 0.5, 0.95])]
            rec["s_c_quantiles_5_50_95_all"] = [float(x) for x in np.quantile(s_c, [0.05, 0.5, 0.95])]
            # numerical-floor checks
            rec["median_abs_c_sigma_3e-7_minus_c_sigma_0_peak"] = float(np.median(np.abs(c_sig["3e-07"][P] - c_sig["0"][P])))
            rec["median_abs_c_sigma_3e-8_minus_c_sigma_0_peak"] = float(np.median(np.abs(c_sig["3e-08"][P] - c_sig["0"][P])))
            rec["frac_t_at_max_zero_peak"] = float(np.mean(t_edge[P] == 0))
            rec["frac_t_at_max_zero_all"] = float(np.mean(t_edge == 0))
            rec["t_at_max_quantiles_5_50_95_peak"] = [float(x) for x in np.quantile(t_edge[P], [0.05, 0.5, 0.95])]
            g_ = ev_all[P] - ln_y0sq[P]
            rec["ln_growth_c_minus_ln_y0sq_quantiles_0_5_50_95_peak"] = [float(x) for x in np.quantile(g_, [0, 0.05, 0.5, 0.95])]
            rec["min_c_edge_peak"] = float(np.min(ev_all[P]))
            vals = np.round(ev_all[P], 6)
            rec["max_multiplicity_of_one_value_(1e-6)"] = int(np.max(np.unique(vals, return_counts=True)[1]))
            rec["min_max_trP_at_edge_peak"] = float(np.exp(np.min(ev_all[P])))
            zp = zf_all[P]
            rec["zero_fit_err_peak"] = summarize(zp)
            if rf_te is not None:
                Pt = P[te_r]
                rec["refit_err_peak_test_rays"] = summarize(rf_te[Pt]) if Pt.any() else None
                rec["refit_err_nonpeak_test_rays"] = summarize(rf_te[~Pt])
            lp[f"c<{cut:g}"] = rec
        out["left_peak"] = lp
        out["below_grid_edges"] = int(np.sum(s_c < s_grid[0]))
        # Gate 3 triple: terms
        zt = np.load(g3 / "model_triple_(1-a1,_a2,_a3).npz")
        _, names = poly_features(np.zeros((1, 3)), 2)
        lab = {"f0": "(1-a1)", "f1": "a2", "f2": "a3"}
        terms = []
        for nm in names:
            for k_, v_ in lab.items():
                nm = nm.replace(k_, v_)
            terms.append(nm)
        cfg3 = yaml.safe_load((g3 / "config_used.yaml").read_text())
        tri_cfg = [c for c in cfg3["candidates"] if c["name"] == "triple (1-a1, a2, a3)"][0]
        out["gate3_triple"] = {"config_entry": tri_cfg, "n_coefficients": int(len(zt["w"])), "terms": terms,
                               "coef_standardised": dict(zip(terms, [float(x) for x in zt["w"]])),
                               "contains_quadratic_terms": bool(len(zt["w"]) == 9 and tri_cfg.get("degree") == 2)}
        return out

    timed("gain_failure", st_gain_failure)
    print("done")


if __name__ == "__main__":
    main()
