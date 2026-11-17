#!/usr/bin/env python
"""Gate 4: DNS of the MFE flow - 4a solver checks, 4b pilot (A, B, T_det) and resolution ladder.
Config: code/configs/gate4_dns.yaml (criteria fixed before running).

    python code/scripts/gate4_dns.py --config code/configs/gate4_dns.yaml --run-dir <dir> --stage 4a|pilot|ladder
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

from committor.dns import Grid, coarse_grid, find_sc, lawson_rk4, run_nonlinear, step_nl
from committor.systems.mfe import LAMINAR, rhs_paper
from committor.utils import atomic_save_npz, atomic_write_json, derive_seed, git_info, read_json

REPO = Path(__file__).resolve().parents[2]
SHARED = ["mirror y about 1 (u,-v,w)(x,2-y,z)", "shift-reflect (u,v,-w)(x+Lx/2,y,-z)", "shift-rotate (-u,-v,w)(-x,-y,z+Lz/2)"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class NanMax:
    """running max that counts NaN explicitly instead of dropping it (fix of 2026-11-17: Python max() drops NaN)."""

    def __init__(self):
        self.v, self.n_nan = 0.0, 0

    def add(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        self.n_nan += int(np.sum(np.isnan(x)))
        if np.isfinite(x).any() or np.isinf(x).any():
            self.v = max(self.v, float(np.max(x[~np.isnan(x)])))
        return self


def make_grid(cfg, N, Re=None):
    S = cfg["system"]
    g = Grid(*N, Re=Re or S["Re"], Lx=S["Lx_over_pi"] * math.pi, Lz=S["Lz_over_pi"] * math.pi, Ly=S["Ly_ext"], device=cfg["device"])
    g.set_enforced(SHARED)
    return g


def stage_4a(cfg, run, res, save):
    C = cfg["checks_4a"]
    seed = int(cfg["seed"])
    N = cfg["grids"][0]
    g = make_grid(cfg, N)
    out = {"grid": N}
    # symmetry: which candidate symmetries leave all nine modes invariant (numerical check)
    sym = {}
    for k, (R, d, D) in g.candidate_symmetries().items():
        r = (g.sym_apply(g.modes, R, d, D) - g.modes).abs().amax(dim=(1, 2, 3, 4)).cpu().numpy()
        sym[k] = {"max_abs_gu_minus_u_per_mode": r.tolist(), "shared_by_all_nine": bool(np.all(r < 1e-12))}
    out["symmetries"] = sym
    out["enforced"] = [k for k, v in sym.items() if v["shared_by_all_nine"]]
    assert sorted(out["enforced"]) == sorted(SHARED), out["enforced"]
    # 1. laminar fixed point
    zero = torch.zeros_like(g.U)
    rhs0 = g.nl_full(zero)                                                     # nu lap 0 = 0
    Fh = g.fwd(torch.stack([torch.as_tensor(math.sqrt(2) * math.pi ** 2 / (4 * g.Re) * np.sin(np.pi * np.broadcast_to(g.ys[:, None, None], (g.Ny, g.Nz, g.Nx)) / 2), device=g.dev),
                            torch.zeros(g.Ny, g.Nz, g.Nx, device=g.dev, dtype=torch.float64),
                            torch.zeros(g.Ny, g.Nz, g.Nx, device=g.dev, dtype=torch.float64)])[None])
    full_lam = g.proj(g.fwd(g.cross(g.U_phys, g.wU_phys)) * g.mask) - g.nu * g.K2 * g.U + Fh
    out["1_laminar"] = {"max_abs_du_dt_perturbation_form": float(g.inv(rhs0).abs().max()),
                        "max_abs_du_dt_full_form": float(g.inv(full_lam).abs().max()),
                        "norm_U": float(g.energy(g.U).sqrt())}
    # 2. Galerkin identity
    rng = np.random.default_rng(derive_seed(seed, "galerkin"))
    d = rng.standard_normal((int(C["galerkin"]["n_states"]), 9))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    s = rng.uniform(0, float(C["galerkin"]["s_max"]), len(d))
    A = LAMINAR[None, :] + s[:, None] * d
    uh = g.from_coeffs(A)
    dudt = g.proj(g.fwd(g.cross(g.inv(uh), g.inv(g.curl(uh)))) * g.mask) - g.nu * g.K2 * uh + Fh
    proj = g.coeffs(dudt).cpu().numpy()
    ref = rhs_paper(A, g.Re, g.Lx, g.Lz)
    out["2_galerkin"] = {"max_abs_diff": float(np.max(np.abs(proj - ref))), "max_abs_rhs": float(np.max(np.abs(ref))),
                         "max_abs_diff_per_mode": np.max(np.abs(proj - ref), axis=0).tolist(),
                         "residual_outside_nine_modes_max_norm": float((g.energy(dudt - g.from_coeffs(proj)).sqrt()).max())}
    # 3. linear analytic solutions
    L = C["linear"]
    eps = float(L["amplitude"])
    rates = {"a1-1": (0, g.beta ** 2 / g.Re), "a3": (2, (g.beta ** 2 + g.gamma ** 2) / g.Re), "a9": (8, 9 * g.beta ** 2 / g.Re)}
    Ain = np.zeros((3, 9))
    for j, (k, (i, _)) in enumerate(rates.items()):
        Ain[j, i] = eps
    u = g.clean(g.from_coeffs(Ain))
    dt = 0.25
    nT = int(round(float(L["t_max"])))
    phi1 = torch.zeros_like(g.U)
    Y3, Z3, X3 = np.meshgrid(g.ys, g.zs, g.xs, indexing="ij")
    p1 = math.sqrt(2) * np.cos(g.gamma * Z3)
    p2 = 2 * np.cos(np.pi * Y3) * np.cos(g.gamma * Z3)
    sh = []
    for pp in (p1, p2):
        f = np.zeros((1, 3) + p1.shape)
        f[0, 0] = pp
        sh.append(g.fwd(torch.as_tensor(f, device=g.dev)))
    k3 = 2 / math.sqrt(4 * g.gamma ** 2 + math.pi ** 2)
    fcoef = [-k3 * g.gamma * math.pi / 2, -k3 * g.gamma * math.pi / (2 * math.sqrt(2))]       # forcing on the shapes per unit a3
    lam = [g.gamma ** 2 / g.Re, (math.pi ** 2 + g.gamma ** 2) / g.Re]
    d3 = rates["a3"][1]
    rel = {k: 0.0 for k in rates}
    rel_streak = [0.0, 0.0]
    other_streak = 0.0
    E1, E2 = g.lap_factor(dt), g.lap_factor(0.5 * dt)
    for t in range(1, nT + 1):
        for _ in range(4):
            u = step_nl(g, u, dt, E1, E2)
        a = g.coeffs(u).cpu().numpy()
        for j, (k, (i, rate)) in enumerate(rates.items()):
            rel[k] = max(rel[k], abs(a[j, i] / (eps * math.exp(-rate * t)) - 1))
        ub = u[1:2] * g.mask_kx0                                                                # kx = 0 part of the a3 run
        streak = ub.clone()
        streak[:, 1:] = 0
        streak[..., :, 0, :] = 0                                                                # kz = 0 removed
        cj = []
        for jj in range(2):
            cj.append(float(g.inner(streak, sh[jj]).item()))
            exact = eps * fcoef[jj] * (math.exp(-d3 * t) - math.exp(-lam[jj] * t)) / (lam[jj] - d3)
            rel_streak[jj] = max(rel_streak[jj], abs(cj[jj] / exact - 1))
        rest = float(g.energy(streak).item()) - cj[0] ** 2 - cj[1] ** 2
        other_streak = max(other_streak, math.sqrt(max(rest, 0.0)) / math.sqrt(cj[0] ** 2 + cj[1] ** 2))
    out["3_linear"] = {"rel_err_decay": rel, "rates": {k: v[1] for k, v in rates.items()},
                       "streak_rel_err": {"cos(gamma z)": rel_streak[0], "cos(pi y) cos(gamma z)": rel_streak[1]},
                       "streak_decay_rates": {"cos(gamma z)": lam[0], "cos(pi y) cos(gamma z)": lam[1]},
                       "streak_other_components_rel_norm_max": other_streak}
    # 4. nonlinear term does no work
    W = C["work"]
    gen = torch.Generator(device=g.dev).manual_seed(derive_seed(seed, "work") % (2 ** 62))
    shape = (int(W["n_fields"]), 3, g.Ny, g.Nz, g.Nxh)
    amp = torch.where(g.K2 > 0, 1.0 / torch.where(g.K2 > 0, g.K2, torch.ones_like(g.K2)), torch.zeros_like(g.K2))
    rr = torch.randn(shape, generator=gen, device=g.dev, dtype=torch.float64) + 1j * torch.randn(shape, generator=gen, device=g.dev, dtype=torch.float64)
    uw = g.proj(g.fwd(g.inv(rr * amp * g.mask)) * g.mask)
    Nw = g.selfcross(uw)
    ratio = (g.inner(uw, Nw).abs() / (g.energy(uw).sqrt() * g.energy(Nw).sqrt())).cpu().numpy()
    out["4_work"] = {"max_ratio": float(ratio.max()), "ratios": ratio.tolist(), "max_div_random_fields": float(g.max_div(uw).max())}
    # 5. dt
    D5 = C["dt"]
    rng = np.random.default_rng(derive_seed(seed, "dt"))
    dd = rng.standard_normal((int(D5["n_dirs"]), 9))
    dd /= np.linalg.norm(dd, axis=1, keepdims=True)
    u0 = g.clean(g.from_coeffs(math.sqrt(float(D5["E0"])) * dd))
    T5 = float(D5["t_max"])

    def run_E(dt_, sym_res=False):
        u = u0.clone()
        n = int(round(T5 / dt_))
        per = int(round(1.0 / dt_))
        Es = [g.energy(u).cpu().numpy()]
        smax, dmax, cmax = NanMax(), NanMax(), NanMax()
        E1, E2 = g.lap_factor(dt_), g.lap_factor(0.5 * dt_)
        for k in range(1, n + 1):
            (un,) = lawson_rk4(g, (u,), lambda s_: (g.nl_full(s_[0]),), dt_, E1, E2)
            un = un * g.mask
            if sym_res and k % per == 0:
                smax.add(g.sym_residual(un).cpu().numpy())
            u = g.symmetrize(un)
            if k % per == 0:
                Es.append(g.energy(u).cpu().numpy())
                if k % (10 * per) == 0:
                    dmax.add(g.max_div(u).cpu().numpy())
                    cmax.add(g.courant(u, dt_).cpu().numpy())
        return np.array(Es), smax, dmax, cmax, u
    cand = [float(x) for x in D5["candidates"]]
    dt_rows = []
    chosen = None
    Eref = {}
    for c_ in cand:
        if c_ not in Eref:
            Eref[c_] = run_E(c_, sym_res=(c_ == cand[0]))
        if c_ / 2 not in Eref:
            Eref[c_ / 2] = run_E(c_ / 2)
        Ea, Eb = Eref[c_][0], Eref[c_ / 2][0]
        relE = float(np.max(np.abs(Ea / Eb - 1)))
        with np.errstate(invalid="ignore", divide="ignore"):
            n_nan_rel = int(np.sum(~np.isfinite(np.abs(Ea / Eb - 1))))
        dt_rows.append({"dt": c_, "max_rel_dE_vs_half": relE, "n_nonfinite_rel_dE": n_nan_rel, "courant_max": Eref[c_][3].v,
                        "courant_n_nan": Eref[c_][3].n_nan})
        log(f"    dt {c_}: max rel dE {relE:.2e} (non-finite {n_nan_rel}), Courant {Eref[c_][3].v:.3f} (NaN {Eref[c_][3].n_nan})")
        if n_nan_rel == 0 and relE < float(D5["tol"]):
            chosen = c_
            break
    out["5_dt"] = {"rows": dt_rows, "chosen": chosen, "sym_residual_max": Eref[cand[0]][1].v, "sym_residual_n_nan": Eref[cand[0]][1].n_nan,
                   "div_max": max(v[2].v for v in Eref.values()), "div_n_nan": int(sum(v[2].n_nan for v in Eref.values())),
                   "E_range_t<=200": [float(np.nanmin(Eref[cand[0]][0])), float(np.nanmax(Eref[cand[0]][0]))],
                   "E_n_nan": int(np.sum(np.isnan(Eref[cand[0]][0])))}
    # 6. alone vs in a batch
    Bc = C["batch"]
    dtc = chosen if chosen is not None else cand[-1]
    rng = np.random.default_rng(derive_seed(seed, "batch"))
    others = rng.standard_normal((int(Bc["batch_size"]), 9))
    others /= np.linalg.norm(others, axis=1, keepdims=True)
    others *= math.sqrt(float(D5["E0"]))
    others[int(Bc["position"])] = math.sqrt(float(D5["E0"])) * dd[0]
    ub = g.clean(g.from_coeffs(others))
    ua = g.clean(g.from_coeffs(math.sqrt(float(D5["E0"])) * dd[:1]))
    E1, E2 = g.lap_factor(dtc), g.lap_factor(0.5 * dtc)
    n = int(round(T5 / dtc))
    per = int(round(1.0 / dtc))
    mdE = 0.0
    for k in range(1, n + 1):
        ua = step_nl(g, ua, dtc, E1, E2)
        ub = step_nl(g, ub, dtc, E1, E2)
        if k % per == 0:
            mdE = max(mdE, abs(float(g.energy(ua)[0] - g.energy(ub)[int(Bc["position"])])))
    out["6_batch"] = {"max_abs_dE": mdE, "max_abs_du_final": float((g.inv(ua) - g.inv(ub[int(Bc["position"]):int(Bc["position"]) + 1])).abs().max())}
    # criteria
    crit = {
        "1 laminar max|du/dt| < 1e-12": max(out["1_laminar"]["max_abs_du_dt_perturbation_form"], out["1_laminar"]["max_abs_du_dt_full_form"]) < float(C["laminar_rhs_max"]),
        "1 max|div u| < 1e-10": out["5_dt"]["div_n_nan"] == 0 and max(out["5_dt"]["div_max"], out["4_work"]["max_div_random_fields"]) < float(C["div_max"]),
        "1 symmetry residual < 1e-12": out["5_dt"]["sym_residual_n_nan"] == 0 and out["5_dt"]["sym_residual_max"] < float(C["sym_residual_max"]),
        "2 Galerkin max diff < 1e-10": out["2_galerkin"]["max_abs_diff"] < float(C["galerkin"]["tol"]),
        "3 decay rates rel err < 1e-8": max(out["3_linear"]["rel_err_decay"].values()) < float(L["tol_decay"]),
        "3 lift-up streak rel err < 1e-6": max(out["3_linear"]["streak_rel_err"].values()) < float(L["tol_streak"]),
        "4 work ratio < 1e-10": out["4_work"]["max_ratio"] < float(W["tol"]),
        "5 dt found": chosen is not None,
        "6 alone vs batch < 1e-12": max(out["6_batch"]["max_abs_dE"], out["6_batch"]["max_abs_du_final"]) < float(Bc["tol"]),
    }
    out["criteria"] = {k: bool(v) for k, v in crit.items()}
    out["all_pass"] = bool(all(crit.values()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stage", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
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
    t0 = time.time()
    if args.stage == "4a":
        log("--- 4a")
        res["4a"] = stage_4a(cfg, run, res, save)
        res["4a"]["seconds"] = time.time() - t0
        save()
        log("4a criteria:", json.dumps(res["4a"]["criteria"], indent=1))
        log("4a all pass:", res["4a"]["all_pass"])


if __name__ == "__main__":
    main()
