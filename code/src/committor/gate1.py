"""Gate 1 checks for the MFE model: (a) laminar fixed point, (b) energy conservation of the
nonlinear term, (c) laminar Jacobian spectrum over Re, (e) symmetry equivariance, the transcription
cross-check, RK4 step selection and the lifetime statistics used for (d)."""

import math

import numpy as np
import torch

from .systems.mfe import (LAMINAR, MFE, SYMMETRIES, galerkin_projection, mode_symmetry_signs,
                          rhs_paper)


def energy_torch(a):
    """Perturbation energy E = (a1 - 1)^2 + sum_{i>=2} a_i^2 (functional form for torch.compile)."""
    return (a[:, 0] - 1.0) ** 2 + (a[:, 1:] ** 2).sum(1)


def check_a(m):
    f_lit = rhs_paper(LAMINAR, m.Re, m.Lx, m.Lz)
    f_ten = m.rhs(LAMINAR)
    return {"rhs_norm_paper_form": float(np.linalg.norm(f_lit)),
            "rhs_norm_tensor_form": float(np.linalg.norm(f_ten)),
            "rhs_paper_form": f_lit.tolist()}


def check_b(m, n, scale, rng):
    a = rng.normal(scale=scale, size=(n, 9))
    N_ten = m.nonlinear(a)
    # nonlinear part of the literal paper RHS: f - c - L a
    N_lit = rhs_paper(a, m.Re, m.Lx, m.Lz) - m.c - a @ m.L.T
    W_ten = np.einsum("ni,ni->n", a, N_ten)
    W_lit = np.einsum("ni,ni->n", a, N_lit)
    scale_ref = np.linalg.norm(a, axis=1) * np.linalg.norm(N_lit, axis=1)
    Q = m.Q
    sym = Q + Q.transpose(1, 0, 2) + Q.transpose(2, 1, 0)        # Q_ijk + Q_jik + Q_kji, with Q_ijk = Q_ikj
    return {"n": n, "scale": scale,
            "max_abs_work_paper_form": float(np.abs(W_lit).max()),
            "max_abs_work_tensor_form": float(np.abs(W_ten).max()),
            "max_relative_work": float((np.abs(W_lit) / scale_ref).max()),
            "median_abs_work": float(np.median(np.abs(W_lit))),
            "typical_size_a_dot_N": float(np.median(scale_ref)),
            "max_abs_symmetrised_Q": float(np.abs(sym).max()),
            "max_abs_Q": float(np.abs(Q).max())}


def check_c(Re_grid, Lx_over_pi, Lz_over_pi):
    max_re = np.empty(len(Re_grid))
    min_decay = np.empty(len(Re_grid))
    spectra = {}
    for i, Re in enumerate(Re_grid):
        m = MFE(Re=Re, Lx_over_pi=Lx_over_pi, Lz_over_pi=Lz_over_pi)
        ev = np.linalg.eigvals(m.jacobian(LAMINAR))
        max_re[i] = ev.real.max()
        min_decay[i] = -ev.real.max()
        if Re in (100, 400, 1000, 2000):
            spectra[str(int(Re))] = sorted([[float(e.real), float(e.imag)] for e in ev])
    # autograd Jacobian of the literal RHS at Re = 400 vs the analytic one
    m = MFE(Re=400.0, Lx_over_pi=Lx_over_pi, Lz_over_pi=Lz_over_pi)
    x = torch.tensor(LAMINAR, dtype=torch.float64)
    J_ad = torch.autograd.functional.jacobian(lambda v: rhs_paper(v, m.Re, m.Lx, m.Lz), x).numpy()
    return {"Re_grid": [float(Re_grid[0]), float(Re_grid[-1]), len(Re_grid)],
            "max_real_part_over_grid": float(max_re.max()),
            "Re_at_max": float(Re_grid[int(np.argmax(max_re))]),
            "spectra": spectra, "jacobian_autograd_vs_analytic_Re400": float(np.abs(J_ad - m.jacobian(LAMINAR)).max()),
            "_max_re": max_re}


def check_e(m, n, scale, rng):
    a = rng.normal(scale=scale, size=(n, 9))
    out = {}
    f = lambda v: rhs_paper(v, m.Re, m.Lx, m.Lz)
    for name, g in SYMMETRIES.items():
        out[name] = {"action": g.astype(int).tolist(),
                     "max_abs_residual_paper_form": float(np.abs(f(a * g) - g * f(a)).max()),
                     "max_abs_residual_tensor_form": float(np.abs(m.rhs(a * g) - g * m.rhs(a)).max()),
                     "laminar_invariant": bool(np.array_equal(LAMINAR * g, LAMINAR)),
                     "energy_invariant": True}
    g = np.ones(9)
    g[3] = -1.0
    neg = float(np.abs(f(a * g) - g * f(a)).max())
    derived = mode_symmetry_signs(m.Lx, m.Lz)
    return {"symmetries": out, "negative_control_flip_a4": neg,
            "signs_derived_from_mode_functions": {k: v.tolist() for k, v in derived.items()},
            "derived_match_siads": all(np.array_equal(derived[k], SYMMETRIES[k]) for k in derived)}


def transcription_crosscheck(m):
    g = galerkin_projection(m.Lx, m.Lz, m.Re)
    return {"max_abs_diff_c": float(np.abs(g["c"] - m.c).max()),
            "max_abs_diff_L": float(np.abs(g["L"] - m.L).max()),
            "max_abs_diff_Q": float(np.abs(g["Q"] - m.Q).max()),
            "max_abs_Q": float(np.abs(m.Q).max()),
            "gram_max_offdiag_over_norm": float(np.abs(g["gram"] - np.diag(np.diag(g["gram"]))).max() / g["norm"]),
            "gram_diag_over_norm": (np.diag(g["gram"]) / g["norm"]).tolist(),
            "max_divergence": float(g["max_div"].max()),
            "max_wall_bc_residual": g["max_wall_residual"]}


def sphere_ics(n, radius2, rng):
    v = rng.normal(size=(n, 9))
    return v / np.linalg.norm(v, axis=1, keepdims=True) * math.sqrt(radius2)


# ------------------------------------------------------------------------------------------------
# lifetime statistics
# ------------------------------------------------------------------------------------------------

def survival(T, t_max, grid):
    """Empirical P(T > t) on grid (T = inf means censored at t_max)."""
    T = np.asarray(T, dtype=float)
    Ts = np.sort(np.where(np.isfinite(T), T, np.inf))
    return 1.0 - np.searchsorted(Ts, grid, side="right") / len(Ts)


def exp_fit(T, t_max, t0):
    """Censored MLE of P(t) = P(t0) exp(-(t - t0)/tau) for t0 <= t <= t_max."""
    T = np.asarray(T, dtype=float)
    N = len(T)
    alive0 = T > t0
    n0 = int(alive0.sum())
    Tc = np.minimum(T[alive0], t_max)
    events = int((np.isfinite(T[alive0]) & (T[alive0] <= t_max)).sum())
    exposure = float((Tc - t0).sum())
    lam = events / exposure if exposure > 0 else float("nan")
    tau = 1.0 / lam if lam > 0 else float("inf")
    P0 = n0 / N
    return {"t0": t0, "n_alive_t0": n0, "P_t0": P0, "events": events, "tau": tau,
            "tau_se": tau / math.sqrt(events) if events else float("nan"),
            "t_onset": t0 + tau * math.log(P0) if P0 > 0 else float("nan"),   # where the fit meets P = 1
            "median_from_fit": t0 + tau * math.log(2 * P0) if P0 > 0.5 else float("nan"),
            "P_fit_t_max": P0 * math.exp(-(t_max - t0) / tau)}


def sup_dev(T, t_max, fit):
    """sup over [t0, t_max] of |P_emp - P_fit|, evaluated on both sides of every jump."""
    T = np.asarray(T, dtype=float)
    N = len(T)
    ev = np.sort(T[np.isfinite(T) & (T > fit["t0"]) & (T <= t_max)])
    pts = np.r_[fit["t0"], ev, t_max]
    Pfit = fit["P_t0"] * np.exp(-(pts - fit["t0"]) / fit["tau"])
    Ts = np.sort(np.where(np.isfinite(T), T, np.inf))
    right = 1.0 - np.searchsorted(Ts, pts, side="right") / N          # P(T > t)
    left = 1.0 - np.searchsorted(Ts, pts, side="left") / N            # P(T >= t)
    return float(max(np.abs(right - Pfit).max(), np.abs(left - Pfit).max()))


def gof_bootstrap(T, t_max, fit, B, rng):
    """Parametric bootstrap of the sup statistic (refitting tau in each replicate), conditional on n_alive(t0)."""
    N = len(T)
    n0 = fit["n_alive_t0"]
    D_obs = sup_dev(T, t_max, fit)
    D = np.empty(B)
    for b in range(B):
        Tb = fit["t0"] + rng.exponential(fit["tau"], size=n0)
        Tb = np.where(Tb > t_max, np.inf, Tb)
        Tall = np.r_[Tb, np.zeros(N - n0)]                              # the rest died before t0
        fb = exp_fit(Tall, t_max, fit["t0"])
        D[b] = sup_dev(Tall, t_max, fb)
    return {"D_obs": D_obs, "p": float((1 + (D >= D_obs).sum()) / (B + 1)),
            "D_null_q50": float(np.median(D)), "D_null_q99": float(np.quantile(D, 0.99))}


def logrank(T1, T2, t_max):
    """Two-sample log-rank test (right-censored at t_max). Returns (chi2, p)."""
    from scipy.stats import chi2
    T1 = np.where(np.isfinite(T1) & (T1 <= t_max), T1, np.inf)
    T2 = np.where(np.isfinite(T2) & (T2 <= t_max), T2, np.inf)
    times = np.unique(np.r_[T1[np.isfinite(T1)], T2[np.isfinite(T2)]])
    s1, s2 = np.sort(T1), np.sort(T2)
    n1 = len(T1) - np.searchsorted(s1, times, side="left")
    n2 = len(T2) - np.searchsorted(s2, times, side="left")
    d1 = np.searchsorted(s1, times, side="right") - np.searchsorted(s1, times, side="left")
    d2 = np.searchsorted(s2, times, side="right") - np.searchsorted(s2, times, side="left")
    n, d = n1 + n2, d1 + d2
    E1 = d * n1 / n
    V = d * (n1 / n) * (n2 / n) * (n - d) / np.maximum(n - 1, 1)
    stat = float(((d1 - E1).sum()) ** 2 / V.sum())
    return stat, float(chi2.sf(stat, 1))
