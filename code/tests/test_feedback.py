"""Gate 3c: quasi-linear structure, W(T), u_th, analytic first crossing and the analytic c* rule."""
import numpy as np
import torch
from scipy.integrate import quad_vec
from scipy.linalg import expm

from committor.feedback import Feedback, analytic_cstar, first_crossing_abs, ray_pq, structure_check, u_th
from committor.systems.mfe import LAMINAR, MFE, rhs_paper
from committor.threshold import RayCoordinate, fit_cstar

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_mfe_is_exactly_quasilinear_and_a3_equation():
    s = structure_check(MFE(Re=400.0))
    for k in ("y_eq_yy", "y_eq_mm", "m_eq_my", "L_ym", "L_my", "c_y", "a3_eq_linear_except_a3", "a3_eq_quadratic_outside_yy"):
        assert s[k] == 0.0, k
    for k in ("y_eq_my", "m_eq_yy", "m_eq_mm"):
        assert s[k] > 0.1, k


def test_W_closed_form_lyapunov_and_ode():
    m = MFE(Re=400.0)
    fb = Feedback(m)
    assert np.abs(fb.B.T @ fb.W_inf + fb.W_inf @ fb.B + fb.F).max() < 1e-12
    Wq, _ = quad_vec(lambda s: np.exp(fb.d3 * s) * expm(fb.Ay.T * s) @ fb.F @ expm(fb.Ay * s), 0, 30.0, epsabs=1e-13, epsrel=1e-12)
    assert np.abs(fb.W(30.0) - Wq).max() < 1e-9
    assert np.abs(fb.W(1e7) - fb.W_inf).max() < 1e-9 and np.all(fb.W(0) == 0)
    # first-order formula against the full nonlinear ODE at small amplitude: a3(T) e^{d3 T} = a3(0) + y0^T W(T) y0
    rng = np.random.default_rng(3)
    y0 = rng.standard_normal(5)
    y0 /= np.linalg.norm(y0)
    eps, dt, n = 1e-5, 0.05, 400
    a = LAMINAR.copy()
    a[2] = 0.3 * eps ** 2
    a[3:8] = eps * y0
    f = lambda z: rhs_paper(z, m.Re, m.Lx, m.Lz)  # noqa: E731
    for _ in range(n):
        k1 = f(a); k2 = f(a + 0.5 * dt * k1); k3 = f(a + 0.5 * dt * k2); k4 = f(a + dt * k3)  # noqa: E702
        a = a + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    lhs = a[2] * np.exp(fb.d3 * n * dt) / eps ** 2
    rhs = 0.3 + y0 @ fb.W(n * dt) @ y0
    assert abs(lhs - rhs) < 1e-4 * max(1.0, abs(rhs))
    # u_th along a ray is p s + q s^2
    d = rng.standard_normal((4, 9))
    p, q = ray_pq(d, fb.W_inf)
    s = 0.013
    assert np.allclose(u_th(s * d, fb.W_inf), p * s + q * s ** 2, rtol=1e-12, atol=0)


def test_first_crossing_abs_matches_grid_rule():
    rng = np.random.default_rng(0)
    n = 400
    p = rng.standard_normal(n) * 0.5
    q = rng.standard_normal(n) * 50.0
    s_grid = np.geomspace(1e-4, 1.0, 20001)
    C = np.abs(p[:, None] * s_grid[None, :] + q[:, None] * s_grid[None, :] ** 2)
    rc = RayCoordinate(C, s_grid, DEV)
    for c in (1e-6, 1e-3, 3e-2, 1.0, 1e3):
        ln_g, _ = rc.ln_s_pred([c], 1)
        ln_g = ln_g[:, 0].cpu().numpy()
        s_a = first_crossing_abs(torch.as_tensor(p), torch.as_tensor(q), torch.tensor(c, dtype=torch.float64), s_grid[0], s_grid[-1]).numpy()
        ln_a = np.log(s_a)
        both = np.isfinite(ln_g) & np.isfinite(ln_a)
        assert np.array_equal(np.isfinite(ln_g), np.isfinite(ln_a)), c
        if both.any():
            assert np.max(np.abs(ln_g[both] - ln_a[both])) < 2e-3, c
    assert not np.isfinite(ln_a).any()          # c = 1e3: nothing crosses on the grid


def test_analytic_cstar_matches_fit_cstar():
    rng = np.random.default_rng(1)
    n = 600
    p = rng.standard_normal(n) * 0.3
    q = rng.standard_normal(n) * 30.0
    s_true = first_crossing_abs(torch.as_tensor(p), torch.as_tensor(q), torch.tensor(1e-3, dtype=torch.float64), 1e-9, 1e9).numpy()
    ln_sc = np.log(s_true) + 0.05 * rng.standard_normal(n)
    s_grid = np.geomspace(1e-4, 1.0, 4001)
    C = np.abs(p[:, None] * s_grid[None, :] + q[:, None] * s_grid[None, :] ** 2)
    rows = np.arange(n)
    fit = fit_cstar(RayCoordinate(C, s_grid, DEV), rows, ln_sc)
    med, c = analytic_cstar(torch.as_tensor(p, device=DEV), torch.as_tensor(q, device=DEV), torch.as_tensor(ln_sc, device=DEV),
                            float(s_grid[0]), float(s_grid[-1]))
    assert fit["side"] == 1
    assert abs(med - fit["train_median_abs"]) < 2e-3
    assert abs(np.log(c / fit["c_star"])) < 0.05
