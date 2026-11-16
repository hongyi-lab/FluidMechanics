import math

import numpy as np
import pytest
import torch

from committor.gate1 import energy_torch, exp_fit, logrank, sup_dev
from committor.ode import run_lifetimes
from committor.systems.mfe import (LAMINAR, MFE, SYMMETRIES, galerkin_projection, perturbation_energy,
                                   rhs_paper)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_tensor_form_equals_paper_form():
    m = MFE(Re=400.0)
    a = np.random.default_rng(0).normal(size=(500, 9))
    assert np.abs(m.rhs(a) - rhs_paper(a, m.Re, m.Lx, m.Lz)).max() < 1e-13
    X = torch.as_tensor(a, device=DEV)
    assert (m.drift(X) - rhs_paper(X, m.Re, m.Lx, m.Lz)).abs().max().item() < 1e-13


def test_jacobian_matches_autograd():
    m = MFE(Re=250.0, Lx_over_pi=1.75, Lz_over_pi=1.2)
    a = np.random.default_rng(1).normal(size=9)
    J = torch.autograd.functional.jacobian(lambda v: rhs_paper(v, m.Re, m.Lx, m.Lz),
                                           torch.tensor(a)).numpy()
    assert np.abs(J - m.jacobian(a)).max() < 1e-13


def test_laminar_fixed_point_and_energy_and_symmetry():
    m = MFE()
    assert np.linalg.norm(rhs_paper(LAMINAR, m.Re, m.Lx, m.Lz)) < 1e-15
    a = np.random.default_rng(2).normal(size=(200, 9))
    assert np.abs(np.einsum("ni,ni->n", a, m.nonlinear(a))).max() < 1e-12
    for g in SYMMETRIES.values():
        assert np.abs(m.rhs(a * g) - g * m.rhs(a)).max() < 1e-14
    assert perturbation_energy(LAMINAR) == 0.0


def test_galerkin_projection_reproduces_coefficients_other_domain():
    """Transcription cross-check in a second geometry (minimal flow unit) and Re."""
    m = MFE(Re=150.0, Lx_over_pi=1.75, Lz_over_pi=1.2)
    g = galerkin_projection(m.Lx, m.Lz, m.Re)
    assert np.abs(g["c"] - m.c).max() < 1e-12
    assert np.abs(g["L"] - m.L).max() < 1e-12
    assert np.abs(g["Q"] - m.Q).max() < 1e-12
    assert np.abs(g["gram"] / g["norm"] - np.eye(9)).max() < 1e-12


def _decay_rhs(a):
    return -a        # E(t) = E(0) exp(-2 t) for a(0) on a sphere around 0 ... shifted below


def test_lifetime_bookkeeping_and_resume(tmp_path):
    """Linear decay toward the laminar state: a - lam = (a0 - lam) exp(-t), so E(t) = E0 exp(-2t)
    and the first passage below th is at t* = log(E0/th)/2 (to within one step)."""
    lam = torch.as_tensor(LAMINAR, device=DEV)
    rhs = lambda a: -(a - lam)
    E0 = np.array([1.0, 0.5, 2.0, 1e-4])
    rng = np.random.default_rng(3)
    d = rng.normal(size=(4, 9))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    A0 = torch.as_tensor(LAMINAR + d * np.sqrt(E0)[:, None], device=DEV)
    th = [0.1, 0.01]
    kw = dict(thresholds=th, return_level=0.3, segment_time=2.0, block_time=0.5, progress=False,
              hist={"log10_min": -8.0, "log10_max": 1.0, "bins": 9, "every": 0.5})
    r = run_lifetimes(rhs, energy_torch, A0, 0.01, 4.0, out_dir=tmp_path / "a", n_record=2, **kw)
    for k, t in enumerate(th):
        expect = np.where(E0 < t, 0.0, np.log(E0 / t) / 2)
        assert np.all(np.abs(r["first_pass"][:, k] - expect) <= 0.0100001)
    assert not r["returned"].any()
    assert r["rec_a"].shape == (8, 2, 9) and np.allclose(r["rec_t"], np.arange(1, 9) * 0.5)
    assert r["hist_counts"].sum() == 8 * 4
    # resume: run segment 1 only (t_max = 2) into a new dir, then extend to t_max = 4 -> identical
    r1 = run_lifetimes(rhs, energy_torch, A0, 0.01, 2.0, out_dir=tmp_path / "b", n_record=2, **kw)
    r2 = run_lifetimes(rhs, energy_torch, A0, 0.01, 4.0, out_dir=tmp_path / "b", n_record=2, **kw)
    for key in ("a_final", "first_pass", "last_above", "returned", "hist_counts", "rec_a"):
        assert np.array_equal(r[key], r2[key]), key
    # graph replay and plain loop agree bit for bit
    r3 = run_lifetimes(rhs, energy_torch, A0, 0.01, 4.0, out_dir=tmp_path / "c", n_record=2,
                       use_graph=False, compile_step=False, **kw)
    assert np.allclose(r["a_final"], r3["a_final"], rtol=0, atol=1e-15)
    assert np.array_equal(r["first_pass"], r3["first_pass"])


def test_mfe_integration_graph_vs_eager(tmp_path):
    m = MFE()
    rhs = lambda a: rhs_paper(a, m.Re, m.Lx, m.Lz)
    rng = np.random.default_rng(4)
    v = rng.normal(size=(8, 9))
    A0 = torch.as_tensor(v / np.linalg.norm(v, axis=1, keepdims=True) * math.sqrt(0.3), device=DEV)
    kw = dict(thresholds=[0.01], return_level=0.3, segment_time=5.0, block_time=1.0, progress=False)
    r1 = run_lifetimes(rhs, energy_torch, A0, 0.05, 10.0, out_dir=tmp_path / "g", **kw)
    r2 = run_lifetimes(rhs, energy_torch, A0, 0.05, 10.0, out_dir=tmp_path / "e", use_graph=False,
                       compile_step=False, **kw)
    assert np.abs(r1["a_final"] - r2["a_final"]).max() < 1e-12


def test_exponential_fit_and_logrank():
    rng = np.random.default_rng(5)
    N, tau, t0, tmax = 20000, 3000.0, 500.0, 6000.0
    T = t0 + rng.exponential(tau, N)
    T = np.where(T > tmax, np.inf, T)
    f = exp_fit(T, tmax, t0)
    assert abs(f["tau"] / tau - 1) < 4 * f["tau_se"] / tau
    assert sup_dev(T, tmax, f) < 0.02
    T2 = t0 + rng.exponential(tau, N)
    T2 = np.where(T2 > tmax, np.inf, T2)
    assert logrank(T, T2, tmax)[1] > 1e-3
    T3 = t0 + rng.exponential(0.8 * tau, N)
    assert logrank(T, np.where(T3 > tmax, np.inf, T3), tmax)[1] < 1e-6


def test_recorded_states_match_record_times_on_cpu(tmp_path):
    """Regression: on CPU, .cpu() returns the stepper's own buffer, so records must be copies."""
    clock = lambda a: torch.ones_like(a)          # a(t) = a0 + t
    A0 = torch.zeros((3, 9), dtype=torch.float64)
    r = run_lifetimes(clock, energy_torch, A0, 0.25, 4.0, thresholds=[1e-3], return_level=1e9,
                      segment_time=2.0, block_time=0.5, out_dir=tmp_path / "c", n_record=2, progress=False,
                      compile_step=False, use_graph=False)
    assert np.allclose(r["rec_a"][:, 0, 0], r["rec_t"]) and np.allclose(r["rec_t"], np.arange(1, 9) * 0.5)


def test_resume_refuses_stale_checkpoint(tmp_path):
    lam = torch.as_tensor(LAMINAR, device=DEV)
    rhs = lambda a: -(a - lam)
    A0 = torch.as_tensor(LAMINAR + 0.3, device=DEV).repeat(4, 1)
    kw = dict(thresholds=[0.1], return_level=0.3, segment_time=1.0, block_time=0.5, progress=False,
              out_dir=tmp_path / "s")
    run_lifetimes(rhs, energy_torch, A0, 0.25, 1.0, **kw)
    with pytest.raises(RuntimeError):
        run_lifetimes(rhs, energy_torch, A0, 0.125, 2.0, **kw)            # different dt
    with pytest.raises(RuntimeError):
        run_lifetimes(rhs, energy_torch, A0 + 1e-3, 0.25, 2.0, **kw)      # different initial conditions
    run_lifetimes(rhs, energy_torch, A0, 0.25, 2.0, **kw)                 # same setup, longer t_max: allowed


def test_mfe_compiled_sde_matches_eager():
    if DEV.type != "cuda":
        pytest.skip("compiled path is used on GPU")
    from committor.sde import simulate_absorbing
    from committor.utils import torch_generator
    m = MFE(sigma=1e-5, r_A=1e-6, E_B=0.29)
    rng = np.random.default_rng(6)
    d = rng.normal(size=(64, 9))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    X0 = torch.as_tensor(LAMINAR + d * np.sqrt(10 ** rng.uniform(-5.5, -2.5, size=(64, 1))), device=DEV).repeat_interleave(8, 0)
    r1 = simulate_absorbing(m, X0, 0.05, 20000, "heun", torch_generator(3, DEV), progress=False, bridge=True)
    r2 = simulate_absorbing(m, X0, 0.05, 20000, "heun", torch_generator(3, DEV), progress=False, bridge=True,
                            compile_step=True)
    assert np.array_equal(r1["outcome"], r2["outcome"]) and np.array_equal(r1["steps"], r2["steps"])
    assert (r1["outcome"] == 1).any() and (r1["outcome"] == 0).any()


def test_dense_first_crossing_finds_first_window():
    """Regression for the pre-study bug: on this ray the deterministic fate switches several times; a 48-point
    grid skipped the first B window (E ~ 1.30e-3) and reported 3.78e-3."""
    if DEV.type != "cuda":
        pytest.skip("GPU test")
    from committor import edge
    m = MFE()
    rhs = lambda a: rhs_paper(a, m.Re, m.Lx, m.Lz)
    d = np.array([-0.67043863, 0.30024444, -0.09527426, -0.23063661, 0.03903567,
                  0.08142279, -0.33093145, -0.39363955, -0.35422269])
    d /= np.linalg.norm(d)
    sg = np.sqrt(np.geomspace(5e-7 * (1 + 1e-9), 0.29 * (1 - 1e-9), 1000))
    r = edge.first_crossing(rhs, d[None], sg, 20, 0.25, 8000.0, 0.29, 5e-7, "rk4", device=DEV, progress=False)
    E_c = float(r["s_c"][0] ** 2)
    assert 1.25e-3 < E_c < 1.35e-3
    assert r["n_switches"][0] >= 3
    # the state just below s_c decays, just above does not
    X = LAMINAR + np.outer([r["lo"][0], r["hi"][0] * 1.0001], d)
    out = edge.classify(rhs, torch.as_tensor(X, device=DEV), 0.25, 8000.0, 0.29, 5e-7, "rk4")["out"]
    assert out[0] == 0 and out[1] != 0


def test_noise_gain_matches_nonlinear_linear_regime():
    """tr P with P(0) = y0 y0^T, sigma = 0 equals |y(t)|^2 of the full model while y stays infinitesimal."""
    from committor import edge
    from committor.gain import noise_gain
    m = MFE()
    rhs = lambda a: rhs_paper(a, m.Re, m.Lx, m.Lz)
    rng = np.random.default_rng(1)
    y = rng.normal(size=5)
    X = LAMINAR.copy()
    X[2] = 2e-3
    X[3:8] = 1e-14 * y / np.linalg.norm(y)
    g = noise_gain(m, X[None], 0.0, T_g=800.0, dt=0.25, device=str(DEV))
    a = torch.as_tensor(X[None], device=DEV)
    best = 0.0
    for _ in range(3200):
        a = edge._advance(rhs, a, 0.25, "rk4")
        best = max(best, float((a[0, 3:8] ** 2).sum()))
    assert abs(g["max_trP"][0] / best - 1) < 1e-4
    # additive part: with y0 = 0 and constant L (laminar base), tr P(t) -> sigma^2 tr S_yy of the Lyapunov solution
    Xl = LAMINAR[None].copy()
    gl = noise_gain(m, Xl, 1.0, T_g=4000.0, dt=0.25, device=str(DEV))
    from scipy.linalg import solve_continuous_lyapunov
    Jyy = m.jacobian(LAMINAR)[3:8, 3:8]
    Syy = solve_continuous_lyapunov(Jyy, -np.eye(5))
    assert abs(gl["max_trP"][0] / np.trace(Syy) - 1) < 1e-3


def test_gain_versions_matches_dop853_large_amplitude():
    """gain_versions (v and Q form, RK4 dt = 0.25) against a DOP853 solve of the same linear system, for a pure-a3 state
    (y0 = 0) and a generic large-amplitude state (y0 != 0, where RK4 on the full P form is known to fail)."""
    from scipy.integrate import solve_ivp
    from committor.gain import gain_versions
    m = MFE()
    rng = np.random.default_rng(3)
    d = rng.normal(size=9)
    d /= np.linalg.norm(d)
    X = np.stack([LAMINAR + 4.4e-3 * np.eye(9)[2], LAMINAR + 0.2 * d])
    sig = [0.0, 1e-3]
    T_g, dt = 200.0, 0.25
    r = gain_versions(m, X, sig, T_g=T_g, dt=dt, device="cpu")
    Y = slice(3, 8)
    LT, QY = np.asarray(m.L)[Y, Y], np.asarray(m.Q)[Y, Y, :]

    def f(t, z):
        a, v, Q = z[:9], z[9:14], z[14:].reshape(5, 5)
        L = LT + 2.0 * np.einsum("ijk,k->ij", QY, a)
        return np.concatenate([rhs_paper(a, m.Re, m.Lx, m.Lz), L @ v, (L @ Q + Q @ L.T + np.eye(5)).ravel()])

    tk = np.arange(0, int(round(T_g / dt)) + 1) * dt
    for n, x in enumerate(X):
        a0 = x.copy()
        a0[Y] = 0.0
        z0 = np.concatenate([a0, x[Y], np.zeros(25)])
        sol = solve_ivp(f, (0, T_g), z0, method="DOP853", rtol=1e-12, atol=1e-30, t_eval=tk)
        v2 = (sol.y[9:14] ** 2).sum(0)
        trQ = sol.y[14:].reshape(5, 5, -1).trace(axis1=0, axis2=1)
        for k, s in enumerate(sig):
            ref = (v2 + s ** 2 * trQ).max()
            if ref == 0.0:                                      # y0 = 0 and sigma = 0
                assert r["max_trP"][n, k] == 0.0
            else:
                assert abs(r["max_trP"][n, k] / ref - 1) < 1e-3
        kpk = int(np.argmax(sol.y[1] ** 2))                    # first maximum of a2^2 on the sampled grid
        assert abs(r["t_a2pk"][n] - tk[kpk]) <= dt
        j = int(round(r["t_a2pk"][n] / dt))
        assert abs(r["v2_a2pk"][n] - v2[j]) <= 1e-3 * max(v2[j], 1e-300) + 1e-300
        assert abs(r["trQ_a2pk"][n] - trQ[j]) <= 1e-3 * max(trQ[j], 1e-300) + 1e-300
