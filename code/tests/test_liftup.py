"""Gate 3d: lift-up quantities (closed forms vs the user's integrals, t*, S along rays, ODE scaling)."""
import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm

from committor.liftup import LiftUp, S_of, mm_of, ray_pq_S
from committor.systems.mfe import LAMINAR, MFE, rhs_paper


def test_constants_and_closed_forms():
    m = MFE(Re=400.0)
    lu = LiftUp(m)
    beta, gamma = np.pi / 2, 2 * np.pi / m.Lz
    assert abs(lu.d2 - (4 * beta ** 2 / 3 + gamma ** 2) / m.Re) < 1e-15
    assert abs(lu.J23 + np.sqrt(1.5) * beta * gamma / np.sqrt(beta ** 2 + gamma ** 2)) < 1e-14
    assert abs(lu.t_star - 103.518) < 1e-3
    tt = np.linspace(0, 400, 400001)
    assert abs(tt[np.argmax(np.abs(lu.L(tt)))] - lu.t_star) < 2e-3
    t = 60.0
    G = lambda X, s: expm(lu.Ay.T * s) @ X @ expm(lu.Ay * s)  # noqa: E731
    Vr, _ = quad_vec(lambda s: lu.L(t - s) * G(lu.F, s), 0, t, epsabs=1e-13, epsrel=1e-12)
    Vd, _ = quad_vec(lambda s: np.exp(-lu.d2 * (t - s)) * G(lu.F2, s), 0, t, epsabs=1e-13, epsrel=1e-12)
    assert np.abs(lu.V_roll(t) - Vr).max() < 1e-8 * max(1.0, np.abs(Vr).max())
    assert np.abs(lu.V_dir(t) - Vd).max() < 1e-10 * max(1.0, np.abs(Vd).max())
    # S along a ray is p s + q s^2
    rng = np.random.default_rng(0)
    d = rng.standard_normal((5, 9))
    c = lu.coeffs()
    p, q = ray_pq_S(d, c)
    s = 0.02
    assert np.allclose(S_of(s * d, c), p * s + q * s ** 2, rtol=1e-12, atol=0)


def test_S_is_a2_at_t_star_to_second_order():
    m = MFE(Re=400.0)
    lu = LiftUp(m)
    c, M = lu.coeffs(), lu.M_mm()
    f = lambda z: rhs_paper(z, m.Re, m.Lx, m.Lz)  # noqa: E731
    n = 2070
    dt = lu.t_star / n
    rng = np.random.default_rng(5)
    for full in (False, True):
        d = rng.standard_normal(9)
        if not full:
            d[[0, 8]] = 0.0
        d /= np.linalg.norm(d)
        r = []
        for eps in (1e-5, 1e-4):
            a = LAMINAR + eps * d
            for _ in range(n):
                k1 = f(a); k2 = f(a + 0.5 * dt * k1); k3 = f(a + 0.5 * dt * k2); k4 = f(a + dt * k3)  # noqa: E702
                a = a + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            r.append((a[1] - S_of(eps * d, c) - mm_of(eps * d, M)) / eps ** 2)
        # remainder of the second-order prediction is O(eps): shrinks about 10x per decade
        assert abs(r[0]) < 0.3 * abs(r[1]) + 1e-3
        if not full:
            assert mm_of(d, M) == 0.0
