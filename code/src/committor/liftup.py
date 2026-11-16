"""Gate 3d: lowest-order lift-up prediction of the streak amplitude a2 at the peak time t* (MFE, laminar state).

At the laminar state the a2 equation is, to the orders used here,
    da2/dt = -d2 a2 + J23 a3 + y^T F2 y + [J23 a3 (a1 - 1 + a9)]        (m-m term in brackets, see below)
and the a3 equation is exactly da3/dt = -d3 a3 + y^T F y (Gate 3c). With a3 = a3(0) e^{-d3 t}, the a2 response to a3(0) is
L(t) a3(0), L(t) = J23 (e^{-d3 t} - e^{-d2 t})/(d2 - d3), largest in magnitude at t* = ln(d2/d3)/(d2 - d3). With
y(s) = e^{A_y s} y0, the y-y forcing of a3 (through L) and of a2 (directly) add
    V_roll(t) = int_0^t L(t - s) G_F(s) ds,  V_dir(t) = int_0^t e^{-d2 (t - s)} G_F2(s) ds,  G_X(s) = e^{A_y^T s} X e^{A_y s},
so that S(t) = e^{-d2 t} a2 + L(t) a3 + y^T (V_roll(t) + V_dir(t)) y is a2(t) to second order for states with a1 = 1 and
a9 = 0. Closed form: int_0^t e^{-d (t-s)} G_X(s) ds = e^{-d t} (Z - e^{B^T t} Z e^{B t}), B = A_y + d/2, B^T Z + Z B = -X.

The m-m term J23 a3 (a1 - 1 + a9) is NOT part of the user's S. For states with a1 - 1 or a9 nonzero it is of the same order
as the y-y terms; its lowest-order contribution to a2(t) is x_m^T M(t) x_m with
M(t) = int_0^t e^{-d2 (t - s)} e^{J_mm^T s} Q2mm e^{J_mm s} ds (m = (a1, a2, a3, a9), J_mm the m block of the Jacobian).
"""
import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm, solve_continuous_lyapunov

from .feedback import M_IDX, Y_IDX, blocks
from .systems.mfe import LAMINAR


def _sym(A):
    return 0.5 * (A + A.T)


class LiftUp:
    def __init__(self, m):
        J = m.jacobian(LAMINAR)
        self.J = J
        self.d2 = float(-J[1, 1])
        self.J23 = float(J[1, 2])
        self.Ay, self.F, self.d3 = blocks(m)
        self.F2 = _sym(m.Q[1][np.ix_(Y_IDX, Y_IDX)].copy())
        self.Q2mm = _sym(m.Q[1][np.ix_(M_IDX, M_IDX)].copy())
        self.Jmm = J[np.ix_(M_IDX, M_IDX)].copy()
        self.t_star = float(np.log(self.d2 / self.d3) / (self.d2 - self.d3))
        self._Z = {}

    def L(self, tau):
        tau = np.asarray(tau, dtype=np.float64)
        return self.J23 * (np.exp(-self.d3 * tau) - np.exp(-self.d2 * tau)) / (self.d2 - self.d3)

    def K(self, d, X, t, key):
        """int_0^t e^{-d (t-s)} e^{A_y^T s} X e^{A_y s} ds (closed form)."""
        if t == 0:
            return np.zeros((5, 5))
        B = self.Ay + 0.5 * d * np.eye(5)
        if (key, d) not in self._Z:
            self._Z[(key, d)] = _sym(solve_continuous_lyapunov(B.T, -X))
        Z = self._Z[(key, d)]
        E = expm(B * float(t))
        return _sym(np.exp(-d * float(t)) * (Z - E.T @ Z @ E))

    def V_roll(self, t):
        return self.J23 / (self.d2 - self.d3) * (self.K(self.d3, self.F, t, "F") - self.K(self.d2, self.F, t, "F"))

    def V_dir(self, t):
        return self.K(self.d2, self.F2, t, "F2")

    def coeffs(self, t=None):
        """(c_a2, c_a3, V) of S(t) = c_a2 a2 + c_a3 a3 + y^T V y; t = t* by default."""
        t = self.t_star if t is None else float(t)
        return float(np.exp(-self.d2 * t)), float(self.L(t)), self.V_roll(t) + self.V_dir(t)

    def M_mm(self, t=None, epsabs=1e-14, epsrel=1e-12):
        """lowest-order m-m contribution (not in S): a2(t) += x_m^T M x_m (quadrature)."""
        t = self.t_star if t is None else float(t)
        d2, Jm, Q = self.d2, self.Jmm, self.Q2mm
        M, _ = quad_vec(lambda s: np.exp(-d2 * (t - s)) * expm(Jm.T * s) @ Q @ expm(Jm * s), 0.0, t, epsabs=epsabs, epsrel=epsrel)
        return _sym(M)


def S_of(X, coeffs):
    """S for states X (..., 9) given as x = a - lam or as a (a2, a3 and y are the same: lam has a2 = a3 = y = 0)."""
    c2, c3, V = coeffs
    X = np.asarray(X)
    y = X[..., 3:8]
    return c2 * X[..., 1] + c3 * X[..., 2] + np.einsum("...i,ij,...j->...", y, V, y)


def mm_of(X, M):
    """x_m^T M x_m with x = a - lam (a1 enters as a1 - 1)."""
    X = np.asarray(X)
    xm = X[..., M_IDX]
    return np.einsum("...i,ij,...j->...", xm, M, xm)


def ray_pq_S(dirs, coeffs):
    """S along a ray x = s d: S(s) = p s + q s^2."""
    c2, c3, V = coeffs
    d = np.asarray(dirs)
    return c2 * d[:, 1] + c3 * d[:, 2], np.einsum("ni,ij,nj->n", d[:, 3:8], V, d[:, 3:8])
