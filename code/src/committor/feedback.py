"""Gate 3c: lowest-order y -> a3 feedback of the MFE model and the theory quantity u_th = a3 + y^T W y.

MFE is exactly quasi-linear in y = (a4..a8): the y equations contain only m*y products (m = (a1, a2, a3, a9)), the
m equations only m*m and y*y products, there is no linear m-y coupling and no forcing of y (a consequence of the
T_{Lx/2} symmetry, which flips exactly the y modes). The a3 equation is exactly
    da3/dt = -d3 a3 + y^T F y,     d3 = (beta^2 + gamma^2)/Re,
so a3(T) e^{d3 T} = a3(0) + int_0^T e^{d3 s} y(s)^T F y(s) ds. With y(s) = e^{A_y s} y0 (A_y: y block of the Jacobian at
the laminar state, i.e. the lowest order in the deviation of m from laminar),
    a3(T) e^{d3 T} = a3(0) + y0^T W(T) y0,   W(T) = int_0^T e^{d3 s} e^{A_y^T s} F e^{A_y s} ds.
With B = A_y + (d3/2) I (stable at Re = 400), W(T) = W_inf - e^{B^T T} W_inf e^{B T}, B^T W_inf + W_inf B = -F.
"""
import numpy as np
import torch
from scipy.linalg import expm, solve_continuous_lyapunov

from .systems.mfe import LAMINAR

M_IDX = [0, 1, 2, 8]
Y_IDX = [3, 4, 5, 6, 7]


def structure_check(m):
    """max |coefficient| of every block of the MFE polynomial (equation block, product block)."""
    L, Q, c = m.L, m.Q, m.c
    ix = np.ix_
    M_, Y_ = M_IDX, Y_IDX
    return {"y_eq_yy": float(np.abs(Q[ix(Y_, Y_, Y_)]).max()), "y_eq_mm": float(np.abs(Q[ix(Y_, M_, M_)]).max()),
            "y_eq_my": float(np.abs(Q[ix(Y_, M_, Y_)]).max()), "m_eq_my": float(np.abs(Q[ix(M_, M_, Y_)]).max()),
            "m_eq_yy": float(np.abs(Q[ix(M_, Y_, Y_)]).max()), "m_eq_mm": float(np.abs(Q[ix(M_, M_, M_)]).max()),
            "L_ym": float(np.abs(L[ix(Y_, M_)]).max()), "L_my": float(np.abs(L[ix(M_, Y_)]).max()),
            "c_y": float(np.abs(c[Y_]).max()),
            "a3_eq_linear_except_a3": float(np.abs(np.delete(L[2], 2)).max()),
            "a3_eq_quadratic_outside_yy": float(max(np.abs(Q[2][ix(M_, M_)]).max(), np.abs(Q[2][ix(M_, Y_)]).max()))}


def blocks(m):
    """(A_y (5, 5), F (5, 5) symmetric, d3) at the laminar state."""
    J = m.jacobian(LAMINAR)
    Ay = J[np.ix_(Y_IDX, Y_IDX)].copy()
    F = m.Q[2][np.ix_(Y_IDX, Y_IDX)].copy()
    F = 0.5 * (F + F.T)
    d3 = float(-m.L[2, 2])
    return Ay, F, d3


class Feedback:
    def __init__(self, m):
        self.Ay, self.F, self.d3 = blocks(m)
        self.B = self.Ay + 0.5 * self.d3 * np.eye(5)
        self.eig_B = np.linalg.eigvals(self.B)
        if not np.all(self.eig_B.real < 0):
            raise ValueError("A_y + d3/2 is not stable: W_inf does not exist")
        W = solve_continuous_lyapunov(self.B.T, -self.F)
        self.W_inf = 0.5 * (W + W.T)

    def W(self, T):
        """W(T) for T >= 0 (closed form)."""
        if T == 0:
            return np.zeros((5, 5))
        if np.isinf(T):
            return self.W_inf.copy()
        E = expm(self.B * float(T))
        W = self.W_inf - E.T @ self.W_inf @ E
        return 0.5 * (W + W.T)


def u_th(X, W, kappa=1.0):
    """u_th = x3 + kappa y^T W y for states X (..., 9) given as x = a - lam or as a (a3 and y are the same: lam has
    a3 = y = 0)."""
    X = np.asarray(X)
    y = X[..., 3:8]
    return X[..., 2] + kappa * np.einsum("...i,ij,...j->...", y, W, y)


def ray_pq(dirs, W):
    """u_th along a ray x = s d: u(s) = p s + q s^2 with p = d3, q = d_y^T W d_y."""
    d = np.asarray(dirs)
    return d[:, 2].copy(), np.einsum("ni,ij,nj->n", d[:, 3:8], W, d[:, 3:8])


def first_crossing_abs(p, q, c, s_lo, s_hi):
    """First s > 0 with |p s + q s^2| >= c (c > 0), with the Gate 3 grid conventions: values below s_lo are returned
    as s_lo (below grid), values above s_hi (or no crossing) as +inf. torch tensors, broadcasting."""
    ap, aq = p.abs(), q.abs()
    same = (p * q >= 0)
    inf = torch.full_like(ap * aq * c, float("inf"))
    # same sign (or q = 0): |u| = s (|p| + |q| s), increasing
    disc_s = ap ** 2 + 4 * aq * c
    s_same = torch.where(aq > 0, 2 * c / (ap + torch.sqrt(disc_s)), torch.where(ap > 0, c / ap.clamp_min(1e-300), inf))
    # opposite signs: |u| rises to p^2/(4|q|) at s = |p|/(2|q|), returns to 0 at |p|/|q|, then rises
    hump = ap ** 2 / (4 * aq.clamp_min(1e-300))
    disc_h = (ap ** 2 - 4 * aq * c).clamp_min(0.0)
    s_hump = 2 * c / (ap + torch.sqrt(disc_h))                     # smaller root of |q| s^2 - |p| s + c = 0
    s_after = (ap + torch.sqrt(ap ** 2 + 4 * aq * c)) / (2 * aq.clamp_min(1e-300))
    s_opp = torch.where(hump >= c, s_hump, s_after)
    s = torch.where(same, s_same, s_opp)
    s = torch.where(s < s_lo, torch.full_like(s, s_lo), s)
    return torch.where(s > s_hi, inf, s)


def _median_rows(E):
    s = torch.sort(E, dim=-1).values
    n = E.shape[-1]
    return 0.5 * (s[..., (n - 1) // 2] + s[..., n // 2])


def analytic_cstar(p, q, ln_sc, s_lo, s_hi, n_levels=1000, n_refine=1000):
    """The Gate 3 c* rule for |u_th| (side +1) with the analytic first crossing: candidate thresholds = quantiles of
    |u| at the training edges, then a local refinement between the neighbours of the best one. p, q, ln_sc: torch (n,).
    Returns (train median |ln|, c*)."""
    sc = torch.exp(ln_sc)
    edge = (p * sc + q * sc ** 2).abs()
    edge = edge[torch.isfinite(edge) & (edge > 0)]
    lev = torch.unique(torch.quantile(edge, torch.linspace(0, 1, n_levels, dtype=edge.dtype, device=edge.device)))

    def med(cs):
        s = first_crossing_abs(p[None, :], q[None, :], cs[:, None], s_lo, s_hi)
        e = (torch.log(s) - ln_sc[None, :]).abs()
        e = torch.where(torch.isnan(e), torch.full_like(e, float("inf")), e)
        return _median_rows(e)

    m0 = med(lev)
    k = int(torch.argmin(m0))
    lo, hi = lev[max(k - 1, 0)], lev[min(k + 1, len(lev) - 1)]
    fine = torch.unique(torch.cat([torch.linspace(float(lo), float(hi), n_refine, dtype=lev.dtype, device=lev.device), lev[k:k + 1]]))
    m1 = med(fine)
    j = int(torch.argmin(m1))
    return float(m1[j]), float(fine[j])
