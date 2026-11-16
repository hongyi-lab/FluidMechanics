"""Linear noise gain of the x-dependent modes along the x-independent transient (Gate 3 candidate, user item 4).

a = (u, y), u = (a1, a2, a3, a9) (x-independent), y = (a4..a8) (x-dependent). The set y = 0 is invariant
(fixed set of T_{Lx/2}), so the x-independent dynamics u(t) is closed. Along u(t) the y-perturbation covariance
obeys dP/dt = L(t) P + P L(t)^T + sigma^2 I, with L(t) = d(dy/dt)/dy at (u(t), y = 0) (5x5), P(0) = y0 y0^T.
The coordinate is log max_{0<=t<=T_g} tr P(t). With y0 = 0 and sigma = 1, max tr P is g(s) of the prediction
sigma^2 g(s50) = const on the pure-a3 ray.
"""
import math

import numpy as np
import torch

from .systems.mfe import rhs_paper

Y = slice(3, 8)


@torch.no_grad()
def noise_gain(m, X, sigma, T_g=4000.0, dt=0.25, device="cuda", chunk=200000):
    """X: (n, 9) states. Returns dict max_trP (n,), log_max_trP, t_at_max, trP_end (numpy).

    WARNING (Gate 3 review): RK4 on the full 5x5 Lyapunov equation with a rank-1 P(0) = y0 y0^T is badly inaccurate
    at dt = 0.25 when y0 != 0 and the transient is strongly non-normal (tr P can even turn negative). It is accurate for
    y0 = 0 (the only use in Gate 2). Use gain_versions (v and Q form) for y0 != 0, and do not use this function as a
    reference for it."""
    X = np.asarray(X, dtype=np.float64)
    outs = []
    for i in range(0, len(X), chunk):
        outs.append(_gain_chunk(m, X[i:i + chunk], float(sigma), float(T_g), float(dt), device))
    res = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
    res["log_max_trP"] = np.log(np.maximum(res["max_trP"], 1e-300))
    return res


def _gain_chunk(m, X, sigma, T_g, dt, device):
    dev = torch.device(device)
    a = torch.as_tensor(X, device=dev).clone()
    y0 = a[:, Y].clone()
    a[:, Y] = 0.0
    P = y0[:, :, None] * y0[:, None, :]
    LT = torch.as_tensor(m.L, device=dev)[Y, Y]                                  # (5, 5)
    QY = torch.as_tensor(m.Q, device=dev)[Y, Y, :]                               # (5, 5, 9)
    I5 = torch.eye(5, dtype=a.dtype, device=dev)
    s2 = sigma ** 2
    Re, Lx, Lz = m.Re, m.Lx, m.Lz

    def Lof(a):
        return LT[None] + 2.0 * torch.einsum("ijk,nk->nij", QY, a)              # Jacobian block dy/dy at y = 0

    def f(a, P):
        L = Lof(a)
        return rhs_paper(a, Re, Lx, Lz), L @ P + P @ L.transpose(1, 2) + s2 * I5

    def step(a, P):
        k1a, k1P = f(a, P)
        k2a, k2P = f(a + 0.5 * dt * k1a, P + 0.5 * dt * k1P)
        k3a, k3P = f(a + 0.5 * dt * k2a, P + 0.5 * dt * k2P)
        k4a, k4P = f(a + dt * k3a, P + dt * k3P)
        return (a + dt / 6.0 * (k1a + 2 * k2a + 2 * k3a + k4a),
                P + dt / 6.0 * (k1P + 2 * k2P + 2 * k3P + k4P))

    stepc = torch.compile(step, dynamic=False, fullgraph=True) if (dev.type == "cuda" and len(X) >= 2) else step
    tr = P.diagonal(dim1=1, dim2=2).sum(1)
    best = tr.clone()
    t_best = torch.zeros_like(best)
    n = int(round(T_g / dt))
    for k in range(1, n + 1):
        a, P = stepc(a, P)
        tr = P.diagonal(dim1=1, dim2=2).sum(1)
        better = tr > best
        best = torch.where(better, tr, best)
        t_best = torch.where(better, torch.full_like(t_best, k * dt), t_best)
    if not torch.isfinite(best).all():
        raise FloatingPointError("non-finite noise gain")
    return {"max_trP": best.cpu().numpy(), "t_at_max": t_best.cpu().numpy(), "trP_end": tr.cpu().numpy(),
            "y_max_abs_end": a[:, Y].abs().max(1).values.cpu().numpy()}


# ---- Gate 3: two versions of the gain coordinate, several sigma in one integration ----------------------------------
# By linearity P(t) = v(t) v(t)^T + sigma^2 Q(t) with dv/dt = L v, v(0) = y0, and dQ/dt = L Q + Q L^T + I, Q(0) = 0,
# so tr P_sigma(t) = |v(t)|^2 + sigma^2 tr Q(t) for every sigma at once.
#   version (i):  c = ln max_{0<=t<=T_g} tr P_sigma(t)
#   version (ii): c = ln tr P_sigma(t*), t* = first time of the maximum of a2(t)^2 on [0, T_g] along the u-transient
@torch.no_grad()
def gain_versions(m, X, sigmas, T_g=4000.0, dt=0.25, device="cuda", chunk=500000):
    """X: (n, 9) states. Returns numpy dict:
    max_trP (n, K) and t_at_max (n, K) for sigmas[k] (version i);
    v2_a2pk, trQ_a2pk, t_a2pk (n,) (version ii: tr P_sigma(t*) = v2_a2pk + sigma^2 trQ_a2pk);
    a2sq_max (n,), v2_end, trQ_end (n,)."""
    X = np.asarray(X, dtype=np.float64)
    pad = min(chunk, len(X))
    outs = [_versions_chunk(m, X[i:i + chunk], [float(s) for s in sigmas], float(T_g), float(dt), device, pad_to=pad)
            for i in range(0, len(X), chunk)]
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


_VSTEP = {}
_U = [0, 1, 2, 8]          # u = (a1, a2, a3, a9); y = 0 along the transient, so L depends on u only
_QIDX = [(i, j) for i in range(5) for j in range(i, 5)]      # 15 unique entries of the symmetric Q


def _versions_step(m, dt, sigmas, device):
    """One compiled RK4 step of Z = (a[9], v[5], Q[15]) written element-wise (so that inductor fuses the whole step
    into one kernel), plus the running maxima of both versions. Mathematically identical to dv/dt = L v,
    dQ/dt = L Q + Q L^T + I with L = LT + sum_k 2 Q[Y, Y, k] a_k."""
    key = (m.Re, m.Lx, m.Lz, dt, tuple(sigmas), str(device))
    if key in _VSTEP:
        return _VSTEP[key]
    dev = torch.device(device)
    LT = np.asarray(m.L)[Y, Y]
    QU = 2.0 * np.asarray(m.Q)[Y, Y, :][:, :, _U]
    terms = [[(float(LT[i, j]), [(k, float(QU[i, j, kk])) for kk, k in enumerate(_U) if QU[i, j, kk] != 0.0])
              for j in range(5)] for i in range(5)]
    s2 = torch.as_tensor(np.square(sigmas), dtype=torch.float64, device=dev)[None, :]
    Re, Lx, Lz = m.Re, m.Lx, m.Lz
    qpos = {ij: 14 + n for n, ij in enumerate(_QIDX)}

    def qget(Z, i, j):
        return Z[:, qpos[(i, j) if i <= j else (j, i)]]

    def Lmat(a):
        L = [[None] * 5 for _ in range(5)]
        for i in range(5):
            for j in range(5):
                c0, tk = terms[i][j]
                e = None
                for k, c in tk:
                    e = c * a[:, k] if e is None else e + c * a[:, k]
                if c0 != 0.0:
                    e = (torch.full_like(a[:, 0], c0) if e is None else e + c0)
                L[i][j] = e
        return L

    def f(Z):
        a = Z[:, :9]
        L = Lmat(a)
        da = rhs_paper(a, Re, Lx, Lz)
        cols = [da[:, k] for k in range(9)]
        for i in range(5):
            e = None
            for k in range(5):
                if L[i][k] is not None:
                    t = L[i][k] * Z[:, 9 + k]
                    e = t if e is None else e + t
            cols.append(e)
        LQ = {}
        for i in range(5):
            for j in range(5):
                e = None
                for k in range(5):
                    if L[i][k] is not None:
                        t = L[i][k] * qget(Z, k, j)
                        e = t if e is None else e + t
                LQ[(i, j)] = e
        for (i, j) in _QIDX:
            e = LQ[(i, j)] + LQ[(j, i)]
            cols.append(e + 1.0 if i == j else e)
        return torch.stack(cols, dim=1)

    def step(Z, t, best, t_best, a2best, t_a2, v2_a2, trQ_a2):
        k1 = f(Z)
        k2 = f(Z + (0.5 * dt) * k1)
        k3 = f(Z + (0.5 * dt) * k2)
        k4 = f(Z + dt * k3)
        Z = Z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t = t + dt
        v2 = (Z[:, 9:14] ** 2).sum(1)
        trQ = Z[:, qpos[(0, 0)]] + Z[:, qpos[(1, 1)]] + Z[:, qpos[(2, 2)]] + Z[:, qpos[(3, 3)]] + Z[:, qpos[(4, 4)]]
        tr = v2[:, None] + s2 * trQ[:, None]
        bt = tr > best
        best = torch.where(bt, tr, best)
        t_best = torch.where(bt, t.expand_as(t_best), t_best)
        a2sq = Z[:, 1] ** 2
        ba = a2sq > a2best
        a2best = torch.where(ba, a2sq, a2best)
        t_a2 = torch.where(ba, t.expand_as(t_a2), t_a2)
        v2_a2 = torch.where(ba, v2, v2_a2)
        trQ_a2 = torch.where(ba, trQ, trQ_a2)
        return Z, t, best, t_best, a2best, t_a2, v2_a2, trQ_a2

    stepc = torch.compile(step, dynamic=False, fullgraph=True) if dev.type == "cuda" else step
    _VSTEP[key] = (step, stepc, s2, qpos)
    return _VSTEP[key]


def _versions_chunk(m, X, sigmas, T_g, dt, device, pad_to=None):
    dev = torch.device(device)
    n0 = len(X)
    if pad_to is not None and n0 < pad_to:                       # fixed shape: one compiled graph for every chunk
        X = np.concatenate([X, np.repeat(X[:1], pad_to - n0, axis=0)])
    a = torch.as_tensor(X, device=dev).clone()
    n = a.shape[0]
    Z = torch.zeros((n, 29), dtype=a.dtype, device=dev)
    Z[:, :9] = a
    Z[:, 9:14] = a[:, Y]
    Z[:, 3:8] = 0.0
    step, stepc, s2, qpos = _versions_step(m, dt, sigmas, device)
    st = stepc if n >= 2 else step
    v2 = (Z[:, 9:14] ** 2).sum(1)
    trQ = torch.zeros(n, dtype=a.dtype, device=dev)
    best = v2[:, None] + s2 * trQ[:, None]                                         # (n, K), includes t = 0
    t_best = torch.zeros_like(best)
    a2best = Z[:, 1] ** 2
    t_a2, v2_a2, trQ_a2 = torch.zeros_like(a2best), v2.clone(), trQ.clone()
    t = torch.zeros((), dtype=a.dtype, device=dev)
    state = (Z, t, best, t_best, a2best, t_a2, v2_a2, trQ_a2)
    for _ in range(int(round(T_g / dt))):
        state = st(*state)
    Z, t, best, t_best, a2best, t_a2, v2_a2, trQ_a2 = state
    v2 = (Z[:, 9:14] ** 2).sum(1)
    trQ = sum(Z[:, qpos[(i, i)]] for i in range(5))
    c = lambda x: x[:n0].cpu().numpy()
    return {"max_trP": c(best), "t_at_max": c(t_best), "v2_a2pk": c(v2_a2), "trQ_a2pk": c(trQ_a2), "t_a2pk": c(t_a2),
            "a2sq_max": c(a2best), "v2_end": c(v2), "trQ_end": c(trQ)}
