"""Deterministic fate classification and ray bisection for the MFE model (Gate 2 pre-study).

classify(): integrate a batch deterministically (RK4 or Heun, compiled, CUDA-graph replay) until
each row first enters B = {E >= E_B} (outcome 1) or A_det = {E < r_det} (outcome 0); rows still
undecided at T get outcome -1. Also returns the hit time and the running maximum of E before the hit.

first_crossing(): along rays a = lam + s d, find the smallest amplitude whose deterministic fate is
B: classify a grid of amplitudes, take the first grid point with outcome != 0 (B or undecided),
then bisect between it and the previous grid point. E_c = s_c^2.
"""

import math

import numpy as np
import torch
from tqdm.auto import tqdm

from .systems.mfe import LAMINAR


def energy(a):
    return (a[:, 0] - 1.0) ** 2 + (a[:, 1:] ** 2).sum(1)


def _advance(rhs, a, dt, scheme):
    if scheme == "rk4":
        k1 = rhs(a)
        k2 = rhs(a + (0.5 * dt) * k1)
        k3 = rhs(a + (0.5 * dt) * k2)
        k4 = rhs(a + dt * k3)
        return a + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if scheme == "heun":
        f0 = rhs(a)
        a1 = a + dt * f0
        return a + (0.5 * dt) * (f0 + rhs(a1))
    raise ValueError(scheme)


class _Runner:
    def __init__(self, step, bufs, block, use_graph):
        self.step, self.block, self.use_graph, self.buf = step, block, use_graph, bufs
        if use_graph:
            backup = [b.clone() for b in bufs]
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._loop()
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._loop()
            for b, x in zip(self.buf, backup):
                b.copy_(x)

    def _loop(self):
        out = tuple(self.buf)
        for _ in range(self.block):
            out = self.step(*out)
        for b, x in zip(self.buf, out):
            b.copy_(x)

    def advance(self):
        self.graph.replay() if self.use_graph else self._loop()


_STEP_CACHE = {}


def _get_step(rhs_key, rhs, dt, scheme, E_B, r_det, compile_step):
    key = (rhs_key, dt, scheme, E_B, r_det, compile_step)
    if key in _STEP_CACHE:
        return _STEP_CACHE[key]

    def step(a, out, thit, emax, t):
        an = _advance(rhs, a, dt, scheme)
        t = t + dt
        active = out == -1
        a = torch.where(active.unsqueeze(1), an, a)
        E = energy(a)
        emax = torch.where(active, torch.maximum(emax, E), emax)
        hB = active & (E >= E_B)
        hA = active & (E < r_det) & ~hB
        out = torch.where(hB, torch.ones_like(out), torch.where(hA, torch.zeros_like(out), out))
        thit = torch.where(hA | hB, t.expand_as(thit), thit)
        return a, out, thit, emax, t

    if compile_step:
        step = torch.compile(step, dynamic=False, fullgraph=True)
    _STEP_CACHE[key] = step
    return step


@torch.no_grad()
def classify(rhs, X0, dt, T, E_B, r_det, scheme="rk4", block_time=1.0, rhs_key="mfe", compile_step=True,
             use_graph=True, chunk=None):
    """Deterministic fate of every row of X0 (torch, float64). Returns numpy dict out, t_hit, E_max."""
    n = X0.shape[0]
    if chunk and n > chunk:
        parts = [classify(rhs, X0[i:i + chunk], dt, T, E_B, r_det, scheme, block_time, rhs_key, compile_step,
                          use_graph) for i in range(0, n, chunk)]
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    device = X0.device
    block = int(round(block_time / dt))
    step = _get_step(rhs_key, rhs, dt, scheme, float(E_B), float(r_det), compile_step)
    E0 = energy(X0)
    out = torch.full((n,), -1, dtype=torch.int8, device=device)
    out = torch.where(E0 >= E_B, torch.ones_like(out), torch.where(E0 < r_det, torch.zeros_like(out), out))
    bufs = [X0.clone(), out, torch.zeros(n, dtype=X0.dtype, device=device), E0.clone(),
            torch.zeros((), dtype=X0.dtype, device=device)]
    runner = _Runner(step, bufs, block, use_graph and device.type == "cuda")
    n_blocks = int(math.ceil(T / block_time))
    for b in range(n_blocks):
        runner.advance()
        if (b + 1) % 50 == 0 and bool((runner.buf[1] != -1).all()):
            break
    a, out, thit, emax, _ = runner.buf
    out_np = out.cpu().numpy().astype(np.int8)
    t_np = thit.cpu().numpy()
    t_np[out_np == -1] = T                                 # undecided rows ran to T
    return {"out": out_np, "t_hit": t_np, "E_max": emax.cpu().numpy()}


@torch.no_grad()
def first_crossing(rhs, dirs, s_grid, n_bisect, dt, T, E_B, r_det, scheme="rk4", device="cuda", chunk=None,
                   progress=True, desc="rays"):
    """Along rays a = lam + s d: the smallest amplitude whose deterministic fate is not A (B or undecided),
    to the resolution of s_grid, refined by bisection inside [last A grid point before it, that grid point].

    s_grid must be increasing and dense: a window of B fate narrower than the grid spacing is missed (the
    pre-study used a 48-point grid, which skipped narrow first windows; Gate 2 uses 1000 points).
    Returns s_c, the bracket, the outcome at the upper bracket, the full grid fate matrix (int8: 0 = A, 1 = B,
    -1 = undecided), the number of fate switches along the grid (A <-> not-A), no_crossing (every grid point A),
    the max E along the final lower-bracket trajectory and the hit times at the final brackets."""
    dirs = np.asarray(dirs, dtype=np.float64)
    s_grid = np.asarray(s_grid, dtype=np.float64)
    N, G = len(dirs), len(s_grid)
    lam = LAMINAR[None, None, :]
    X = (lam + s_grid[None, :, None] * dirs[:, None, :]).reshape(-1, 9)
    r = classify(rhs, torch.as_tensor(X, device=device), dt, T, E_B, r_det, scheme, chunk=chunk)
    outg = r["out"].reshape(N, G)
    notA = outg != 0
    first = np.argmax(notA, axis=1)
    no_cross = ~notA.any(axis=1)
    switches = np.abs(np.diff(notA.astype(np.int8), axis=1)).sum(1)
    hi = np.where(no_cross, s_grid[-1], s_grid[first])
    lo = np.where(first > 0, s_grid[np.maximum(first - 1, 0)], 0.0)
    out_hi = outg[np.arange(N), first].copy()
    Emax_lo = np.zeros(N)
    thit_lo = np.zeros(N)
    thit_hi = np.zeros(N)
    act = ~no_cross & (first > 0)          # first grid point already not-A: s_c is below the grid
    for _ in tqdm(range(n_bisect), desc=desc, disable=not progress):
        if not act.any():
            break
        mid = 0.5 * (lo + hi)
        idx = np.where(act)[0]
        Xm = LAMINAR[None, :] + mid[idx, None] * dirs[idx]
        rm = classify(rhs, torch.as_tensor(Xm, device=device), dt, T, E_B, r_det, scheme, chunk=chunk)
        dec = rm["out"] == 0
        lo[idx] = np.where(dec, mid[idx], lo[idx])
        hi[idx] = np.where(dec, hi[idx], mid[idx])
        out_hi[idx] = np.where(dec, out_hi[idx], rm["out"])
        Emax_lo[idx] = np.where(dec, rm["E_max"], Emax_lo[idx])
        thit_lo[idx] = np.where(dec, rm["t_hit"], thit_lo[idx])
        thit_hi[idx] = np.where(dec, thit_hi[idx], rm["t_hit"])
    s_c = np.where(no_cross | (~no_cross & (first == 0)), np.nan, 0.5 * (lo + hi))   # below_grid: s_c unresolved
    return {"s_c": s_c, "lo": lo, "hi": hi, "out_hi": out_hi, "no_crossing": no_cross, "below_grid": ~no_cross & (first == 0),
            "first_index": first, "n_switches": switches, "E_max_below": Emax_lo, "t_hit_below": thit_lo,
            "t_hit_above": thit_hi, "grid_out": outg}


def random_directions(n, rng):
    v = rng.normal(size=(n, 9))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def lyapunov_laminar(m):
    """S1 with J S1 + S1 J^T + I = 0 at the laminar state (sigma = 1): stationary covariance of the
    linearised SDE. E_ss(sigma) = sigma^2 tr S1."""
    from scipy.linalg import solve_continuous_lyapunov
    J = m.jacobian(LAMINAR)
    S1 = solve_continuous_lyapunov(J, -np.eye(9))
    resid = np.abs(J @ S1 + S1 @ J.T + np.eye(9)).max()
    return S1, float(resid)


def optimal_growth(m, t_grid):
    """Linear transient energy growth G(t) = ||exp(J t)||_2^2 at the laminar state; returns the
    optimal time, G_max and the optimal initial direction (top right singular vector)."""
    from scipy.linalg import expm
    J = m.jacobian(LAMINAR)
    G = []
    for t in t_grid:
        G.append(np.linalg.norm(expm(J * t), 2) ** 2)
    G = np.array(G)
    k = int(np.argmax(G))
    U, s, Vt = np.linalg.svd(expm(J * t_grid[k]))
    return {"t_opt": float(t_grid[k]), "G_max": float(G[k]), "direction": Vt[0], "G": G}
