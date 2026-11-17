"""Second-order theory about the laminar state on the DNS grid (Gate 5 theory quantities, Gate 6 seed gain).

delta(t): linearised NS about U, delta(0) = sum_i d_i u_i (unit amplitude).
q(t):     dq/dt = L0 q - P0[(delta.grad) delta] = L0 q + P0 P[delta x curl delta], q(0) = 0 (P0 = x average, kx = 0).
          (rotational form: -(d.grad)d = d x curl d - grad |d|^2/2; the gradient is removed by the projection.)
Mean-flow prediction at amplitude s: s P0 delta + s^2 q.
Streak = streamwise velocity with kx = 0 and kz != 0. A(t; s)^2 = s^2 <a,a> + 2 s^3 <a,b> + s^4 <b,b>, a = streak(P0 delta),
b = streak(q) (norms of dns.Grid, ||U|| = 1).

Gate 6 seed gain: dy/dt = P Pw [M x curl y + y x curl M] + nu lap y, M(t; s) = U + s P0 delta(t) + s^2 q(t), y(0) = s Pw delta(0),
Pw = 1 - P0 (equal to -Pw[(M.grad)y + (y.grad)M] - grad pi + lap y/Re after projection). delta, q and y are advanced in the same
Lawson RK4 steps (exact coupling at the stages). All fields are dealiased and symmetrised after every step as the DNS.
"""
import math

import numpy as np
import torch

from .dns import lawson_rk4


def streak_mask(grid):
    """(Nz, Nxh) mask of kx = 0, kz != 0 (to multiply the u component)."""
    return (grid.mask_kx0[0] * (grid.l_int != 0).to(torch.float64)[:, None]).contiguous()


def streak(grid, fh, sm=None):
    sm = streak_mask(grid) if sm is None else sm
    out = torch.zeros_like(fh)
    out[:, 0] = fh[:, 0] * sm
    return out


def _lin_self(grid, dh):
    """(lin(d), P[d x curl d]) sharing the transforms."""
    d = grid.inv(dh)
    wd = grid.inv(grid.curl(dh))
    lin = grid.proj(grid.fwd(grid.cross(grid.U_phys, wd) + grid.cross(d, grid.wU_phys)) * grid.mask)
    sc = grid.proj(grid.fwd(grid.cross(d, wd)) * grid.mask)
    return lin, sc


def run_delta_q(grid, dirs, T, dt, every=1.0, store_times=(), max_batch=64, progress=False, desc="theory"):
    """delta and q along rays dirs (R, 9). Returns aa, ab, bb (R, n_out) at t = 0, every, ..., T, and, if store_times, the kx = 0
    fields P0 delta and q at those times: dict t -> (P0 delta (R, 3, ...) cpu, q cpu)."""
    from tqdm.auto import tqdm
    dirs = np.asarray(dirs, dtype=np.float64)
    R = len(dirs)
    n_out = int(round(T / every)) + 1
    spr = int(round(every / dt))
    assert abs(spr * dt - every) < 1e-12
    aa, ab, bb = (np.zeros((R, n_out)) for _ in range(3))
    stored = {float(t): [None, None] for t in store_times}
    sm = streak_mask(grid)
    P0 = grid.mask_kx0
    E1, E2 = grid.lap_factor(dt), grid.lap_factor(0.5 * dt)

    def rhs(s):
        dh, qh = s
        lin_d, sc = _lin_self(grid, dh)
        lin_q = grid.lin(qh)
        return lin_d, (lin_q + sc) * P0

    for b0 in range(0, R, max_batch):
        sl = slice(b0, min(R, b0 + max_batch))
        dh = grid.clean(grid.from_coeffs(dirs[sl]))
        qh = torch.zeros_like(dh)
        parts = {float(t): ([], []) for t in store_times}

        def record(k, dh, qh):
            a = streak(grid, dh * P0, sm)
            b = streak(grid, qh, sm)
            aa[sl, k] = grid.inner(a, a).cpu().numpy()
            ab[sl, k] = grid.inner(a, b).cpu().numpy()
            bb[sl, k] = grid.inner(b, b).cpu().numpy()
            t = k * every
            for ts in stored:
                if abs(ts - t) < 1e-9:
                    parts[ts][0].append((dh * P0).cpu())
                    parts[ts][1].append(qh.cpu())
        record(0, dh, qh)
        for k in tqdm(range(1, n_out), desc=desc, disable=not progress, leave=False, mininterval=30):
            for _ in range(spr):
                dh, qh = lawson_rk4(grid, (dh, qh), rhs, dt, E1, E2)
                dh, qh = grid.clean(dh), grid.clean(qh)
            record(k, dh, qh)
        for ts in stored:
            stored[ts][0] = parts[ts][0][0] if stored[ts][0] is None else torch.cat([stored[ts][0], parts[ts][0][0]])
            stored[ts][1] = parts[ts][1][0] if stored[ts][1] is None else torch.cat([stored[ts][1], parts[ts][1][0]])
    return {"aa": aa, "ab": ab, "bb": bb, "t": np.arange(n_out) * every, "stored": stored}


def A_of(aa, ab, bb, s):
    """A(t; s) for amplitudes s: aa etc (R, n_t), s (R, G) or (G,) -> (R, G, n_t)."""
    s = np.asarray(s, dtype=np.float64)
    if s.ndim == 1:
        s = np.broadcast_to(s, (aa.shape[0], len(s)))
    s = s[:, :, None]
    A2 = s ** 2 * aa[:, None, :] + 2 * s ** 3 * ab[:, None, :] + s ** 4 * bb[:, None, :]
    return np.sqrt(np.maximum(A2, 0.0))


def run_gain(grid, dirs, amps, T, dt, every=1.0, max_rays=16, progress=False, desc="gain"):
    """Seed gain along rays dirs (R, 9) at amplitudes amps (R, K). Returns yy (R, K, n_out) = ||y||^2 / ||U||^2 at the output
    times and courant_M (R, K): max over the steps of dt * sum_c max|M_c| kmax_c (the advecting field of y; recorded, no re-run)."""
    from tqdm.auto import tqdm
    dirs = np.asarray(dirs, dtype=np.float64)
    amps = np.asarray(amps, dtype=np.float64)
    R, K = amps.shape
    n_out = int(round(T / every)) + 1
    spr = int(round(every / dt))
    assert abs(spr * dt - every) < 1e-12
    yy = np.zeros((R, K, n_out))
    cM = np.zeros((R, K))
    kmax = torch.tensor(grid.kmax, device=grid.dev, dtype=torch.float64)
    P0 = grid.mask_kx0
    Pw = 1.0 - P0
    E1, E2 = grid.lap_factor(dt), grid.lap_factor(0.5 * dt)
    for b0 in range(0, R, max_rays):
        sl = slice(b0, min(R, b0 + max_rays))
        r = sl.stop - sl.start
        s = torch.as_tensor(amps[sl], device=grid.dev, dtype=torch.float64)           # (r, K)
        sv = s.reshape(r, K, 1, 1, 1, 1)
        dh = grid.clean(grid.from_coeffs(dirs[sl]))
        qh = torch.zeros_like(dh)
        yh = (dh * Pw)[:, None].expand(r, K, *dh.shape[1:]).reshape(r * K, *dh.shape[1:]) * s.reshape(r * K, 1, 1, 1, 1)
        yh = grid.clean(yh.contiguous())

        box = {}

        def rhs(st):
            dh_, qh_, yh_ = st
            lin_d, sc = _lin_self(grid, dh_)
            q = grid.inv(qh_)
            wq = grid.inv(grid.curl(qh_))
            lin_q = grid.proj(grid.fwd(grid.cross(grid.U_phys, wq) + grid.cross(q, grid.wU_phys)) * grid.mask)
            d0 = dh_ * P0
            p0d = grid.inv(d0)
            wp0d = grid.inv(grid.curl(d0))
            M = (grid.U_phys[:, None] + sv * p0d[:, None] + sv ** 2 * q[:, None]).reshape(r * K, 3, grid.Ny, grid.Nz, grid.Nx)
            if "c" not in box:
                box["c"] = (M.abs().amax(dim=(-3, -2, -1)) * kmax).sum(1)
            wM = (grid.wU_phys[:, None] + sv * wp0d[:, None] + sv ** 2 * wq[:, None]).reshape(r * K, 3, grid.Ny, grid.Nz, grid.Nx)
            y = grid.inv(yh_)
            wy = grid.inv(grid.curl(yh_))
            dy = grid.proj(grid.fwd(grid.cross(M, wy) + grid.cross(y, wM)) * grid.mask) * Pw
            return lin_d, (lin_q + sc) * P0, dy

        yy[sl, :, 0] = grid.energy(yh).reshape(r, K).cpu().numpy()
        for k in tqdm(range(1, n_out), desc=desc, disable=not progress, leave=False, mininterval=30):
            for _ in range(spr):
                box = {}
                dh, qh, yh = lawson_rk4(grid, (dh, qh, yh), rhs, dt, E1, E2)
                dh, qh, yh = grid.clean(dh), grid.clean(qh), grid.clean(yh)
                cM[sl] = np.maximum(cM[sl], dt * box["c"].reshape(r, K).cpu().numpy())
            yy[sl, :, k] = grid.energy(yh).reshape(r, K).cpu().numpy()
    return {"yy": yy, "t": np.arange(n_out) * every, "courant_M": cM}
