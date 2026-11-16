"""Batched pseudo-spectral DNS of the MFE flow (sinusoidal shear flow between free-slip walls), float64 on the GPU.

Flow (NJP 2004 section 2): y in [-1, 1] with free-slip walls (v = 0, du/dy = dw/dy = 0), periodic x (Lx) and z (Lz),
body force F = (sqrt2 pi^2/(4 Re)) sin(pi y/2) e_x, laminar U = sqrt2 sin(pi y/2) e_x. Units as the paper.

The y direction is extended to period 4 by the mirror about y = 1, (u, v, w)(x, 2 - y, z) = (u, -v, w)(x, y, z), so that all
three directions are Fourier. Grid x_j = j Lx/Nx, y_j = -1 + 4 j/Ny, z_j = j Lz/Nz; physical arrays (..., Ny, Nz, Nx),
spectral arrays (..., Ny, Nz, Nx//2 + 1) (real FFT along x), coefficients f = sum_k f_k e^{i k.(x - x0)}, x0 = (0, -1, 0).
Norms: ||f||^2 = (1/V) int |f|^2 = sum_k w_k |f_k|^2 (w = 2 for 0 < m < Nx/2), equal on the extended and the physical
domain; ||U|| = 1, so E = ||u - U||^2.

Perturbation form: u = U + u', du'/dt = P[(U + u') x (w_U + w')] + nu lap u' (nu lap U + F = 0 and P[U x w_U] = 0).
Rotational form, 2/3 dealiasing (|m| <= ceil(N/3) - 1 per axis), Leray projection P = I - k k^T / |k|^2, Lawson RK4 with the
viscous integrating factor. Symmetries: g u = D u(R x + d) applied in Fourier space; enforced by averaging over the group.
"""
import math

import numpy as np
import torch

from .systems.mfe import LAMINAR, rhs_paper  # noqa: F401  (rhs_paper used by callers / tests)


def _mode_arrays(X, Y, Z, al, be, ga):
    """The nine NJP modes (7)-(16) evaluated on arrays X, Y, Z: returns (9, 3, *X.shape)."""
    s2, s3 = math.sqrt(2.0), math.sqrt(3.0)
    cy, sy = np.cos(np.pi * Y / 2), np.sin(np.pi * Y / 2)
    cx, sx = np.cos(al * X), np.sin(al * X)
    cz, sz = np.cos(ga * Z), np.sin(ga * Z)
    o = np.zeros_like(X)
    N8 = 2 * s2 / math.sqrt((al ** 2 + ga ** 2) * (4 * al ** 2 + 4 * ga ** 2 + np.pi ** 2))
    k3 = 2 / math.sqrt(4 * ga ** 2 + np.pi ** 2)
    k6 = 4 * s2 / math.sqrt(3 * (al ** 2 + ga ** 2))
    k7 = 2 * s2 / math.sqrt(al ** 2 + ga ** 2)
    m = [
        (s2 * sy, o, o),
        (4 / s3 * cy ** 2 * cz, o, o),
        (o, k3 * 2 * ga * cy * cz, k3 * np.pi * sy * sz),
        (o, o, 4 / s3 * cx * cy ** 2),
        (o, o, 2 * sx * sy),
        (k6 * (-ga) * cx * cy ** 2 * sz, o, k6 * al * sx * cy ** 2 * cz),
        (k7 * ga * sx * sy * sz, o, k7 * al * cx * sy * cz),
        (N8 * np.pi * al * sx * sy * sz, N8 * 2 * (al ** 2 + ga ** 2) * cx * cy * sz, N8 * (-np.pi) * ga * cx * sy * cz),
        (s2 * np.sin(3 * np.pi * Y / 2), o, o),
    ]
    return np.array([[np.broadcast_to(c, X.shape) for c in mm] for mm in m])


class Grid:
    def __init__(self, Nx, Ny, Nz, Re=400.0, Lx=4 * math.pi, Lz=2 * math.pi, Ly=4.0, device="cuda"):
        self.Nx, self.Ny, self.Nz = int(Nx), int(Ny), int(Nz)
        self.Nxh = self.Nx // 2 + 1
        self.Re, self.nu = float(Re), 1.0 / float(Re)
        self.Lx, self.Ly, self.Lz = float(Lx), float(Ly), float(Lz)
        self.dev = torch.device(device)
        self.alpha, self.ky0, self.gamma = 2 * math.pi / Lx, 2 * math.pi / Ly, 2 * math.pi / Lz
        self.beta = math.pi / 2
        dev, f64 = self.dev, torch.float64
        m = torch.arange(self.Nxh, device=dev, dtype=f64)
        n = torch.fft.fftfreq(self.Ny, 1.0 / self.Ny, device=dev, dtype=f64)
        l = torch.fft.fftfreq(self.Nz, 1.0 / self.Nz, device=dev, dtype=f64)
        self.m_int, self.n_int, self.l_int = m, n, l
        self.KX = (self.alpha * m)[None, None, :]
        self.KY = (self.ky0 * n)[:, None, None]
        self.KZ = (self.gamma * l)[None, :, None]
        self.K2 = self.KX ** 2 + self.KY ** 2 + self.KZ ** 2
        self.K2inv = torch.where(self.K2 > 0, 1.0 / torch.where(self.K2 > 0, self.K2, torch.ones_like(self.K2)), torch.zeros_like(self.K2))
        cut = lambda N: math.ceil(N / 3) - 1  # noqa: E731
        self.cut = (cut(self.Nx), cut(self.Ny), cut(self.Nz))
        mask = ((m.abs() <= self.cut[0])[None, None, :] & (n.abs() <= self.cut[1])[:, None, None] & (l.abs() <= self.cut[2])[None, :, None])
        self.mask = mask.to(f64)
        w = torch.full((self.Nxh,), 2.0, device=dev, dtype=f64)
        w[0] = 1.0
        if self.Nx % 2 == 0:
            w[-1] = 1.0
        self.w = w[None, None, :]
        self.kmax = (self.alpha * self.cut[0], self.ky0 * self.cut[1], self.gamma * self.cut[2])
        # physical grid
        xs = np.arange(self.Nx) * self.Lx / self.Nx
        ys = -1.0 + np.arange(self.Ny) * self.Ly / self.Ny
        zs = np.arange(self.Nz) * self.Lz / self.Nz
        self.xs, self.ys, self.zs = xs, ys, zs
        Y, Z, X = np.meshgrid(ys, zs, xs, indexing="ij")
        modes = _mode_arrays(X, Y, Z, self.alpha, self.beta, self.gamma)                     # (9, 3, Ny, Nz, Nx)
        self.modes_phys = torch.as_tensor(modes, device=dev, dtype=f64)
        self.modes = self.fwd(self.modes_phys)                                               # (9, 3, ...)
        self.U = self.modes[0:1].clone()                                                     # (1, 3, ...) laminar
        self.U_phys = self.inv(self.U)
        self.wU_phys = self.inv(self.curl(self.U))
        self.mask_kx0 = (m == 0).to(f64)[None, None, :]

    # ---------------------------------------------------------------- transforms and operators
    def fwd(self, f):
        return torch.fft.rfftn(f, dim=(-3, -2, -1), norm="forward")

    def inv(self, fh):
        return torch.fft.irfftn(fh, s=(self.Ny, self.Nz, self.Nx), dim=(-3, -2, -1), norm="forward")

    def curl(self, uh):
        u, v, w = uh[:, 0], uh[:, 1], uh[:, 2]
        return torch.stack([1j * (self.KY * w - self.KZ * v), 1j * (self.KZ * u - self.KX * w), 1j * (self.KX * v - self.KY * u)], dim=1)

    def div(self, uh):
        return 1j * (self.KX * uh[:, 0] + self.KY * uh[:, 1] + self.KZ * uh[:, 2])

    def proj(self, nh):
        kd = (self.KX * nh[:, 0] + self.KY * nh[:, 1] + self.KZ * nh[:, 2]) * self.K2inv
        out = torch.stack([nh[:, 0] - self.KX * kd, nh[:, 1] - self.KY * kd, nh[:, 2] - self.KZ * kd], dim=1)
        out[..., 0, 0, 0] = 0.0
        return out

    def lap_factor(self, h):
        return torch.exp(-self.nu * self.K2 * h)

    @staticmethod
    def cross(a, b):
        return torch.stack([a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1], a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2], a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]], dim=1)

    def inner(self, ah, bh):
        """<a, b> per batch element (sum over components and wavenumbers, rfft weights)."""
        return (self.w * (ah.conj() * bh).real).sum(dim=(-4, -3, -2, -1))

    def energy(self, uh):
        return self.inner(uh, uh)

    def coeffs(self, uh):
        """projections a_i = <u, u_i> (B, 9)."""
        return torch.einsum("bcyzx,icyzx->bi", (uh * self.w).conj(), self.modes).real

    def from_coeffs(self, A):
        """sum_i A_i u_i for A (B, 9) (numpy or torch)."""
        A = torch.as_tensor(np.asarray(A), device=self.dev, dtype=torch.float64)
        return torch.einsum("bi,icyzx->bcyzx", A.to(torch.complex128), self.modes)

    # ---------------------------------------------------------------- right-hand sides
    def nl_full(self, uph):
        """P[trunc((U + u') x (w_U + w'))] for the perturbation u' (B, 3, ...), evaluated in the algebraically identical
        split form P[trunc(U x w' + u' x (w_U + w'))] (P[U x w_U] = 0 exactly), so that no O(1) product enters (round-off
        relative to the perturbation, not to U)."""
        up = self.inv(uph)
        wp = self.inv(self.curl(uph))
        return self.proj(self.fwd(self.cross(self.U_phys, wp) + self.cross(up, self.wU_phys + wp)) * self.mask)

    def lin(self, dh):
        """linearised about U: P[trunc(U x w_d + d x w_U)]."""
        d = self.inv(dh)
        wd = self.inv(self.curl(dh))
        return self.proj(self.fwd(self.cross(self.U_phys, wd) + self.cross(d, self.wU_phys)) * self.mask)

    def bil(self, ah, bh):
        """P[trunc(a x curl b + b x curl a)] (symmetric bilinear form; bil(a, a) = 2 P[a x w_a])."""
        a, b = self.inv(ah), self.inv(bh)
        wa, wb = self.inv(self.curl(ah)), self.inv(self.curl(bh))
        return self.proj(self.fwd(self.cross(a, wb) + self.cross(b, wa)) * self.mask)

    def selfcross(self, ah):
        """P[trunc(a x curl a)]."""
        a = self.inv(ah)
        wa = self.inv(self.curl(ah))
        return self.proj(self.fwd(self.cross(a, wa)) * self.mask)

    # ---------------------------------------------------------------- symmetries
    def sym_apply(self, fh, R, d, D):
        """g f = D f(R x + d) for vector fields fh (B, 3, ...); R, D: 3-tuples of +-1; d: shifts."""
        sx, sy, sz = R
        g = fh
        flip_y = (sy == -1) if sx == 1 else (sy == 1)
        flip_z = (sz == -1) if sx == 1 else (sz == 1)
        if flip_y:
            g = torch.roll(torch.flip(g, dims=[-3]), shifts=1, dims=-3)
        if flip_z:
            g = torch.roll(torch.flip(g, dims=[-2]), shifts=1, dims=-2)
        if sx == -1:
            g = g.conj()
        phase = torch.exp(1j * (sx * self.KX * d[0] + self.KY * (-1.0 + sy * (d[1] + 1.0)) + sz * self.KZ * d[2]))
        Dt = torch.tensor(D, device=self.dev, dtype=torch.float64).view(1, 3, 1, 1, 1)
        return g * phase * Dt

    def sym_apply_phys(self, f, R, d, D):
        """the same action on physical arrays (B, 3, Ny, Nz, Nx) by exact index maps (check of sym_apply)."""
        sx, sy, sz = R
        jx = (sx * np.arange(self.Nx) + int(round(d[0] / (self.Lx / self.Nx)))) % self.Nx
        dyi = (-sy + 1 + d[1]) / (self.Ly / self.Ny)
        assert abs(dyi - round(dyi)) < 1e-9
        jy = (sy * np.arange(self.Ny) + int(round(dyi))) % self.Ny
        jz = (sz * np.arange(self.Nz) + int(round(d[2] / (self.Lz / self.Nz)))) % self.Nz
        g = f[:, :, jy][:, :, :, jz][:, :, :, :, jx]
        return g * torch.tensor(D, device=self.dev, dtype=torch.float64).view(1, 3, 1, 1, 1)

    def candidate_symmetries(self):
        Lx, Lz = self.Lx, self.Lz
        return {
            "mirror y about 1 (u,-v,w)(x,2-y,z)": ((1, -1, 1), (0.0, 2.0, 0.0), (1, -1, 1)),
            "shift-reflect (u,v,-w)(x+Lx/2,y,-z)": ((1, 1, -1), (Lx / 2, 0.0, 0.0), (1, 1, -1)),
            "shift-rotate (-u,-v,w)(-x,-y,z+Lz/2)": ((-1, -1, 1), (0.0, 0.0, Lz / 2), (-1, -1, 1)),
            "reflect z (u,v,-w)(x,y,-z)": ((1, 1, -1), (0.0, 0.0, 0.0), (1, 1, -1)),
            "shift x Lx/2": ((1, 1, 1), (Lx / 2, 0.0, 0.0), (1, 1, 1)),
            "shift z Lz/2": ((1, 1, 1), (0.0, 0.0, Lz / 2), (1, 1, 1)),
            "shift x Lx/2 and z Lz/2": ((1, 1, 1), (Lx / 2, 0.0, Lz / 2), (1, 1, 1)),
            "rotate (-u,-v,w)(-x,-y,z)": ((-1, -1, 1), (0.0, 0.0, 0.0), (-1, -1, 1)),
            "reflect x (-u,v,w)(-x,y,z)": ((-1, 1, 1), (0.0, 0.0, 0.0), (-1, 1, 1)),
            "reflect y about 0 (u,-v,w)(x,-y,z)": ((1, -1, 1), (0.0, 0.0, 0.0), (1, -1, 1)),
        }

    def set_enforced(self, names):
        c = self.candidate_symmetries()
        self.enforced = [(k, c[k]) for k in names]
        self._sym_cache = []
        for _, (R, d, D) in self.enforced:
            sx, sy, sz = R
            flip_y = (sy == -1) if sx == 1 else (sy == 1)
            flip_z = (sz == -1) if sx == 1 else (sz == 1)
            phase = torch.exp(1j * (sx * self.KX * d[0] + self.KY * (-1.0 + sy * (d[1] + 1.0)) + sz * self.KZ * d[2]))
            Dt = torch.tensor(D, device=self.dev, dtype=torch.float64).view(1, 3, 1, 1, 1)
            self._sym_cache.append((flip_y, flip_z, sx == -1, (phase[None, None] * Dt).contiguous()))

    def _apply_cached(self, fh, cache):
        flip_y, flip_z, conj, pd = cache
        g = fh
        if flip_y:
            g = torch.roll(torch.flip(g, dims=[-3]), shifts=1, dims=-3)
        if flip_z:
            g = torch.roll(torch.flip(g, dims=[-2]), shifts=1, dims=-2)
        if conj:
            g = g.conj()
        return g * pd

    def symmetrize(self, fh):
        for cache in self._sym_cache:
            fh = 0.5 * (fh + self._apply_cached(fh, cache))
        return fh

    def sym_residual(self, fh):
        """max |g u - u| in physical space over the enforced generators (per batch element)."""
        f = self.inv(fh)
        r = torch.zeros(fh.shape[0], device=self.dev, dtype=torch.float64)
        for _, (R, d, D) in self.enforced:
            r = torch.maximum(r, (self.inv(self.sym_apply(fh, R, d, D)) - f).abs().amax(dim=(1, 2, 3, 4)))
        return r

    def courant(self, uph, dt):
        up = self.inv(uph) + self.U_phys
        mx = up.abs().amax(dim=(-3, -2, -1))                        # (B, 3)
        return dt * (mx[:, 0] * self.kmax[0] + mx[:, 1] * self.kmax[1] + mx[:, 2] * self.kmax[2])

    def max_div(self, uph):
        return self.inv(self.div(uph)).abs().amax(dim=(-3, -2, -1))

    def clean(self, fh):
        return self.symmetrize(fh * self.mask)


def lawson_rk4(grid, state, rhs, h, E1=None, E2=None):
    """One Lawson RK4 step for a tuple of spectral fields with linear part -nu k^2 (all fields). rhs(state) -> tuple."""
    if E1 is None:
        E1, E2 = grid.lap_factor(h), grid.lap_factor(0.5 * h)
    k1 = rhs(state)
    s2 = tuple(E2 * (u + 0.5 * h * k) for u, k in zip(state, k1))
    k2 = rhs(s2)
    s3 = tuple(E2 * u + 0.5 * h * k for u, k in zip(state, k2))
    k3 = rhs(s3)
    s4 = tuple(E1 * u + h * E2 * k for u, k in zip(state, k3))
    k4 = rhs(s4)
    return tuple(E1 * u + (h / 6.0) * (E1 * a + 2.0 * E2 * (b + c) + d) for u, a, b, c, d in zip(state, k1, k2, k3, k4))


def step_nl(grid, uph, h, E1=None, E2=None):
    (u,) = lawson_rk4(grid, (uph,), lambda s: (grid.nl_full(s[0]),), h, E1, E2)
    return grid.clean(u)


def run_nonlinear(grid, uph0, dt, T, r_A=None, E_B=None, record_every=1.0, absorb=True, max_batch=256, progress=False,
                  desc="dns", keep_final=False, courant_every=10.0, extra=None, time_limit_s=None):
    """Integrate B perturbation fields. Fate: 0 = A (E < r_A), 1 = B (E >= E_B), -1 = undecided at T. With absorb, decided
    cases are removed from the batch. Records E every record_every (NaN after absorption). Returns dict of numpy arrays."""
    from tqdm.auto import tqdm
    import time as _time
    B = uph0.shape[0]
    n_rec = int(round(T / record_every)) + 1
    steps_per_rec = int(round(record_every / dt))
    assert abs(steps_per_rec * dt - record_every) < 1e-12
    E_hist = np.full((B, n_rec), np.nan)
    fate = np.full(B, -1, dtype=np.int8)
    t_hit = np.full(B, np.nan)
    cmax = np.zeros(B)
    finals = [None] * B if keep_final else None
    E1, E2 = grid.lap_factor(dt), grid.lap_factor(0.5 * dt)
    t_start = _time.time()
    for b0 in range(0, B, max_batch):
        idx = np.arange(b0, min(B, b0 + max_batch))
        u = uph0[idx].clone()
        e = grid.energy(u).cpu().numpy()
        E_hist[idx, 0] = e
        act = np.ones(len(idx), dtype=bool)
        if absorb:
            for j in range(len(idx)):
                if r_A is not None and e[j] < r_A:
                    fate[idx[j]], t_hit[idx[j]], act[j] = 0, 0.0, False
                elif E_B is not None and e[j] >= E_B:
                    fate[idx[j]], t_hit[idx[j]], act[j] = 1, 0.0, False
            u = u[torch.as_tensor(np.where(act)[0], device=grid.dev)]
            ids = idx[act]
        else:
            ids = idx.copy()
        it = tqdm(range(1, n_rec), desc=desc, disable=not progress, leave=False)
        for r in it:
            if len(ids) == 0:
                break
            for _ in range(steps_per_rec):
                u = step_nl(grid, u, dt, E1, E2)
            t = r * record_every
            e = grid.energy(u).cpu().numpy()
            E_hist[ids, r] = e
            if courant_every and (abs(t / courant_every - round(t / courant_every)) < 1e-9):
                c = grid.courant(u, dt).cpu().numpy()
                cmax[ids] = np.maximum(cmax[ids], c)
            bad = ~np.isfinite(e)
            if absorb:
                newA = (e < r_A) if r_A is not None else np.zeros(len(ids), bool)
                newB = ((e >= E_B) | bad) if E_B is not None else bad
                dec = newA | newB
                if dec.any():
                    fate[ids[newA]] = 0
                    fate[ids[newB & ~newA]] = 1
                    t_hit[ids[dec]] = t
                    if keep_final:
                        for j in np.where(dec)[0]:
                            finals[ids[j]] = u[j:j + 1].clone()
                    keep = np.where(~dec)[0]
                    u = u[torch.as_tensor(keep, device=grid.dev)]
                    ids = ids[keep]
            if time_limit_s is not None and _time.time() - t_start > time_limit_s:
                break
        if keep_final:
            for j, i in enumerate(ids):
                finals[i] = u[j:j + 1].clone()
    out = {"fate": fate, "t_hit": t_hit, "E": E_hist, "courant_max": cmax}
    if keep_final:
        out["final"] = finals
    return out


def coarse_grid(r_A, E_B, n):
    return np.sqrt(np.geomspace(r_A * (1 + 1e-9), E_B * (1 - 1e-9), int(n)))


def find_sc(grid, dirs, r_A, E_B, T_det, dt, n_coarse=12, rel_tol=1e-3, max_batch=256, progress=False, desc="s_c",
            log=print):
    """Gate 2 rule along rays u' = s sum d_i u_i: coarse grid of n_coarse amplitudes (E from r_A to E_B, log-spaced);
    first grid point whose fate is not A (B or undecided by T_det); bisection inside [previous point, that point] until
    (hi - lo)/lo <= rel_tol; s_c = midpoint. Returns dict (numpy)."""
    import time as _time
    dirs = np.asarray(dirs, dtype=np.float64)
    R = len(dirs)
    sg = coarse_grid(r_A, E_B, n_coarse)
    t0 = _time.time()
    A = (sg[None, :, None] * dirs[:, None, :]).reshape(-1, 9)
    u0 = grid.clean(grid.from_coeffs(A))
    r = run_nonlinear(grid, u0, dt, T_det, r_A, E_B, record_every=1.0, absorb=True, max_batch=max_batch, progress=progress, desc=desc + " coarse")
    fg = r["fate"].reshape(R, n_coarse)
    tg = r["t_hit"].reshape(R, n_coarse)
    notA = fg != 0
    first = np.argmax(notA, axis=1)
    no_cross = ~notA.any(axis=1)
    switches = np.abs(np.diff(notA.astype(np.int8), axis=1)).sum(1)
    hi = np.where(no_cross, sg[-1], sg[first])
    lo = np.where(first > 0, sg[np.maximum(first - 1, 0)], 0.0)
    out_hi = fg[np.arange(R), first].copy()
    act = ~no_cross & (first > 0)
    n_round = 0
    undec = int(np.sum(fg == -1))
    n_runs = R * n_coarse
    t_lo_hit = np.where(first > 0, tg[np.arange(R), np.maximum(first - 1, 0)], np.nan)
    t_hi_hit = tg[np.arange(R), first]
    log(f"    {desc}: coarse {R}x{n_coarse} in {_time.time() - t0:.0f} s; undecided {undec}; no crossing {int(no_cross.sum())}; below grid {int((~no_cross & (first == 0)).sum())}")
    while act.any() and np.any((hi[act] - lo[act]) / lo[act] > rel_tol):
        n_round += 1
        idx = np.where(act & ((hi - lo) / np.where(lo > 0, lo, 1.0) > rel_tol))[0]
        mid = 0.5 * (lo[idx] + hi[idx])
        u0 = grid.clean(grid.from_coeffs(mid[:, None] * dirs[idx]))
        rr = run_nonlinear(grid, u0, dt, T_det, r_A, E_B, record_every=1.0, absorb=True, max_batch=max_batch, progress=False)
        n_runs += len(idx)
        undec += int(np.sum(rr["fate"] == -1))
        isA = rr["fate"] == 0
        lo[idx] = np.where(isA, mid, lo[idx])
        hi[idx] = np.where(isA, hi[idx], mid)
        out_hi[idx] = np.where(isA, out_hi[idx], rr["fate"])
        t_lo_hit[idx] = np.where(isA, rr["t_hit"], t_lo_hit[idx])
        t_hi_hit[idx] = np.where(isA, t_hi_hit[idx], rr["t_hit"])
        log(f"    {desc}: bisection round {n_round}: {len(idx)} rays, max rel width {np.max((hi[idx] - lo[idx]) / lo[idx]):.2e}, {_time.time() - t0:.0f} s")
    s_c = np.where(no_cross | (~no_cross & (first == 0)), np.nan, 0.5 * (lo + hi))
    return {"s_c": s_c, "lo": lo, "hi": hi, "out_hi": out_hi, "no_crossing": no_cross, "below_grid": ~no_cross & (first == 0),
            "n_switches": switches, "coarse_fate": fg, "coarse_t_hit": tg, "s_grid": sg, "n_rounds": n_round,
            "n_undecided_runs": undec, "n_runs": n_runs, "t_hit_lo": t_lo_hit, "t_hit_hi": t_hi_hit, "seconds": _time.time() - t0}
