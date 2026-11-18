"""Gate 8 (gate8_dns.yaml): the third condition (fast channel = oblique-wave route) and the 8d records.

wave_gain:   G = max_{0 <= t <= T} ||Pw delta(t)||^2 / ||U||^2 (||U|| = 1), delta linearised about U, delta(0) = sum_i d_i u_i (unit
             amplitude), the maximum taken over every time step (h2). Also ||Pw delta||^2 at every output time.
theory_kz:   delta and q as dns_theory.run_delta_q, with the streak (u, kx = 0, kz != 0) and the roll ((v, w), kx = 0, kz != 0) of
             P0 delta and q split into kz bins |l| = 1, 2, >= 3 (h10).
diag_kz:     the Gate 7 diagnostic runs (dns_local, 5b re-run rule) with the extra kz bins of the streak and roll energies (h10).
gain_alpha:  dns_theory.run_gain at given amplitudes, recording ||y||^2 and the kx = alpha part ||y_alpha||^2 (h11).
dns.py, dns_theory.py and dns_local.py are not changed (they are the record of Gates 4-7).
"""
import numpy as np
import torch

from .dns import _kmax_t, lawson_rk4, step_nl_c
from .dns_local import PARTS, partition, partition_masks
from .dns_theory import _lin_self, streak_mask

KZ_BINS = ["kz=gamma", "kz=2gamma", "kz>=3gamma"]
PARTS8 = PARTS + [f"streak {b}" for b in KZ_BINS] + [f"roll {b}" for b in KZ_BINS]


def _kz_bin_masks(grid):
    """(Nz, 1) masks of |l| = 1, |l| = 2, |l| >= 3 (to combine with kx = 0)."""
    al = grid.l_int.abs()
    return [((al == 1).to(torch.float64))[:, None], ((al == 2).to(torch.float64))[:, None], ((al >= 3).to(torch.float64))[:, None]]


def wave_gain(grid, dirs, T=20.0, dt=0.0625, every=1.0, max_batch=64):
    """G (R,), t_at_max (R,), series (R, n_out) of ||Pw delta||^2 at t = 0, every, ..., T; the max is over every step."""
    dirs = np.asarray(dirs, dtype=np.float64)
    R = len(dirs)
    n_out = int(round(T / every)) + 1
    spr = int(round(every / dt))
    assert abs(spr * dt - every) < 1e-12
    Pw = 1.0 - grid.mask_kx0
    E1, E2 = grid.lap_factor(dt), grid.lap_factor(0.5 * dt)
    G = np.zeros(R)
    tmax = np.zeros(R)
    ser = np.zeros((R, n_out))
    for b0 in range(0, R, max_batch):
        sl = slice(b0, min(R, b0 + max_batch))
        dh = grid.clean(grid.from_coeffs(dirs[sl]))
        e = grid.energy(dh * Pw).cpu().numpy()
        ser[sl, 0] = e
        G[sl], tmax[sl] = e, 0.0
        step = 0
        for k in range(1, n_out):
            for _ in range(spr):
                (dh,) = lawson_rk4(grid, (dh,), lambda s: (grid.lin(s[0]),), dt, E1, E2)
                dh = grid.clean(dh)
                step += 1
                e = grid.energy(dh * Pw).cpu().numpy()
                up = e > G[sl]
                G[sl] = np.where(up, e, G[sl])
                tmax[sl] = np.where(up, step * dt, tmax[sl])
            ser[sl, k] = e
    return {"G": G, "t_at_max": tmax, "series": ser, "t": np.arange(n_out) * every}


def theory_kz(grid, dirs, T, dt, every=1.0, max_batch=16, progress=False, desc="theory kz"):
    """aa, ab, bb per kz bin for the streak (u) and the roll ((v, w)) of a = P0 delta, b = q: arrays (R, 3, n_out) in a dict
    {'streak': {'aa','ab','bb'}, 'roll': {...}}; the bins are orthogonal, so the bin sums are the totals of run_delta_q."""
    from tqdm.auto import tqdm
    dirs = np.asarray(dirs, dtype=np.float64)
    R = len(dirs)
    n_out = int(round(T / every)) + 1
    spr = int(round(every / dt))
    assert abs(spr * dt - every) < 1e-12
    out = {p: {k: np.zeros((R, 3, n_out)) for k in ("aa", "ab", "bb")} for p in ("streak", "roll")}
    sm = streak_mask(grid)                                   # (Nz, Nxh): kx = 0, kz != 0
    bins = [sm * b for b in _kz_bin_masks(grid)]
    P0 = grid.mask_kx0
    E1, E2 = grid.lap_factor(dt), grid.lap_factor(0.5 * dt)

    def rhs(s):
        dh, qh = s
        lin_d, sc = _lin_self(grid, dh)
        return lin_d, (grid.lin(qh) + sc) * P0

    def part(fh, comp, mk):
        o = torch.zeros_like(fh)
        o[:, comp] = fh[:, comp] * mk
        return o

    for b0 in range(0, R, max_batch):
        sl = slice(b0, min(R, b0 + max_batch))
        dh = grid.clean(grid.from_coeffs(dirs[sl]))
        qh = torch.zeros_like(dh)

        def record(k):
            a0, b0_ = dh * P0, qh
            for p, comps in (("streak", [0]), ("roll", [1, 2])):
                for j, mk in enumerate(bins):
                    a = sum(part(a0, c, mk) for c in comps)
                    b = sum(part(b0_, c, mk) for c in comps)
                    out[p]["aa"][sl, j, k] = grid.inner(a, a).cpu().numpy()
                    out[p]["ab"][sl, j, k] = grid.inner(a, b).cpu().numpy()
                    out[p]["bb"][sl, j, k] = grid.inner(b, b).cpu().numpy()
        record(0)
        for k in tqdm(range(1, n_out), desc=desc, disable=not progress, leave=False, mininterval=30):
            for _ in range(spr):
                dh, qh = lawson_rk4(grid, (dh, qh), rhs, dt, E1, E2)
                dh, qh = grid.clean(dh), grid.clean(qh)
            record(k)
    out["t"] = np.arange(n_out) * every
    return out


def partition8(grid, uh, masks, bins):
    """PARTS8 energies (B, len(PARTS8)): the Gate 7 parts and the streak / roll energies in the kz bins."""
    base = partition(grid, uh, masks)
    e = grid.w * (uh.real ** 2 + uh.imag ** 2)
    ext = [(e[:, 0] * mk).sum(dim=(1, 2, 3)) for mk in bins] + [(e[:, 1:] * mk).sum(dim=(1, 2, 3, 4)) for mk in bins]
    return torch.cat([base, torch.stack(ext, dim=1)], dim=1)


def _diag8_level(grid, u0, dt, T, every, masks, bins, courant_limit, stop_on_flag, desc="diag8"):
    from tqdm.auto import tqdm
    n_rec = int(round(T / every)) + 1
    spr = int(round(every / dt))
    assert abs(spr * dt - every) < 1e-12
    E1, E2 = grid.lap_factor(dt), grid.lap_factor(0.5 * dt)
    km = _kmax_t(grid)
    u = u0.clone()
    B = u.shape[0]
    rec = np.full((B, n_rec, len(PARTS8)), np.nan)
    rec[:, 0] = partition8(grid, u, masks, bins).cpu().numpy()
    cmax = torch.zeros(B, device=grid.dev, dtype=torch.float64)
    flag = torch.zeros(B, dtype=torch.bool, device=grid.dev)
    for k in tqdm(range(1, n_rec), desc=desc, leave=False, mininterval=30):
        for _ in range(spr):
            u, c = step_nl_c(grid, u, dt, E1, E2, km)
            c = torch.nan_to_num(c, nan=float("inf"))
            cmax = torch.maximum(cmax, c)
            flag |= c > courant_limit
        rec[:, k] = partition8(grid, u, masks, bins).cpu().numpy()
        flag |= ~torch.isfinite(torch.as_tensor(rec[:, k, 0], device=grid.dev))
        if stop_on_flag and bool(flag.all()):
            break
    return rec, cmax.cpu().numpy(), flag.cpu().numpy()


def diag_kz(grid, dirs_amp, dt, T=400.0, every=1.0, high_frac=2.0 / 3.0, courant_limit=2.5, max_levels=2, log=print):
    """dns_local.run_diag with PARTS8 (same 5b re-run rule: a flagged case is re-run from t = 0 at dt/2, then dt/4)."""
    masks = partition_masks(grid, high_frac)
    sm = streak_mask(grid)
    bins = [(sm * b)[None] for b in _kz_bin_masks(grid)]          # (1, Nz, Nxh) to broadcast over (B, Ny, Nz, Nxh)
    u0 = grid.clean(grid.from_coeffs(np.asarray(dirs_amp)))
    rec, cm, fl = _diag8_level(grid, u0, dt, T, every, masks, bins, courant_limit, max_levels > 0, desc=f"diag8 dt {dt:g}")
    dt_used = np.full(len(rec), dt)
    todo = np.where(fl)[0]
    for lv in range(1, max_levels + 1):
        if len(todo) == 0:
            break
        h = dt / 2 ** lv
        log(f"    diag8: re-running {len(todo)} flagged cases with dt = {h:g}")
        sub, cms, fls = _diag8_level(grid, u0[torch.as_tensor(todo, device=grid.dev)], h, T, every, masks, bins, courant_limit, lv < max_levels,
                                     desc=f"diag8 dt {h:g}")
        rec[todo], cm[todo], dt_used[todo] = sub, cms, h
        fl[todo] = fls
        todo = todo[fls] if lv < max_levels else todo[:0]
    return {"rec": rec, "t": np.arange(rec.shape[1]) * every, "dt_used": dt_used, "courant_max": cm, "flag_last": fl, "parts": PARTS8}


def gain_alpha(grid, dirs, amps, T, dt, every=1.0, max_rays=16, progress=False, desc="gain"):
    """dns_theory.run_gain (identical equations and steps) at amplitudes amps (R, K), recording yy = ||y||^2 and ya = ||y_alpha||^2
    (kx = alpha, m = 1) at every output time, and courant_M."""
    from tqdm.auto import tqdm
    dirs = np.asarray(dirs, dtype=np.float64)
    amps = np.asarray(amps, dtype=np.float64)
    R, K = amps.shape
    n_out = int(round(T / every)) + 1
    spr = int(round(every / dt))
    assert abs(spr * dt - every) < 1e-12
    yy = np.zeros((R, K, n_out))
    ya = np.zeros((R, K, n_out))
    cM = np.zeros((R, K))
    kmax = torch.tensor(grid.kmax, device=grid.dev, dtype=torch.float64)
    P0 = grid.mask_kx0
    Pw = 1.0 - P0
    Pa = (grid.m_int == 1).to(torch.float64)[None, None, :]
    E1, E2 = grid.lap_factor(dt), grid.lap_factor(0.5 * dt)
    for b0 in range(0, R, max_rays):
        sl = slice(b0, min(R, b0 + max_rays))
        r = sl.stop - sl.start
        s = torch.as_tensor(amps[sl], device=grid.dev, dtype=torch.float64)
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

        def rec(k):
            yy[sl, :, k] = grid.energy(yh).reshape(r, K).cpu().numpy()
            ya[sl, :, k] = grid.energy(yh * Pa).reshape(r, K).cpu().numpy()
        rec(0)
        for k in tqdm(range(1, n_out), desc=desc, disable=not progress, leave=False, mininterval=30):
            for _ in range(spr):
                box = {}
                dh, qh, yh = lawson_rk4(grid, (dh, qh, yh), rhs, dt, E1, E2)
                dh, qh, yh = grid.clean(dh), grid.clean(qh), grid.clean(yh)
                cM[sl] = np.maximum(cM[sl], dt * box["c"].reshape(r, K).cpu().numpy())
            rec(k)
    return {"yy": yy, "ya": ya, "t": np.arange(n_out) * every, "courant_M": cM}
