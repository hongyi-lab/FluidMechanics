"""Gate 7 (gate7_dns.yaml): s_c by a local grid around a previous estimate plus bisection (h2-h7), and the 7c energy-partition
diagnostics (h10). Uses dns.run_det (5b re-run rule) for every nonlinear run; dns.py itself is unchanged."""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .dns import _kmax_t, run_det, step_nl_c
from .utils import atomic_write_json


def _rec(s, src, r, j):
    return {"s": float(s), "src": src, "fate": int(r["fate"][j]), "t_hit": (float(r["t_hit"][j]) if np.isfinite(r["t_hit"][j]) else None),
            "dt_used": float(r["dt_used"][j]), "courant_max": float(r["courant_max"][j]), "nan_final": bool(r["nan_final"][j])}


def _run_points(grid, dirs, pts, r_A, E_B, T_det, dt, courant_limit, max_levels, max_batch, log, desc, progress):
    """pts: list of (ray index, s, src). Runs them in one run_det batch. Returns list of records (same order) and counts."""
    if not pts:
        return [], None
    A = np.array([s * dirs[i] for i, s, _ in pts])
    u0 = grid.clean(grid.from_coeffs(A))
    r = run_det(grid, u0, dt, T_det, r_A, E_B, absorb=True, max_batch=max_batch, courant_limit=courant_limit, max_levels=max_levels,
                progress=progress, desc=desc, log=log)
    return [_rec(s, src, r, j) for j, (_, s, src) in enumerate(pts)], r["counts"]


def _bracket(points):
    """points: grid / extension records of one ray. Returns (status, i_lo, i_hi) on the s-sorted list, plus switches."""
    P = sorted(points, key=lambda p: p["s"])
    notA = np.array([p["fate"] != 0 for p in P])
    switches = int(np.abs(np.diff(notA.astype(np.int8))).sum()) if len(P) > 1 else 0
    if not notA.any():
        return "all A", P, None, switches
    first = int(np.argmax(notA))
    if first == 0:
        return "first not A", P, None, switches
    if P[first]["fate"] == -1:
        return "undecided", P, first, switches
    return "ok", P, first, switches


def find_sc_local(grid, dirs, names, s_prev, factors, ext_up, ext_down, r_A, E_B, T_det, dt, rel_width=0.01, courant_limit=2.5,
                  max_levels=2, max_batch=64, ckpt=None, log=print, desc="s_c", progress=True):
    """gate7_dns.yaml (h2-h7). dirs (R, 9) unit rays, s_prev (R,) previous-level s_c. Resumable through the JSON checkpoint `ckpt`
    (written after the grid, the extension and every bisection round). Returns the state dict (per ray: points, bisection
    history, lo, hi, s_c, t_hit_lo, t_hit_hi, status)."""
    dirs = np.asarray(dirs, dtype=np.float64)
    R = len(dirs)
    t0 = time.time()
    key = {"names": list(names), "s_prev": [float(x) for x in s_prev], "dt": float(dt), "T_det": float(T_det), "r_A": float(r_A),
           "E_B": float(E_B), "factors": list(map(float, factors)), "rel_width": float(rel_width), "dirs": np.asarray(dirs).round(15).tolist()}
    st = None
    if ckpt is not None and Path(ckpt).exists():
        Z = json.loads(Path(ckpt).read_text())
        if Z.get("key") == key:
            st = Z
            st["seconds_prev"] = float(st.get("seconds", 0.0))           # review fix: keep the time of earlier sessions
            log(f"    {desc}: resumed from {ckpt} (phase {st['phase']}, round {st['n_round']})")
            if st["phase"] == "done":
                return st
    if st is None:
        st = {"key": key, "phase": "grid", "n_round": 0, "counts": [], "rays": [{"name": n, "s_prev": float(s), "points": [], "bisect": []}
                                                                               for n, s in zip(names, s_prev)]}

    def save():
        st["seconds"] = st.get("seconds_prev", 0.0) + (time.time() - t0)
        if ckpt is not None:
            atomic_write_json(ckpt, st)

    run = lambda pts, d, prog=False: _run_points(grid, dirs, pts, r_A, E_B, T_det, dt, courant_limit, max_levels, max_batch, log, d, prog)  # noqa: E731
    if st["phase"] == "grid":
        pts = [(i, st["rays"][i]["s_prev"] * f, f"grid x{f:g}") for i in range(R) for f in factors]
        recs, c = run(pts, desc + " grid", progress)
        for (i, _, _), rc in zip(pts, recs):
            st["rays"][i]["points"].append(rc)
        st["counts"].append(c)
        st["phase"] = "extend"
        save()
        log(f"    {desc}: local grid {R}x{len(factors)} in {time.time() - t0:.0f} s; fates "
            + "; ".join(f"{r['name']} " + "".join({0: "A", 1: "B", -1: "u", 2: "d"}[p["fate"]] for p in sorted(r["points"], key=lambda p: p["s"])) for r in st["rays"]))
    if st["phase"] == "extend":
        pts = []
        for i, r in enumerate(st["rays"]):
            status, _, _, _ = _bracket(r["points"])
            r["grid_status"] = status
            if status == "all A":
                pts += [(i, r["s_prev"] * f, f"ext up x{f:g}") for f in ext_up]
            elif status == "first not A":
                pts += [(i, r["s_prev"] * f, f"ext down x{f:g}") for f in ext_down]
        if pts:
            log(f"    {desc}: extending {len(set(p[0] for p in pts))} rays once ({len(pts)} runs)")
            recs, c = run(pts, desc + " extend")
            for (i, _, _), rc in zip(pts, recs):
                st["rays"][i]["points"].append(rc)
            st["counts"].append(c)
        for r in st["rays"]:
            status, P, first, sw = _bracket(r["points"])
            r["n_switches"] = sw
            r["status"] = {"ok": "bisect", "all A": "not bracketed (all A)", "first not A": "not bracketed (first point not A)",
                           "undecided": "undecided (first not-A point undecided at T_det)"}[status]
            if status == "ok":
                r["lo"], r["hi"] = P[first - 1]["s"], P[first]["s"]
                r["t_hit_lo"], r["t_hit_hi"] = P[first - 1]["t_hit"], P[first]["t_hit"]
                r["fate_hi"] = P[first]["fate"]
        st["phase"] = "bisect"
        save()
    while st["phase"] == "bisect":
        act = [i for i, r in enumerate(st["rays"]) if r["status"] == "bisect" and (r["hi"] - r["lo"]) / r["lo"] > rel_width]
        if not act:
            break
        pts = [(i, 0.5 * (st["rays"][i]["lo"] + st["rays"][i]["hi"]), f"bisect {st['n_round'] + 1}") for i in act]
        recs, c = run(pts, desc + " bisect")
        st["counts"].append(c)
        for (i, s, _), rc in zip(pts, recs):
            r = st["rays"][i]
            r["bisect"].append(rc)
            if rc["fate"] == 0:
                r["lo"], r["t_hit_lo"] = s, rc["t_hit"]
            elif rc["fate"] == -1:
                r["status"] = "undecided (bisection midpoint undecided at T_det)"
            else:
                r["hi"], r["t_hit_hi"], r["fate_hi"] = s, rc["t_hit"], rc["fate"]
        st["n_round"] += 1
        save()
        w = [(st["rays"][i]["hi"] - st["rays"][i]["lo"]) / st["rays"][i]["lo"] for i in act]
        log(f"    {desc}: bisection round {st['n_round']}: {len(act)} rays, max rel width {max(w):.2e}, {time.time() - t0:.0f} s")
    for r in st["rays"]:
        if r["status"] == "bisect":
            r["status"] = "ok"
        r["s_c"] = 0.5 * (r["lo"] + r["hi"]) if r["status"] == "ok" else None
    st["phase"] = "done"
    save()
    return st


# ------------------------------------------------------------------------------------------------ 7c diagnostics (h10)
PARTS = ["E", "streak kx=0", "roll kx=0", "mean kx=0 kz=0", "kx=alpha", "kx>=2alpha", "high third x", "high third y", "high third z"]


def partition_masks(grid, high_frac):
    m, n, l = grid.m_int, grid.n_int, grid.l_int
    m0 = (m == 0)[None, None, :]
    lz0 = (l == 0)[None, :, None]
    one = torch.ones((grid.Ny, grid.Nz, grid.Nxh), dtype=torch.bool, device=grid.dev)
    M = {
        "streak kx=0": (m0 & ~lz0) & one,
        "roll kx=0": (m0 & ~lz0) & one,
        "mean kx=0 kz=0": (m0 & lz0) & one,
        "kx=alpha": (m == 1)[None, None, :] & one,
        "kx>=2alpha": (m >= 2)[None, None, :] & one,
        "high third x": (m.abs() > high_frac * grid.cut[0])[None, None, :] & one,
        "high third y": (n.abs() > high_frac * grid.cut[1])[:, None, None] & one,
        "high third z": (l.abs() > high_frac * grid.cut[2])[None, :, None] & one,
    }
    return {k: v.to(torch.float64) for k, v in M.items()}


def partition(grid, uh, masks):
    """(B, len(PARTS)) energies: E, streak (u), roll (v, w), mean, kx = alpha, kx >= 2 alpha (all components), high bands
    (absolute energies; fractions are taken by the caller)."""
    e = grid.w * (uh.real ** 2 + uh.imag ** 2)                     # (B, 3, Ny, Nz, Nxh)
    out = [e.sum(dim=(1, 2, 3, 4))]
    for k in PARTS[1:]:
        mk = masks[k]
        if k == "streak kx=0":
            out.append((e[:, 0] * mk).sum(dim=(1, 2, 3)))
        elif k == "roll kx=0":
            out.append((e[:, 1:] * mk).sum(dim=(1, 2, 3, 4)))
        else:
            out.append((e * mk).sum(dim=(1, 2, 3, 4)))
    return torch.stack(out, dim=1)


def _diag_level(grid, u0, dt, T, every, masks, courant_limit, stop_on_flag, desc="diag"):
    from tqdm.auto import tqdm
    n_rec = int(round(T / every)) + 1
    spr = int(round(every / dt))
    assert abs(spr * dt - every) < 1e-12
    E1, E2 = grid.lap_factor(dt), grid.lap_factor(0.5 * dt)
    km = _kmax_t(grid)
    u = u0.clone()
    B = u.shape[0]
    rec = np.full((B, n_rec, len(PARTS)), np.nan)
    rec[:, 0] = partition(grid, u, masks).cpu().numpy()
    cmax = torch.zeros(B, device=grid.dev, dtype=torch.float64)
    flag = torch.zeros(B, dtype=torch.bool, device=grid.dev)
    for k in tqdm(range(1, n_rec), desc=desc, leave=False, mininterval=30):
        for _ in range(spr):
            u, c = step_nl_c(grid, u, dt, E1, E2, km)
            c = torch.nan_to_num(c, nan=float("inf"))
            cmax = torch.maximum(cmax, c)
            flag |= c > courant_limit
        rec[:, k] = partition(grid, u, masks).cpu().numpy()
        flag |= ~torch.isfinite(torch.as_tensor(rec[:, k, 0], device=grid.dev))
        if stop_on_flag and bool(flag.all()):
            break
    return rec, cmax.cpu().numpy(), flag.cpu().numpy()


def run_diag(grid, dirs_amp, dt, T=400.0, every=1.0, high_frac=2.0 / 3.0, courant_limit=2.5, max_levels=2, log=print):
    """dirs_amp (B, 9) = s * d. Runs to T without absorption; flagged cases (Courant > limit or NaN) re-run from t = 0 at dt/2,
    dt/4 (5b). Returns rec (B, n_t, len(PARTS)) absolute energies, t, dt_used, courant_max, flag_last."""
    masks = partition_masks(grid, high_frac)
    u0 = grid.clean(grid.from_coeffs(np.asarray(dirs_amp)))
    rec, cm, fl = _diag_level(grid, u0, dt, T, every, masks, courant_limit, max_levels > 0, desc=f"diag dt {dt:g}")
    dt_used = np.full(len(rec), dt)
    todo = np.where(fl)[0]
    for lv in range(1, max_levels + 1):
        if len(todo) == 0:
            break
        h = dt / 2 ** lv
        log(f"    diag: re-running {len(todo)} flagged cases with dt = {h:g}")
        sub, cms, fls = _diag_level(grid, u0[torch.as_tensor(todo, device=grid.dev)], h, T, every, masks, courant_limit, lv < max_levels)
        rec[todo], cm[todo], dt_used[todo] = sub, cms, h
        fl[todo] = fls
        todo = todo[fls] if lv < max_levels else todo[:0]
    return {"rec": rec, "t": np.arange(rec.shape[1]) * every, "dt_used": dt_used, "courant_max": cm, "flag_last": fl, "parts": PARTS}
