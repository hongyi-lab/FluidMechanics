"""Gate 9 (gate9_dns.yaml): blind test outside the nine NJP patterns.

Rays of classes G, O, V are low-wavenumber fields (|kx| <= 2 alpha, |kz| <= 2 gamma, wall-normal index |n| <= 3, n the index of
the y wavenumber n pi/2 of the mirror-extended domain). They are built on a reference grid from physical white noise (an isotropic
Gaussian in the energy norm), restricted to the band and the class selection, made divergence-free by the Leray projection and
symmetric by the DNS group average (grid.proj, grid.clean), and normalised to E = 1 (h1-h4). They are stored as compact
coefficient arrays C (3, 7, 5, 3) over (component, n = -3..3, l = -2..2, m = 0..2) and embedded identically into any grid, so the
same continuous field is used at 64, 96 and 128. `extend_basis` appends them to grid.modes after the nine NJP modes; every
existing routine (grid.from_coeffs) then takes a Gate 9 ray as a unit direction (h6). dns.py, dns_theory.py and dns_local.py are
not changed.

Also: the 9.3 local-grid s_c search (h12; Gate 7 bracket and bisection with a wider grid and a two-sided extension) and the 9.4
statistics (h15)."""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .dns_local import _bracket, _run_points
from .utils import atomic_write_json

NB, LB, MB = 3, 2, 2                                     # |n| <= 3, |l| <= 2, 0 <= m <= 2


def band_idx(grid):
    """index arrays (n, l, m) of the compact band in the grid's spectral layout."""
    t = lambda a: torch.as_tensor(a, device=grid.dev, dtype=torch.long)  # noqa: E731
    return t(np.arange(-NB, NB + 1) % grid.Ny), t(np.arange(-LB, LB + 1) % grid.Nz), t(np.arange(0, MB + 1))


def embed(grid, C):
    """compact C (B, 3, 7, 5, 3) complex -> full spectral (B, 3, Ny, Nz, Nxh)."""
    C = torch.as_tensor(np.asarray(C), device=grid.dev, dtype=torch.complex128)
    out = torch.zeros((C.shape[0], 3, grid.Ny, grid.Nz, grid.Nxh), device=grid.dev, dtype=torch.complex128)
    n, l, m = band_idx(grid)
    out[:, :, n[:, None, None], l[None, :, None], m[None, None, :]] = C
    return out


def extract(grid, fh):
    """full spectral -> (compact C (B, 3, 7, 5, 3) numpy, energy outside the band (B,))."""
    n, l, m = band_idx(grid)
    C = fh[:, :, n[:, None, None], l[None, :, None], m[None, None, :]]
    rest = grid.energy(fh) - grid.energy(embed(grid, C.cpu().numpy()))
    return C.cpu().numpy(), rest.cpu().numpy()


def band_select(grid, cls, k=None):
    """(3, Ny, Nz, Nxh) 0/1 mask of the class selection (h2-h4): 'G' the whole band; 'O' m = k[0], |l| = k[1]; 'V' m = 0,
    |l| = k[1], v and w only."""
    m, n, l = grid.m_int, grid.n_int, grid.l_int
    inb = (n.abs() <= NB)[:, None, None] & (l.abs() <= LB)[None, :, None] & (m <= MB)[None, None, :]
    if cls == "G":
        sel = inb
    elif cls == "O":
        sel = inb & (m == k[0])[None, None, :] & (l.abs() == k[1])[None, :, None]
    elif cls == "V":
        sel = inb & (m == 0)[None, None, :] & (l.abs() == k[1])[None, :, None]
    else:
        raise ValueError(cls)
    S = sel.to(torch.float64)[None].expand(3, -1, -1, -1).clone()
    if cls == "V":
        S[0] = 0.0                                                    # u = 0 (h4)
    return S


def random_field(grid, seed, cls, k=None):
    """one projected, normalised field of the class (spectral, (1, 3, ...)); white noise N(0, 1) per grid point and component."""
    rng = np.random.default_rng(int(seed))
    w = torch.as_tensor(rng.standard_normal((1, 3, grid.Ny, grid.Nz, grid.Nx)), device=grid.dev, dtype=torch.float64)
    fh = grid.fwd(w) * band_select(grid, cls, k)[None]
    fh = grid.clean(grid.proj(fh))
    e = float(grid.energy(fh)[0])
    if e <= 1e-300:
        return None
    return fh / math.sqrt(e)


def wavenumber2(grid, fh):
    """energy-weighted mean |k|^2 (kx^2 + ky^2 + kz^2) of a field (B,)."""
    e = grid.w * (fh.real ** 2 + fh.imag ** 2)
    return ((e * grid.K2[None, None]).sum(dim=(1, 2, 3, 4)) / e.sum(dim=(1, 2, 3, 4))).cpu().numpy()


def build_rays(grid, spec):
    """spec: the 'rays9' block of the config. Returns dict name -> record (compact C, metadata) for the G, O, V rays."""
    out = {}
    for j, sd in enumerate(spec["G"]["seeds"]):
        name = f"G#{j + 1}"
        seed, tries = int(sd), []
        while True:
            fh = random_field(grid, seed, "G")
            a = grid.coeffs(fh)[0].cpu().numpy()
            frac9 = float(np.sum(a ** 2))
            tries.append({"seed": seed, "nine_mode_fraction": frac9})
            if frac9 <= float(spec["G"]["max_nine_mode_fraction"]):
                break
            seed += int(spec["G"]["reseed_step"])                     # (h2)
        out[name] = {"cls": "G", "field": fh, "seed": seed, "tries": tries}
    for j, (kk, sd) in enumerate(zip(spec["O"]["k"], spec["O"]["seeds"])):
        name = f"O#{j + 1}"
        fh = random_field(grid, int(sd), "O", kk)
        out[name] = {"cls": "O", "field": fh, "seed": int(sd), "k": list(kk)}
    for j, (lk, sd, ss) in enumerate(zip(spec["V"]["l"], spec["V"]["seeds"], spec["V"]["seed_seeds"])):
        name = f"V#{j + 1}"
        v = random_field(grid, int(sd), "V", (0, lk))
        gseed = random_field(grid, int(ss), "G")
        fh = v + float(spec["V"]["seed_rel_amplitude"]) * gseed       # (h4) amplitude ratio 1e-2 (energy 1e-4)
        fh = fh / math.sqrt(float(grid.energy(fh)[0]))
        out[name] = {"cls": "V", "field": fh, "seed": int(sd), "seed_seed": int(ss), "l": int(lk)}
    for name, r in out.items():
        if r["field"] is None:
            raise RuntimeError(f"{name}: the projection left no energy")
        a = grid.coeffs(r["field"])[0].cpu().numpy()
        r["nine_mode_fraction"] = float(np.sum(a ** 2))
        r["nine_mode_coeffs"] = a.tolist()
        r["k2_mean"] = float(wavenumber2(grid, r["field"])[0])
        r["max_div"] = float(grid.max_div(r["field"])[0])
        r["sym_residual"] = float(grid.sym_residual(r["field"])[0])
        r["C"], rest = extract(grid, r["field"])
        r["energy_outside_band"] = float(rest[0])
        r["C"] = r["C"][0]
        del r["field"]
    return out


def extend_basis(grid, C_list):
    """append the compact fields (list of (3, 7, 5, 3)) to grid.modes (h6); returns the number of NJP modes (9)."""
    n9 = 9
    assert grid.modes.shape[0] == n9, "extend_basis called twice"
    if len(C_list):
        extra = embed(grid, np.stack(C_list))
        grid.modes = torch.cat([grid.modes, extra], dim=0)
    return n9


# ------------------------------------------------------------------------------------------------------------ 9.3 search
def find_sc_grid9(grid, dirs, names, s_center, factors, ext_up, ext_down, r_A, E_B, T_det, dt, rel_width=0.01, courant_limit=2.5,
                  max_levels=2, max_batch=64, ckpt=None, log=print, desc="s_c", progress=True):
    """(h12) local grid s_center x factors; rays not bracketed get ext_down and ext_up (both sides, once); bracket = the first
    not-A point and the one before (Gate 7 rule; an undecided first not-A point is flagged); more than one A / not-A switch on the
    grid flagged 'non-monotone' (bisection still on the first switch); bisection to rel_width. Resumable via the JSON `ckpt`."""
    dirs = np.asarray(dirs, dtype=np.float64)
    R = len(dirs)
    t0 = time.time()
    key = {"names": list(names), "s_center": [float(x) for x in s_center], "dt": float(dt), "T_det": float(T_det), "r_A": float(r_A),
           "E_B": float(E_B), "factors": list(map(float, factors)), "ext_up": list(map(float, ext_up)), "ext_down": list(map(float, ext_down)),
           "rel_width": float(rel_width), "dirs": np.asarray(dirs).round(15).tolist()}
    st = None
    if ckpt is not None and Path(ckpt).exists():
        Z = json.loads(Path(ckpt).read_text())
        if Z.get("key") == key:
            st = Z
            st["seconds_prev"] = float(st.get("seconds", 0.0))
            log(f"    {desc}: resumed from {ckpt} (phase {st['phase']}, round {st['n_round']})")
            if st["phase"] == "done":
                return st
    if st is None:
        st = {"key": key, "phase": "grid", "n_round": 0, "counts": [], "rays": [{"name": n, "s_center": float(s), "points": [], "bisect": []}
                                                                               for n, s in zip(names, s_center)]}

    def save():
        st["seconds"] = st.get("seconds_prev", 0.0) + (time.time() - t0)
        if ckpt is not None:
            atomic_write_json(ckpt, st)

    def fates(r):
        return "".join({0: "A", 1: "B", -1: "u", 2: "d"}[p["fate"]] for p in sorted(r["points"], key=lambda p: p["s"]))

    run = lambda pts, d, prog=False: _run_points(grid, dirs, pts, r_A, E_B, T_det, dt, courant_limit, max_levels, max_batch, log, d, prog)  # noqa: E731
    if st["phase"] == "grid":
        pts = [(i, st["rays"][i]["s_center"] * f, f"grid x{f:g}") for i in range(R) for f in factors]
        recs, c = run(pts, desc + " grid", progress)
        for (i, _, _), rc in zip(pts, recs):
            st["rays"][i]["points"].append(rc)
        st["counts"].append(c)
        st["phase"] = "extend"
        save()
        log(f"    {desc}: local grid {R}x{len(factors)} in {time.time() - t0:.0f} s; fates " + "; ".join(f"{r['name']} {fates(r)}" for r in st["rays"]))
    if st["phase"] == "extend":
        pts = []
        for i, r in enumerate(st["rays"]):
            status, _, _, _ = _bracket(r["points"])
            r["grid_status"] = status
            if status in ("all A", "first not A"):
                pts += [(i, r["s_center"] * f, f"ext down x{f:g}") for f in ext_down] + [(i, r["s_center"] * f, f"ext up x{f:g}") for f in ext_up]
        if pts:
            log(f"    {desc}: extending {len(set(p[0] for p in pts))} rays on both sides ({len(pts)} runs)")
            recs, c = run(pts, desc + " extend")
            for (i, _, _), rc in zip(pts, recs):
                st["rays"][i]["points"].append(rc)
            st["counts"].append(c)
        for r in st["rays"]:
            status, P, first, sw = _bracket(r["points"])
            r["n_switches"] = sw
            r["non_monotone"] = bool(sw > 1)
            r["fates"] = fates(r)
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


# ------------------------------------------------------------------------------------------------------------ statistics
def conformal_halfwidth(abs_err, p):
    """split conformal (h11): the ceil((n+1) p)-th smallest |err|; inf if that rank exceeds n."""
    a = np.sort(np.asarray(abs_err, dtype=np.float64))
    k = math.ceil((len(a) + 1) * p)
    return float(a[k - 1]) if k <= len(a) else math.inf


def boot_median_ci(x, n_boot=10000, seed=0, level=0.95):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = np.median(x[idx], axis=1)
    lo, hi = np.quantile(m, [(1 - level) / 2, 1 - (1 - level) / 2])
    return [float(lo), float(hi)]


def binom_two_sided(k, n, p):
    """exact two-sided binomial test (sum of probabilities <= P(k)), as scipy.stats.binomtest."""
    if n == 0:
        return None
    try:
        from scipy.stats import binomtest
        return float(binomtest(int(k), int(n), float(p)).pvalue)
    except ImportError:                                               # fallback, same definition
        pmf = np.array([math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)])
        return float(min(1.0, pmf[pmf <= pmf[k] * (1 + 1e-7)].sum()))
