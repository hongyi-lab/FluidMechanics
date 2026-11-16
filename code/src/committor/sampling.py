"""State sampling for committor labels.

(i)  Ray bisection: random unit directions d from the A-side reference state x_ref; along
     x(s) = x_ref + s d, a noisy bisection (M_bisect paths per evaluation) locates the
     amplitudes where q crosses q_lo, 0.5 and q_hi; n_dense states are then placed (stratified)
     inside [s_lo, s_hi] = [s(q_lo) - margin*w, s(q_hi) + margin*w], w = s(q_hi) - s(q_lo).
(ii) Trajectory states: from one random (i) state per ray, one noisy path is integrated until
     absorption and up to k of its states (recorded every traj_record_every steps) are kept.

Every state carries the id of its ray (group); trajectory states inherit their ray's id.
"""

import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .sde import counts_from_outcomes, simulate_absorbing
from .utils import atomic_save_npz, derive_seed, load_npz, torch_generator


def estimate_q(system, states, M, sim, seed, device, desc=None, progress=True):
    states = np.ascontiguousarray(states, dtype=np.float64)
    X0 = torch.from_numpy(states).to(device).repeat_interleave(M, dim=0)
    res = simulate_absorbing(system, X0, sim["dt"], sim["max_steps"], sim["scheme"],
                             torch_generator(seed, device), sim["compact_every"], desc=desc,
                             progress=progress, bridge=sim["bridge"])
    nB, nA, nT = counts_from_outcomes(res["outcome"], len(states), M)
    return nB / np.maximum(nA + nB, 1), nT


def sample_rays(system, scfg, sim, seed, run_dir, device, progress=True):
    run_dir = Path(run_dir)
    final = run_dir / "rays.npz"
    if final.exists():
        return load_npz(final)
    prog_path = run_dir / "rays_progress.npz"
    ref = np.asarray(system.reference_state(), dtype=np.float64)
    R, s_max = int(scfg["n_rays"]), float(scfg["s_max"])
    Mb, nit = int(scfg["M_bisect"]), int(scfg["n_bisect"])
    q_lo, q_hi = float(scfg["q_lo"]), float(scfg["q_hi"])

    rng = np.random.default_rng(derive_seed(seed, "ray_directions"))
    D = rng.standard_normal((R, system.dim))
    D /= np.linalg.norm(D, axis=1, keepdims=True)

    P = load_npz(prog_path) if prog_path.exists() else {"seconds": np.array(0.0), "n_rays": np.array(R)}
    if int(P["n_rays"]) != R:
        raise RuntimeError(f"{prog_path} was started with n_rays={int(P['n_rays'])}, now {R}")
    t_mark = [time.perf_counter()]

    def save():
        now = time.perf_counter()
        P["seconds"] = np.array(float(P["seconds"]) + now - t_mark[0])
        t_mark[0] = now
        atomic_save_npz(prog_path, **P)

    if "q_smax" not in P:
        q, _ = estimate_q(system, ref + s_max * D, Mb, sim, derive_seed(seed, "ray_smax"), device,
                          "rays: q at s_max", progress)
        P["q_smax"] = q
        save()
    valid = P["q_smax"] >= q_hi
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        raise RuntimeError("no ray reaches q >= q_hi within s_max")
    nv = len(idx)

    for name, target in (("s50", 0.5), ("s05", q_lo), ("s95", q_hi)):
        if f"{name}_done" in P:
            continue
        if f"{name}_lo" not in P:
            if name == "s50":
                lo, hi = np.zeros(nv), np.full(nv, s_max)
            elif name == "s05":
                lo, hi = np.zeros(nv), P["s50_mid"].copy()
            else:
                lo, hi = P["s50_mid"].copy(), np.full(nv, s_max)
            P[f"{name}_lo"], P[f"{name}_hi"], P[f"{name}_it"] = lo, hi, np.array(0)
            save()
        for it in tqdm(range(int(P[f"{name}_it"]), nit), desc=f"bisect {name}", disable=not progress):
            lo, hi = P[f"{name}_lo"], P[f"{name}_hi"]
            mid = 0.5 * (lo + hi)
            q, _ = estimate_q(system, ref + mid[:, None] * D[idx], Mb, sim,
                              derive_seed(seed, "bisect", name, it), device, None, progress)
            up = q >= target
            P[f"{name}_hi"] = np.where(up, mid, hi)
            P[f"{name}_lo"] = np.where(up, lo, mid)
            P[f"{name}_it"] = np.array(it + 1)
            save()
        P[f"{name}_mid"] = 0.5 * (P[f"{name}_lo"] + P[f"{name}_hi"])
        P[f"{name}_done"] = np.array(True)
        save()

    out = {"directions": D, "ref": ref, "q_smax": P["q_smax"], "valid": valid, "ray_ids": idx,
           "s05": P["s05_mid"], "s50": P["s50_mid"], "s95": P["s95_mid"], "s_max": np.array(s_max),
           "seconds": P["seconds"]}
    atomic_save_npz(final, **out)
    return out


def build_states(system, rays, scfg, sim, seed, run_dir, device, progress=True):
    run_dir = Path(run_dir)
    final = run_dir / "states.npz"
    if final.exists():
        return load_npz(final)
    ref, D, idx = rays["ref"], rays["directions"], rays["ray_ids"]
    s05, s50, s95, s_max = rays["s05"], rays["s50"], rays["s95"], float(rays["s_max"])
    nR, d = len(idx), system.dim

    w = np.maximum(s95 - s05, float(scfg["min_width_frac"]) * s50)
    s_lo = np.clip(s05 - float(scfg["margin"]) * w, 0.0, s_max)
    s_hi = np.clip(s95 + float(scfg["margin"]) * w, 0.0, s_max)
    nd = int(scfg["n_dense"])
    rng = np.random.default_rng(derive_seed(seed, "dense"))
    u = rng.random((nR, nd))
    s = s_lo[:, None] + (np.arange(nd)[None, :] + u) / nd * (s_hi - s_lo)[:, None]
    Xr = ref[None, None, :] + s[:, :, None] * D[idx][:, None, :]

    # (ii) one noisy path per ray from a random (i) state of that ray
    start = rng.integers(0, nd, size=nR)
    X0 = np.ascontiguousarray(Xr[np.arange(nR), start])
    res = simulate_absorbing(system, torch.from_numpy(X0).to(device), sim["dt"], sim["max_steps"],
                             sim["scheme"], torch_generator(derive_seed(seed, "traj"), device),
                             sim["compact_every"], record_every=int(scfg["traj_record_every"]),
                             desc="trajectory sampling", progress=progress, bridge=sim["bridge"])
    ri, rs, rX = res["rec_idx"], res["rec_step"], res["rec_X"]
    k = int(scfg["traj_states_per_ray"])
    order = np.argsort(ri, kind="stable")
    bounds = np.searchsorted(ri[order], np.arange(nR + 1))
    picks = []
    for j in range(nR):
        rows = order[bounds[j]:bounds[j + 1]]
        if len(rows):
            picks.append(np.sort(rng.choice(rows, size=min(k, len(rows)), replace=False)))
    sel = np.concatenate(picks) if picks else np.zeros(0, np.int64)

    X = np.concatenate([Xr.reshape(-1, d), rX[sel]])
    group = np.concatenate([np.repeat(idx, nd), idx[ri[sel]]])
    origin = np.concatenate([np.zeros(nR * nd, np.int8), np.ones(len(sel), np.int8)])
    s_param = np.concatenate([s.ravel(), np.full(len(sel), np.nan)])
    t_rec = np.concatenate([np.zeros(nR * nd), rs[sel] * sim["dt"]])

    Xt = torch.from_numpy(np.ascontiguousarray(X))
    in_AB = (system.in_A(Xt) | system.in_B(Xt)).numpy()
    keep = ~in_AB
    out = {"X": X[keep], "group": group[keep], "origin": origin[keep], "s": s_param[keep],
           "t_rec": t_rec[keep], "n_dropped_in_AB": np.array(int(in_AB.sum())),
           "traj_outcome": res["outcome"], "traj_steps": res["steps"],
           "s_lo": s_lo, "s_hi": s_hi}
    atomic_save_npz(final, **out)
    return out
