"""Batched SDE integration with absorption in A or B (float64, GPU)."""

import math

import numpy as np
import torch
from tqdm.auto import tqdm


@torch.no_grad()
def _advance_one(system, X, alive, loc_out, loc_step, noise, u, k, dt, scheme, bridge):
    """One step for all rows (the math of the loop below, factored out so it can be compiled).
    noise and u are drawn by the caller in the same order as the original loop (noise, then u)."""
    one = torch.ones_like(loc_out)
    zero = torch.zeros_like(loc_out)
    if bridge:
        lA0, lB0, vA, vB = system.boundary_levels(X)
    f = system.drift(X)
    Xn = X + f * dt + noise
    if scheme == "heun":
        Xn = X + 0.5 * (f + system.drift(Xn)) * dt + noise
    X = torch.where(alive.unsqueeze(1), Xn, X)
    enterB = system.in_B(X)
    enterA = system.in_A(X)
    if bridge:
        lA1, lB1, _, _ = system.boundary_levels(X)
        pA = torch.exp(-2.0 * lA0.clamp(min=0.0) * lA1.clamp(min=0.0) / (vA * dt))
        pB = torch.exp(-2.0 * lB0.clamp(min=0.0) * lB1.clamp(min=0.0) / (vB * dt))
        enterA = enterA | (u[:, 0] < pA)
        enterB = enterB | (u[:, 1] < pB)
    hB = alive & enterB
    hA = alive & enterA & ~hB
    hit = hA | hB
    loc_out = torch.where(hB, one, torch.where(hA, zero, loc_out))
    loc_step = torch.where(hit, k.expand_as(loc_step), loc_step)
    alive = alive & ~hit
    return X, alive, loc_out, loc_step


_COMPILED_STEPS = {}


def simulate_absorbing(system, X0, dt, max_steps, scheme="heun", generator=None,
                       compact_every=100, record_every=None, desc=None, progress=True, bridge=False,
                       compile_step=False, return_final=False):
    """Integrate dX = f(X) dt + S dW for every row of X0 until it first enters A or B.

    Membership of A and B is tested after every step (and once on the initial state).
    Finished rows are frozen and periodically dropped from the working batch.

    bridge=True adds the Brownian-bridge crossing test: a step whose two end points are both
    outside A (level h_A > 0) is still absorbed in A with probability
    exp(-2 h_A(X_n) h_A(X_{n+1}) / (var_A dt)), and likewise for B. This removes the O(sqrt(dt))
    bias of checking the boundary only at grid times (exact for flat boundaries and constant noise;
    a local linearisation otherwise). The system must provide boundary_levels(X).
    If both the A and the B test fire in the same step, B is taken (as for direct entry).

    Returns a dict of numpy arrays
      outcome : int8 (n,)   1 = entered B first, 0 = entered A first, -1 = unresolved after max_steps
      steps   : int64 (n,)  step index of absorption (0 if X0 already in A or B; max_steps if unresolved)
    and, if record_every=k, the unabsorbed states every k steps:
      rec_idx (row of X0), rec_step, rec_X
    """
    if scheme not in ("em", "heun"):
        raise ValueError(f"unknown scheme {scheme!r}")
    if X0.dtype != torch.float64:
        raise TypeError("X0 must be float64")
    if bridge and not system.has_boundary_levels():
        raise ValueError(f"{system.name} does not provide boundary_levels; bridge correction unavailable")
    device, dtype = X0.device, X0.dtype
    n = X0.shape[0]
    S = system.noise_std(device=device, dtype=dtype) * math.sqrt(dt)
    one = torch.tensor(1, dtype=torch.int8, device=device)
    zero = torch.tensor(0, dtype=torch.int8, device=device)

    final_X = X0.clone() if return_final else None     # state at absorption (or at max_steps)
    outcome = torch.full((n,), -1, dtype=torch.int8, device=device)
    steps = torch.full((n,), max_steps, dtype=torch.int64, device=device)
    inB = system.in_B(X0)
    inA = system.in_A(X0) & ~inB
    outcome[inB] = 1
    outcome[inA] = 0
    steps[inA | inB] = 0
    keep = ~(inA | inB)
    gidx = torch.nonzero(keep).squeeze(1)
    X = X0[keep].clone()
    loc_out = torch.full((X.shape[0],), -1, dtype=torch.int8, device=device)
    loc_step = torch.full((X.shape[0],), max_steps, dtype=torch.int64, device=device)
    alive = torch.ones(X.shape[0], dtype=torch.bool, device=device)

    rec_idx, rec_step, rec_X = [], [], []
    pbar = tqdm(total=max_steps, desc=desc, leave=False, disable=not progress, mininterval=2.0)
    last = 0
    if compile_step:
        key = (id(system), float(dt), scheme, bool(bridge))
        if key not in _COMPILED_STEPS:
            _COMPILED_STEPS[key] = torch.compile(
                lambda X, a, lo, ls, nz, u, k: _advance_one(system, X, a, lo, ls, nz, u, k, dt, scheme, bridge),
                dynamic=True, fullgraph=True)
        step_fn = _COMPILED_STEPS[key]
    eager_fn = lambda X, a, lo, ls, nz, u, k: _advance_one(system, X, a, lo, ls, nz, u, k, dt, scheme, bridge)
    if not compile_step:
        step_fn = eager_fn
    k_dev = torch.zeros((), dtype=torch.int64, device=device)      # step counter kept on the device (no sync)
    for k in range(1, max_steps + 1):
        if X.shape[0] == 0:
            break
        k_dev = k_dev + 1
        noise = torch.randn(X.shape, generator=generator, device=device, dtype=dtype) * S
        u = torch.rand((X.shape[0], 2), generator=generator, device=device, dtype=dtype) if bridge else None
        fn = step_fn if X.shape[0] >= 2 else eager_fn          # inductor mis-launches batch-size-1 kernels
        X, alive, loc_out, loc_step = fn(X, alive, loc_out, loc_step, noise, u, k_dev)
        if record_every and k % record_every == 0:
            rec_idx.append(gidx[alive].cpu())
            rec_step.append(torch.full((int(alive.sum()),), k, dtype=torch.int64))
            rec_X.append(X[alive].cpu())
        if k % compact_every == 0 or k == max_steps:
            n_alive = int(alive.sum())
            pbar.update(k - last)
            last = k
            pbar.set_postfix(alive=n_alive, refresh=False)
            if n_alive < alive.numel():
                if return_final:
                    final_X[gidx] = X                  # absorbed rows are frozen at their absorption state
                outcome[gidx] = loc_out
                steps[gidx] = loc_step
                X, gidx, loc_out, loc_step = X[alive], gidx[alive], loc_out[alive], loc_step[alive]
                alive = alive[alive]
            if n_alive == 0:
                break
    pbar.close()
    outcome[gidx] = loc_out
    steps[gidx] = loc_step

    res = {"outcome": outcome.cpu().numpy(), "steps": steps.cpu().numpy()}
    if return_final:
        final_X[gidx] = X
        res["final_X"] = final_X.cpu().numpy()
    if record_every:
        if rec_idx:
            res["rec_idx"] = torch.cat(rec_idx).numpy()
            res["rec_step"] = torch.cat(rec_step).numpy()
            res["rec_X"] = torch.cat(rec_X).numpy()
        else:
            res["rec_idx"] = np.zeros(0, dtype=np.int64)
            res["rec_step"] = np.zeros(0, dtype=np.int64)
            res["rec_X"] = np.zeros((0, X0.shape[1]))
    return res


def counts_from_outcomes(outcome, n_states, M):
    out = outcome.reshape(n_states, M)
    return (out == 1).sum(1), (out == 0).sum(1), (out == -1).sum(1)
