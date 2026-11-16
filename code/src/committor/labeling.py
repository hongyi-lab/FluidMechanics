"""Monte Carlo committor labels: M independent paths per state, saved in resumable chunks."""

import math
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .sde import counts_from_outcomes, simulate_absorbing
from .utils import (array_hash, atomic_save_npz, atomic_write_json, derive_seed, load_npz,
                    read_json, stable_hash, torch_generator)


def label_states(system, states, M, sim, seed, out_dir, chunk_states, device,
                 tag="label", progress=True):
    """Return n_B, n_A, n_T (unresolved) per state, computing only chunks not already on disk.

    Each chunk file carries a key derived from the system, integrator, M, seed and the exact
    state array; a chunk with a different key is an error, never silently reused or overwritten.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    states = np.ascontiguousarray(states, dtype=np.float64)
    N = len(states)
    key = stable_hash({"system": system.params(), "M": int(M), "sim": sim, "seed": int(seed),
                       "chunk_states": int(chunk_states), "states": array_hash(states), "tag": tag,
                       "device": torch.device(device).type})
    n_chunks = math.ceil(N / chunk_states)
    st_path = out_dir / "chunks_status.json"
    st = read_json(st_path, {"key": key, "n_states": N, "n_chunks": n_chunks, "done": [], "seconds": {}})
    if st["key"] != key:
        raise RuntimeError(f"{st_path} belongs to a different labeling configuration")

    for c in tqdm(range(n_chunks), desc=f"{tag} chunks", disable=not progress):
        path = out_dir / f"chunk_{c:05d}.npz"
        if path.exists():
            if str(load_npz(path)["key"]) != key:
                raise RuntimeError(f"{path} was produced with a different configuration")
            continue
        lo, hi = c * chunk_states, min(N, (c + 1) * chunk_states)
        X0 = torch.from_numpy(states[lo:hi]).to(device).repeat_interleave(M, dim=0)
        gen = torch_generator(derive_seed(seed, tag, c), device)
        t0 = time.perf_counter()
        res = simulate_absorbing(system, X0, sim["dt"], sim["max_steps"], sim["scheme"], gen,
                                 sim["compact_every"], desc=f"{tag} chunk {c}", progress=progress,
                                 bridge=sim["bridge"])
        el = time.perf_counter() - t0
        nB, nA, nT = counts_from_outcomes(res["outcome"], hi - lo, M)
        stp = res["steps"].reshape(hi - lo, M).astype(np.float64)
        absorbed = res["outcome"].reshape(hi - lo, M) >= 0
        n_abs = absorbed.sum(1)
        mean_t = np.where(n_abs > 0, (stp * absorbed).sum(1) / np.maximum(n_abs, 1), np.nan) * sim["dt"]
        atomic_save_npz(path, key=np.array(key), lo=np.array(lo), hi=np.array(hi),
                        n_B=nB.astype(np.int64), n_A=nA.astype(np.int64), n_T=nT.astype(np.int64),
                        mean_hit_time=mean_t, seconds=np.array(el))
        st["done"] = sorted(set(st["done"]) | {c})
        st["seconds"][str(c)] = el
        atomic_write_json(st_path, st)

    nB = np.zeros(N, np.int64)
    nA = np.zeros(N, np.int64)
    nT = np.zeros(N, np.int64)
    mt = np.full(N, np.nan)
    covered = np.zeros(N, bool)
    secs = 0.0
    for c in range(n_chunks):
        d = load_npz(out_dir / f"chunk_{c:05d}.npz")
        lo, hi = int(d["lo"]), int(d["hi"])
        nB[lo:hi], nA[lo:hi], nT[lo:hi], mt[lo:hi] = d["n_B"], d["n_A"], d["n_T"], d["mean_hit_time"]
        covered[lo:hi] = True
        secs += float(d["seconds"])
    if not covered.all():
        raise RuntimeError("label chunks do not cover all states")
    if not np.all(nB + nA + nT == M):
        raise RuntimeError("label counts do not add up to M")
    return {"n_B": nB, "n_A": nA, "n_T": nT, "M": np.full(N, M, np.int64),
            "mean_hit_time": mt, "seconds": secs, "key": key}
