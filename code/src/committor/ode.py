"""Deterministic batched RK4 integration with energy-threshold bookkeeping (Gate 1 lifetimes).

One RK4 step plus bookkeeping is a single function, compiled with torch.compile (fused kernels);
`block` consecutive steps are captured once in a CUDA graph and replayed. Bookkeeping after every
RK4 step, per trajectory n and threshold k:
  first_pass[n, k]  first time E < thresholds[k]                    (+inf if never; 0 if E(0) < threshold)
  last_above[n, k]  last time E >= thresholds[k]                    (0 if E < threshold for all t > 0)
  returned[n, k]    E > return_level at some time after first_pass[n, k]
Results are checkpointed every `segment_time` (one file per segment, written atomically); a
restart resumes from the last completed segment.
"""

import math
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .utils import array_hash, atomic_save_npz, atomic_write_json, json_safe, load_npz, read_json


def make_step(rhs, dt, energy, thresholds, return_level):
    """One RK4 step with bookkeeping: (a, t, fp, la, ret) -> (a, t, fp, la, ret, E)."""
    th = thresholds

    def step(a, t, fp, la, ret):
        k1 = rhs(a)
        k2 = rhs(a + (0.5 * dt) * k1)
        k3 = rhs(a + (0.5 * dt) * k2)
        k4 = rhs(a + dt * k3)
        a = a + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t = t + dt
        E = energy(a)
        below = E.unsqueeze(1) < th
        fp = torch.where(below & torch.isinf(fp), t, fp)
        la = torch.where(below, la, t.expand_as(la))
        ret = ret | (torch.isfinite(fp) & (E.unsqueeze(1) > return_level))
        return a, t, fp, la, ret, E

    return step


class Stepper:
    """Advance static state buffers by `block` steps per call (CUDA graph replay, or a plain loop)."""

    def __init__(self, step, state, block, use_graph):
        self.step, self.block, self.use_graph = step, int(block), use_graph
        self.buf = [x.clone() for x in state]                     # a, t, fp, la, ret, E
        if use_graph:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            backup = [x.clone() for x in self.buf]
            with torch.cuda.stream(s):                          # warm-up (compilation, allocator)
                for _ in range(2):
                    self._loop()
            torch.cuda.current_stream().wait_stream(s)
            for b, x in zip(self.buf, backup):
                b.copy_(x)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._loop()
            for b, x in zip(self.buf, backup):                    # capture does not execute; restore anyway
                b.copy_(x)

    def _loop(self):
        out = tuple(self.buf[:5])
        for _ in range(self.block):
            out = self.step(*out[:5])
        for b, x in zip(self.buf, out):
            b.copy_(x)

    def set(self, state):
        for b, x in zip(self.buf, state):
            b.copy_(x)

    def advance(self):
        if self.use_graph:
            self.graph.replay()
        else:
            self._loop()
        return self.buf


@torch.no_grad()
def run_lifetimes(rhs, energy, A0, dt, t_max, thresholds, return_level, segment_time, out_dir,
                  block_time=1.0, n_record=0, hist=None, compile_step=True, use_graph=True,
                  desc="lifetimes", progress=True, status=None, stage=None):
    """Integrate every row of A0 to t_max. State is recorded / histogrammed every block_time.

    Returns numpy arrays: a_final, first_pass, last_above, returned, hist_counts (log10 E occupation,
    with under/overflow bins), hist_edges, rec_t, rec_a (first n_record rows), and elapsed_s."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device, dtype = A0.device, A0.dtype
    n, K, D = A0.shape[0], len(thresholds), A0.shape[1]
    block = int(round(block_time / dt))
    if not math.isclose(block * dt, block_time, rel_tol=1e-12):
        raise ValueError("block_time must be a multiple of dt")
    blocks_per_seg = int(round(segment_time / block_time))
    n_seg = int(round(t_max / segment_time))
    if not (math.isclose(blocks_per_seg * block_time, segment_time) and math.isclose(n_seg * segment_time, t_max)):
        raise ValueError("t_max must be a multiple of segment_time, and segment_time of block_time")
    use_graph = use_graph and device.type == "cuda"
    edges = np.linspace(hist["log10_min"], hist["log10_max"], hist["bins"] + 1) if hist else np.zeros(0)
    nb = len(edges) + 1 if hist else 0

    th = torch.as_tensor(thresholds, device=device, dtype=dtype).unsqueeze(0)
    step = make_step(rhs, dt, energy, th, return_level)
    if compile_step:
        step = torch.compile(step, dynamic=False, fullgraph=True)

    meta = {"dt": float(dt), "t_max": float(t_max), "thresholds": [float(x) for x in thresholds],
            "return_level": float(return_level), "segment_time": float(segment_time), "block_time": float(block_time),
            "n_record": int(n_record), "hist": hist or None, "A0_hash": array_hash(A0.detach().cpu().numpy())}
    meta_path = out_dir / "run_meta.json"
    old_meta = read_json(meta_path)
    if old_meta is None:
        if any(out_dir.glob("segment_*.npz")):
            raise RuntimeError(f"{out_dir} has segments but no run_meta.json; refusing to resume")
        atomic_write_json(meta_path, meta)
    else:
        for k in meta:
            if k == "t_max":
                continue                         # extending a run to a later t_max is allowed
            if old_meta.get(k) != json_safe(meta[k]):
                raise RuntimeError(f"{out_dir} was written with {k}={old_meta.get(k)!r}, now {meta[k]!r}")
        if meta["t_max"] != old_meta["t_max"]:
            atomic_write_json(meta_path, {**old_meta, "t_max": meta["t_max"]})

    start_seg = 0
    for s in range(n_seg, 0, -1):
        if (out_dir / f"segment_{s:03d}.npz").exists():
            start_seg = s
            break
    if start_seg:
        ck = load_npz(out_dir / f"segment_{start_seg:03d}.npz")
        if ck["a"].shape != tuple(A0.shape):
            raise RuntimeError(f"checkpoint in {out_dir} has a different batch shape")
        a = torch.as_tensor(ck["a"], device=device)
        fp = torch.as_tensor(ck["first_pass"], device=device)
        la = torch.as_tensor(ck["last_above"], device=device)
        ret = torch.as_tensor(ck["returned"], device=device)
        hcounts = ck["hist_counts"].astype(np.int64)
        elapsed = float(ck["elapsed_s"])
    else:
        a = A0.clone()
        E0 = energy(a)
        fp = torch.where(E0.unsqueeze(1) < th, torch.zeros((n, K), device=device, dtype=dtype),
                         torch.full((n, K), math.inf, device=device, dtype=dtype))
        la = torch.zeros((n, K), device=device, dtype=dtype)
        ret = torch.zeros((n, K), device=device, dtype=torch.bool)
        hcounts = np.zeros(nb, dtype=np.int64)
        elapsed = 0.0

    t = torch.zeros((), device=device, dtype=dtype)
    stepper = Stepper(step, (a, t, fp, la, ret, energy(a)), block, use_graph)
    pbar = tqdm(total=n_seg * blocks_per_seg, initial=start_seg * blocks_per_seg, desc=desc,
                disable=not progress, mininterval=2.0)
    for s in range(start_seg, n_seg):
        t0w = time.time()
        rec_t, rec_a = [], []
        stepper.set((a, torch.tensor(s * segment_time, device=device, dtype=dtype), fp, la, ret))
        for b in range(blocks_per_seg):
            a, _, fp, la, ret, E = stepper.advance()
            tb = s * segment_time + (b + 1) * block_time
            if n_record:
                rec_t.append(tb)
                rec_a.append(a[:n_record].detach().cpu().numpy().copy())   # copy: on CPU .cpu() aliases the buffer
            if hist:
                lE = torch.log10(E.clamp(min=1e-300)).cpu().numpy()
                hcounts += np.bincount(np.digitize(lE, edges), minlength=nb)
            pbar.update(1)
        a, fp, la, ret = (x.clone() for x in (a, fp, la, ret))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.time() - t0w
        if not torch.isfinite(a).all():
            raise FloatingPointError(f"non-finite state in segment {s + 1}")
        atomic_save_npz(out_dir / f"segment_{s + 1:03d}.npz", a=a.cpu().numpy(), first_pass=fp.cpu().numpy(),
                        last_above=la.cpu().numpy(), returned=ret.cpu().numpy(), hist_counts=hcounts,
                        elapsed_s=np.float64(elapsed), t_end=np.float64((s + 1) * segment_time),
                        rec_t=np.asarray(rec_t, dtype=np.float64),
                        rec_a=(np.stack(rec_a) if rec_a else np.zeros((0, n_record, D))))
        if status is not None and stage is not None:
            status.update(stage, segments_done=s + 1, n_segments=n_seg, seconds=elapsed)
    pbar.close()

    ck = load_npz(out_dir / f"segment_{n_seg:03d}.npz")
    segs = [load_npz(out_dir / f"segment_{s:03d}.npz") for s in range(1, n_seg + 1)]
    return {"a_final": ck["a"], "first_pass": ck["first_pass"], "last_above": ck["last_above"],
            "returned": ck["returned"], "hist_counts": ck["hist_counts"], "hist_edges": edges,
            "elapsed_s": float(ck["elapsed_s"]),
            "rec_t": np.concatenate([g["rec_t"] for g in segs]),
            "rec_a": np.concatenate([g["rec_a"] for g in segs])}
