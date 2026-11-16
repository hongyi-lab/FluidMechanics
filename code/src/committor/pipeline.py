"""Shared stages: ray sampling -> trajectory sampling -> Monte Carlo labels -> dataset."""

import copy
import time
from pathlib import Path

import numpy as np

from .labeling import label_states
from .sampling import build_states, sample_rays
from .utils import Status, atomic_save_npz, load_npz


def sim_params(cfg_sim, **override):
    sim = {"dt": float(cfg_sim["dt"]),
           "max_steps": int(round(float(cfg_sim["t_max"]) / float(cfg_sim["dt"]))),
           "scheme": str(cfg_sim["scheme"]),
           "compact_every": int(cfg_sim.get("compact_every", 100)),
           "bridge": bool(cfg_sim.get("boundary_bridge", False))}
    sim.update(override)
    return sim


def scaled_sampling(scfg, frac):
    """Sampling config at a fraction of full scale (for the 1% cost estimate)."""
    s = copy.deepcopy(scfg)
    if frac < 1.0:
        s["n_rays"] = max(8, int(round(frac * scfg["n_rays"])))
    return s


def run_sampling(system, cfg, run_dir, device, frac=1.0, progress=True, status=None):
    run_dir = Path(run_dir)
    status = status if status is not None else Status(run_dir / "status.json")
    sim = sim_params(cfg["sim"])
    seed = int(cfg["seed"])
    scfg = scaled_sampling(cfg["sampling"], frac)

    if not status.done("rays"):
        rays = sample_rays(system, scfg, sim, seed, run_dir, device, progress)
        status.update("rays", done=True, seconds=float(rays["seconds"]),
                      n_rays=int(len(rays["valid"])), n_valid=int(rays["valid"].sum()))
    rays = load_npz(run_dir / "rays.npz")

    if not status.done("states"):
        t0 = time.perf_counter()
        st = build_states(system, rays, scfg, sim, seed, run_dir, device, progress)
        status.update("states", done=True, seconds=time.perf_counter() - t0, n_states=int(len(st["X"])),
                      n_ray_states=int((st["origin"] == 0).sum()), n_traj_states=int((st["origin"] == 1).sum()),
                      n_dropped_in_AB=int(st["n_dropped_in_AB"]))
    return load_npz(run_dir / "states.npz")


def run_labels(system, cfg, run_dir, device, st, progress=True, status=None, tag="label", sim=None):
    """MC labels for the states in st; tag names the stage, its chunk dir and its labels file."""
    run_dir = Path(run_dir)
    status = status if status is not None else Status(run_dir / "status.json")
    sim = sim if sim is not None else sim_params(cfg["sim"])
    lcfg = cfg["labeling"]
    out = run_dir / f"{tag}s.npz"
    if not status.done(tag):
        t0 = time.perf_counter()
        lab = label_states(system, st["X"], int(lcfg["M"]), sim, int(cfg["seed"]), run_dir / f"{tag}_chunks",
                           int(lcfg["chunk_states"]), device, tag=tag, progress=progress)
        atomic_save_npz(out, **{k: v for k, v in lab.items() if k not in ("seconds", "key")})
        status.update(tag, done=True, seconds=time.perf_counter() - t0, gpu_seconds=lab["seconds"],
                      key=lab["key"], n_paths=int(len(st["X"]) * int(lcfg["M"])), sim=sim)
    return load_npz(out)


def run_sampling_and_labels(system, cfg, run_dir, device, frac=1.0, progress=True, status=None):
    status = status if status is not None else Status(Path(run_dir) / "status.json")
    st = run_sampling(system, cfg, run_dir, device, frac, progress, status)
    lab = run_labels(system, cfg, run_dir, device, st, progress, status)
    return assemble(st, lab)


def assemble(st, lab):
    M_eff = lab["n_A"] + lab["n_B"]
    keep = M_eff > 0
    data = {"X": st["X"][keep], "group": st["group"][keep], "origin": st["origin"][keep],
            "s": st["s"][keep], "n_B": lab["n_B"][keep], "n_A": lab["n_A"][keep], "n_T": lab["n_T"][keep],
            "M": lab["M"][keep], "M_eff": M_eff[keep], "mean_hit_time": lab["mean_hit_time"][keep]}
    data["n_excluded_all_timeout"] = int((~keep).sum())
    data["timeout_fraction"] = float(lab["n_T"].sum() / lab["M"].sum())
    return data
