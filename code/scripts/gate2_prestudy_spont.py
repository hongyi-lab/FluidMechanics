#!/usr/bin/env python
"""Spontaneous laminar -> B fraction within T_max at the two proposed sigma (A empty, B absorbing).

    python code/scripts/gate2_prestudy_spont.py code/configs/gate2_prestudy_spont.yaml experiments/20261115_gate2_prestudy_spont_v1
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import beta as beta_dist

from committor.sde import simulate_absorbing
from committor.systems.mfe import LAMINAR, MFE
from committor.utils import atomic_write_json, derive_seed, git_info, read_json, stable_hash, torch_generator

REPO = Path(__file__).resolve().parents[2]
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
run = Path(sys.argv[2])
run.mkdir(parents=True, exist_ok=True)
(run / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
atomic_write_json(run / "manifest.json", {"start": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv,
                                          "config_hash": stable_hash(cfg), "git": git_info(REPO)})
P = read_json(REPO / cfg["prestudy_run"] / "results.json")
E_B, dt, T_max = P["histograms"]["E_B"], P["heun_dt"]["chosen"], 6000.0
tr = P["lyapunov"]["trace_S1"]
out = read_json(run / "results.json", {}) or {}
for name in ("sigma_lo", "sigma_hi"):
    if name in out:
        continue
    s = P["sigma"][name]
    t0 = time.perf_counter()
    m = MFE(sigma=s, r_A=None, E_B=E_B)
    X0 = torch.as_tensor(LAMINAR, device=cfg["device"]).repeat(int(cfg["n_paths"]), 1)
    r = simulate_absorbing(m, X0, dt, int(round(T_max / dt)), "heun",
                           torch_generator(derive_seed(cfg["seed"], name), cfg["device"]), compact_every=200,
                           record_every=int(round(cfg["record_every_time"] / dt)), progress=False, bridge=True,
                           compile_step=True)
    k, n = int((r["outcome"] == 1).sum()), len(r["outcome"])
    keep = r["rec_step"] * dt >= cfg["burn_in"]
    X = r["rec_X"][keep]
    E = (X[:, 0] - 1) ** 2 + (X[:, 1:] ** 2).sum(1)
    per_path = np.bincount(r["rec_idx"][keep], weights=E, minlength=n) / np.maximum(np.bincount(r["rec_idx"][keep], minlength=n), 1)
    out[name] = {"sigma": s, "n": n, "n_reach_B": k, "fraction": k / n,
                 "cp95_upper": 1.0 if k == n else float(beta_dist.ppf(0.975, k + 1, n - k)),
                 "max_E_recorded": float(E.max()), "mean_E": float(E.mean()), "mean_E_se_over_paths": float(per_path.std() / np.sqrt(n)),
                 "E_ss_lyapunov": s ** 2 * tr, "seconds": time.perf_counter() - t0}
    atomic_write_json(run / "results.json", out)
    print(name, out[name], flush=True)
