#!/usr/bin/env python
"""Gate 4a.5 descriptive diagnostic (not a criterion): why no dt passes. Same 4 directions as 4a.5 (derive_seed(seed, "dt")),
E(0) = 1e-3 and 1e-6, dt = 0.25 .. 0.015625, t <= 200, E every 1.0; figure + npz into the run dir.

    python code/scripts/gate4_dt_diagnostic.py experiments/20261117_gate4_dns_v2
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from committor.dns import Grid, step_nl  # noqa: E402
from committor.plotting import STYLE, _save  # noqa: E402
from committor.utils import atomic_save_npz, atomic_write_json, derive_seed  # noqa: E402

run = Path(sys.argv[1])
cfg = yaml.safe_load((run / "config_used.yaml").read_text())
g = Grid(*cfg["grids"][0])
g.set_enforced(["mirror y about 1 (u,-v,w)(x,2-y,z)", "shift-reflect (u,v,-w)(x+Lx/2,y,-z)", "shift-rotate (-u,-v,w)(-x,-y,z+Lz/2)"])
rng = np.random.default_rng(derive_seed(int(cfg["seed"]), "dt"))
dd = rng.standard_normal((4, 9))
dd /= np.linalg.norm(dd, axis=1, keepdims=True)
dts = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
E = {}
for E0 in (1e-3, 1e-6):
    for dt in dts:
        u = g.clean(g.from_coeffs(math.sqrt(E0) * dd))
        E1, E2 = g.lap_factor(dt), g.lap_factor(0.5 * dt)
        per = int(round(1 / dt))
        rows = [g.energy(u).cpu().numpy()]
        cm = 0.0
        for t in range(1, 201):
            for _ in range(per):
                u = step_nl(g, u, dt, E1, E2)
            rows.append(g.energy(u).cpu().numpy())
            if t % 10 == 0:
                cm = max(cm, float(np.nanmax(g.courant(u, dt).cpu().numpy())))
        E[(E0, dt)] = (np.array(rows), cm)
        print(E0, dt, "done", flush=True)
out = {"directions": dd.tolist(), "rows": []}
for E0 in (1e-3, 1e-6):
    for a, b in zip(dts[:-1], dts[1:]):
        Ea, Eb = E[(E0, a)][0], E[(E0, b)][0]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.abs(Ea / Eb - 1)
        per_dir = [float(np.nanmax(r[:, k])) if np.isfinite(r[:, k]).all() else None for k in range(4)]
        # first time the relative change exceeds 1e-4, per direction
        t_exc = [int(np.argmax(r[:, k] > 1e-4)) if (r[:, k] > 1e-4).any() else None for k in range(4)]
        out["rows"].append({"E0": E0, "dt": a, "dt_half": b, "max_rel_dE_per_direction": per_dir, "first_t_rel_dE_above_1e-4": t_exc,
                            "courant_max": E[(E0, a)][1]})
Eref = E[(1e-3, dts[-1])][0]
out["max_E_per_direction_E0_1e-3"] = Eref.max(axis=0).tolist()
out["first_t_E_above_0.1_E0_1e-3"] = [int(np.argmax(Eref[:, k] > 0.1)) if (Eref[:, k] > 0.1).any() else None for k in range(4)]
atomic_write_json(run / "dt_diagnostic.json", out)
atomic_save_npz(run / "dt_diagnostic.npz", **{f"E_{E0:g}_{dt:g}": v[0] for (E0, dt), v in E.items()})
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    t = np.arange(201)
    cols = ["#4C72B0", "#C44E52", "#55A868", "#DD8452"]
    for k in range(4):
        ax[0].semilogy(t, E[(1e-3, 0.03125)][0][:, k], color=cols[k], lw=1.4, label=f"direction {k + 1}")
        ax[0].semilogy(t, E[(1e-3, 0.015625)][0][:, k], color=cols[k], lw=0.8, ls="--")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("E")
    ax[0].legend(fontsize=8, frameon=False)
    ax[0].set_title("(a) E(0) = 1e-3: dt = 0.03125 (solid), 0.015625 (dashed)", fontsize=9)
    for k in range(4):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.abs(E[(1e-3, 0.03125)][0][:, k] / E[(1e-3, 0.015625)][0][:, k] - 1)
        ax[1].semilogy(t[1:], np.maximum(r[1:], 1e-17), color=cols[k], lw=1.2, label=f"direction {k + 1}")
    ax[1].axhline(1e-4, color="k", ls=":", lw=0.8, label="criterion 1e-4")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("|E_dt / E_dt/2 - 1|")
    ax[1].legend(fontsize=8, frameon=False)
    ax[1].set_title("(b) E(0) = 1e-3: relative change when dt is halved (0.03125 -> 0.015625)", fontsize=9)
    for k in range(4):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.abs(E[(1e-6, 0.25)][0][:, k] / E[(1e-6, 0.125)][0][:, k] - 1)
        ax[2].semilogy(t[1:], np.maximum(r[1:], 1e-17), color=cols[k], lw=1.2, label=f"direction {k + 1}")
    ax[2].axhline(1e-4, color="k", ls=":", lw=0.8)
    ax[2].set_xlabel("t")
    ax[2].set_title("(c) E(0) = 1e-6 (below the edge): dt 0.25 -> 0.125 (descriptive)", fontsize=9)
    fig.tight_layout()
    _save(fig, run / "fig_dt_diagnostic")
print("written")
