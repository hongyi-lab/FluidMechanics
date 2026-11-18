"""MFE time to B at s = s_c (1 + delta) on the W rays (and the 14 fast DNS rays), to compare with the user's 900-1200."""
import json, sys
from pathlib import Path
import numpy as np, torch, yaml
REPO = Path("/srv/hgao0864/project_main")
from committor import edge
from committor.dns_rays import build_rays
from committor.systems.mfe import MFE, rhs_paper, LAMINAR
from committor.utils import atomic_write_json
out = Path(sys.argv[1])
cfg5 = yaml.safe_load((REPO / "experiments/20261117_gate5_dns_v1/config_used.yaml").read_text())
rays = build_rays(cfg5, REPO)
cfg2 = rays["gate2_cfg"]; D2 = cfg2["deterministic"]
m0 = MFE(Re=400.0, Lx_over_pi=4.0, Lz_over_pi=2.0)
R = json.loads((out / "results.json").read_text())
names = list(R["mfe"]["all_rays_s_c_mfe_recomputed"].keys())
sc = np.array([R["mfe"]["all_rays_s_c_mfe_recomputed"][n] for n in names], dtype=float)
D = np.concatenate([rays["F"][0], rays["R"][0], rays["W"][0]])
deltas = [1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1]
rhs = lambda x: rhs_paper(x, m0.Re, m0.Lx, m0.Lz)
res = {"deltas": deltas, "E_B": [0.29, 0.24], "t_hit": {}}
for EB in (0.29, 0.24):
    T = np.zeros((len(D), len(deltas))); F = np.zeros_like(T)
    for j, d in enumerate(deltas):
        X = LAMINAR[None, :] + (sc * (1 + d))[:, None] * D
        r = edge.classify(rhs, torch.as_tensor(X, device="cuda"), 0.25, 8000.0, EB, 5e-7, "rk4")
        T[:, j], F[:, j] = r["t_hit"], r["out"]
    res["t_hit"][f"E_B={EB}"] = {n: T[i].tolist() for i, n in enumerate(names)}
    res[f"fate_E_B={EB}"] = {n: F[i].tolist() for i, n in enumerate(names)}
    W = [i for i, n in enumerate(names) if n.startswith("W#")]
    print(f"E_B {EB}: W t_hit range per delta:", [(d, int(T[W, j].min()), int(T[W, j].max()), int((F[W, j] == 1).sum())) for j, d in enumerate(deltas)])
atomic_write_json(out / "mfe_offsets.json", res)
