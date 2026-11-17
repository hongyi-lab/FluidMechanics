"""Ray sets of Gate 5 (gate5_dns.yaml f1, f2): F (vortex + seed, angle 0), R (first 24 Gate 2 test rays), W (pure wave), with
the MFE s_c where it exists (F: Gate 2 family_det / Gate 3d item3_det; R: Gate 2 scan; W: computed by the caller)."""
import json
import math
from pathlib import Path

import numpy as np
import yaml

from .utils import derive_seed


def build_rays(cfg5, repo):
    repo = Path(repo)
    D = cfg5["data"]
    cfg2 = yaml.safe_load((repo / D["gate2_config"]).read_text())
    g2, g3d = repo / D["gate2_run"], repo / D["gate3d_run"]
    # F: Gate 2 RNG stream exactly as Gate 3d (d8)
    F2c = cfg2["noise_family"]
    rngf = np.random.default_rng(derive_seed(int(cfg2["seed"]), "family"))
    r_by = {}
    for ang in F2c["angles_deg"]:
        for k in range(int(F2c["draws_per_eps"])):
            r = np.zeros(9)
            r[3:8] = rngf.normal(size=5)
            r[3:8] /= np.linalg.norm(r[3:8])
            r_by[(float(ang), k)] = r

    def fdir(ang, eps, k):
        t = math.radians(ang)
        base = np.zeros(9)
        base[2], base[1] = math.cos(t), math.sin(t)
        return base.copy() if eps == 0 else (base + eps * r_by[(float(ang), k)]) / math.sqrt(1 + eps ** 2)

    Fc = cfg5["rays"]["F"]
    ang = float(Fc["angle_deg"])
    F_meta, F_d = [], []
    for eps in Fc["eps"]:
        for k in range(int(Fc["draws"])):
            F_meta.append({"set": "F", "eps": float(eps), "draw": k, "angle": ang})
            F_d.append(fdir(ang, float(eps), k))
    F_d = np.array(F_d)
    # MFE s_c of F
    R2 = json.loads((g2 / "results.json").read_text())
    FD = np.load(g2 / "family_det.npz")
    fam = R2["family_det"]["rays"]
    I3 = np.load(g3d / "item3_det.npz")
    F_mfe = np.full(len(F_meta), np.nan)
    F_check = []
    for i, m in enumerate(F_meta):
        hit = [j for j, f in enumerate(fam) if f["angle"] == ang and f["eps"] == m["eps"] and f["draw"] == m["draw"]]
        if hit:
            F_mfe[i] = FD["s_c"][hit[0]]
            F_check.append(float(np.max(np.abs(FD["d"][hit[0]] - F_d[i]))))
            continue
        hit = np.where((I3["angle"] == ang) & (I3["eps"] == m["eps"]) & (I3["draw"] == m["draw"]))[0]
        if len(hit):
            F_mfe[i] = I3["s_c"][hit[0]]
            F_check.append(float(np.max(np.abs(I3["d"][hit[0]] - F_d[i]))))
    # R: first 24 split == 2 rays of Gate 2
    RY, SC = np.load(g2 / "rays.npz"), np.load(g2 / "scan.npz")
    n_R = int(cfg5["rays"]["R"]["n"])
    R_idx = np.where(RY["split"] == 2)[0][:n_R]
    R_d = RY["dirs"][R_idx]
    R_mfe = SC["s_c"][R_idx]
    n_cal = int(cfg5["rays"]["R"]["n_cal"])
    R_meta = [{"set": "R", "gate2_ray": int(j), "k": i + 1, "role": "cal" if i < n_cal else "test"} for i, j in enumerate(R_idx)]
    # W: pure wave
    seed5 = int(cfg5["seed"])
    W_d = []
    for i in range(int(cfg5["rays"]["W"]["n"])):
        rng = np.random.default_rng(derive_seed(seed5, "W", i))
        r = np.zeros(9)
        r[3:8] = rng.normal(size=5)
        r[3:8] /= np.linalg.norm(r[3:8])
        W_d.append(r)
    W_d = np.array(W_d)
    W_meta = [{"set": "W", "k": i + 1, "role": "test"} for i in range(len(W_d))]
    return {"F": (F_d, F_meta, F_mfe), "R": (R_d, R_meta, R_mfe), "W": (W_d, W_meta, np.full(len(W_d), np.nan)),
            "F_direction_check_max_abs": max(F_check) if F_check else None, "F_n_matched": len(F_check), "gate2_cfg": cfg2}


def ladder_rays(rays):
    """the 6 resolution / dt check rays (gate4_dns.yaml e12): F(1e-6, d0), F(1e-2, d0), F(1, d0), R #1, R #13, W #1."""
    F_d, F_meta, _ = rays["F"]
    pick = []
    names = []
    for eps in (1e-6, 1e-2, 1.0):
        j = [i for i, m in enumerate(F_meta) if m["eps"] == eps and m["draw"] == 0][0]
        pick.append(F_d[j])
        names.append(f"F eps={eps:g} draw 0")
    R_d = rays["R"][0]
    pick += [R_d[0], R_d[12]]
    names += ["R #1", "R #13"]
    pick.append(rays["W"][0][0])
    names.append("W #1")
    return np.array(pick), names
