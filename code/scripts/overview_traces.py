#!/usr/bin/env python
"""Energy traces for the project overview figure (descriptive only, no criterion).

Two rays at the chosen resolution 64x32x32 (dt 0.0625, Re 400): W#1 (fast channel) and R#15 (slow control). For each, the two
cases of the final Gate 5 bracket (lo -> A, hi -> B; relative width ~1e-3) are run without absorption to t = T, E every `every`.

    python code/scripts/overview_traces.py --run-dir experiments/20261118_overview_traces_v1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate7_dns import Ctx, REPO, log, sc_at  # noqa: E402
from committor.dns_local import PARTS, run_diag  # noqa: E402
from committor.utils import atomic_save_npz  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--rays", default="W#1,R#15")
    ap.add_argument("--T", type=float, default=4000.0)
    ap.add_argument("--every", type=float, default=2.0)
    a = ap.parse_args()
    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    out = run / "traces_64.npz"
    if out.exists():
        log("done earlier")
        return
    cfg = yaml.safe_load((REPO / "code/configs/gate7_dns.yaml").read_text())
    ctx = Ctx(cfg)
    g, dt = ctx.grid(64)
    names, which, amps, svals, tref = [], [], [], [], []
    for n in a.rays.split(","):
        p = sc_at(ctx, run, "64", n)
        for tag, s, th in (("lo", p[1], p[3]), ("hi", p[2], p[4])):
            names.append(n)
            which.append(tag)
            amps.append(s * ctx.dir(n))
            svals.append(s)
            tref.append(th)
    log(f"traces at 64 for {names} {which}; s = {svals}; Gate 5 t_hit = {tref}")
    t0 = time.time()
    r = run_diag(g, np.array(amps), dt, a.T, a.every, 2.0 / 3.0, ctx.climit, ctx.lv, log=log)
    atomic_save_npz(out, rec=r["rec"], t=r["t"], dt_used=r["dt_used"], courant_max=r["courant_max"], flag_last=r["flag_last"],
                    names=np.array(names), which=np.array(which), s=np.array(svals), t_hit_gate5=np.array(tref), parts=np.array(PARTS),
                    r_A=ctx.r_A, E_B=ctx.E_B)
    (run / "traces_64.json").write_text(json.dumps({"names": names, "which": which, "s": svals, "t_hit_gate5": tref,
                                                     "dt_used": np.asarray(r["dt_used"]).tolist(),
                                                     "courant_max": np.asarray(r["courant_max"]).tolist(),
                                                     "seconds": time.time() - t0}, indent=1))
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
