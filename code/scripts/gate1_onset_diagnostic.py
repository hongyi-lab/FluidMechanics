#!/usr/bin/env python
"""Gate 1 onset diagnostic (description only): stricter laminarisation thresholds and the
alternative reading of the paper's initial-condition shell, compared with NJP figure 10 (Re = 400).

    python code/scripts/gate1_onset_diagnostic.py --config code/configs/gate1_onset_diag.yaml \
        --run-dir experiments/20261115_gate1_onset_diag_v1
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from committor import gate1 as G
from committor.ode import run_lifetimes
from committor.plotting import C_ALT, C_FIT, C_PTS, C_REF, STYLE, _save
from committor.systems import build_system
from committor.systems.mfe import LAMINAR, rhs_paper
from committor.utils import (Status, atomic_write_json, derive_seed, git_info, read_json, stable_hash,
                             tree_hash)

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    device = torch.device(cfg["device"])
    src = REPO / "code" / "src" / "committor"
    ident = {"config_hash": stable_hash(cfg), "sim_code_hash": tree_hash(
        [src / f for f in ("ode.py", "gate1.py", "utils.py", "systems/mfe.py", "systems/base.py")])}
    old = read_json(run_dir / "run_identity.json")
    if old is None:
        atomic_write_json(run_dir / "run_identity.json", ident)
    elif old != ident:
        sys.exit(f"{run_dir} was started with {old}, now {ident}; use a new run dir")
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    manifest = read_json(run_dir / "manifest.json", {"invocations": []})
    manifest["invocations"].append({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv,
                                    "git": git_info(REPO), "torch": torch.__version__,
                                    "gpu": torch.cuda.get_device_name(0)})
    atomic_write_json(run_dir / "manifest.json", manifest)
    status = Status(run_dir / "status.json")

    m = build_system(cfg["system"])
    rhs = lambda a: rhs_paper(a, m.Re, m.Lx, m.Lz)
    seed, N, R = int(cfg["seed"]), int(cfg["n_ic"]), float(cfg["energy_shell"])
    ths = [float(x) for x in cfg["thresholds"]]
    t_max = float(cfg["t_max"])
    A0 = {"A_total_shell": G.sphere_ics(N, R, np.random.default_rng(derive_seed(seed, "lifetime_ics"))),
          "B_perturbation_shell": LAMINAR + G.sphere_ics(N, R, np.random.default_rng(derive_seed(seed, "ics_perturbation_shell")))}
    out = {}
    for name, ic in A0.items():
        r = run_lifetimes(rhs, G.energy_torch, torch.as_tensor(ic, device=device), float(cfg["dt"]), t_max,
                          thresholds=ths, return_level=float(cfg["return_level"]),
                          segment_time=float(cfg["segment_time"]), block_time=1.0,
                          out_dir=run_dir / f"lifetimes_{name}", desc=name, progress=not args.no_progress,
                          status=status, stage=name)
        out[name] = r

    # consistency with Gate 1 (reading A, threshold 1e-3)
    g1 = REPO / "experiments" / "20261115_gate1_mfe_v2" / "lifetimes.npz"
    same_as_gate1 = None
    if g1.exists():
        same_as_gate1 = bool(np.array_equal(np.load(g1)["first_pass"][:, 2], out["A_total_shell"]["first_pass"][:, 0]))

    T_paper = np.array([np.inf if x is None else float(x) for x in
                        read_json(REPO / "literature" / "notes" / "MFE2004_fig10_re400_vector.json")["lifetimes"]])
    tc = cfg["compare_times"]
    res = {"gate1_reproduced_reading_A_1e-3": same_as_gate1, "predicted_shift_per_decade": cfg["predicted_shift_per_decade"],
           "exact_shift_per_decade_linear": math.log(10) / (2 * (math.pi / 2) ** 2 / m.Re), "readings": {}}
    for name, r in out.items():
        fp = r["first_pass"]
        Ts = [np.where(fp[:, k] <= t_max, fp[:, k], np.inf) for k in range(len(ths))]
        rows = {}
        for k, th in enumerate(ths):
            T = Ts[k]
            pw = []
            for t in tc:
                Po, Pp = float((T > t).mean()), float((T_paper > t).mean())
                se = math.sqrt(Po * (1 - Po) / len(T) + Pp * (1 - Pp) / len(T_paper))
                pw.append({"t": t, "P_ours": Po, "P_paper": Pp, "z": (Po - Pp) / se if se > 0 else None})
            lr = G.logrank(T, T_paper, t_max)
            early = np.minimum(T, 3000.0)
            early[T > 3000.0] = np.inf                       # censor both at t = 3000: early-part test
            early_p = np.where(T_paper > 3000.0, np.inf, T_paper)
            lr_early = G.logrank(early, early_p, 3000.0)
            rows[str(th)] = {"first_decay": float(T[np.isfinite(T)].min()), "P_t_max": float((T > t_max).mean()),
                             "fit_t0_1000": G.exp_fit(T, t_max, 1000.0), "fit_t0_2000": G.exp_fit(T, t_max, 2000.0),
                             "pointwise_vs_paper": pw,
                             "logrank_vs_paper_all": {"chi2": lr[0], "p": lr[1]},
                             "logrank_vs_paper_t_le_3000": {"chi2": lr_early[0], "p": lr_early[1]},
                             "n_returned_after_passage": int(r["returned"][:, k].sum())}
        shifts = {}
        for k in range(len(ths) - 1):
            both = np.isfinite(Ts[k]) & np.isfinite(Ts[k + 1])
            d = Ts[k + 1][both] - Ts[k][both]
            shifts[f"{ths[k]}->{ths[k + 1]}"] = {"n": int(both.sum()), "median": float(np.median(d)),
                                                 "mean": float(d.mean()), "q01": float(np.quantile(d, 0.01)),
                                                 "q99": float(np.quantile(d, 0.99)), "min": float(d.min()),
                                                 "max": float(d.max())}
        res["readings"][name] = {"per_threshold": rows, "shift_between_thresholds": shifts}
    res["paper"] = {"first_decay": float(T_paper[np.isfinite(T_paper)].min()), "P_t_max": float((T_paper > t_max).mean()),
                    "fit_t0_1000": G.exp_fit(T_paper, t_max, 1000.0), "fit_t0_2000": G.exp_fit(T_paper, t_max, 2000.0)}
    res["seconds"] = {k: v.get("seconds") for k, v in status.data["stages"].items()}
    atomic_write_json(run_dir / "results.json", res)

    # figure: survival curves (each reading, all thresholds) with the paper's curve; shift per decade
    import matplotlib.pyplot as plt
    grid = np.linspace(0, t_max, 4001)
    cols = [C_PTS, C_FIT, C_ALT, "#55A868"]
    Tp = np.sort(np.where(np.isfinite(T_paper), T_paper, np.inf))
    Pp = 1 - np.searchsorted(Tp, grid, side="right") / len(Tp)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.9))
        for ax, (name, r) in zip(axes[:2], out.items()):
            for k, th in enumerate(ths):
                T = np.sort(np.where(r["first_pass"][:, k] <= t_max, r["first_pass"][:, k], np.inf))
                ax.step(grid, 1 - np.searchsorted(T, grid, side="right") / len(T), where="post", color=cols[k], lw=1.1,
                        label=f"E < 1e{int(round(math.log10(th)))}")
            ax.step(grid, Pp, where="post", color=C_REF, lw=1.4, ls="--", label="NJP fig. 10 (1000 ICs)")
            ax.set_xlim(0, 4000)
            ax.set_ylim(0.65, 1.01)
            ax.set_xlabel("t")
            ax.set_ylabel("P(lifetime > t)")
            ax.set_title({"A_total_shell": r"(a) ICs: $\sum_{i=1}^{9} a_i^2 = 0.3$ (Gate 1)",
                          "B_perturbation_shell": r"(b) ICs: $\sum_i (a_i - a_i^{lam})^2 = 0.3$"}[name], fontsize=10)
            ax.legend(fontsize=8, loc="lower left")
        ax = axes[2]
        rA = out["A_total_shell"]["first_pass"]
        for k in range(len(ths) - 1):
            both = (rA[:, k] <= t_max) & (rA[:, k + 1] <= t_max)
            ax.hist(rA[both, k + 1] - rA[both, k], bins=np.arange(150, 265, 1.0), histtype="step", color=cols[k + 1],
                    label=f"1e{int(round(math.log10(ths[k])))} -> 1e{int(round(math.log10(ths[k + 1])))}")
        ax.axvline(cfg["predicted_shift_per_decade"], color=C_REF, ls=":", lw=1.2, label="predicted 186.6")
        ax.set_xlabel("lifetime shift per decade of threshold")
        ax.set_ylabel("trajectories")
        ax.set_title("(c) shift per decade, reading A", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        (run_dir / "figures").mkdir(exist_ok=True)
        _save(fig, run_dir / "figures" / "fig5_onset_diagnostic")
    print(f"gate1 reproduced: {same_as_gate1}")
    for name in out:
        rr = res["readings"][name]
        print(name, {k: round(v["median"], 2) for k, v in rr["shift_between_thresholds"].items()})
        for th, row in rr["per_threshold"].items():
            z = {p["t"]: round(p["z"], 1) for p in row["pointwise_vs_paper"]}
            print(f"  E<{th}: first {row['first_decay']:.0f}, P(1e4) {row['P_t_max']:.4f}, tau {row['fit_t0_1000']['tau']:.0f}, "
                  f"z {z}, logrank all p {row['logrank_vs_paper_all']['p']:.3g}, t<=3000 p {row['logrank_vs_paper_t_le_3000']['p']:.3g}")
    print("paper:", res["paper"]["first_decay"], res["paper"]["P_t_max"])


if __name__ == "__main__":
    main()
