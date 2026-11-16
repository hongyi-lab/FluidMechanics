#!/usr/bin/env python
"""Gate 0: positive control on 2D overdamped Langevin with a known committor.

Runs the shared pipeline (ray/trajectory sampling -> MC labels -> ranking) and checks, with the
thresholds fixed in the config before the run:
  P1   MC q agrees with the exact q within binomial error
  P2   skill(x) close to 1
  P2b  skill(x) inside the exact-label null distribution
  P3   (spec wording) |skill(y)| close to 0
  P3b  skill(y) inside the exact-label null distribution
  P3c  adding y to x gains no skill (logistic quad(x, y), MLP)
The exact-label null (labels drawn from Binomial(M, q_exact) on the same states and split) is computed
before any MC label exists and written to predictions.json. Re-running the same command resumes.

    python code/scripts/gate0_langevin2d.py --config code/configs/gate0_langevin2d.yaml \
        --run-dir experiments/20261115_gate0_langevin2d_v2 [--frac 0.01]
"""

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from committor import plotting
from committor.gate0 import analytic_label_check, edge_bin_pattern, exact_label_null
from committor.pipeline import assemble, run_labels, run_sampling, sim_params
from committor.ranking import group_split, rank, ranking_table
from committor.systems import build_system
from committor.utils import (Status, atomic_write_json, derive_seed, git_info, read_json, stable_hash,
                             tree_hash)

REPO = Path(__file__).resolve().parents[2]
SIM_CODE = ["sde.py", "labeling.py", "sampling.py", "pipeline.py", "utils.py", "systems/base.py",
            "systems/langevin2d.py", "systems/__init__.py"]


def check_identity(run_dir, identity, allow_code_change):
    path = run_dir / "run_identity.json"
    old = read_json(path)
    if old is None:
        atomic_write_json(path, identity)
        return
    for k in ("config_hash", "frac", "device"):
        if old[k] != identity[k]:
            sys.exit(f"{run_dir} was started with {k}={old[k]!r}, now {identity[k]!r}; use a new run dir")
    if old["sim_code_hash"] != identity["sim_code_hash"] and not allow_code_change:
        sys.exit(f"simulation code changed since {run_dir} was started "
                 f"({old['sim_code_hash']} -> {identity['sim_code_hash']}); use a new run dir "
                 "or pass --allow-code-change if the change cannot affect the cached results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of full scale (cost estimate)")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--allow-code-change", action="store_true")
    args = ap.parse_args()
    progress = not args.no_progress

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        device = cfg["device"]
    elif args.allow_cpu:
        device = "cpu"
    else:
        sys.exit("CUDA not available; pass --allow-cpu to run on CPU (different RNG stream)")
    src = REPO / "code" / "src" / "committor"
    identity = {"config_hash": stable_hash(cfg), "frac": args.frac, "device": torch.device(device).type,
                "sim_code_hash": tree_hash([src / f for f in SIM_CODE])}
    check_identity(run_dir, identity, args.allow_code_change)
    frozen = run_dir / "config_used.yaml"
    if not frozen.exists():
        frozen.write_text(yaml.safe_dump(cfg, sort_keys=False))

    code_files = sorted(src.rglob("*.py")) + [Path(__file__).resolve()]
    manifest = read_json(run_dir / "manifest.json", {"invocations": []})
    manifest["invocations"].append({
        "start": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv, "code_hash": tree_hash(code_files),
        "git": git_info(REPO), "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "device": device,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None})
    atomic_write_json(run_dir / "manifest.json", manifest)

    system = build_system(cfg["system"])
    status = Status(run_dir / "status.json")
    seed = int(cfg["seed"])
    rcfg, dcfg, crit = cfg["ranking"], cfg["diagnostics"], cfg["pass_criteria"]
    M = int(cfg["labeling"]["M"])
    t_all = time.perf_counter()

    # ---- sampling ----
    st = run_sampling(system, cfg, run_dir, device, frac=args.frac, progress=progress, status=status)
    split_states = group_split(st["group"], rcfg["split"], derive_seed(seed, "split"))
    qa_states = system.analytic_q(st["X"][:, 0])

    # ---- pre-registered prediction: skills under exact-q labels (no MC labels used) ----
    pred_path = run_dir / "predictions.json"
    if not pred_path.exists():
        t0 = time.perf_counter()
        null = exact_label_null(system, st["X"], split_states, qa_states, np.full(len(qa_states), M),
                                dcfg["exact_label_null_features"], dcfg["exact_label_null_replicates"],
                                derive_seed(seed, "exact_null"), rcfg["iso_pseudo"], progress)
        atomic_write_json(pred_path, {"written": time.strftime("%Y-%m-%d %H:%M:%S"),
                                      "note": "computed from states and exact q only, before MC labeling",
                                      "n_states": int(len(qa_states)), "null": null})
        status.update("exact_label_null", done=True, seconds=time.perf_counter() - t0)
        print("predictions (exact-label null, mean [0.5%, 99.5%]): " + "; ".join(
            f"{k} {v['mean']:.4f} [{v['q005']:.4f}, {v['q995']:.4f}]" for k, v in null.items()))
    null = read_json(pred_path)["null"]

    # ---- MC labels (bridge setting from config) ----
    lab = run_labels(system, cfg, run_dir, device, st, progress=progress, status=status, tag="label")
    data = assemble(st, lab)
    if data["n_excluded_all_timeout"]:
        print(f"WARNING: {data['n_excluded_all_timeout']} states excluded (all paths timed out); "
              "split/null computed on the full state set may differ")
    N = len(data["n_B"])
    qhat = data["n_B"] / data["M_eff"]
    qa = system.analytic_q(data["X"][:, 0])
    in_layer = (qa >= 0.05) & (qa <= 0.95)
    print(f"states: {N}  (ray {int((data['origin'] == 0).sum())}, trajectory {int((data['origin'] == 1).sum())}); "
          f"exact q in [0.05, 0.95]: {int(in_layer.sum())}; timeout fraction {data['timeout_fraction']:.2e}")

    # ---- P1 ----
    t0 = time.perf_counter()
    check, pit = analytic_label_check(qa, data["n_B"], data["M_eff"], crit["labels"],
                                      derive_seed(seed, "gate0_check"), progress)
    status.update("label_check", done=True, seconds=time.perf_counter() - t0)
    print(f"P1 labels: deviance/state {check['deviance_per_state']:.4f} (null mean/state "
          f"{check['deviance_pb_mean'] / N:.4f}, p={check['deviance_pb_p']:.3f}); pooled z {check['pooled_bias_z']:+.2f}; "
          f"max |bin z| {check['max_abs_qbin_z']:.2f}; coverage95 {check['coverage95']:.4f} -> "
          f"{'PASS' if check['pass'] else ('FAIL' if check['evaluable'] else 'not evaluable')}")

    # ---- diagnostic: same states, bridge off ----
    diag = {"edge_bins_bridge": edge_bin_pattern(qa, data["n_B"], data["M_eff"])}
    if dcfg.get("relabel_without_bridge") and cfg["sim"].get("boundary_bridge"):
        lab0 = run_labels(system, cfg, run_dir, device, st, progress=progress, status=status,
                          tag="label_nobridge", sim=sim_params(cfg["sim"], bridge=False))
        d0 = assemble(st, lab0)
        diag["edge_bins_no_bridge"] = edge_bin_pattern(system.analytic_q(d0["X"][:, 0]), d0["n_B"], d0["M_eff"])
        print("bin z (bridge):    " + " ".join(f"{z:+.2f}" if z is not None else "  na " for z in diag["edge_bins_bridge"]["z"]))
        print("bin z (no bridge): " + " ".join(f"{z:+.2f}" if z is not None else "  na " for z in diag["edge_bins_no_bridge"]["z"]))

    results = {"gate": 0, "name": cfg["name"], "frac": args.frac, "n_states": N,
               "n_states_q_in_0.05_0.95_exact": int(in_layer.sum()),
               "n_states_q_in_0.05_0.95_mc": int(((qhat >= 0.05) & (qhat <= 0.95)).sum()),
               "n_ray_states": int((data["origin"] == 0).sum()),
               "n_traj_states": int((data["origin"] == 1).sum()),
               "timeout_fraction": data["timeout_fraction"],
               "n_excluded_all_timeout": data["n_excluded_all_timeout"],
               "label_check": check, "diagnostics": diag, "predictions_file": str(pred_path)}

    # ---- ranking (same code as Gate 3) ----
    groups_ok = len(np.unique(data["group"])) >= 6
    if groups_ok:
        t0 = time.perf_counter()
        res, preds, split = rank(system, data, rcfg, seed, device,
                                 references=[("exact q", lambda X: system.analytic_q(X[:, 0]))],
                                 progress=progress)
        status.update("ranking", done=True, seconds=time.perf_counter() - t0)
        if data["n_excluded_all_timeout"] == 0:
            assert np.array_equal(split, split_states)
        results["ranking"] = res
        table = ranking_table(res)
        (run_dir / "ranking_table.md").write_text(table + "\n")
        print(table)

        by = {r["name"]: r for r in res["rows"]}
        rc = crit["ranking"]
        lvl = rc["null_quantile_level"]
        qlo, qhi = ("q005", "q995") if abs(lvl - 0.99) < 1e-9 else ("q025", "q975")

        def in_null(name, feat):
            return bool(null[feat][qlo] <= by[name]["skill"] <= null[feat][qhi])

        gain = max(by["MLP (all coords)"]["skill"], by["logistic quad(x, y)"]["skill"]) - by["x"]["skill"]
        checks = {
            "P1_labels_match_exact_q": check["pass"],
            "P2_skill_x_close_to_1": bool(by["x"]["skill"] >= rc["skill_x_min"]),
            "P2b_skill_x_in_exact_label_null": in_null("x", "x"),
            "P3_spec_literal_abs_skill_y_le_0.03": bool(abs(by["y"]["skill"]) <= rc["skill_y_abs_max"]),
            "P3b_skill_y_in_exact_label_null": in_null("y", "y"),
            "P3c_no_gain_from_adding_y": bool(gain <= rc["conditional_gain_max"]),
        }
        verdict = {
            "checks": checks,
            "conditional_gain": gain,
            "GATE0_PASS_spec_literal": bool(checks["P1_labels_match_exact_q"] and checks["P2_skill_x_close_to_1"]
                                            and checks["P3_spec_literal_abs_skill_y_le_0.03"]),
            "GATE0_PASS_amended": bool(checks["P1_labels_match_exact_q"] and checks["P2_skill_x_close_to_1"]
                                       and checks["P2b_skill_x_in_exact_label_null"]
                                       and checks["P3b_skill_y_in_exact_label_null"]
                                       and checks["P3c_no_gain_from_adding_y"]),
        }
        results["verdict"] = verdict
        print("verdict:", verdict)

        # ---- figures ----
        t0 = time.perf_counter()
        fig_dir = run_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        te = split == 2
        cands = [r for r in res["rows"] if r["kind"] not in ("reference", "control")]
        best = max(cands, key=lambda r: r["skill"])
        best_single = max([r for r in cands if r["kind"] == "feature"], key=lambda r: r["skill"])
        xg = np.linspace(-1.6, 1.6, 801)
        qg = system.analytic_q(xg)
        panels = []
        for r in [best] + ([best_single] if best_single["name"] != best["name"] else []):
            pr = preds[r["name"]]
            panels.append({"c": pr["c"][te], "qhat": qhat[te], "p": pr["p"][te],
                           "title": f"{r['name']}: test skill {r['skill']:.3f}",
                           "xlabel": r["name"] if r["kind"] == "feature" else f"{r['name']} (logit)",
                           "analytic": (xg, qg) if r["name"] == "x" else None})
        plotting.q_vs_coordinate(panels, fig_dir / "fig1_q_vs_best_coordinate")
        hsel = []
        for nm in [best["name"], "x", "E", "y", "|y|"]:
            if nm not in hsel:
                hsel.append(nm)
        plotting.histogram_test([{"title": nm, "counts": np.array(by[nm]["hist_counts"]),
                                  "expected": np.array(by[nm]["hist_expected_if_exact"]),
                                  "n": by[nm]["hist_n"],
                                  "std": by[nm]["hist_std_q"] if by[nm]["hist_std_q"] is not None else float("nan"),
                                  "std_exact": (by[nm]["hist_std_if_exact"]
                                                if by[nm]["hist_std_if_exact"] is not None else float("nan"))}
                                 for nm in hsel], res["hist_bins"], fig_dir / "fig2_committor_histogram_test",
                                window=rcfg["hist_window"])
        E = system.features()["E"](data["X"])
        plotting.q_vs_E(E, qhat, data["origin"], fig_dir / "fig3_q_vs_E",
                        xlabel=r"E = ((x+1)$^2$ + y$^2$)/2  (distance$^2$/2 from A-side reference)")
        plotting.gate0_analytic(data["X"][:, 0], qa, qhat, data["M_eff"], pit,
                                fig_dir / "fig0_mc_vs_exact_q", xg, qg)
        status.update("figures", done=True, seconds=time.perf_counter() - t0)
        results["figures"] = sorted(str(p.name) for p in fig_dir.glob("*.png"))
    else:
        print("too few groups for a ranking at this scale; ranking skipped")

    results["wall_seconds_this_invocation"] = time.perf_counter() - t_all
    results["stages"] = status.data["stages"]
    results["identity"] = identity
    atomic_write_json(run_dir / "results.json", results)
    print(f"done in {results['wall_seconds_this_invocation']:.1f} s -> {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
