#!/usr/bin/env python
"""Gate 1: MFE nine-mode model (Moehlis, Faisst & Eckhardt 2004), implementation checks.

  (a) |f(laminar)| < 1e-12
  (b) |sum_j a_j N_j(a)| < 1e-10 for 1000 random a
  (c) all laminar Jacobian eigenvalues have negative real part, Re = 100..2000
  (d) turbulent-lifetime distribution of 1e4 random initial conditions is roughly exponential
      (straight on semi-log axes); compared with NJP figure 10 (same domain, Re = 400)
  (e) f(g a) = g f(a) for the symmetries listed in SIADS 2005 (3.24)-(3.25)
plus a transcription cross-check (Galerkin projection of the paper's modes; diagnostic only).
All thresholds are in the config, fixed before the first lifetime run. Re-running resumes.

    python code/scripts/gate1_mfe.py --config code/configs/gate1_mfe.yaml \
        --run-dir experiments/20261115_gate1_mfe_v1
"""

import argparse
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from committor import gate1 as G
from committor import plotting
from committor.ode import run_lifetimes
from committor.systems import build_system
from committor.systems.mfe import rhs_paper, rk4_step
from committor.utils import (Status, atomic_save_npz, atomic_write_json, derive_seed, git_info, load_npz,
                             read_json, stable_hash, tree_hash)

REPO = Path(__file__).resolve().parents[2]
SIM_CODE = ["ode.py", "gate1.py", "utils.py", "systems/base.py", "systems/mfe.py", "systems/__init__.py"]


def check_identity(run_dir, identity, allow_code_change):
    path = run_dir / "run_identity.json"
    old = read_json(path)
    if old is None:
        atomic_write_json(path, identity)
        return
    for k in ("config_hash", "device"):
        if old[k] != identity[k]:
            sys.exit(f"{run_dir} was started with {k}={old[k]!r}, now {identity[k]!r}; use a new run dir")
    if old["sim_code_hash"] != identity["sim_code_hash"] and not allow_code_change:
        sys.exit(f"simulation code changed since {run_dir} was started; use a new run dir "
                 "or pass --allow-code-change if the change cannot affect the cached results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--allow-code-change", action="store_true")
    ap.add_argument("--stop-after", choices=["checks", "dt", "timing", "lifetimes"], default=None)
    args = ap.parse_args()
    progress = not args.no_progress

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device(cfg["device"])
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        sys.exit("CUDA not available; pass --allow-cpu to run on CPU")
    src = REPO / "code" / "src" / "committor"
    identity = {"config_hash": stable_hash(cfg), "device": device.type,
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
        "numpy": np.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None})
    atomic_write_json(run_dir / "manifest.json", manifest)

    status = Status(run_dir / "status.json")
    seed = int(cfg["seed"])
    m = build_system(cfg["system"])
    C = cfg["checks"]
    t_all = time.perf_counter()
    results = read_json(run_dir / "results.json", {}) or {}
    results.update({"gate": 1, "name": cfg["name"], "system": m.params()})

    # ---------------- (a), (b), (c), (e), transcription cross-check ----------------
    chk_path = run_dir / "checks.json"
    if not chk_path.exists():
        t0 = time.perf_counter()
        ca = G.check_a(m)
        cb = G.check_b(m, C["b_energy"]["n_random"], C["b_energy"]["scale"],
                       np.random.default_rng(derive_seed(seed, "b")))
        cc = C["c_jacobian"]
        Re_grid = np.arange(cc["Re_min"], cc["Re_max"] + 0.5 * cc["Re_step"], cc["Re_step"], dtype=float)
        ccr = G.check_c(Re_grid, m.Lx_over_pi, m.Lz_over_pi)
        np.savez(run_dir / "jacobian_scan.npz", Re=Re_grid, max_real=ccr.pop("_max_re"))
        ce = G.check_e(m, C["e_symmetry"]["n_random"], C["e_symmetry"]["scale"],
                       np.random.default_rng(derive_seed(seed, "e")))
        ct = G.transcription_crosscheck(m)
        checks = {"a": ca, "b": cb, "c": ccr, "e": ce, "transcription": ct}
        atomic_write_json(chk_path, checks)
        status.update("checks", done=True, seconds=time.perf_counter() - t0)
    checks = read_json(chk_path)
    pa = checks["a"]["rhs_norm_paper_form"] < C["a_laminar"]["rhs_norm_max"] and \
        checks["a"]["rhs_norm_tensor_form"] < C["a_laminar"]["rhs_norm_max"]
    pb = max(checks["b"]["max_abs_work_paper_form"], checks["b"]["max_abs_work_tensor_form"]) < C["b_energy"]["abs_max"]
    pc = checks["c"]["max_real_part_over_grid"] < 0.0
    sym_res = max(max(v["max_abs_residual_paper_form"], v["max_abs_residual_tensor_form"])
                  for v in checks["e"]["symmetries"].values())
    pe = sym_res < C["e_symmetry"]["abs_max"] and \
        checks["e"]["negative_control_flip_a4"] > C["e_symmetry"]["negative_control_min"]
    ct = checks["transcription"]
    p_tr = max(ct["max_abs_diff_c"], ct["max_abs_diff_L"], ct["max_abs_diff_Q"]) < C["transcription_crosscheck"]["galerkin_tol"]
    print(f"(a) |f(lam)| = {checks['a']['rhs_norm_paper_form']:.2e} -> {pa}")
    print(f"(b) max |a.N(a)| = {checks['b']['max_abs_work_paper_form']:.2e} (rel {checks['b']['max_relative_work']:.1e}) -> {pb}")
    print(f"(c) max Re(lambda), Re 100-2000 = {checks['c']['max_real_part_over_grid']:.3e} "
          f"at Re {checks['c']['Re_at_max']:.0f} -> {pc}")
    print(f"(e) max symmetry residual = {sym_res:.2e}, flip-a4 control {checks['e']['negative_control_flip_a4']:.2e} -> {pe}")
    print(f"transcription: Galerkin diffs c {ct['max_abs_diff_c']:.1e} L {ct['max_abs_diff_L']:.1e} "
          f"Q {ct['max_abs_diff_Q']:.1e} -> {p_tr}")
    results["checks"] = checks
    results["pass"] = {"a": pa, "b": pb, "c": pc, "e": pe, "transcription_crosscheck": p_tr}
    atomic_write_json(run_dir / "results.json", results)
    if args.stop_after == "checks":
        return

    # ---------------- RK4 step selection ----------------
    rhs = lambda a: rhs_paper(a, m.Re, m.Lx, m.Lz)
    L = cfg["lifetimes"]
    dts = cfg["dt_selection"]
    dt_path = run_dir / "dt_selection.json"
    if not dt_path.exists():
        t0 = time.perf_counter()
        rng = np.random.default_rng(derive_seed(seed, "dt_selection"))
        X = torch.as_tensor(G.sphere_ics(dts["n_states"], L["ic_energy_shell"], rng), device=device)
        step0 = torch.compile(lambda a: rk4_step(rhs, a, 0.01), dynamic=False, fullgraph=True)
        from tqdm.auto import tqdm
        for _ in tqdm(range(int(round(dts["spinup_time"] / 0.01))), desc="dt spin-up", disable=not progress,
                      mininterval=2.0):
            X = step0(X)
        E_spun = G.energy_torch(X).cpu().numpy()

        def integrate(dt):
            Y = X.clone()
            f = torch.compile(lambda a: rk4_step(rhs, a, dt), dynamic=False, fullgraph=True)
            for _ in range(int(round(dts["horizon"] / dt))):
                Y = f(Y)
            return Y.cpu().numpy()
        ref = integrate(dts["reference_dt"])
        errs = {str(dt): float(np.abs(integrate(dt) - ref).max()) for dt in dts["candidates"]}
        ok = [dt for dt in dts["candidates"] if errs[str(dt)] <= dts["tolerance"]]
        if not ok:
            sys.exit(f"no candidate dt meets the tolerance: {errs}")
        chosen = max(ok)
        order = {f"{a}->{b}": math.log(errs[str(a)] / errs[str(b)]) / math.log(a / b)
                 for a, b in zip(dts["candidates"][:-1], dts["candidates"][1:]) if errs[str(b)] > 1e-14}
        atomic_write_json(dt_path, {"errors_at_horizon": errs, "observed_order": order, "chosen_dt": chosen,
                                    "E_after_spinup": {"min": float(E_spun.min()), "median": float(np.median(E_spun)),
                                                       "max": float(E_spun.max())}})
        status.update("dt_selection", done=True, seconds=time.perf_counter() - t0)
    dsel = read_json(dt_path)
    dt = float(dsel["chosen_dt"])
    print(f"RK4 dt = {dt}  (errors at t={dts['horizon']}: {dsel['errors_at_horizon']})")
    results["dt_selection"] = dsel
    atomic_write_json(run_dir / "results.json", results)
    if args.stop_after == "dt":
        return

    # ---------------- lifetimes ----------------
    rng = np.random.default_rng(derive_seed(seed, "lifetime_ics"))
    A0 = G.sphere_ics(int(L["n_ic"]), L["ic_energy_shell"], rng)
    common = dict(thresholds=L["thresholds"], return_level=L["return_level"], segment_time=L["segment_time"],
                  block_time=L["record_every"], hist=L["energy_hist"], progress=progress, status=status)

    # 1% timing run (same code path, first 1% of the ICs)
    n_small = max(1, int(round(L["timing_frac"] * L["n_ic"])))
    tim = run_lifetimes(rhs, G.energy_torch, torch.as_tensor(A0[:n_small], device=device), dt, L["t_max"],
                        out_dir=run_dir / "lifetimes_frac0.01", n_record=0, desc="timing 1%",
                        stage="lifetimes_frac0.01", **common)
    print(f"1% run ({n_small} ICs): {tim['elapsed_s']:.1f} s")
    if "timing" not in results:
        # per-step cost vs batch size (cost is not linear in the number of trajectories on the GPU)
        bench = {}
        for nb in (n_small, int(L["n_ic"])):
            r = run_lifetimes(rhs, G.energy_torch, torch.as_tensor(A0[:nb], device=device), dt, 10 * L["record_every"],
                              thresholds=L["thresholds"], return_level=L["return_level"],
                              segment_time=10 * L["record_every"], block_time=L["record_every"],
                              out_dir=run_dir / "timing_bench" / str(nb), hist=L["energy_hist"], progress=False)
            bench[str(nb)] = r["elapsed_s"] / (10 * L["record_every"] / dt)
        est = tim["elapsed_s"] * bench[str(int(L["n_ic"]))] / bench[str(n_small)]
        results["timing"] = {"frac0.01_seconds": tim["elapsed_s"], "seconds_per_step": bench,
                             "full_estimate_seconds": est,
                             "dt_half_check_estimate_seconds_upper": 2.0 * est,
                             "note": "elapsed times exclude torch.compile + CUDA-graph capture (~10-20 s per run)"}
        atomic_write_json(run_dir / "results.json", results)
    print(f"timing: {results['timing']}")
    if args.stop_after == "timing":
        return

    full = run_lifetimes(rhs, G.energy_torch, torch.as_tensor(A0, device=device), dt, L["t_max"],
                         out_dir=run_dir / "lifetimes", n_record=L["n_record"], desc="lifetimes",
                         stage="lifetimes", **common)
    n_half = int(L["dt_half_check"]["n_ic"])
    half = run_lifetimes(rhs, G.energy_torch, torch.as_tensor(A0[:n_half], device=device), dt / 2, L["t_max"],
                         out_dir=run_dir / "lifetimes_dt_half", n_record=0, desc="dt/2 check",
                         stage="lifetimes_dt_half", **common)
    atomic_save_npz(run_dir / "lifetimes.npz", A0=A0, first_pass=full["first_pass"], last_above=full["last_above"],
                    returned=full["returned"], a_final=full["a_final"], hist_counts=full["hist_counts"],
                    hist_edges=full["hist_edges"], rec_t=full["rec_t"], rec_a=full["rec_a"],
                    half_first_pass=half["first_pass"], thresholds=np.asarray(L["thresholds"]))
    if args.stop_after == "lifetimes":
        return

    # ---------------- (d) analysis ----------------
    t_max = float(L["t_max"])
    ths = list(L["thresholds"])
    kp = ths.index(L["primary_threshold"])
    T = full["first_pass"][:, kp]
    T = np.where(T <= t_max, T, np.inf)
    fit = G.exp_fit(T, t_max, L["fit"]["t0"])
    fit_alt = G.exp_fit(T, t_max, L["fit"]["t0_alt"])
    gof = G.gof_bootstrap(T, t_max, fit, int(L["fit"]["gof_bootstrap"]),
                          np.random.default_rng(derive_seed(seed, "gof")))
    P_tmax = float((T > t_max).mean())
    finite = np.sort(T[np.isfinite(T)])
    median_emp = float(finite[len(T) // 2]) if (np.isfinite(T).sum() > len(T) / 2) else None
    per_threshold = {}
    for k, th in enumerate(ths):
        Tk = np.where(full["first_pass"][:, k] <= t_max, full["first_pass"][:, k], np.inf)
        # last exit: last time E >= threshold; censored if still above at t_max
        still_above = G.energy_torch(torch.as_tensor(full["a_final"])).numpy() >= th
        Tl = np.where(still_above, np.inf, full["last_above"][:, k])
        passed = np.isfinite(Tk)
        per_threshold[str(th)] = {
            "fit_first_passage": G.exp_fit(Tk, t_max, L["fit"]["t0"]),
            "fit_last_exit": G.exp_fit(Tl, t_max, L["fit"]["t0"]),
            "P_t_max_first_passage": float((Tk > t_max).mean()),
            "n_passed": int(passed.sum()),
            "n_returned_after_passage": int((full["returned"][:, k] & passed).sum()),
            "fraction_returned": float((full["returned"][:, k] & passed).sum() / max(passed.sum(), 1)),
            "max_abs_first_minus_last_exit": float(np.nanmax(np.abs(np.where(passed & np.isfinite(Tl), Tk - Tl, np.nan))))
            if (passed & np.isfinite(Tl)).any() else None,
        }
    Th = np.where(half["first_pass"][:, kp] <= t_max, half["first_pass"][:, kp], np.inf)
    chi2, p_lr = G.logrank(T[:n_half], Th, t_max)
    fit_h = G.exp_fit(Th, t_max, L["fit"]["t0"])
    fit_same = G.exp_fit(T[:n_half], t_max, L["fit"]["t0"])
    paper = L["paper"]
    ratio = fit["tau"] / paper["tau"]
    D = gof["D_obs"]
    pd_straight = D <= L["fit"]["roughly_straight_max_abs_dev"]
    pd_paper = (paper["tau_ratio_range"][0] <= ratio <= paper["tau_ratio_range"][1]) and \
        abs(P_tmax - paper["P_at_t_max"]) <= paper["P_at_t_max_abs_tol"]
    life = {"n_ic": int(len(T)), "dt": dt, "primary_threshold": L["primary_threshold"],
            "fit": fit, "fit_alt_t0": fit_alt, "gof": gof, "P_t_max": P_tmax, "median_empirical": median_emp,
            "n_decayed_before_t0": int((T <= L["fit"]["t0"]).sum()),
            "per_threshold": per_threshold,
            "dt_half": {"n_ic": n_half, "logrank_chi2": chi2, "logrank_p": p_lr, "fit_dt": fit_same, "fit_dt_half": fit_h,
                        "identical_lifetime_fraction": float(np.mean(np.isclose(T[:n_half], Th, atol=1.0) |
                                                                     (np.isinf(T[:n_half]) & np.isinf(Th))))},
            "paper_comparison": {"tau_ratio": ratio, "P_t_max_diff": P_tmax - paper["P_at_t_max"],
                                 "t_onset_ours": fit["t_onset"], "t_onset_paper": paper["t_onset"],
                                 "median_fit_ours": fit["median_from_fit"], "median_fit_paper": paper["median_from_fit"]},
            "seconds": {"full": full["elapsed_s"], "dt_half": half["elapsed_s"]}}
    results["lifetimes"] = life
    results["pass"].update({"d_roughly_straight": pd_straight, "d_consistent_with_paper": pd_paper,
                            "d_exponential_gof_strict_reported_only": gof["p"] >= 0.01})
    results["pass"]["GATE1_PASS"] = bool(pa and pb and pc and pe and pd_straight and pd_paper)
    print(f"(d) tau = {fit['tau']:.0f} +- {fit['tau_se']:.0f} (paper {paper['tau']:.0f}, ratio {ratio:.3f}); "
          f"P(t_max) = {P_tmax:.3f} (paper {paper['P_at_t_max']}); sup|P-Pfit| = {D:.4f} "
          f"(GOF p {gof['p']:.3f}); dt/2 log-rank p = {p_lr:.3f}")
    print("pass:", results["pass"])

    # ---------------- figures ----------------
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    jac = np.load(run_dir / "jacobian_scan.npz")
    # Post-hoc comparison with the paper's Re = 400 curve extracted from the PDF vector graphics
    # (code/scripts/extract_njp_fig10_vector.py). Written after the run: description only, no criterion.
    vec_path = REPO / "literature" / "notes" / "MFE2004_fig10_re400_vector.json"
    T_paper = None
    if vec_path.exists():
        T_paper = np.array([np.inf if x is None else float(x) for x in read_json(vec_path)["lifetimes"]])
        grid = [1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9999]
        cond = 3000.0
        lr_all = G.logrank(T, T_paper, t_max)
        lr_late = G.logrank(T[T > cond], T_paper[T_paper > cond], t_max)
        results["lifetimes"]["paper_vector_comparison"] = {
            "note": "paper curve from PDF vector graphics, N = 1000; post hoc, description only",
            "pointwise": [{"t": t, "P_paper": float((T_paper > t).mean()), "P_ours": float((T > t).mean()),
                           "se_paper": float(np.sqrt((T_paper > t).mean() * (1 - (T_paper > t).mean()) / len(T_paper))),
                           "se_ours": float(np.sqrt((T > t).mean() * (1 - (T > t).mean()) / len(T)))} for t in grid],
            "first_decay": {"paper": float(T_paper[np.isfinite(T_paper)].min()), "ours": float(T[np.isfinite(T)].min())},
            "fit_paper_t0_1000": G.exp_fit(T_paper, t_max, 1000.0), "fit_paper_t0_2000": G.exp_fit(T_paper, t_max, 2000.0),
            "fit_paper_t0_3000": G.exp_fit(T_paper, t_max, 3000.0), "fit_ours_t0_3000": G.exp_fit(T, t_max, 3000.0),
            "logrank_all": {"chi2": lr_all[0], "p": lr_all[1]},
            "logrank_given_survival_to_3000": {"chi2": lr_late[0], "p": lr_late[1]},
        }
    plotting.gate1_lifetimes(T, t_max, fit, paper, L["fit"]["t0"], fig_dir / "fig1_lifetimes", T_paper=T_paper)
    plotting.gate1_jacobian(jac["Re"], jac["max_real"], fig_dir / "fig2_laminar_spectral_abscissa")
    plotting.gate1_timeseries(full["rec_t"], full["rec_a"], T[:L["n_record"]], fig_dir / "fig3_example_transient")
    plotting.gate1_energy_hist(full["hist_edges"], full["hist_counts"], ths, fig_dir / "fig4_energy_occupation")
    results["figures"] = sorted(p.name for p in fig_dir.glob("*.png"))
    results["wall_seconds_this_invocation"] = time.perf_counter() - t_all
    results["stages"] = status.data["stages"]
    results["identity"] = identity
    atomic_write_json(run_dir / "results.json", results)
    print(f"done in {results['wall_seconds_this_invocation']:.1f} s -> {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
