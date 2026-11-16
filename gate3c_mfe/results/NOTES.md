# Gate 3c full run

- Config: `code/configs/gate3c_mfe.yaml` (criteria and choices c1-c13 fixed before the 1% run; unchanged since).
- 1% timing run: `experiments/20261116_gate3c_mfe_frac0.01_v1` (estimate 2-2.5 h).
- Command: `python code/scripts/gate3c_mfe.py --config code/configs/gate3c_mfe.yaml --run-dir experiments/20261116_gate3c_mfe_v1 --no-progress`
- Started: 2026-11-16 19:40 (the independent code review was still running; started on the user's go).
- Code review (3 lenses, each with an adversarial verifier) finished at about 19:50, after the start. No finding changes a reported
  number or verdict, so no code or config change was made. Points for the report:
  1. first_crossing_abs corner case (whole |u| hump below the grid start): only in the kappa/T search, never in a reported number;
     verified at full scale (analytic vs grid train medians agree to 3e-5).
  2. W_inf, W(T) and F satisfy the T_Lz/2 constraint S W S = -W exactly; the random controls do not. A reviewer's post-hoc check with
     symmetry-restricted random W gave the same range (0.11-0.15), so the control conclusion stands.
  3. The item-0 ODE check covers small amplitude from laminar m; at 0.9 s_c on ray states the lowest-order a3 correction holds in the
     median (ratio 0.96) with wide tails (10-90%: 0.41-1.91).
  4. Item 2 restores only the a3 part of the y -> m feedback; the y-y forcing of a1, a2, a9 is 2-3 times larger in coefficient
     (a2's lowest-order shift at the edges, median 0.0045, exceeds |a2| 0.0029). A negative item 2 does not exclude seed x gain
     with the full feedback.
  5. Criterion reading (c8/c10): the forward family LEVEL (about +0.18 at every eps in the 1% run) is not in the criterion; report it.
  Reviewers replicated item 1 at full scale during the review (CPU/GPU, outside the run); nothing was changed in response.
- Finished: 2026-11-16 21:41 (compute 2.02 h: item 1 31 min, merged gain grid 87 min, family 2 min; estimate was 2-2.5 h).
- Results: item 1 holds ((c) W(T), T = 44.97: test 0.0347 [0.0321, 0.0387]; (b) kappa = 0.343: 0.0475 [0.0420, 0.0534]);
  item 2 does not hold (forward 0.0735 [0.0679, 0.0785], reverse 0.2172 [0.2127, 0.2229]; family eps trend +0.008, family level
  +0.14 at every eps with the forward c* = 13.24). Post-hoc family check of item 1: `experiments/20261116_gate3c_posthoc_family_v1`.
- Figures: `python code/scripts/gate3c_figures.py experiments/20261116_gate3c_mfe_v1 docs/reports/figs/gate3c`.
- The user asked for the results directly instead of a written report (2026-11-16 ~22:30); pushed to GitHub on request.
