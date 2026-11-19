# Gate 9 (DNS) v1: blind test outside the nine patterns (28 new rays G/O/V/M)

- **Config:** `code/configs/gate9_dns.yaml`, written before any run (snapshot `config_source.yaml`). Criteria and constants are the
  user's: S* = 0.16553, c* = −2.8204, P* = 0.0020953 (full precision from Gate 8 `predictions_8a.json`).
- **Code:** `code/scripts/gate9_dns.py`, `code/src/committor/dns_gate9.py` (snapshots in this folder). The G, O, V fields are
  appended to `grid.modes` (h6); dns.py, dns_theory.py, dns_local.py, dns_gate8.py are not changed.
- **Order:** posthoc8, rays, calib (CPU / seconds) → predict (9.2; frozen, committed and pushed) → driver `run_gate9.sh`:
  truth G, O, M, (budget) V, (budget) 128, analysis. One process at a time.

## Log

- **00:20, posthoc8** (`posthoc8.json`): 8-ray median 0.172, bootstrap 95% [0.056, 0.463]. 25 rays: s_pred ≤ 0.03 n = 19,
  median |err| 0.0133, max 0.1565; s_pred > 0.03 n = 6, |err| 0.027, 0.227, 0.286, 0.463, 0.872, 0.952, all predicted low.
- **00:20, rays** (`rays9.npz`, `rays9.json`): 22 fields (G 10, O 6, V 6). At 64/96/128: E = 1 to 15 digits, max div ≤ 3.2e-13,
  symmetry residual ≤ 1e-13, energy outside the band 0. No G ray needed a new seed (nine-mode fractions 0.14–0.39).
  - Nine-mode fractions: O#1 0.63, O#2 0.62, O#3 0.19, O#4–O#6 0 (wavenumbers outside the nine modes); V#3 0.57.
  - **Degeneracy (parameter-type, recorded, not changed):** at kx = 0, kz = 2γ, with u = 0, the walls, the three enforced
    symmetries and n ≤ 3, the vortex has a single allowed shape (ψ ∝ sin(πy) sin(2γz)). V#4, V#5, V#6 therefore have the same
    vortex (overlaps 0.99987–0.99991, same sign by chance) and differ only in their 1e-2 G-class seeds. V#1–V#3 (kz = γ) span a
    2-D space (cos(πy/2), cos(3πy/2)).
  - 128 rays (h13): O#6, G#1 (largest |k|² in G), V#1 (largest |k|² in V).
- **00:21, calib** (`calib.json`): slow n = 24, h50 0.0097, h80 0.0136; fast n = 8, h50 0.0558, h80 0.1565 (user 0.010 / 0.014;
  0.056 / 0.157). All calibration rays in range and of their stated route under the 9.0 rule.
- **00:22, checks:** G96 of X#1 and W#1 on the extended basis = Gate 8 values (19.9549, 3.6242). run_gain throughput: 28 rays ×
  13 amplitudes to T = 1500 ≈ 3.5 h (linear in the batch; the GPU is saturated).
- **00:23:37, predict (9.2) started**; the 16 h budget clock starts here (`budget.json`).
- **00:25–00:38, code review** (3 reviewers, each finding adversarially verified; 9 findings, 5 distinct real ones). All fixed
  before any freeze:
  1. V#4–V#6 one vortex (the |n| ≤ 3 band was my choice h4; the user bounded the index for G and O only) → **h4 revised:**
     V uses |n| ≤ 6 (3-D shape space at kz = γ and 2γ); compact storage widened to |n| ≤ 6. G and O fields are bit-identical to
     the first build. New V overlaps, pairwise among the new rays: V#1–V#3 −0.80, 0.55, −0.76; V#4–V#6 0.18, 0.31, −0.82 (two roundings corrected 13:50; new vs first build: 0.94, 0.68, 0.97, 0.89, 0.55, 0.00). 128 V ray is now V#6 (largest |k|²).
     V#5, V#6 carry n up to 6; the 64 theory truncates their quadratic products above n = 10 (recorded; the 128 run on V#6
     tests the grid).
  2. 9.4 analysis would crash on a ray whose two-condition baseline is None (s_i never crosses) → None baselines are left out of
     that baseline's statistics, count reported.
  3. No O fallback for the 128 ray (h13) → O#6 then the other O rays by |k|².
  4. No-prediction rays dropped silently → listed with their truth.
  5. Censored truths (not bracketed) dropped from coverage and the range rule → h18: counted as not covered / by their bound,
     listed apart from the error statistics.
- **00:39:40, predict stopped** (16 min in: theory done, gain block 1 at 7/1500) for fix 1. The first build, its theory and log
  are in `superseded_pre_freeze/`. No prediction had been written. The budget clock keeps its 00:23:37 start.
- **00:41:23, predict (9.2) restarted** with the revised rays (`config_source_rev0045.yaml`, new snapshots).
- **04:25:17, gain block 1 done** (3.7 h; no ray needed the second block: all 28 s_ii flags "ok"). Max Courant of the gain runs
  ≤ 2.23 except G#10 (2.84 at its top amplitude s = 0.072, far above its crossing at 0.030; recorded, no re-run, as Gates 6/8).
  No NaN.
- **04:26:04, 9.2 frozen** (`predictions_9.json`, `.md`; 225 min). New V nine-mode fractions: V#1 0.026, V#2 0.054, V#3 0.535,
  V#4–V#6 ≈ 1e-5 (seed only). All s_i cross S*.
  - In range 25, out of range 3 (O#5 0.0993, V#5 0.0375, V#6 0.170). No ray without a prediction.
  - Predicted fast 14 (G#1–4, G#6–8, G#10; O#1–4, O#6; M#6), slow 14.
  - O#5 (2α, γ): G = 1 (E_w maximal at t = 0), no fast route. V: g 4.5–20 but E_w(0) ≈ 8e-5, so s_fast 1.1–2.5 (slow).
  - M: g = 19.95 for every θ (the a3 part carries no wave; E_w ∝ sin²θ), as X#1.
  - G_96/G_64 ≥ 0.9994 (lowest V#5, V#6, the n ≤ 6 rays).
- **04:28:59, 9.2 pushed:** commit b9b4d21 (GitHub main), `gate9_dns/results_9_2/`. `PUSHED_9.txt` written; 9.3 may start.
- **04:29:08–07:37:16, truth G** (188 min; 10 rays, all "ok", one switch each, final rel width ≤ 0.0099). Parameter ladder: grid
  30 cases at dt/2, 1 of them then at dt/4; bisection 2–4 cases per round at dt/2. Descriptive only (no verdict):
  the 8 rays predicted fast all have s_c above s_pred (ln err −0.075 to −0.998); G#5 and G#9 (predicted slow) +0.024, +0.005.
  Truth t_hit_hi: G#2 1124, G#4 933, G#5 1212, G#9 922; the others 79–160.
- **07:37:16–08:05:18, truth O** (28 min; 6 rays, all "ok", one switch each). Ladder: grid 21 cases at dt/2, 3 of them at dt/4.
  Descriptive: O#1 −0.258, O#2 +0.165, O#3 −0.040, O#4 −0.979, O#6 −1.059 (all predicted fast; truth t_hit_hi 61–174);
  O#5 (out of range) s_c 0.0861 ≥ 0.8 s_pred = 0.0795.
- **09:22, partial G/O pushed at the user's request** (commit 2ad60f7; `partial_GO_0920/` here = the 9.4 analysis stage run on a scratch copy with only the G and O truths; the run dir's own results.json is not written). M keeps running; whether to finish M, V and 128 is the user's call.
- **08:05:18–10:09:15, truth M** (124 min; 6 rays, all "ok", one switch each). Ladder: grid 10 cases at dt/2; bisection 1 case
  at dt/2 in each of 4 rounds. Descriptive: ln err M#1 +0.014, M#2 +0.005, M#3 +0.005, M#4 −0.003, M#5 −0.009, M#6 −0.022 (fast,
  t_hit_hi 129; the others slow, 1034–1586). M figure coordinates (x = s_c/s_slow, y = s_c/s_fast): M#1–M#5 x 0.986–1.009,
  y 0.056–0.762; M#6 x 0.687, y 1.022 — all on or next to the right angle within 2.2% (corrected 13:50 after the report check: M#1's whole bracket is below s_slow, 1.4% inside, like V#3 which has no fast route; M#2, M#3 touch s_slow with their upper end).
- **10:09:15, budget before V:** elapsed 9.76 h + est V 1.55 h + est 128 3.0 h = 14.3 h ≤ 16 → V runs (started 10:09:16).
- **10:09:16–11:26:56, truth V** (78 min; 6 rays, all "ok", one switch each). Ladder: grid 17 cases at dt/2, 2 of them at dt/4;
  bisection 1–3 cases per round at dt/2. Descriptive: kz = γ (V#1–V#3) ln err −0.003, −0.003, +0.014, truth slow (t_hit_hi
  1034–1198); kz = 2γ (V#4–V#6) +0.314, +0.246, +0.234, predicted slow but truth fast (t_hit_hi 95–172); V#5 and V#6 (out of
  range) s_c < 0.8 s_pred (bound not met). On V#4–V#6, s_i alone is within +0.02 to +0.08 (s_ii > s_i raised s_slow).
- **11:26:56, budget before 128:** elapsed 11.06 h + 3.0 h = 14.06 h ≤ 16 → 128 runs (O#6, G#1, V#6; started 11:26:57).
- **11:26:57–13:14:40, truth 128** (108 min; O#6, G#1, V#6; local grid of 6 factors around s_c(96), all bracketed with one switch,
  bisection to rel width (hi−lo)/lo ≤ 0.0095). Ladder: grid 7 cases at dt/2 (0.015625), none at dt/4; bisection none (max Courant 2.26);
  no NaN. s_c 128 vs 96: O#6 0.030968 vs 0.029581 (Δln +0.0458), G#1 0.031164 vs 0.029074 (+0.0694), V#6 0.13394 vs 0.13457
  (−0.0047). t_hit_hi 96 → 128: O#6 61 → 79, G#1 79 → 580, V#6 95 → 95. V#6 top grid point s = 0.54 is B at t = 0 (s² > E_B).
- **13:14:42, 9.4 analysis done** (driver step "analysis", unchanged code): results.json. u per class now from the 128 rays
  (G ≈ 0.070, O ≈ 0.046, V 0.0056–0.0067; M from the nine-mode values by route). All other 9.4 numbers equal the pre-128 draft
  (they use the 96 truth by spec). Tables: `tables_9_4.md` (gate9_tables.py); figures: docs/reports/figs/gate9/ (gate9_figures.py);
  report: docs/reports/gate9_dns_report_zh.md. Total wall time 12.85 h (00:23:37–13:14:42), nothing cut.
