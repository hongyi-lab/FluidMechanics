# Gate 8 (DNS) v1: blind test of the third condition (fast channel = oblique-wave route); 8d records; 8c completes Gate 7

- **Config:** `code/configs/gate8_dns.yaml`, written before any run; snapshot in `config_source.yaml`. The criteria and
  constants are the user's:
  - S* = 0.1655 (Gate 5)
  - c* = −2.82 (Gate 6)
  - P* from the W#1 128 truth (Gate 7)
- **Code:** `code/scripts/gate8_dns.py` and `code/src/committor/dns_gate8.py` (snapshots in this folder). It reuses the Gate 7
  `find_sc_local` and the Gate 5/6 theory code; dns.py, dns_theory.py and dns_local.py are not changed.
- **Order (h13):** one process at a time.
  1. posthoc
  2. predict (8a): frozen, then pushed to GitHub
  3. truth (8b)
  4. d8 (8d)
  5. c8_prepare, then the unchanged Gate 7 script in `8c_gate7_128/` (8c)
  6. analysis

## Log

- **Before 18:02, function checks before any run** (time corrected 23:08; it was logged as "~18:40" by mistake) (64 grid, short runs; scratch, not kept):
  - `theory_kz` bins sum to `run_delta_q` aa, ab, bb within 1e-16.
  - `gain_alpha` reproduces `run_gain` exactly (diff 0).
  - `diag_kz` reproduces the Gate 7 `run_diag` parts exactly (diff 0); the streak and roll bins sum to the totals within 4e-16.
  - The linear wave y stays at kx = α (M has kx = 0 only), so ‖y_α‖² = ‖y‖².
- **18:02–18:13, posthoc and 8a** (`stdout_posthoc.log`, `stdout_predict.log`):
  - G at 64 and 96 for the 52 rays and X#1 (`g_all.npz`); G_96 / G_64 = 1 within 1e-8.
  - X#1 at 64: s_i = 0.01672, s_ii = 0.01579 (ok); `x1_gate56.json`.
  - **8a frozen 18:13:23:** `predictions_8a.json` and `.md`. P* = 0.0020953.
  - **Pushed to GitHub as 40cd95a** (`PUSHED_8a.txt`), before any 8b truth run.
- **Before 18:14, review fix** (one reviewer; time corrected 23:08, it was logged as "18:2x" by mistake). `st_analysis`: `relative_to(REPO)` on a relative run dir would raise → `.resolve()`.
  It would only have crashed the final analysis; no number is affected. New snapshot `script_snapshot_gate8_dns_rev1825.py`.
  - Not done, by choice: the Gate 7 7c diagnostics at 128 for W#3, W#7, W#9. The Gate 7 verdict needs only 7a and the
    theory (already at 128).
- **Driver:** `run_gate8.sh` (8b → 8d → 8c → analysis, one process at a time; a failing stage stops the chain).
- **18:14:09, first driver start failed** (SyntaxError from my inline comment in the review fix; no run had started). Fixed (comment
  moved above the line, py_compile), log renamed `stdout_8b_truth_failed_start.log`, restarted 18:14:40. The broken file was in the
  8a push 40cd95a; the fix was pushed as a6e31fa.
- **18:14:40–20:09:59, 8b** (`truth_96_blind.json`, 115.3 min): 9 rays, local grid without extension, 6 bisection rounds; dt/2 re-runs
  42 (4 of them then dt/4); 0 NaN, 0 undecided.
- **20:10:00–20:34:38, 8d** (24.6 min): theory kz 5.4 min, diag kz 10.3 min (4 re-runs at dt/2, 1 of them then dt/4), linear wave 8.9 min.
- **20:34:39–23:04:31, 8c** (`8c_gate7_128/`, 149.8 min for the 128 truth of W#3, W#7, W#9): grid ABBBBB for all three, 5 bisection
  rounds (path AAAAB for all three, the same as W#1, so the same bracket factor [0.98125, 0.990625] and the same dln -0.014163);
  dt/2 re-runs 12; Gate 7 analysis: **real route** (5/5), controls pass.
- **23:04:32, analysis** (`results.json`): 8b n = 8, median |err| 3 conditions 0.1717, 2 conditions 0.2856, energy 1.5552 →
  **partly**; routes 6/8; X#1 −0.0216.
- **23:05–23:15, figures** (`docs/reports/figs/gate8/`): fig9 label offsets and axis range 0.15 (R#13 at 0.125 was outside); the
  fig9 (c) fast cluster named in one text line. Report `docs/reports/gate8_dns_report_zh.md`.
- **23:20–23:35, report check** (3 independent verifiers recomputed every number in the report from the data). 26 findings (24 distinct): counts, ranges, double rounding of four section 2 cells, wording. 23 fixed in the report; the a6e31fa push was confirmed from the git log of the GitHub clone. No verdict, criterion or frozen prediction is affected.
