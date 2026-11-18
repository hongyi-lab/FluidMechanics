# Gate 7 (DNS) v1: fast channel, numerical artefact or real route?

- **Config:** `code/configs/gate7_dns.yaml`, written before any run; snapshot in `config_source.yaml`. Criteria and constants
  are the user's: S* = 0.1655 (Gate 5), c* = −2.82 (Gate 6).
- **Code:**
  - `code/scripts/gate7_dns.py`
  - `code/src/committor/dns_local.py`
  - snapshots of both in this folder
- **Driver:** `run_gate7.sh` runs four process chains on the one GPU, then the analysis. Every stage resumes from its checkpoint.

      bash experiments/20261118_gate7_dns_v1/run_gate7.sh

## Log

- **10:01, 1% timing** (`timing.json`, `stdout_timing.log`):
  - Per-case-step cost: 1.1 ms at 64, 4.2 ms at 96, 9.5 ms at 128. It does not depend on batch size (the GPU is saturated
    at batch 1).
  - Full 128 set estimate: 38 h by the preset estimator, about 13 h case by case. Both exceed 12 h.
  - So 128 is reduced to W#1, R#22, W#11 (user rule).
- **10:02, start.** The four chains:
  - **P1:** 96 for R#22, W#11 → 128 for R#22, W#11 → 7c at 128.
  - **P2:** 96 for W#3, W#7, W#9, R#15 → 7c at 96 for those four and R#13.
  - **P3:** 128 for W#1 → 7c at 128.
  - **P4:** theory at 64, 96, 128 → 7c at 64 (all 8 rays) → W#1 fine scan at 64, then 96.
- **10:12, solver correctness at the new resolutions** (`check_4a_128.py`, `check_4a_128.json`). This check is not in the
  preset plan.
  - Gate 4a items 1, 2, 4 and 6 at 128×64×64 (dt 0.03125) and 96×48×48 (dt 0.0625), with the mode symmetries and a short
    nonlinear run (E0 = 1e-3, t ≤ 30).
  - All pass. Every residual is at round-off: laminar 1.5e-15, Galerkin 2.2e-16, work 3e-18, symmetry 4e-15,
    divergence 2e-14, batch 0.
- **10:27, code review** (4 reviewers and adversarial verification; 7 findings confirmed, all minor). None changes a number
  on the normal path. The fixes are in place before any downstream stage started, and the running processes are untouched.
  New snapshots are `script_snapshot_*_rev1040.py`; the first snapshots are kept.
  1. A flagged ray (undecided or not bracketed) used to stop the other rays of the same chain through an assert. It is now
     skipped per ray and written to `skipped_<stage>.json`.
  2. Theory and diag stages now skip work already done, so the driver can be re-run safely.
  3. The truth checkpoint time is no longer reset on resume.
  4. Diag runs now show a progress bar.
  5. The analysis now works with expected levels per ray (128 only for the effective 128 set). A missing or flagged s_c at an
     expected level is reported. Such a decision ray counts for neither verdict ("incomplete"), and a control with a missing
     level fails. An s_i with no crossing gives err(i) = +inf, as in Gate 5.
  6. The first-order ratio p is not formed for pure-wave rays, where p(64) is zero up to round-off (P0 δ = 0).
     The second-order ratio r and the ratio S(s_c(64)) are reported instead.
  - Check: theory at 64, recomputed here, matches the Gate 5 `theory.npz` to 4e-16 (`theory_64_vs_gate5.json`). err(i) at 64
    reproduces the Gate 5 per-ray table.
- **17:25, all done** (7.4 h; `results.json`, `stdout_analysis.log`).
  - **Controls pass:** R#15 64→96 +0.0031; W#11 64→96 +0.0031, 96→128 −0.0047 (limit 0.02).
  - **Verdict (preset): undetermined.** "Real route" holds for W#1 and R#22 (4 needed); "artefact" for none; incomplete 0.
    - W#3, W#7, W#9 have no 128 (reduction), so their top two levels are (64, 96), with Δln 0.42–0.44 > 0.05.
    - They already meet t_hit_hi < 600 and |err(i)| ≥ 0.25. So 128 for these three alone decides between "real route"
      (all three |Δln| ≤ 0.05) and "undetermined". Not run; waiting for the user.
  - **Descriptive:**
    - 64→96: s_c up 41–55% on all 5 decision rays. 96→128: W#1 −1.4%, R#22 −0.5%.
    - err(i) at the highest level is 0.42–0.56 (64: 0.85–0.93). s_i does not depend on the level (ratio 0.9998).
    - t_hit_hi at the highest level is 109–130.
    - R#13 is no longer fast at 96 (t_hit_hi 1506; s_c +10%).
    - W#1 fine scan: monotone at 64 and 96 (one switch each).
  - **Counts:** 0 undecided runs, 0 NaN, 0 flagged rays.
    - s_c searches: 34 re-runs at dt/2 and 1 at dt/4.
    - Fine scans: 18 at dt/2 and 2 at dt/4.
  - **Report:** `docs/reports/gate7_dns_report_zh.md`. Figures in `docs/reports/figs/gate7/`; overview in
    `docs/reports/figs/overview/`.
