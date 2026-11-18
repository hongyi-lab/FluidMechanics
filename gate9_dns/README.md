# Gate 9 (DNS): blind test outside the nine patterns (2026-11-19, unattended)

**Status (2026-11-19 09:20):**

- 9.2 is frozen and pushed (b9b4d21).
- 9.3 truth at 96 is done for G (10 rays) and O (6 rays).
- M is running; V and 128 follow, subject to the 16 h budget.
- A partial snapshot is in `results_partial_GO_0920/` (see below). It is pushed at the user's request so they can look at it before deciding whether to run the rest.

## Partial results: G and O only (16 of 28 rays; descriptive, no verdict)

Files: `results_partial_GO_0920/`.

- `tables_partial_GO.md`: all 9.4 tables, with a per-ray list.
- `results.json`: the unchanged 9.4 analysis stage, run on a copy that holds only the G and O truths.
- `fig13`, `fig14`, `fig16`: the figures.
- `truth_96_G.json`, `truth_96_O.json`: the truth checkpoints.
- `NOTES.md`: the run log.

All 16 rays are bracketed: none undecided, none non-monotone, rel width ≤ 0.01.

Uncertainty u is 0.005–0.015. No 128 run has been done yet, so u uses the nine-mode 96→128 values.

**In range (n = 15), ln(s_pred / s_c):**

- Median −0.238; IQR [−0.611, −0.058]; 90% quantile of |err| 0.990; max |err| 1.059.
- |err| median 0.238, bootstrap 95% [0.075, 0.385].

**Coverage, with the exact binomial test against nominal:**

| Interval | Covered | p |
|---|---|---|
| 50% | 2/15 | 0.0074 |
| 80% | 3/15 | 1.0e-6 |

**Paired |err_rule| − |err_baseline| (median, bootstrap 95%):**

| Baseline | Median | 95% interval | Rule closer |
|---|---|---|---|
| Two conditions | −0.148 | [−0.272, 0.101] | 8/15 |
| Energy only | −0.626 | [−0.881, −0.321] | 14/15 |
| Gate 8a rule | 0 | — | — |

On in-range rays the Gate 8a rule is identical to this round's rule.

**By route:**

- **Predicted fast (13 rays):**
  - 12 have s_pred < s_c. The truth is harder than the fast estimate.
  - Errors: G#1 −0.30, G#2 −0.39, G#3 −0.22, G#4 −0.24, G#6 −0.19, G#7 −0.08, G#8 −1.00, G#10 −0.84, O#1 −0.26, O#2 +0.17, O#3 −0.04, O#4 −0.98, O#6 −1.06.
  - 11 of 13 truths lie between s_fast and s_slow. The exceptions are G#8 (above s_slow) and O#2 (below s_fast).
- **Predicted slow, in range:** G#5 +0.024 and G#9 +0.005.
- **Out of range:** O#5 has s_c 0.0861 ≥ 0.8·s_pred = 0.0795, so the bound holds (1/1).

**Routes (truth fast = t_hit_hi < 600):**

| | true fast | true slow |
|---|---|---|
| pred fast | 11 | 2 (G#2, G#4) |
| pred slow | 1 (O#5) | 2 |

Draft of the full report (sections 1–4: Gate 8 post-hoc check, rays, calibration, frozen predictions): `report_zh_draft.md`.

**Frozen predictions (9.2):**

- Files: `results_9_2/predictions_9.md` (and `.json`).
- Frozen 2026-11-19 04:26:04.
- Committed before any 9.3 truth run.

## The rule (9.0, fixed before any run)

- **Slow estimate:** s_slow = max(s_i, s_ii). S* = 0.16553 and c* = −2.8204 are unchanged from Gates 5/6.
- **Wave energy:** E_w(t) = ‖P_w δ(t)‖² / ‖U‖². δ is the linearised response to the unit ray, computed at 96×48×48.
- **Wave gain:** G = max over 0 ≤ t ≤ 20 of E_w(t), and g = G / E_w(0).
- **Fast estimate:** if g ≥ 2, s_fast = √(P*/G), with P* = 0.0020953. Otherwise there is no fast route.
- **Prediction:** s_pred = min(s_fast, s_slow). The route is "fast" if s_fast < s_slow, otherwise "slow".
- **Range:**
  - For s_pred ≤ 0.03 (in range), the prediction comes with split-conformal 50% and 80% intervals by predicted route. Half-widths in ln s:
    - slow: 0.0097 and 0.0136 (n = 24);
    - fast: 0.0558 and 0.1565 (n = 8).
  - Otherwise (out of range), only the bound s_c ≥ 0.8·s_pred is claimed.
- **Baselines:**
  - the Gate 8a rule, min(√(P*/G), s_slow);
  - two conditions, s_slow;
  - energy only, 0.0070263.

## The 28 new rays (9.1)

All rays are projected for symmetry and divergence and normalised to E = 1.

| Class | Rays | Shape | Nine-mode fraction |
|---|---|---|---|
| G | 10 | Band \|kx\| ≤ 2α, \|kz\| ≤ 2γ, wall-normal index ≤ 3, Gaussian | 0.14–0.39 |
| O | 6 | One (kx, ±kz) pair, random normal shape: (α,γ)×3, (α,2γ), (2α,γ), (2α,2γ) | O#1–3: 0.63, 0.62, 0.19; O#4–6: 0 |
| V | 6 | kx = 0 (v, w) vortices, u = 0, kz = γ×3 and 2γ×3, plus a 1e-2 G-class seed | see note |
| M | 6 | cosθ·a3 + sinθ·a8, θ = 10, 25, 40, 55, 70, 80° | 1 |

- **V class:** the wall-normal index runs to |n| ≤ 6. This is revision h4, made before the freeze. With |n| ≤ 3, the kz = 2γ vortex has a single allowed shape.
- **Seeds and build details:** `code/configs/gate9_dns.yaml`, choices h1–h18.

## Summary of the frozen predictions

- **Range:** 25 rays in range and 3 out of range: O#5 (0.0993), V#5 (0.0375) and V#6 (0.170).
- **Routes:** 14 predicted fast and 14 predicted slow.
  - Fast: G#1–4, G#6–8, G#10, O#1–4, O#6 and M#6.
- **O#5** has no fast route: G = 1.
- **V rays:** the fast estimate is 1.1–2.5 because E_w(0) ≈ 8e-5, so all six are predicted slow.

## Files

- **Code:**
  - `code/scripts/gate9_dns.py` (stages);
  - `code/src/committor/dns_gate9.py` (rays, the truth search, statistics);
  - `code/scripts/gate9_figures.py` and `code/scripts/gate9_tables.py` (figures and tables from `results.json`);
  - `code/configs/gate9_dns.yaml`.
- **`results_9_2/`:**
  - predictions;
  - rays (`rays9.npz` holds the coefficients);
  - calibration (`calib.json`);
  - the Gate 8 post-hoc check (`posthoc8.json`);
  - theory and gain data at 64;
  - wave gain at 64 and 96;
  - `NOTES.md` (run log);
  - the driver `run_gate9.sh`.
