# Gate 9 (DNS): blind test outside the nine patterns (2026-11-19, unattended)

**Status:** done 2026-11-19 13:14:42 (12.85 h, within the 16 h budget, nothing cut). The run order was 9.2 → 9.3 (G, O, M, V) → 128 → 9.4.

By design there is no "holds / does not hold" verdict. The results are given as:

- error distributions;
- interval coverage;
- comparison with the baselines;
- the uncertainty of the truth.

The full report (Chinese) is `report_zh.md`.

## Results (descriptive)

Error: err = ln(s_pred / s_c(96)). A negative err means the prediction is too low.

**Truth:**

- All 28 rays are bracketed at 96×48×48, none undecided, none non-monotone, no NaN.
- The rays O#6, G#1 and V#6 were rerun at 128×64×64:

| Ray | s_c(96) | s_c(128) | Δln | t_hit_hi, 96 → 128 |
|---|---|---|---|---|
| O#6 | 0.029581 | 0.030968 | +0.046 | 61 → 79 |
| G#1 | 0.029074 | 0.031164 | +0.069 | 79 → 580 |
| V#6 | 0.13457 | 0.13394 | −0.005 | 95 → 95 |

- These shifts set the truth uncertainty u for each class:
  - G ≈ 0.070, O ≈ 0.046, V 0.0056–0.0067.
  - M has no 128 ray, so it uses the nine-mode 96→128 values by route: u 0.0056–0.0145.
  - u is only drawn as error bars; every 9.4 statistic uses the 96 truth.

**In range (s_pred ≤ 0.03, n = 25):**

| | n | \|err\| median (bootstrap 95%) | 50% interval covered | 80% interval covered |
|---|---|---|---|---|
| All | 25 | 0.075 [0.014, 0.258] | 9/25, p 0.23 | 10/25, p 1.4e-5 |
| Predicted slow | 11 | 0.005 [0.003, 0.014] | 7/11, p 0.55 | 7/11, p 0.25 |
| Predicted fast | 14 | 0.248 [0.165, 0.682] | 2/14, p 0.013 | 3/14, p 4.1e-6 |

The p values are exact binomial tests against the nominal rate.

- **Predicted fast:** 13 of 14 are predicted too low; only O#2 is high (+0.165). The largest errors are O#6 −1.06, G#8 −1.00, O#4 −0.98 and G#10 −0.84.
  - O#4 and O#6 have wavenumbers outside the nine modes.
  - At 128 the truths of G#1 and O#6 are higher, so the errors there grow further.
- **Predicted slow:** 4 rays are not covered.
  - Three lie just past the 80% half-width of 0.0136: V#3 and M#1 at +0.014 (over by 0.0006), and G#5 at +0.024 (over by 0.010).
  - The fourth, V#4, is on the wrong route.
- **Baselines** (paired |err_rule| − |err_baseline|, median with bootstrap 95%):

| Baseline | Paired median | Rule closer | Note |
|---|---|---|---|
| Energy only | −0.626 [−0.742, −0.382] | 24/25 | |
| Two conditions | 0 [−0.148, 0] | 9/25 | The two differ only on the 14 predicted fast: −0.178 [−0.395, 0.320], 9/14 |
| Gate 8a rule | 0 | — | Identical in range |

**Out of range (n = 3):** the bound s_c ≥ 0.8·s_pred holds for 1 of 3.

- O#5 holds.
- V#5 (0.0293 < 0.0300) and V#6 (0.1346 < 0.1360) miss by about 2% and 1%.

**Routes (truth fast = t_hit_hi < 600):** 22 of 28 are right.

| | true fast | true slow |
|---|---|---|
| pred fast | 12 | 2 (G#2, G#4) |
| pred slow | 4 (O#5, V#4, V#5, V#6) | 10 |

- At 96, t_hit_hi falls in two clusters: 61–174 and 922–1586.
- The 4 rays predicted slow but truly fast have either no wave growth (O#5, G = 1) or almost no wave energy (V#4–V#6, kz = 2γ vortices, E_w(0) ≈ 8e-5).

**M class (cosθ·a3 + sinθ·a8):** all 6 points lie on or next to the right angle, within 2.2%.

- The two nearest the corner are outside it: M#5 at x = 1.009 and M#6 at y = 1.022.
- Only M#1 (θ = 10°, farthest from the corner) is slightly inside. Its whole bracket is 1.4% below s_slow. V#3, which has no fast route, is also 1.4% below.

## The rule (9.0, fixed before any run)

- **Slow estimate:** s_slow = max(s_i, s_ii), with S* = 0.16553 and c* = −2.8204.
- **Wave gain:** G = max over 0 ≤ t ≤ 20 of E_w(t), and g = G / E_w(0).
  - E_w(t) = ‖P_w δ(t)‖² / ‖U‖², where δ is the linearised response to the unit ray at 96×48×48.
- **Fast estimate:** if g ≥ 2, s_fast = √(P*/G), with P* = 0.0020953.
- **Prediction:** s_pred = min(s_fast, s_slow). The route is "fast" if s_fast < s_slow.
- **In range (s_pred ≤ 0.03):** split-conformal 50% and 80% intervals by route, as half-widths in ln s:
  - slow: 0.0097 and 0.0136 (n = 24);
  - fast: 0.0558 and 0.1565 (n = 8).
- **Out of range:** only s_c ≥ 0.8·s_pred is claimed.
- **Baselines:** the Gate 8a rule, two conditions (s_slow), and energy only (0.0070263).

## The 28 rays (9.1)

| Class | Rays | Shape |
|---|---|---|
| G | 10 | Band \|kx\| ≤ 2α, \|kz\| ≤ 2γ, wall-normal index ≤ 3, Gaussian; nine-mode fraction 0.14–0.39 |
| O | 6 | One (kx, ±kz) pair: (α,γ)×3, (α,2γ), (2α,γ), (2α,2γ) |
| V | 6 | kx = 0 (v, w) vortices, u = 0, kz = γ×3 and 2γ×3, wall-normal index ≤ 6, plus a 1e-2 G-class seed |
| M | 6 | cosθ·a3 + sinθ·a8, θ = 10, 25, 40, 55, 70, 80° |

## Files

- **`report_zh.md`:** the full report (Chinese). Its paths map as follows:
  - `docs/reports/figs/gate9/` → this folder;
  - `experiments/20261119_gate9_dns_v1/` → `results/`.
- **Figures (white background):**
  - `fig13_gate9_pred_vs_truth`: prediction against truth, with intervals and u.
  - `fig14_gate9_errors`: error distributions of the rule and the three baselines.
  - `fig15_gate9_M`: the M right-angle figure.
  - `fig16_gate9_routes`: the t_hit_hi histogram and the route table.
- **`results/`:** the full run directory.
  - `results.json` holds the 9.4 output; `tables_9_4.md` renders it, including a per-ray table.
  - Truths: `truth_96_{G,O,M,V}.json` and `truth_128.json`.
  - Also: `predictions_9.*`, `budget.json`, `NOTES.md` (run log), and the stdout logs.
- **`results_9_2/`:** the frozen predictions as pushed before any truth run (b9b4d21, 2026-11-19 04:28:59).
- **`results_partial_GO_0920/`:** the partial G/O snapshot pushed at 09:22 for inspection (2ad60f7).
- **Code:**
  - `code/scripts/gate9_dns.py`;
  - `code/src/committor/dns_gate9.py`;
  - `code/scripts/gate9_figures.py` and `code/scripts/gate9_tables.py`;
  - `code/configs/gate9_dns.yaml`.
