# Gate 9 (DNS): blind test outside the nine patterns (2026-11-19, unattended)

**Status: 9.2 frozen; 9.3 truth runs not started at the time of this commit.**

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
  - `code/scripts/gate9_figures.py`;
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
