# Gate 3c: is the edge a state threshold, or seed × gain = threshold? (2026-11-16)

This is the MFE nine-mode model at Re = 400, Lx = 4π, Lz = 2π.

- **Data:** existing data only, from Gate 2 and Gate 3. No new MC.
- **Split and metric:** the same as Gate 3. The primary metric is |ln(s_pred / s_c)| on the 1448 test rays, reported as the median, the q90 and a 95% bootstrap CI.
- **Criteria:** fixed in `code/configs/gate3c_mfe.yaml` before the run. Choices the user did not specify are listed as c1–c13 in that file.
- **Code:**
  - `code/src/committor/feedback.py`
  - `code/scripts/gate3c_mfe.py`, with figures from `code/scripts/gate3c_figures.py`
  - post-hoc check: `code/scripts/gate3c_posthoc_family.py`
  - tests: `code/tests/test_feedback.py`
- **Run outputs:** in `results/`. `results/results.json` holds the full run; `results/posthoc_family_results.json` holds the post-hoc check.
- **Status:** awaiting the user's judgement on Gate 3c. There is no written report, because the user asked for the results directly.

Notation: m = (a1, a2, a3, a9), y = (a4..a8), and x = a − a_lam.

## Main result

### Item 0: check of the background derivation
The background claims hold. They were checked as stated in the user's message, because `check_m1.py` did not reach us.

- **Quasi-linearity.** MFE is exactly quasi-linear in y.
  - The y equations contain only m·y terms.
  - y–y products enter only the m equations.
  - There is no linear m–y coupling.
  - This structure is forced by the T_{Lx/2} shift, which flips exactly a4..a8.
- **The a3 equation.**
  - It is exactly da3/dt = −d3 a3 + yᵀF y, with d3 = (β² + γ²)/Re = 0.00867.
  - Gain (i) is exact except that it integrates m(t) with y = 0, which drops y's feedback on m.
- **W(T).**
  - Closed form: W(T) = W_inf − e^{BᵀT} W_inf e^{BT}, with B = A_y + d3/2 (largest real part of an eigenvalue: −0.0035).
  - The closed form matches numerical quadrature to 8e-13.
  - Against the full ODE at small amplitude, starting from laminar m, the first-order formula has a relative error of 1e-7, 1e-5 and 1e-3 at ε = 1e-5, 1e-4 and 1e-3.
- **Matching W(T) to M1.**
  - cos(W(45), W_M1) = 0.998, where W_M1 is the y–y block of Gate 3b's M1 divided by its a3 coefficient.
  - The cosine alone does not fix T: it is ≥ 0.99 for every T ≥ 21, and 0.993 at T = ∞.
  - T is fixed by magnitude: |W(T)|_F = |W_M1|_F at T = 45.2.
  - The instantaneous coefficient F has a cosine of only 0.395 with W_M1.

### Item 1: the theory quantity u_th = a3 + yᵀW y with threshold |u_th| = c*
The theory quantity holds under the criterion (median ≤ 0.05 for (b) or (c)).

| Variant | Fitted parameters | Test median [95% CI] | q90 |
|---|---|---|---|
| (a) W = W_inf | c* only | 0.225 [0.201, 0.255] | 0.80 |
| (b) W = κ W_inf | κ = 0.343, c* | 0.0475 [0.0420, 0.0534] | 0.31 |
| (c) W = W(T) | T = 45.0, c* | **0.0347** [0.0321, 0.0387] | 0.30 |
| (d) W = κ F (instantaneous coefficient) | κ, c* | 0.125 [0.114, 0.137] | 0.86 |
| (d) 100 random symmetric W, same norm | κ, c* | 0.108–0.149 (none ≤ (b), none ≤ 0.05) | |
| Reference: Gate 3b M1 | 54 coefficients plus a small MLP | 0.0394 | 0.28 |
| Reference: Gate 3 \|a3\| | c* | 0.139 | 1.02 |
| Reference: 9D MLP ceiling | | 0.0199 | 0.094 |

- **(c) compared with M1.** (c) − M1 = −0.0047 [−0.0073, −0.0013]. The fitted T = 45.0 is the same T that item 0 obtained independently from M1's magnitude.
- **What M1's quantity is.** Its physical content is a3 plus the nonlinear forcing of a3 by y, accumulated over about 45 time units of linear y dynamics.
- **Tails.** The tails are not improved: q90 is 0.30 against 0.28 for M1.
- **Random controls.** W_inf, W(T) and F satisfy the T_{Lz/2} constraint S W S = −W; the random controls do not. A reviewer's post-hoc check with symmetry-restricted random W gave the same range, 0.11–0.15.

### Item 2: merged candidate (gain (i) started from (a1, a2, u_th, a9), with κ from (b))
The merged candidate does not hold under the criterion.

- **Forward:** c* = 13.24, fitted on the random train rays. Test median 0.0735 [0.0679, 0.0785], q90 0.30.
- **Reverse:** c*₀ = 7.151 from Gate 3, with no fit. Test median 0.217 [0.213, 0.223], with a signed error of −0.21.
- **Rule:** the candidate fails if either direction is ≥ 0.10, and the reverse direction is.

What improved compared with Gate 3 gain (i):

- **Random test rays:** 0.149 → 0.0735. The difference is −0.075 [−0.088, −0.063], and q90 drops from 0.99 to 0.30.
- **The 680 left-peak rays:** 1.34 → 0.064.
- **Dependence on the a3 share:** the error no longer grows as the a3 share of the ray shrinks.

The noise family (ε > 0, 48 rays, σ = 0):

- **No ε trend:** e(ε = 1e-6) − e(ε = 1e-2) = +0.008.
- **A constant offset:** with the forward c*, the signed error is +0.14 at every ε (+0.143, +0.146 and +0.136).
- **Two different thresholds:** with c*₀ the family error is 0.023, which is descriptive only. The random rays need c* = 13.24 and the pure-a3 family needs 7.15. That is a difference of 6.1 in ln gain, or about 440×.

Comparison with the state threshold:

- **Merged forward against item 1 (b):** +0.026 [+0.021, +0.030]. On the random rays, the gain dynamics make the state threshold worse, not better.

### Post-hoc, descriptive (not a criterion): the item-1 state threshold on the noise family
- **Drift with ε.** The frozen u_th of (b) and (c) has signed errors of −0.48, −0.29 and −0.09 at ε = 1e-6, 1e-4 and 1e-2. The ε trend is −0.39.
- **Same as M1.** The drift matches Gate 3b's M1 (−0.50, −0.30, −0.10). On the family, u_th equals a3 to a relative 4e-6 or better.
- **The 90° rays.** The two rays at 90° (pure a2 direction) never cross.

### Taken together
In this form, neither a state threshold nor seed × gain gives one threshold for both families.

- **The state threshold u_th** is good on the random rays (0.035) but drifts with ε on the family.
- **Seed × gain** is flat in ε on the family, but needs a threshold about 440× higher in gain on the random rays than on the pure-a3 family.

### Caveats
- **Only the a3 feedback is restored.** Item 2 puts back only the a3 part of y's feedback on m. The y–y forcing of a1, a2 and a9 is 2–3× larger in coefficient. At the edges, a2's lowest-order shift (median 0.0045) exceeds |a2| (0.0029). A negative item 2 therefore does not exclude seed × gain with the full feedback.
- **The item-0 ODE check is local.** It holds at small amplitude from laminar m. On ray states at 0.9 s_c, the lowest-order a3 correction is right in the median (ratio 0.96), but the tails are wide (10–90% range: 0.41–1.91).
- **The family level is outside the criterion.** The forward offset of +0.14 at every ε is not part of the item-2 criterion, which uses only the ε trend.

## `fig1_theory_vs_M1.png`: W(T) against M1's learned y–y form
- **(a) Cosine against T:** it reaches its plateau by T ≈ 21.
- **(b) Frobenius norm against T:** it crosses |W_M1| at T = 45.2.
- **(c–f) The matrices:** F, W(45), W_inf and W_M1.

## `fig2_theory_quantity.png`: item 1
- **(a) κ search:** W_inf, F and the 100 random W.
- **(b) T search.**
- **(c) Test medians with CIs:** every candidate, with the criterion lines at 0.05 and 0.10.

## `fig3_seed_x_gain.png`: item 2
- **(a) Forward errors** on the test rays.
- **(b) Reverse errors** on the test rays.
- **(c) The noise family by ε.** The two points at +1.5 for ε = 1e-6 are the 90° rays, which never cross and are drawn at the clip limit.
- **(d) Error against the a3 share.**
- **(e) The 680 left-peak rays.**
- **(f) The merged coordinate at the random-ray edges.** The 5/50/95% quantiles are 0.6, 13.7 and 27.2.
