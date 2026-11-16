# Gate 3d: is the edge two conditions at once? (2026-11-16)

This is the MFE nine-mode model at Re = 400, Lx = 4π, Lz = 2π.

**The two conditions:**
- **(i) Lift-up streak condition:** |S| ≥ S*, where S is the predicted peak of the streak a2. It has no fitted time.
- **(ii) Seed × gain condition from Gate 3c:** merged gain ≥ c*₀ = 7.151.

The composite prediction is s_pred = max(s_i, s_ii).

**Data:**
- Random rays and the noise family: existing data only (Gate 2, Gate 3, Gate 3c).
- Transition family (ε = 0.03–3): new deterministic s_c from the Gate 2 routine. No MC.

**Split and metric:** the same as Gate 3. The primary metric is |ln(s_pred / s_c)| on the 1448 test rays, reported as the median, the q90 and a 95% bootstrap CI.

**Criteria:** fixed in `code/configs/gate3d_mfe.yaml` before the run. Choices the user did not specify are listed as d1–d10 in that file.

**Code:**
- `code/src/committor/liftup.py`
- `code/scripts/gate3d_mfe.py`, with figures from `code/scripts/gate3d_figures.py`
- tests: `code/tests/test_liftup.py`

**Run outputs:** in `results/`.

**Status:** results are in. The written report and the independent code review were still in progress when this was pushed. The user's replication attachment did not reach the server.

Notation: y = (a4..a8).

## Main result

| Item | Criterion | Result |
|---|---|---|
| 1. \|S\| = S* (S* fitted on train only) | test median ≤ 0.05 and ≤ 0.01 worse than Gate 3c (c) | **holds**: 0.0358 vs 0.0347 (+0.0011) |
| 2. composite on random rays and the noise family | random ≤ 0.05, family ≤ 0.05, \|ε trend\| ≤ 0.05 | **holds**: 0.0363, 0.0225, +0.0494 (marginal) |
| 3. composite on the transition family (70 rays, 90° excluded) | pooled median ≤ 0.05 | **holds**: 0.0293. ε = 3 alone is 0.100 > 0.05 |

### Item 0: the derivation
- **Constants:**
  - d2 = (4β²/3 + γ²)/Re = 0.010725
  - J23 = −√(3/2) βγ/κ_βγ = −1.0332
  - t* = 103.52; |L| peaks there
  - L(t*) = −39.27
  - e^{−d2 t*} = 0.329
- **Closed forms against quadrature:** V_roll and V_dir match direct quadrature of the user's integrals to 1e-10.
- **S against the full nonlinear ODE:** S equals a2(t*) to second order when a1 = 1 and a9 = 0. The residual divided by ε² goes to 0 linearly in ε.
- **What S leaves out:** S does not include the m–m term J23·a3·(a1 − 1 + a9) of da2/dt. On directions that excite a1 or a9, that term is of the same order as the y–y terms. Adding it (S_hat) changes nothing on the rays: 0.0358 in both cases.
- **Link to Gate 3c:** S/L(t*) = a3 − 0.0084 a2 + yᵀ(V/L)y.
  - cos(V/L, W(45)) = 0.998, and |V/L| / |W(45)| = 1.08.
  - cos(V/L, W_M1) = 0.999.
  - The T ≈ 45 fitted in Gate 3c is therefore the lift-up time scale, expressed in a3 units.

### Item 1
- **S\*:** 0.1233, side +1.
- **Test median:** 0.0358 [0.0326, 0.0393], q90 0.30.
- **Paired differences:**
  - against Gate 3c (c): +0.0011 [−0.0021, +0.0039]
  - against Gate 3b M1: −0.0037 [−0.0058, −0.0006]
  - against the 9D MLP ceiling: +0.0159
- **t scan (descriptive):** the train median is smallest at t = 92.6, close to t* = 103.5.

### Item 2
- **Random test rays:**
  - composite 0.0363 [0.0329, 0.0403]; (i) alone 0.0358; (ii) alone 0.217
  - (i) binds on 93.6% of the rays
- **Noise family (ε = 1e-6, 1e-4, 1e-2; 48 rays incl. 90°):**
  - composite 0.0225, with signed medians of +0.024, +0.011 and −0.028
  - (ii) binds on 100%, 94% and 88% of the rays per ε
  - (i) alone drifts with ε: −0.48 → −0.09
- **ε trend:** +0.0494, which passes the 0.05 line by 0.0006. Without the 90° rays it is +0.0505 (descriptive).

### Item 3: transition family
- **Per ε (composite median |ln|):**

  | ε | 0.03 | 0.1 | 0.3 | 1 | 3 |
  |---|---|---|---|---|---|
  | median \|ln\| | 0.038 | 0.020 | 0.009 | 0.029 | **0.100** |
  | signed | −0.038 | −0.017 | +0.005 | +0.021 | +0.086 |
- **Which condition binds:** (i) binds on 50% of the rays at ε = 0.03, 93% at 0.1 and 100% from 0.3 up.
- **(ii) alone:** it gets worse as ε grows, from 0.048 to 0.27.
- **The 90° rays (reported separately):**
  - The composite is off by +0.39 to +0.53.
  - One of the two rays at ε = 0.03 never crosses.

### Checks
- **(ii) alone reproduces Gate 3c exactly:** on the random rays it matches the Gate 3c item 2 reverse errors, and on the noise family the Gate 3c errors with c*₀ (difference 0).
- **Gate 2 reproduced exactly:**
  - The noise-family directions rebuilt from Gate 2's RNG stream match exactly.
  - Re-running the deterministic scan reproduces Gate 2's s_c on the 16 rays with ε = 1e-2.

## Figures
- **`fig1_liftup_theory.png`:**
  - (a) L(τ), with t* marked
  - (b) S against the full ODE
  - (c–h) F2, V_roll/L, V_dir/L, V/L, W(45) and W_M1
- **`fig2_item1_S.png`:** (a) the t scan; (b) test medians with CIs; (c) test error histograms against Gate 3c (c).
- **`fig3_item2_composite.png`:** (a) random test errors; (b) the noise family by ε; (c) binding fractions.
- **`fig4_item3_transition.png`:**
  - (a) signed errors across both families, ε from 1e-6 to 3
  - (b) where each condition is met along the ray, against s_c
  - (c) binding fractions by ε
