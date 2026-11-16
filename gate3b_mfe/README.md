# Gate 3b: can the edge be described by a few quadratic mechanism quantities? (2026-11-16)

This is the MFE nine-mode model at Re = 400, Lx = 4π, Lz = 2π.

- **Data:** Gate 2 data only: 10 000 random rays (train / val / test = 7072 / 1480 / 1448) and 200 000 offset states, plus the deterministic s_c of the noise-family rays. No new MC.
- **Metric:** same as Gate 3. The primary metric is |ln(s_pred / s_c)| on the 1448 test rays (median, 90% quantile, 95% bootstrap CI).
- **Ladder M_k:** c(x) = g(u_1..u_k), with x = a − a_lam and u_i = xᵀQ_i x + b_iᵀx. Q_i is symmetric and not forced to be definite, g is a small MLP with no monotonicity constraint, and k = 1..4.
- **Criterion:** fixed before the run: the smallest k with test median ≤ 0.04; "not enough" if k = 4 is still ≥ 0.10.
- **Full report** (Chinese): `report_zh.md`. Run outputs are in `results/`. Code is in `../code` (`code/scripts/gate3b_mfe.py`, `code/src/committor/mechanism.py`, config `code/configs/gate3b_mfe.yaml`).

These results await the user's confirmation of Gate 3b.

## Main result

- **One quadratic mechanism quantity unifies the edge over all loading directions of the random rays to about 4%.**
  - M1 test median: 0.0394 [0.0363, 0.0427].
  - The criterion records k = 1. The pass is marginal: the CI straddles 0.04, and the other three converged restarts score 0.041–0.042.
  - k = 2 passes robustly: 0.0308 [0.0286, 0.0341].
- **The quantity is readable.** u1 ≈ a3 + 54.7 a4a8 + 11.3 a4a7 + 10.9 a5a6 + 8.8 a5a8 + …, normalised by the a3 coefficient.
  - The edge sits at |u1| ≈ const: g is V-shaped around u1 = 0, and half of the edge states have u1 < 0.
  - The y–y products it uses are the quadratic terms of the MFE da3/dt equation, though their weights do not match the instantaneous coefficients.
  - One reading is "the streamwise-vortex amplitude plus its nonlinear forcing by y". This is an interpretation that has not been tested separately.
- **The ladder reaches the 9D MLP ceiling.**
  - M3: 0.0191, no difference from the ceiling 0.0199.
  - M4: 0.0177, median slightly better than the ceiling, tail slightly worse (q90 0.116 vs 0.094).
- **Several simpler models do not reach 0.04.**
  - Strict single-quadric threshold (monotone g): 0.120.
  - Physics k = 2 with u1 = a3², u2 = |y|²: 0.125 with g free, 0.128 for the additive version.
  - The missing ingredients are the sign of a3 and the y–y cross products.
- **The unification does not extend to the small-ε noise family.**
  - Every model trained on random rays predicts too early there, including M1–M4 and the MLP ceiling. The ln error is about −0.1 at ε = 1e-2, −0.3 at 1e-4 and −0.5 at 1e-6.
  - Random rays contain no states with a tiny y seed, so the ln ε dependence of s_c is not learned.
  - Only the linear-gain coordinate gain (i) is ε-independent (+0.15).

## `fig1_mechanism_planes.png`: edge states of the 10 000 rays in the mechanism planes (description, no fit)
- (|a3|, |y|): a sharp vertical stem at |a3| ≈ 3e-3 for |y| ≲ 5e-3. For larger |y| the lower envelope bends to small |a3|, and the points spread widely above it.
- The points do not collapse onto one curve. The ratio of within-bin to overall IQR is 0.62–1.29.
- Colouring by 1−a1 and a9 shows no pattern: signed correlations with the residual are ±0.01. The scatter follows the a3 share of the direction and the amplitude s_c.

## `fig2_ladder.png`: test error of the ladder and the physics models
- Grey dots show every restart.
- Horizontal lines: Gate 3 references and the two criterion lines.

## `fig3_k2_planes.png`: k = 2 planes
- (a) Learned M2 in the asinh(u/τ) plane.
  - Edge states sit on two vertical bands, u1 = ±const, closed by a top line u2 = const.
  - B lies outside the arch: |u1| above a threshold OR u2 above a threshold. This is a max-type combination, not additive.
- (b) Physics k = 2 in (log a3², log |y|²).
  - The critical line is an L: a3² ≈ 9.6e-6 at small |y|, and |y|² ≈ 3.7e-4 at small a3.
  - The additive version (α3 = 3.26e-3, αy = 2.16e-2) almost coincides with it.

## `fig4_quadratic_forms.png`: principal axes of the learned quadratic forms (canonical basis)
- Each heat map shows the modal composition of the top eigenvectors, plus one row for the linear term b.
- Only a3 (linear and squared), the y–y cross products (a4a8, a4a7, a5a6, a5a8) and the y energies (a8², a4², a5²) appear.
- a1 and a9 never appear among the main components.

## `fig5_family.png`: frozen models predicting the deterministic s_c of the 48 noise-family rays with ε > 0
- Errors grow as ε decreases and are nearly the same for every model trained on random rays.
- Only gain (i) at σ = 0 stays flat.

## `fig6_gain_failure.png`: why gain (i) fails on random rays
- (a, b) Error against the initial y energy fraction: only a weak trend.
- (d, e) Error against the a3 energy fraction: strong.
  - a3-poor rays are predicted much too late (+1.75 at an a3 fraction of 1e-4).
  - a3-rich rays are early with the frozen c*₀ and accurate after a refit.
- (c, f) The left peak of Gate 3 fig3 (b1), c < −4.5, consists of 680 rays (6.8%). 99.7% of them have an a3 fraction < 0.01 (66% < 0.001), and a4 and a8 are enriched.
- The peak is not a numerical floor:
  - c is independent of σ (differences ~1e-8);
  - the maximum lies at t > 0 for 98%;
  - there are no repeated values;
  - tr P is far from underflow.
- Without a3 there is no lift-up transient, so the y perturbation grows only about 15×.
