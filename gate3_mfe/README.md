# Gate 3: coordinate ranking and the zero-fit test of the linear-gain coordinate (2026-11-16)

This is the MFE nine-mode model at Re = 400, Lx = 4π, Lz = 2π.
- **Data:** Gate 2 data only: 10 000 random rays (split 7072 / 1480 / 1448 into train / val / test), 200 000 offset states, and the 159 resolved noise-family MC profiles. No new MC.
- **Primary metric:** the threshold-prediction error ln(s_pred / s_c).
  - s_pred is the first amplitude along a ray at which the coordinate reaches its critical value c*.
  - c* and the side of the threshold are fitted on the training rays.
  - E is the baseline; an MLP classifier is the ceiling.
- **Linear-gain coordinate:** P(0) = y0 y0ᵀ with the σ²I term, y = (a4..a8), along the x-independent transient. Two versions:
  - (i) c = ln max_t tr P;
  - (ii) c = ln tr P at the first maximum of a2².

Gate 3 was accepted by the user on 2026-11-16. The user's record: H2 does not hold; a generic critical linear gain does not hold; the zero-fit of gain (i) holds only within the noise family, with 90° as the exception. Gate 3b is in ../gate3b_mfe.

## `fig1_threshold_error.png`: threshold-prediction error on the 1448 test rays
- (a) Median |ln(s_pred / s_c)| with 95% bootstrap CIs.
  - Ceiling (MLP, polar input): 0.020. The best low-dimensional coordinates are 7× worse:
    - triple (1−a1, a2, a3) 0.138;
    - |a3| 0.139;
    - gain (i) 0.149;
    - gain (ii) 0.207.
  - Baseline E: 0.518.
  - Adding a1, a2 or the gain to |a3| does not improve the median: gain (i) − |a3| = +0.010 [−0.002, +0.016].
- (b) 90% quantile. Open symbols marked [grid floor] are candidates whose fitted threshold is already met at the first grid point on every ray (signed single modes except a3, |a1|, 9D linear logistic; signed a3 never crosses on 51% of rays). Their error only measures where the grid starts.
- The gain results agree to printed precision for σ = 0, 3e-9, 3e-8, 3e-7 (per-ray differences ≤ 1e-4). At generic edges |y0|² dominates σ² tr Q, so on generic rays gain (i) acts as a deterministic transient-growth coordinate.

## `fig2_accuracy_vs_u.png`: classification error of offset states s_c(1 + u) on test rays
- At |u| ≤ 1e-3 every candidate is at 0.5, because threshold errors are far larger than |u|.
- At |u| = 0.1 the accuracy is: MLP 0.95, |a3| / triple 0.71, gain (i) 0.69, E 0.56.
- Dashed line: irreducible error from decay windows above the edge (0.16% of u > 0 states; at most 0.76% at any one u).

## `fig3_zero_fit.png`: zero-fit test (critical value calibrated only on the pure-a3 ray, three σ, then frozen)
- (a1) Version (i), c*₀ = 7.151.
  - Predicting s50 of the 156 non-calibration noise-family profiles gives median |ln| 0.012, the same as a refit (leave-one-angle-out 0.012).
  - Exception: the 19 profiles at 90° (pure a2 plus y) are all late by 15–33%.
- (a2) Version (ii), c*₀ = −18.66: median 0.470. At 90° the a2² maximum is at t = 0, so the coordinate has no noise term and 74% of those profiles never cross.
- (b1) Version (i) on the 10 000 random rays: zero-fit median |ln| 0.256, systematically early (signed −0.18).
  - The refitted c* gives 0.149.
  - c at the edges spans −5.1 to 42.7 (5–95%), so there is no universal critical gain.
- (b2) Version (ii): 99.6% of rays already exceed c*₀ at the first grid point; the error is ≥ 2.88 (about 4.7 when extrapolated below the grid).

## `fig4_p_eff_versions.png`: p_eff (σ^p g(s50) = const) on the pure-a3 ray, isotonic s50 of Gate 2
- Version (i): 1.83 ± 0.04 and 1.82 ± 0.04, as in Gate 2.
- Version (ii): 0.43 ± 0.01 and 0.45 ± 0.01, far from 2.
  - a2² peaks at t ≈ 100, before the gain builds up (tr P peaks at t ≈ 430–560).
  - With an 8× smaller time step the values are 0.445 and 0.448.

## Notes
- The analysis code went through an independent review (3 lenses, each with a verifier). Its findings were fixed before the final run:
  - inf-safe tail statistics and paired differences;
  - below-grid flags;
  - accounting for version (ii) at t* = 0;
  - a time-step check for version (ii).

  No conclusion changed.
- On the noise family (descriptive, threshold refitted leaving one angle out), gain (i) ranks first: 0.0032, against |a3| 0.048, MLP 0.052, E 0.257.
