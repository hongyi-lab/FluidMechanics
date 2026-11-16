# Gate 2: deterministic labels and noise-family committor (2026-11-16)

The model is the MFE nine-mode model at Re = 400, Lx = 4π, Lz = 2π, with da = f(a) dt + σ dW.
- A = {E < 5e-7}, B = {E ≥ 0.29}, where E = Σ(aᵢ − aᵢ^lam)².
- SDE: stochastic Heun, dt = 0.01, Brownian bridge, T_max = 6000.

Gate 2 was accepted by the user on 2026-11-16; Gate 3 is in ../gate3_mfe.

## `fig1_det_edge.png`: deterministic edge on 10 000 random rays from the laminar state
- (a) First-transition energy E_c ranges from 1.0e-5 to 7.8e-2 (max/min = 7664; median 1.6e-4).
  - Every ray transitions below E_B.
  - The 20 rays with the smallest E_c are all dominated by a3 (68–91% of the energy).
- (b) Fraction of rays whose fate switches at least twice along the ray: 4.8% / 16% / 27% / 42% at 50 / 250 / 500 / 1000 grid points, and 62% / 74% at 2000 / 4000 points on a 500-ray subset.
  - The fraction does not converge.
  - The first-crossing amplitude s_c is unchanged under refinement (|Δln s_c| ≤ 3e-10).
- (c) Labels of the 200 000 offset states at s_c(1 + u).
  - Above s_c, 0.1–0.5% of states fall in decay windows.
  - Below s_c there is one B state (ray 4914, u = −0.003). It reveals a 0.25%-wide B window that the grid missed, so that ray's true first crossing is 0.38% lower. Reported, not corrected.

## `fig2_noise_family.png`: MC committor on 56 rays × 3 σ (3e-9, 3e-8, 3e-7)
The directions are cos θ e3 + sin θ e2, plus an a4..a8 component of relative size ε.
- (a) s50 (MC) over the deterministic edge s_c.
  - ε = 0 (invariant subspace): 0.33 / 0.30 / 0.27, the same for all 7 angles to within 0.3%.
  - ε = 1e-2: 1.000, so noise is irrelevant there.
- (b) Relative layer width is 0.08–0.16 when noise dominates, and 5e-5 to 5e-2 when the deterministic edge dominates.
- 159 of the 168 profiles are resolved. The 9 unresolved ones are flagged with ×.

## `fig3_noise_gain_prediction.png`: test of σ²·g(s50) = const on the pure-a3 ray
Here g = max_t tr P / σ² from the linear noise gain along the deterministic transient.
- (a) Committor profiles at the three σ (M = 400).
- (b) ln σ²g(s50) against σ.
  - Predicting the outer σ from the middle one gives errors of +0.89% ± 0.20% and −1.12% ± 0.22% (isotonic s50).
  - The effective exponent p_eff (σ^p g = const) is 1.83 ± 0.04 and 1.82 ± 0.04 in the two decades, against p = 2 predicted.
  - So the relation holds to about 1% in s50 but is not exactly constant.
  - The null "s50 independent of σ" would give −9.5% / +11.4%.
