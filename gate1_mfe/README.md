# Gate 1 figures: MFE nine-mode model (2026-11-15)

This is the Moehlis–Faisst–Eckhardt (2004) nine-mode shear-flow model at Re = 400, Lx = 4π, Lz = 2π.
- The equations are transcribed term by term from NJP 6:56 (2004).
- Galerkin projection of the paper's own mode functions reproduces the transcribed coefficients to 1e-14.

Checks:
- The right-hand side vanishes at the laminar state.
- The nonlinear term conserves energy to 7e-15.
- Every laminar Jacobian eigenvalue has a negative real part for Re = 100–2000.
- The dynamics are exactly equivariant under the D2 symmetry group (SIADS 2005, eqs. 3.24–3.25).

Figures:
- `fig1_lifetimes.png`: lifetimes of 1e4 initial conditions on the shell Σaᵢ² = 0.3.
  - (a) Lifetime histogram on semi-log axes.
  - (b) Survival probability with the exponential fit (τ = 10502 ± 140), compared with NJP fig. 10. The paper's 1000 lifetimes were extracted from the PDF vectors.
- `fig2_laminar_spectral_abscissa.png`: slowest laminar decay rate against Re. It is always positive and equals β²/Re.
- `fig3_example_transient.png`: the nine amplitudes of one transient with lifetime 4365 (compare NJP fig. 5).
- `fig4_energy_occupation.png`: occupation of log10 of the perturbation energy E = Σ(aᵢ − aᵢ^lam)².
- `fig5_onset_diagnostic.png`: stricter laminarisation thresholds (E < 1e-3 … 1e-6) and an alternative initial-condition reading.
  - At E < 1e-5 the early part matches the paper.
  - Each decade stricter shifts lifetimes by about 180–190 time units, against a prediction of 186.6.
