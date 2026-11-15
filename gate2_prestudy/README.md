# Gate 2 pre-study: noise level, A and B (2026-11-15)

The model is the MFE nine-mode model at Re = 400, Lx = 4π, Lz = 2π, with isotropic additive noise da = f(a) dt + σ dW.
The sets are A = {E < r_A} and B = {E ≥ E_B}, where E = Σ(aᵢ − aᵢ^lam)².

These values are proposed and still await the user's confirmation; no full committor labels have been computed.

| Item | Value |
|---|---|
| σ | 3e-8, 3e-7 |
| r_A | 5e-7 |
| E_B | 0.29 |
| Integrator | stochastic Heun, dt = 0.01, Brownian-bridge crossing test |
| T_max | 6000 |

`figP_prestudy.png` panels:
- (a, b) Occupation histograms of E and a1 for 1e4 deterministic trajectories. The valley gives E_B; it is a plateau over roughly 0.1–0.36.
- (c) Deterministic edge energy E_c along 4096 random rays from the laminar state (min ≈ 1e-5, max ≈ 0.13), together with r_A and the noise floor E_ss.
- (d) Stationary mean of E from the linearised Lyapunov equation, E_ss = σ²·2.99e5 (lift-up a3 → a2), checked by Monte Carlo, and the fraction of paths from the laminar state that reach B within T_max.
- (e, f) Committor profiles q(s) along 5 rays at the two σ.
  - The transition layer is smooth and monotone but thin: relative width about 2e-4 and 2e-3, proportional to σ.
  - It is centred on the deterministic edge.
- (g) The pure-a3 ray lies in the invariant subspace a4…a8 = 0, where turbulence cannot develop. Noise breaks the symmetry, and the transition moves to about 0.3 × the deterministic value.
- (h) B commitment: paths entering B from the transition layer decay within 1000 time units only 1.8–4% of the time, consistent with genuine turbulence.

Checks:
- From states just outside A, 0 of 26600 paths reach B at either σ.
- Spontaneous laminar → B transitions: 0 of 10⁴ at each σ.
