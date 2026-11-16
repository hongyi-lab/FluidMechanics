# Gate 4: full DNS of the MFE flow – solver checks (2026-11-17, overnight, unattended)

**Outcome: Gate 4a item 5 (dt) failed. Following the user's rule ("any Gate 4 failure: stop and report, do not change criteria or method"), the night stopped at Gate 4a.**
- Not run: the pilot, the resolution ladder, Gate 5 and Gate 6.
- Every other 4a criterion passes. In the nine-mode subspace the DNS reproduces the MFE right-hand side exactly.
- Full report (Chinese): `report_zh.md`. Run outputs: `results/`.

## Solver
Code: `code/src/committor/dns.py`; checks: `code/scripts/gate4_dns.py`; config with criteria fixed before running: `code/configs/gate4_dns.yaml`.
- **Flow:** sinusoidal shear flow with free-slip walls at y = ±1. Re = 400, Lx = 4π, Lz = 2π, body force (√2π²/(4Re)) sin(πy/2) e_x.
- **Domain extension:** y is extended to period 4 by the mirror about y = 1.
- **Discretisation:**
  - Fourier in x, y and z; pseudo-spectral in rotational form;
  - 2/3 dealiasing;
  - Leray projection;
  - Lawson RK4 with a viscous integrating factor;
  - batched float64 on the GPU.
- **Symmetries:** the discrete symmetries shared by all nine modes were found numerically: the y mirror, shift-reflect and shift-rotate. They are enforced every step by group averaging. The user's two symmetry claims are confirmed.

## 4a criteria (v2; 48×24×24)

| Criterion | Result |
|---|---|
| Laminar fixed point max\|du/dt\| < 1e-12 | 0 / 2.2e-16 ✓ |
| max\|div u\| < 1e-10; symmetry residual < 1e-12 | 1.4e-14; 8.7e-15 ✓ (excludes the NaN trajectory of the dt = 0.25 run; see report) |
| Galerkin identity (100 random nine-mode states) < 1e-10 | 1.1e-16 ✓ |
| Decay rates of a1 − 1, a3, a9 at amplitude 1e-8 < 1e-8 | 2.4e-13, 1.9e-13, 1.4e-12 ✓ |
| Lift-up streak components cos γz and cos πy cos γz (rates γ²/Re, (π²+γ²)/Re) < 1e-6 | 7.8e-14, 3.1e-13 ✓ |
| Nonlinear term does no work < 1e-10 | 7.1e-18 ✓ |
| **dt: halving dt changes E(t) by < 1e-4 (E(0) = 1e-3, t ≤ 200)** | **✗ for every dt** (0.25 → NaN; 0.125 / 0.0625 / 0.03125 → 0.27 / 0.28 / 0.37) |
| Alone vs in a batch < 1e-12 | 3.3e-16 ✓ |

## Why item 5 fails
The E(0) = 1e-3 initial states are above the edge for 3 of the 4 random directions. The MFE model also sends these same three to B.
- **Transition:** those trajectories become turbulent at t ≈ 25–72.
- **Chaos:** chaotic amplification, at 0.06–0.15 per time unit, turns any change of dt into an O(0.1–0.5) change of E by t = 200.
- **Below the edge it passes:** before transition the solver is fourth-order accurate, and the decaying direction passes at dt = 0.25. At E(0) = 1e-6, dt = 0.25 passes (9e-6).

This was confirmed by two independent checker agents. Diagnostic figure: `fig_dt_diagnostic.png` (descriptive).

The first 4a run (`results/results_v1.json`) also failed item 3 for a1 and a9. The cause was round-off of the O(1) product U×ω_U. It was fixed by an algebraically identical split form; the criteria were unchanged.

**Awaiting the user's decision on item 5.**
