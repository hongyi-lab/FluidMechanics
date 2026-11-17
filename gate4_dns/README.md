# Gate 4: full DNS of the MFE flow – solver checks (2026-11-17, overnight, unattended)

**Outcome: Gate 4a item 5 (dt) failed. Following the user's rule ("any Gate 4 failure: stop and report, do not change criteria or method"), the night stopped at Gate 4a.**

**Update (2026-11-18):** after the user's decision of 2026-11-17 07:40, Gate 4b, Gate 5 and Gate 6 were run. See the Gate 4b section below and `../gate5_dns/report_zh.md`.
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

**User decision (2026-11-17 07:40):** item 5 is recorded as "does not pass" as written; the cause is the criterion itself (E(0) = 1e-3 lies above the edge). It is replaced by 5a/5b/5c (post-hoc). From then on only the solver-correctness criteria stop the run.

## Gate 4b (2026-11-17, unattended)

Full report (Chinese): `../gate5_dns/report_zh.md`, section 1. Run outputs: `results_4b_v1/` and `results_4b_v2/`. Figure: `fig1_gate4b_pilot_ladder.png`.

- **5a (post-hoc; E(0) = 1e-6, dt vs dt/2):** max relative ΔE is 5.3e-8 at 48×24×24 and 5.6e-8 at 64×32×32. Pass.
- **5b:** the spectral Courant number is recorded every step. Cases above 2.5 are re-run from t = 0 at dt/2, then at dt/4. No NaN in any run.
- **Pilot (Re 400):** 8 directions (cut by the 45-min guard). E_B = 0.24 (valley at 0.245), r_A = 5e-7, rebound 0/72. The pilot-rule T_det is 2250.
- **T_det = 8000 (user decision, post-hoc):** v1 (`results_4b_v1/`, T_det 2250) found s_c set by runs still undecided at T_det, caused by slow streak decay. The ladder was re-run as v2 with T_det 8000.
- **Resolution ladder (v2):**
  - 48 vs 64: median |Δln s_c| 1.41 → next rung.
  - 64 vs 96: median 0.0010 → 64×32×32. Per ray the values are [0, 0, 0, 0.002, 0.103, 0.437], so R#13 and W#1 are not converged.
- **5c:** dt 0.0625 vs 0.03125 gives median 0.0000. Pass (limited by the bisection tolerance 1e-3).
- **Undecided runs:** 0 at the chosen resolution.
- **Choice:** Re 400, 64×32×32, dt 0.0625, r_A 5e-7, E_B 0.24, T_det 8000.

Gate 5 and Gate 6 results: `../gate5_dns/README.md`, `../gate6_dns/README.md`.
