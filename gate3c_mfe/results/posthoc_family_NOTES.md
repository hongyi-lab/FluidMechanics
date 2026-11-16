# Gate 3c post-hoc: item-1 state thresholds on the noise family (descriptive, not a criterion)

- Script: `code/scripts/gate3c_posthoc_family.py` (no arguments; reads `experiments/20261116_gate3c_mfe_v1/results.json` and the
  Gate 2 family `family_det.npz`). Run 2026-11-16 ~22:20, seconds.
- Frozen from the Gate 3c full run: (b) kappa = 0.34297, c* = 0.0031298; (c) T = 44.965, c* = 0.0031311. Same family rays
  (eps = 1e-6, 1e-4, 1e-2; 16 each), deterministic s_c, ray grid (4000 points) and grid rule as Gate 3b section 6 / Gate 3c item 2.
- Why: item 2 showed the merged seed x gain candidate is flat in eps on the family; the user's question is state threshold vs
  seed x gain, so the state threshold's own family behaviour is needed for the comparison. Not in the user's spec; reported as post-hoc.
- Result (signed median ln(s_pred/s_c), with no-crossing rays as +inf):
  - (b): -0.485, -0.293, -0.092 at eps = 1e-6, 1e-4, 1e-2; eps trend (1e-6 minus 1e-2, 14 finite pairs) -0.386.
  - (c): -0.484, -0.293, -0.092; trend -0.386.
  - Gate 3b M1 on the same rays: -0.497, -0.296, -0.097.
  - The 2 rays at 90 deg (pure a2 direction, a3 = 0) never cross at any eps (12.5% per eps).
- Reading: on the family u_th = a3 to relative 4e-6 or better, so the state threshold predicts the same s for every eps, while s_c
  moves with eps; the error therefore drifts with eps, by the same amount as M1.
