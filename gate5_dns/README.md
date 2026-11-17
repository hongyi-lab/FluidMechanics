# Gate 5 (DNS): ground truth and the first gate (2026-11-17/18, unattended)

**Outcome: 5.2 partly holds.** Test median |ln(s_pred/s_c)| is 0.231. The R test rays alone give 0.020 and the W rays 0.803; the energy-only control gives 0.704. W is the preset key set, and it fails. 5.1: (a) yes, (b) yes, (c) no.

- **Full report (Chinese, Gates 4b–6):** `report_zh.md`. Its figure paths `docs/reports/figs/gate456/` map to the gate folders here; its `experiments/...` paths map to the `results*/` folders.
- **Run outputs:** `results/`. The 48 MB `theory_check_fields.pt` and the checkpoints are not included.
- **Parameters, from Gate 4b** (`../gate4_dns/results_4b_v2/choice.json`): Re 400, 64×32×32, dt 0.0625, r_A 5e-7, E_B 0.24, T_det 8000 (the T_det value is a post-hoc user decision).
- **Undecided runs (user rule):** undecided does not count as a transition. Every s_c must be bracketed by one run that reaches A and one that reaches B.

## Ground truth
- **Rays:** 52 in total: F 16, R 24 (12 calibration + 12 test), W 12. 1131 runs; every ray is bracketed A → B.
- **Undecided and NaN:** 0 undecided runs, 0 flagged rays, 0 NaN. 96 cases were re-run at dt/2 by the Courant rule (5b).
- **DNS vs the nine-mode model (MFE):** on the F rays with ε ≤ 1 the ratio is 0.87–1.05. On the R and W rays it ranges from 0.40 to 2.67.

## 5.2, first gate
S* = 0.1655 is fitted on the first 12 R rays only.

| Predictor | Test median | R test | W |
|---|---|---|---|
| S (first gate) | **0.231** | 0.020 | 0.803 |
| energy only | 0.704 | 0.548 | 0.765 |
| \|a3\| only / first order only | no crossing on W | 0.047 / 0.052 | no crossing |

- **Two kinds of W ray:** 7 of the 12 W rays reach the DNS edge with S = 0.026–0.037, about 6× below S*. Their predictions are too high by ln 0.76–0.93.
- **Resolution caveat:** W#1 is the only W ray checked at 96×48×48. There its s_c is 55% larger than at 64. The other W rays were not checked at 96, so whether the W failure survives resolution is open.

## Figures
- `fig2_gate5_truth_first_gate.png`: (a) DNS vs MFE; (b) 5.1 shape; (c) 5.2 errors by predictor.
- `fig3_gate5_S_curves.png`: S(s) per ray set, with the DNS edge marked.
