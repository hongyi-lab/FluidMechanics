# Gate 3d full run

- Config: `code/configs/gate3d_mfe.yaml` (criteria and choices d1-d10 fixed before the 1% run; unchanged since).
- 1% timing run: `experiments/20261116_gate3d_mfe_frac0.01_v1` (estimate 6-8 min).
- Command: `python code/scripts/gate3d_mfe.py --config code/configs/gate3d_mfe.yaml --run-dir experiments/20261116_gate3d_mfe_v1 --no-progress`
- Started together with an independent code review (workflow); a review finding that changes a number would mean a new run dir (v2).
- Finished: 2026-11-16 23:11 (compute about 7 min: theory 13 s, item 1 11 s, item 2 85 s, item 3 det 28 s, item 3 gain 293 s).
- Results: item 1 holds (S test 0.0358 vs 3c (c) 0.0347, +0.0011); item 2 holds (random 0.0363, family 0.0225, eps trend
  +0.0494, marginal); item 3 holds (pooled 0.0293; eps = 3 at 0.100 > 0.05). Checks: (ii) alone = 3c reverse and 3c family
  c*_0 errors exactly (diff 0); Gate 2 directions and det s_c reproduced exactly.
- Figures: `python code/scripts/gate3d_figures.py experiments/20261116_gate3d_mfe_v1 docs/reports/figs/gate3d`.
