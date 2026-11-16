# 20261116_gate3b_mfe_v1

Gate 3b full run: can the MFE edge be described by a few quadratic mechanism quantities? Gate 2 data only, no new MC.

- Config: `code/configs/gate3b_mfe.yaml` (criterion fixed before any run: smallest k with test median <= 0.04; k = 4
  median >= 0.10 -> "quadratic mechanism quantities are not enough"). The config history lists every change made after
  the 1% tests and the code review, all before this run and without test data.
- Script: `code/scripts/gate3b_mfe.py`; model module `code/src/committor/mechanism.py`; figures `code/scripts/gate3b_figures.py`.
- Command: `python code/scripts/gate3b_mfe.py --config code/configs/gate3b_mfe.yaml --run-dir experiments/20261116_gate3b_mfe_v1 --no-progress`
- 1% tests: `20261116_gate3b_mfe_frac0.01_v1` (time estimate), `_v2`, `_v3`, `_v4` (pipeline checks of the fixes).
- Time estimate (v1 benchmark at full size): training 0.5-2.3 h (2000-10000 epochs per run), plus ~10 min for the
  convex quadratic logistic; evaluation minutes.
- Started 2026-11-16 16:06, finished 17:33 (1.45 h of compute), no interruption.
- Outcome: see `docs/reports/gate3b_report_zh.md`. Criterion line in results.json -> evaluate.criterion.
- Figures: `python code/scripts/gate3b_figures.py experiments/20261116_gate3b_mfe_v1 docs/reports/figs/gate3b`.
