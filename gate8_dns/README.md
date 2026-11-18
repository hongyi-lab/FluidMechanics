# Gate 8 (DNS): blind test of the third condition (fast channel = oblique-wave route) (2026-11-18, unattended)

**Outcome (preset criteria):**

- **8b, third condition: partly holds.** The median |ln(s_pred / s_c(96))| over the 8 blind rays is 0.172. "Holds" needs ≤ 0.10; "does not hold" is ≥ 0.25.
  - Controls on the same 8 rays: the two-condition composite gives 0.286, energy only gives 1.555.
  - Routes: 6 of 8 right. W#6 and W#12 were predicted fast; the truth is slow.
  - X#1 (pure a8, reported alone): error −0.022, route right.
- **8c, Gate 7 re-judged with W#3, W#7, W#9 at 128 (spec unchanged): real route.** All 5 decision rays qualify and the controls pass.

**Descriptive (not part of the verdict):**

- The errors sit on the three G ≈ 1 rays predicted fast (W#6 −0.87, W#12 −0.46, W#8 −0.29). On these rays the oblique waves do not grow.
- Blind rays with wave growth (G ≥ 4) are within 0.12.
- The non-blind pure-wave rays W#3, W#7 and W#9 are within 0.016 at 96 and within 0.007 at 128.
- 8d: the fast rays W#1, W#3 and X#1 first make kz = 2γ streaks, which hold 97–100% of the streak energy at t = 20. This is the oblique-wave signature. The slow W#11 stays at kz = γ.

**Frozen predictions (8a):** `results_8a/predictions_8a.md` (and `.json`). They were frozen 2026-11-18 18:13:23 and committed (40cd95a) before any 8b truth run.

## Method

- **Third condition:** G = max over 0 ≤ t ≤ 20 of ‖Pw δ(t)‖² / ‖U‖².
  - δ is linearised about the laminar state, with unit amplitude. The maximum is taken over every step, at 96×48×48.
  - P* = s_c(W#1, 128)² · G(W#1) = 0.0020953.
  - s_fast = √(P*/G).
- **Prediction:** s_pred = min(s_fast, s_slow), with s_slow = max(s_i, s_ii). S* and c* are unchanged from Gates 5/6.
- **Route:** "fast" if s_fast < s_slow, otherwise "slow".
- **8b, blind truth:** s_c at 96×48×48 for W#2, W#4, W#5, W#6, W#8, W#10, W#12 and R#14, plus X#1. The spec is that of Gate 7 (7a).
- **8c:** the unchanged Gate 7 script, run in `results/8c_gate7_128/`.
- **8d (record only):**
  - streak energy by kz (γ, 2γ, ≥ 3γ), theory and DNS;
  - the linear wave against the DNS kx = α energy at s = s_c(96).
- **Counts:** 0 undecided, 0 NaN.

## Files

- **Full report (Chinese):** `report_zh.md`. Its paths map as follows:
  - `docs/reports/figs/gate8/` → this folder;
  - `experiments/20261118_gate8_dns_v1/` → `results/`.
- **Figures (white background):**
  - `fig9_gate8_blind.png`: the 8b blind test (prediction vs truth, errors with controls, routes).
  - `fig10_gate8_posthoc.png`: the post-hoc observations 1–3.
  - `fig11_gate8_8d.png`: 8d.
  - `fig12_gate7_levels_8c.png`: Gate 7 levels after 8c.
- **Run outputs:** `results/`:
  - `results.json`: the verdict and all per-ray quantities.
  - `truth_96_blind.json`: every 8b case.
  - `d8_*.npz`: the 8d data.
  - `8c_gate7_128/`: the 8c run.
  - logs, manifests and `NOTES.md`.
- **Config, written before any run:** `code/configs/gate8_dns.yaml`.
- **Code:**
  - `code/scripts/gate8_dns.py`
  - `code/src/committor/dns_gate8.py`
  - `code/scripts/gate8_figures.py`
