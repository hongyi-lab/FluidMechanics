# Slow route at Re = 500: quick test (2026-11-19)

After Gate 9 the user asked to look only at the slow route and to try another Reynolds number. This quick test does that. It is descriptive and gives no verdict.

## Question

Does the slow rule still predict s_c when Re changes from 400 to 500?

- The rule is s_slow = max(s_i, s_ii).
- The two constants are kept at their Re = 400 values: S* = 0.16553 and c* = −2.8204.
- The theory is recomputed at Re = 500. This covers the streak quantity S(s), behind s_i, and the seed gain c(s), behind s_ii.
- The baseline is "no change with Re": predict s_c(500) = s_c(400).

## Setup

- **Rays:** the 10 Gate 9 rays that were predicted slow and truly slow at Re = 400: G#5, G#9, V#1–V#3 and M#1–M#5. None of them was used to fit S* or c*.
- **Truth:** at 64×32×32, with the Gate 9 search (local grid, then bisection to a relative width of 0.01).
  - T_det is scaled with Re, from 8000 to 10000. r_A and E_B are unchanged.
  - Resolution control: V#1 and V#2 were also run at 96×48×48.
- **Code check:** the same script run at Re = 400 reproduces the Gate 9 s_i of all 10 rays bit for bit.
- **Choices r1–r8:** listed in `code/configs/slowre_dns.yaml`.

## Results

| Ray | s_c(400) | s_c(500) | ln(s_c500/s_c400) | Route at 500 (t_hit_hi) | Rule err at 400 | Rule err at 500 | s_i alone | s_ii alone |
|---|---|---|---|---|---|---|---|---|
| V#1 | 0.019568 | 0.013375 | −0.381 | slow (1272) | −0.003 | +0.043 | +0.043 | +0.015 |
| V#2 | 0.015056 | 0.010203 | −0.389 | slow (1661) | −0.003 | +0.019 | −0.052 | +0.019 |
| V#3 | 0.0047271 | 0.0032275 | −0.382 | slow (1476) | +0.014 | +0.027 | −0.052 | +0.027 |
| M#1 | 0.0032982 | 0.0022838 | −0.368 | slow (1307) | +0.014 | +0.022 | −0.003 | +0.022 |
| M#2 | 0.0034456 | 0.0023695 | −0.374 | slow (1430) | +0.005 | +0.043 | +0.043 | +0.025 |
| M#3 | 0.0039528 | 0.0027416 | −0.366 | slow (1396) | +0.005 | +0.065 | +0.065 | +0.019 |
| M#4 | 0.0051654 | 0.0035879 | −0.364 | slow (1259) | −0.003 | +0.086 | +0.086 | +0.013 |
| G#5 | 0.0099188 | 0.0063234 | −0.450 | **fast** (175) | +0.024 | +0.258 | +0.258 | +0.119 |
| G#9 | 0.013207 | 0.0085407 | −0.436 | **fast** (230) | +0.005 | +0.222 | +0.222 | +0.065 |
| M#5 | 0.0083082 | 0.0057672 | −0.365 | **fast** (180) | −0.009 | +0.127 | +0.127 | +0.013 |

err = ln(prediction / s_c). The Re = 400 truth is at 96 (Gate 9); the Re = 500 truth is at 64.

- **The threshold falls faster than 1/Re.**
  - On the 7 rays that stay slow, ln(s_c500/s_c400) is −0.364 to −0.389. That is s_c ∝ Re^(−1.63 to −1.74) between 400 and 500.
  - "No change with Re" is therefore off by 0.36–0.39.
- **The rule as it stands over-predicts at Re = 500.**
  - On the 7 slow rays the error is +0.019 to +0.086 (median +0.043).
  - All 7 lie outside the slow 80% interval (±0.0136). At Re = 400 the same 7 rays were within 0.014.
- **The two conditions behave differently.**
  - s_i (S* fixed) falls exactly as 1/Re (−0.223), so it is now too high. It is the larger of the two on V#1 and M#2–M#4, and lifts the rule there.
  - s_ii (c* fixed) falls by −0.349 to −0.369, close to the truth. On the 7 slow rays it errs by +0.013 to +0.027.
  - Using s_ii alone is a post-hoc observation, not a preset rule.
- **Three rays switch to the fast route at Re = 500.** G#5, G#9 and M#5 have t_hit_hi 175–230; at Re = 400 these were 922–1358. The slow rule over-predicts them by +0.13 to +0.26.
- **Checks:**
  - **Resolution:** V#1 and V#2 at 96 have the same fate as at 64 at every grid and bisection point, so the two levels give the same bracket.
  - **Truth runs:** no undecided case and no NaN.
  - **Pilot:** at Re = 500, cases that cross E_B = 0.24 stay turbulent (median E 0.74–0.77). At Re = 400 one of the three dropped back.

## Files

- **Figure:** `fig17_slowre500.png` and `.pdf`.
  - (a) prediction against truth at both Re.
  - (b) error of the rule and of "no change".
  - (c) error of each condition alone. Shaded rays are fast at Re = 500.
- **`results/`:** the run directory.
  - `results.json` and `tables.md` hold the results;
  - `NOTES.md` is the run log, including the code review and the changes made during the run;
  - also the truth, theory and gain files, and the logs.
- **Code:** `code/scripts/slowre_dns.py`, `code/scripts/slowre_figures.py` and `code/configs/slowre_dns.yaml`.
