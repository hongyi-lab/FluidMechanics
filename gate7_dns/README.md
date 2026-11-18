# Gate 7 (DNS): fast channel, numerical artefact or real route? (2026-11-18, unattended)

**Update (Gate 8c, 2026-11-18 23:04):** with W#3, W#7 and W#9 run at 128 (spec unchanged), the same preset verdict gives **real route** (5 of 5 decision rays; controls pass). See `../gate8_dns/` (report section 5).

**Outcome as first reported (preset criteria): undetermined.** The controls pass. "Real route" holds for 2 of the 5 decision rays (W#1, R#22); 4 are needed. "Numerical artefact" holds for none.

- **Full report (Chinese):** `report_zh.md`. Its paths map as follows:
  - `docs/reports/figs/gate7/` → this folder;
  - `experiments/20261118_gate7_dns_v1/` → `results/`;
  - `docs/reports/figs/overview/` → `../overview/`.
- **Config, written before any run:** `code/configs/gate7_dns.yaml`. The criteria and constants are the user's: S* = 0.1655 (Gate 5) and c* = −2.82 (Gate 6).
- **Code:**
  - `code/scripts/gate7_dns.py`
  - `code/src/committor/dns_local.py`
  - `code/scripts/gate7_figures.py`
- **Levels:**

  | Level | Grid | dt |
  |---|---|---|
  | 64 (the Gate 5 choice) | 64×32×32 | 0.0625 |
  | 96 | 96×48×48 | 0.0625 |
  | 128 | 128×64×64 | 0.03125 |

  The 1% timing put the full 128 set over the 12 h budget. So 128 was reduced to W#1, R#22 and W#11 (user rule). The run itself took 7.4 h.
- **Counts:** 0 undecided runs, 0 NaN, 0 flagged rays.

## Verdict table (each decision ray at its highest level)

| Ray | Top two levels | Δln s_c | t_hit_hi | err(i) | Real route? |
|---|---|---|---|---|---|
| W#1 | 96, 128 | −0.014 | 120 | +0.506 | yes |
| W#3 | 64, 96 | +0.423 | 130 | +0.456 | no (Δln) |
| W#7 | 64, 96 | +0.441 | 118 | +0.456 | no (Δln) |
| W#9 | 64, 96 | +0.435 | 121 | +0.416 | no (Δln) |
| R#22 | 96, 128 | −0.005 | 109 | +0.557 | yes |

**Controls** (|Δln s_c| ≤ 0.02 required):

| Control | 64 → 96 | 96 → 128 |
|---|---|---|
| R#15 | +0.0031 | not run |
| W#11 | +0.0031 | −0.0047 |

## What the data show (descriptive, no verdict)

- **64 → 96:** the true threshold s_c of all 5 decision rays moves up 41–55%.
- **96 → 128:** the two rays run at 128 move only −1.4% (W#1) and −0.5% (R#22).
- **Error stays large:** the first-gate prediction s_i does not depend on resolution (ratio 0.9998). At the highest level the error is still ln 0.42–0.56, so the prediction is 1.5–1.75× the truth.
- **The fast channel stays:** t_hit_hi is 109–130 at the highest level.
- **R#13 (descriptive ray)** is no longer fast at 96: t_hit_hi goes from 170 to 1506.
- **What would settle it:** W#3, W#7 and W#9 already meet the other two "real route" conditions. So the verdict now depends only on whether their s_c moves ≤ 5% from 96 to 128. That run (~4 h) has not been done; it waits for the user's decision.

## Figures

- `fig5_posthoc_fast_slow.png`: the post-hoc fast/slow observation on the Gate 5/6 data (report section 8 in `../gate5_dns/report_zh.md`).
- `fig6_gate7_levels.png`:
  - (a) s_c vs resolution;
  - (b) t_hit_hi;
  - (c) err(i) at the same resolution.
- `fig7_gate7_diag_hi.png` and `fig7_gate7_diag_lo.png`: 7c, the energy parts and the high-wavenumber fractions, 64 vs the highest level.
- `fig8_gate7_W1_scan.png`: the W#1 fine scan. The fate is monotone at both 64 and 96, with one switch each.
