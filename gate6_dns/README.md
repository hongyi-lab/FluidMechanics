# Gate 6 (DNS): second gate (seed × gain) and the composite max(s_i, s_ii) (2026-11-18, unattended)

**Outcome:**
- **6.1 holds:** on 14 F rays the composite median error is 0.0086. Each gate alone gives (i) 0.116 and (ii) 0.0097.
- **6.2 holds:** the constant-free slope ratio is 0.962.
- **6.3 partly holds:** the composite median error is 0.231. Split by set (descriptive): R 0.013 and W 0.803.

On W the second gate adds nothing. Its error is 0.05–0.15 below that of the first gate, with the same sign. Seven W rays transition with c ≈ −7 at the edge, while c* = −2.82.

- **Report (Chinese):** `../gate5_dns/report_zh.md`, section 3.
- **Run outputs:** `results/`.
- **Config:** `code/configs/gate6_dns.yaml`. Its criteria are unchanged; its implementation notes are flagged in the report.

## Details
- **c\*:** −2.82, the mean of c at the edge of the two ε = 1e-6 F rays (−2.75 and −2.89).
- **Amplitude grid:** ln s_c ± 0.9 in 13 points. No maximum of ‖y‖² fell at t = 1500.
- **Courant number of the advecting field M:** recorded only; maximum 1.37 on F and 2.81 on R/W.

## Figure
`fig4_gate6_seed_gain.png`: (a) c(s) on the F rays; (b) 6.1 errors; (c) 6.2 slopes.
