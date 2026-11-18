# Overview traces v1 (descriptive, for the project overview figure)

- **Purpose:** energy traces E(t) for panel 1 of the one-page overview figure (user request 2026-11-18 ~13:50: an overview a
  non-specialist physicist or aero engineer can read). No criterion, nothing decided from it.
- **Script:** `code/scripts/overview_traces.py` (uses the Gate 7 context: same grid, Re, dt, 5b re-run rule).
- **Cases:** 64×32×32, dt 0.0625, Re 400, no absorption, t ≤ 4000, E every 2.0. Two rays, the final Gate 5 bracket of each
  (lo went to A, hi went to B in Gate 5; relative width ~1e-3):
  - W#1 (fast channel): s = 0.0157556 / 0.0157666; Gate 5 t_hit 1483 / 219.
  - R#15 (slow control): s = 0.0094077 / 0.0094137; Gate 5 t_hit 3441 / 1741.
- **Run:** started 13:55 on the shared GPU next to Gate 7 (adds a few minutes to Gate 7, which is not otherwise touched).
- **Output:** `traces_64.npz` (rec = absolute energies of the Gate 7 7c partition, column 0 = E; t; dt_used; courant_max),
  `traces_64.json`, `stdout.log`.
- **Figure:** `code/scripts/overview_figure.py` → `docs/reports/figs/overview/overview_en.{png,pdf}`. It is in English, as the user
  asked. Re-rendered after Gate 7 finished (17:25) with the final result; printout in `overview_figure_printout.json`.
- **Result (14:35, 39.9 min; the hi cases re-run at dt/2 by the 5b rule):** W#1 lo → A at 1484, hi → B at 220;
  R#15 lo → A at 3442, hi → B at 1742. Same as Gate 5.
