# Post-hoc check of the fast/slow observation (Gate 5/6 data), descriptive

- **What:** the user's observation of 2026-11-18, checked against the run data. No criterion, no refit.
- **Inputs:** `experiments/20261117_gate5_dns_v1` (truth, analysis) and `experiments/20261117_gate6_dns_v1` (composite errors).
- **Recomputed:** only MFE ODE quantities (Gate 2 routine).
- **Script:** `code/scripts/gate56_posthoc_fast.py` (snapshot `script_snapshot.py`) and `mfe_offsets.py`.
- **Outputs:** `results.json`, `mfe_offsets.json`, `stdout.log`.

## Result (09:56)

- **Grouping by t_hit_hi (64, Gate 5 truth), line at 600:**
  - 14 fast rays: R#6, 12, 13, 14, 22 and W#1, 2, 3, 5, 6, 7, 8, 9, 10. Their t_hit_hi is at most 297.
  - The other 38 are slow, with t_hit_hi of at least 1218.
- **Error.** Composite for every ray, except (i) for the R calibration rays.
  - Fast rays: smallest |error| is 0.238 (R#14), median 0.87.
  - Slow rays: median 0.0098, largest 0.224 (W#4).
  - None of the slow rays reaches 0.24. Only R#14 among the fast rays is below 0.24 (0.2376).
- **MFE, the same 12 W rays: time to B just above the MFE edge.**
  - At the final upper bracket after 24 bisections (about 1e-9 above s_c), it is 3619–4282.
  - At s = 1.001 s_c (E_B 0.29), it is 884–1189. This is the user's 900–1200, and comparable to the DNS bracket, which is
    within 5e-4 of s_c.
  - At s = 1.001 s_c, every one of the 52 rays takes 884–1189 in MFE, including the 14 rays that are fast in the DNS. The MFE
    has no fast channel.
- **MFE criterion with frozen MFE constants on the MFE W thresholds** (Gate 5 f7 code path, DNS s-grid):
  - With S* = 0.12333 (Gate 3d): composite median |ln| 0.075, max 0.62. Condition (i) alone gives 0.061, (ii) alone 0.239.
    (i) binds on 10 of 12 rays.
  - With S* = 0.1236 (the value in the user's message): composite 0.075, (i) 0.063.
