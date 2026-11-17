# 20261117_gate4b_dns_v2

Gate 4b re-run of the resolution ladder and 5c with T_det = 8000 (user decision 2026-11-17 ~10:55, post-hoc; see
`code/configs/gate4_dns.yaml` history). v1 (`20261117_gate4b_dns_v1`, T_det = 2250 from the pilot rule) found that at 64x32x32
the s_c of 5 of 6 check rays was set by runs undecided at T_det (slow streak decay), not by a transition; v1 is kept unchanged.

Copied from v1 (not re-run): 5a at 48x24x24 dt 0.0625 (pass), the item-1 measures on stable runs, the pilot (E_B = 0.24,
r_A = 5e-7, pilot-rule T_det 2250; 8 directions). Added check: 0 undecided runs in the s_c searches at the chosen resolution.

The s_c searches of the rungs run concurrently (sc-only processes, each writing sc_<key>.json), then the driver:

    python code/scripts/gate4b_dns.py --config code/configs/gate4_dns.yaml --cfg5 code/configs/gate5_dns.yaml --run-dir experiments/20261117_gate4b_dns_v2 [--sc-grid 64x32x32 --sc-dt 0.0625]

## Interim (2026-11-17 13:55)

- 48x24x24 dt 0.0625 done 12:00 (57 min): 131 runs, 0 undecided, 1 case re-run at dt/2, 0 NaN.
- 64x32x32 dt 0.0625 done 13:52 (169 min): s_c = [0.00526, 0.00358, 0.00411, 0.0960, 0.1128, 0.0158]; 131 runs, 0 undecided,
  14 cases re-run at dt/2, 0 NaN, 0 rays flagged.
- 48 vs 64: |ln ratio| = [1.42, 1.39, 1.40, 1.71, 2.40, 0.33], median 1.41 > 0.02 -> 48 fails (decision is the driver's).
- 96x48x48 dt 0.0625 and 64x32x32 dt 0.03125 still running.

## Result (2026-11-17 17:10, driver)

- 96x48x48 dt 0.0625 done 17:08 (365 min): s_c = [0.00526, 0.00358, 0.00411, 0.0962, 0.1250, 0.0244]; 130 runs, 0 undecided,
  18 re-run at dt/2, 0 NaN. 64x32x32 dt 0.03125 done 15:31: s_c identical to dt 0.0625 (same bisection decisions).
- Ladder: 48 vs 64 median 1.41 -> next; 64 vs 96 median 0.0010 <= 0.02 -> 64x32x32. Per ray 64 vs 96:
  [0, 0, 0, 0.002, 0.103, 0.437] -> R#13 and W#1 are not converged at 64 (median rule passes; reported, not acted on).
- 5c: median 0.0000 <= 0.01 pass (resolution of the check limited by the bisection tolerance 1e-3). 5a at 64: 5.6e-8 pass.
- Zero-undecided check at 64: 0 and 0 -> pass. choice.json: Re 400, 64x32x32, dt 0.0625, r_A 5e-7, E_B 0.24, T_det 8000.
- Ladder time 226 min > 60-min cap: reported only (e22).
