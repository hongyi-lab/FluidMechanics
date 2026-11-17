# 20261117_gate5_dns_v1

Gate 5 (DNS) with the criteria of `code/configs/gate5_dns.yaml` (unchanged). Parameters from Gate 4b
(`experiments/20261117_gate4b_dns_v2/choice.json`): Re 400, 64x32x32, dt 0.0625, r_A 5e-7, E_B 0.24, T_det 8000 (user
decision ~10:55, post-hoc). User rule ~11:05: undecided is not a transition; s_c must be bracketed by one run to A and one to B;
rays still undecided at 8000 are flagged and counted, no s_c.

Resolution note (from 4b): 64 vs 96 passed by the median rule (0.0010 <= 0.02), but per ray |d ln s_c| =
[0, 0, 0, 0.002, 0.103, 0.437] (F x3, R#1, R#13, W#1): the R and especially W truth at 64x32x32 may carry resolution errors of
10-50 %. Reported, not acted on (preset criterion).

    python code/scripts/gate5_dns.py --config code/configs/gate5_dns.yaml --gate4 code/configs/gate4_dns.yaml \
        --choice experiments/20261117_gate4b_dns_v2/choice.json --run-dir experiments/20261117_gate5_dns_v1 --stage all

## Result (2026-11-18 00:34)

- Truth: F 16 rays (137 min), R+W 36 rays (252 min); all bracketed A -> B, 0 undecided runs, 0 flagged rays, 3 + 93 cases
  re-run at dt/2, 0 NaN. Theory 53 min, second-order check 0.6 min.
- 5.1: (a) yes, (b) yes, (c) no (draw-mean slope 0.1 -> 1 is about the same as 1e-6 -> 1e-2, no flattening).
- 5.2 first gate (S* = 0.1655 fitted on R first 12, cal median 0.011): test median 0.231 (R 0.020, W 0.803); energy-only
  0.704 -> partly holds. W is the key set and fails: 7 of 12 W rays reach the DNS edge with S ~ 0.03 << S* (prediction too high
  by ln 0.75-0.93); the other 5 have S >= 0.065. Controls |a3| only and first order only: no crossing on all W rays (P0 delta = 0).
- Caveat: W#1 truth is 0.0158 at 64 but 0.0244 at 96 (4b); at 96 its error would be ~0.49 instead of 0.93. Resolution of the
  W truth is a confounder; not acted on (preset rule).
