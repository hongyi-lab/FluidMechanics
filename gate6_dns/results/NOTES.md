# 20261117_gate6_dns_v1

Gate 6 (DNS): seed x gain about the second-order mean flow, the second gate and the composite max(s_i, s_ii). Criteria of
`code/configs/gate6_dns.yaml` (unchanged). Rays, truth s_c and the first gate from `experiments/20261117_gate5_dns_v1`
(64x32x32, dt 0.0625, T_det 8000). Rays without a truth s_c (flagged undecided / unresolved) are skipped and counted.
Started 2026-11-18 ~00:35.

    python code/scripts/gate6_dns.py --config code/configs/gate6_dns.yaml --cfg5 code/configs/gate5_dns.yaml \
        --gate4 code/configs/gate4_dns.yaml --gate5-run experiments/20261117_gate5_dns_v1 --run-dir experiments/20261117_gate6_dns_v1 --stage all

## Result (2026-11-18 05:39)

- gain_F 124 min (16 rays x 13 amplitudes), gain_RW 180 min (24 rays); no ray skipped; 0 maxima at t = 1500; Courant of M max
  1.37 (F), 2.81 (R/W), recorded only.
- c* = -2.82 (mean of c at the edge of the two eps = 1e-6 F rays: -2.75, -2.89).
- 6.1 (14 F rays): composite median 0.0086 -> holds; (i) alone 0.116, (ii) alone 0.0097; (ii) binds 11/14, (i) 3/14 (eps >= 1).
- 6.2: predicted slope -0.0407, measured -0.0423, ratio 0.962 -> holds (draws 0.957, 0.968).
- 6.3 (R last 12 + W 12): composite 0.231 -> partly holds; R 0.013, W 0.803 (descriptive split). On W, err(ii) is 0.05-0.15
  below err(i) with the same sign: the second gate adds nothing there; 7 W rays have c ~ -7 at the edge (c* = -2.82).
- Report: docs/reports/gate456_dns_report_zh.md.
