Re = 500 against Re = 400; S* = 0.16553, c* = -2.8204 fixed; T_det 10000.

| ray | s_i | s_ii | s_slow (bind) | s_c(500, 64) | t_hit_hi | err(500) | s_slow(400) | s_c(400, 96) | err(400) | shift ln(s_c500/s_c400) | S(s_c)/S* 400 → 500 | c(s_c) − c* 400 → 500 | gain t_max to crossing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G#5 | 0.008184 | 0.0071199 | 0.008184 (i) | 0.0063234 | 175 | +0.258 | 0.010157 | 0.0099188 | +0.024 | -0.450 | 0.974 → 0.755 | -0.022 → -3.972 | 831 |
| G#9 | 0.010665 | 0.0091149 | 0.010665 (i) | 0.0085407 | 230 | +0.222 | 0.013269 | 0.013207 | +0.005 | -0.436 | 0.995 → 0.786 | +1.484 → -1.981 | 745 |
| V#1 | 0.013964 | 0.013583 | 0.013964 (i) | 0.013375 | 1272 | +0.043 | 0.019507 | 0.019568 | -0.003 | -0.381 | 1.121 → 0.958 | +0.126 → -0.672 | 836 |
| V#2 | 0.0096888 | 0.0104 | 0.0104 (ii) | 0.010203 | 1661 | +0.019 | 0.015009 | 0.015056 | -0.003 | -0.389 | 1.243 → 1.053 | +0.133 → -0.887 | 915 |
| V#3 | 0.0030647 | 0.0033165 | 0.0033165 (ii) | 0.0032275 | 1476 | +0.027 | 0.0047945 | 0.0047271 | +0.014 | -0.382 | 1.234 → 1.053 | -0.609 → -1.276 | 909 |
| M#1 | 0.0022767 | 0.0023354 | 0.0023354 (ii) | 0.0022838 | 1307 | +0.022 | 0.0033453 | 0.0032982 | +0.014 | -0.368 | 1.159 → 1.003 | -0.602 → -1.054 | 912 |
| M#2 | 0.0024739 | 0.0024303 | 0.0024739 (i) | 0.0023695 | 1430 | +0.043 | 0.0034618 | 0.0034456 | +0.005 | -0.374 | 1.114 → 0.958 | -0.183 → -1.071 | 834 |
| M#3 | 0.0029268 | 0.002795 | 0.0029268 (i) | 0.0027416 | 1396 | +0.065 | 0.0039714 | 0.0039528 | +0.005 | -0.366 | 1.080 → 0.937 | -0.184 → -0.818 | 834 |
| M#4 | 0.0039085 | 0.0036338 | 0.0039085 (i) | 0.0035879 | 1259 | +0.086 | 0.0051493 | 0.0051654 | -0.003 | -0.364 | 1.057 → 0.918 | +0.123 → -0.541 | 834 |
| M#5 | 0.0065501 | 0.0058429 | 0.0065501 (i) | 0.0057672 | 180 | +0.127 | 0.0082311 | 0.0083082 | -0.009 | -0.365 | 1.015 → 0.880 | +0.386 → -0.574 | 829 |

- n with s_c: 10; not ok: []
- |err(500)| median 0.0542, max 0.2579; |err(400)| median 0.0047, max 0.0237
- shift median -0.3775, range [-0.4501734978435711, -0.3644300108290503]; rule closer than 'no change': 10 / 10
- 96 control V#1: s_c(96) 0.013375, ln(96/64) +0.0000; err_96 +0.0431, shift_96 -0.3805
- 96 control V#2: s_c(96) 0.010203, ln(96/64) +0.0000; err_96 +0.0191, shift_96 -0.3890

Pilot (s = 3 s_c(400), T = 2000, no absorption):

| Re | ray | t first E ≥ E_B | E min after | E median after | fraction below E_B after | non-finite | dt used | Courant max |
|---|---|---|---|---|---|---|---|---|
| 400 | G#9 | 21 | 0.302 | 0.731 | 0.000 | 0 | 0.03125 | 1.55 |
| 400 | V#1 | 38 | 0.000 | 0.004 | 0.668 | 0 | 0.03125 | 1.63 |
| 400 | M#4 | 37 | 0.247 | 0.755 | 0.000 | 0 | 0.03125 | 1.63 |
| 500 | G#9 | 19 | 0.315 | 0.738 | 0.000 | 0 | 0.03125 | 1.67 |
| 500 | V#1 | 29 | 0.345 | 0.764 | 0.000 | 0 | 0.03125 | 1.92 |
| 500 | M#4 | 31 | 0.267 | 0.765 | 0.000 | 0 | 0.03125 | 1.62 |

counts 64: [{'n_cases': 80, 'flagged_per_level': [21, 0], 'nan_per_level': [0, 0], 'courant_max_level0': 2.8487790210587605, 'n_rerun_dt_half': 21, 'n_rerun_dt_quarter': 0, 'n_nan_final_counted_B': 0, 'n_flag_at_last_level': 0}, {'n_cases': 10, 'flagged_per_level': [2, 0], 'nan_per_level': [0, 0], 'courant_max_level0': 2.5731982382481884, 'n_rerun_dt_half': 2, 'n_rerun_dt_quarter': 0, 'n_nan_final_counted_B': 0, 'n_flag_at_last_level': 0}, {'n_cases': 10, 'flagged_per_level': [2, 0], 'nan_per_level': [0, 0], 'courant_max_level0': 2.722697910723764, 'n_rerun_dt_half': 2, 'n_rerun_dt_quarter': 0, 'n_nan_final_counted_B': 0, 'n_flag_at_last_level': 0}, {'n_cases': 10, 'flagged_per_level': [2, 0], 'nan_per_level': [0, 0], 'courant_max_level0': 2.5720167643796934, 'n_rerun_dt_half': 2, 'n_rerun_dt_quarter': 0, 'n_nan_final_counted_B': 0, 'n_flag_at_last_level': 0}, {'n_cases': 10, 'flagged_per_level': [1, 0], 'nan_per_level': [0, 0], 'courant_max_level0': 2.611585813510878, 'n_rerun_dt_half': 1, 'n_rerun_dt_quarter': 0, 'n_nan_final_counted_B': 0, 'n_flag_at_last_level': 0}, {'n_cases': 10, 'flagged_per_level': [0], 'nan_per_level': [0], 'courant_max_level0': 2.3169768026545094, 'n_rerun_dt_half': 0, 'n_rerun_dt_quarter': 0, 'n_nan_final_counted_B': 0, 'n_flag_at_last_level': 0}, {'n_cases': 8, 'flagged_per_level': [1, 0], 'nan_per_level': [0, 0], 'courant_max_level0': 2.6216998136396903, 'n_rerun_dt_half': 1, 'n_rerun_dt_quarter': 0, 'n_nan_final_counted_B': 0, 'n_flag_at_last_level': 0}]
96 files: {'truth_96.json': {'phase': 'done', 'counts': [{'n_cases': 16, 'flagged_per_level': [2, 0], 'nan_per_level': [0, 0], 'courant_max_level0': 2.682768981627691, 'n_rerun_dt_half': 2, 'n_rerun_dt_quarter': 0, 'n_nan_final_counted_B': 0, 'n_flag_at_last_level': 0}, {'n_cases': 2, 'flagged_per_level': [0], 'nan_per_level': [0], 'courant_max_level0': 1.956446087044317, 'n_rerun_dt_half': 0, 'n_rerun_dt_quarter': 0, 'n_nan_final_counted_B': 0, 'n_flag_at_last_level': 0}, {'n_cases': 2, 'flagged_per_level': [0], 'nan_per_level': [0], 'courant_max_level0': 1.983734772871445, 'n_rerun_dt_half': 0, 'n_rerun_dt_quarter': 0, 'n_nan_final_counted_B': 0, 'n_flag_at_last_level': 0}, {'n_cases': 2, 'flagged_per_level': [0], 'nan_per_level': [0], 'courant_max_level0': 2.472007462441062, 'n_rerun_dt_half': 0, 'n_rerun_dt_quarter': 0, 'n_nan_final_counted_B': 0, 'n_flag_at_last_level': 0}, {'n_cases': 2, 'flagged_per_level': [0], 'nan_per_level': [0], 'courant_max_level0': 1.9127119417591427, 'n_rerun_dt_half': 0, 'n_rerun_dt_quarter': 0, 'n_nan_final_counted_B': 0, 'n_flag_at_last_level': 0}, {'n_cases': 2, 'flagged_per_level': [0], 'nan_per_level': [0], 'courant_max_level0': 1.7645665185194466, 'n_rerun_dt_half': 0, 'n_rerun_dt_quarter': 0, 'n_nan_final_counted_B': 0, 'n_flag_at_last_level': 0}, {'n_cases': 1, 'flagged_per_level': [0], 'nan_per_level': [0], 'courant_max_level0': 2.083431048067127, 'n_rerun_dt_half': 0, 'n_rerun_dt_quarter': 0, 'n_nan_final_counted_B': 0, 'n_flag_at_last_level': 0}]}}
