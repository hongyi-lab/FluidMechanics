# Slow route at Re = 500: quick test (2026-11-19)

User (after Gate 9): "我觉得其它的不用管了，我们专门看慢路线，要不你换个雷诺数测测试试？先换一个能快速验证的跑一下给我看看".
Descriptive, no verdict. Design and choices r1–r8: `code/configs/slowre_dns.yaml` (all mine; the user left the design to me).

- Question: with S* = 0.16553 and c* = −2.8204 fixed at their Re = 400 values and the theory recomputed at Re = 500, does
  s_slow = max(s_i, s_ii) follow s_c? Compared with "no change with Re" (the error of predicting s_c(500) = s_c(400)).
- Rays: the 10 Gate 9 rays predicted slow and truly slow at Re = 400 (G#5, G#9, V#1–V#3, M#1–M#5).
- Truth at 64x32x32 (T_det scaled to 10000), V#1 also at 96 (resolution control). Pilot: E after the first E_B crossing.
- Check: the same code at Re = 400 recomputes s_i for the 10 rays (scratch run), to compare with Gate 9.2.

## Log

- **16:01:56, start** (driver `run_slowre.sh`; script snapshot `script_snapshot_slowre_dns.py`).
- **16:14:03, theory done** (12.1 min). s_i at Re = 500: G#5 0.008184, G#9 0.010665, V#1 0.013964, V#2 0.0096888, V#3 0.0030647,
  M#1 0.0022767, M#2 0.0024739, M#3 0.0029268, M#4 0.0039085, M#5 0.0065501. The driver then started truth 64, truth 96 (V#1),
  gain and pilot at the same time.
- **16:15, code review** (workflow, 2 reviewers + 2 re-checkers). Confirmed and acted on:
  - The 96 control should include V#2 (largest |k|² of the 10, 43.4), not only V#1 (27). Config r6 corrected.
  - The pilot ran with run_nonlinear (no Courant/NaN ladder, Courant sampled every 10 t; a NaN would read as "E stays above E_B").
    Stopped at 16:17 and re-run with run_det (ladder) into pilot_det.npz; the summary now counts non-finite samples.
  - Analysis only: gain t_at_max flag over the amplitudes up to the crossing; 96 controls give same-level shift_96 / err_96
    (s_c(500) at 64 vs s_c(400) at 96 otherwise); per-ray dicts for the implied constants. Snapshot `script_snapshot_slowre_dns_rev1618.py`.
- **16:20, both 96 controls stopped** (5 processes on one GPU: truth 64 started at 5.6 s per time unit, ~15 h). They run after the
  driver ends (`run_followup.sh`, V#1 and V#2 in one batch into truth_96.json). No result is lost: the killed 96 runs had not
  finished their first grid round.
- **16:22, code check at Re = 400** (scratch run, same script with Re = 400, T_det 8000): s_i of the 10 rays equals Gate 9.2
  bit for bit (Δln 0). At Re = 500, s_i is lower by ln 0.8 = −0.223 on V and M rays and −0.216/−0.218 on G#5/G#9: S(s) is to
  leading order a function of s·Re (lift-up), so with S* fixed s_i ∝ 1/Re.
- **18:52 truth 64 done** (158 min; 10 rays, all "ok", one switch, no undecided, no NaN; rel width ≤ 0.0063; ladder: grid 21
  cases at dt/2, bisection 1–2 per round, none at dt/4). **18:59 gain done** (165 min; all s_ii "ok", gain t_max up to the
  crossing 745–915 < 1400). **19:55 96 control done** (V#1, V#2 in one batch): same fates at every point as at 64, so the same
  brackets (64 and 96 agree within the bisection width); t_hit differ (the run is a real 96 run).
- **Results (descriptive):**
  - 7 rays stay slow at Re 500 (V#1–V#3, M#1–M#4; t_hit_hi 1259–1661): shift ln(s_c500/s_c400) −0.364 to −0.389, i.e.
    s_c ∝ Re^(−1.63 to −1.74) between 400 and 500.
  - 3 rays switch to the fast route at Re 500 (G#5 175, G#9 230, M#5 180; at 400 they were 1212, 922, 1358): shift −0.450,
    −0.436, −0.365; the slow rule over-predicts them by +0.26, +0.22, +0.13.
  - Condition i (S* fixed): s_i scales exactly as 1/Re (−0.223), the truth as ~Re^−1.7, so s_i is now too high where it binds:
    err_i on the 7 slow rays −0.052 to +0.086.
  - Condition ii (c* fixed): s_ii scales −0.349 to −0.369, close to the truth; err_ii on the 7 slow rays +0.013 to +0.027
    (at Re 400 −0.003 to +0.014).
  - Rule err(500) on the 7 slow rays +0.019 to +0.086 (median +0.043); at 400 −0.003 to +0.014. "No change with Re" errs by 0.36–0.39.
  - Post hoc (not a preset rule): s_ii alone would give +0.013 to +0.027 on the 7 slow rays.
  - Pilot: at Re 500 all three cases stay above E_B after the first crossing (E median 0.74–0.77, minimum 0.27–0.35); at Re 400
    V#1 drops back (median 0.004 after the crossing).
- Figure docs/reports/figs/slowre/fig17_slowre500.png; table tables.md. Not pushed (the user asked to see it first).
