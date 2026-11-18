# Gate 8 (DNS): blind test of the third condition (fast channel = oblique-wave route) — in progress

**Frozen predictions (8a), committed before any 8b truth run:** `results_8a/predictions_8a.md` (and `.json`), frozen 2026-11-18 18:13.

- **Third condition:** G = max over 0 ≤ t ≤ 20 of ‖Pw δ(t)‖² / ‖U‖². Here δ is linearised about the laminar state with unit amplitude, the maximum is taken over every step, and G is evaluated at 96×48×48.
  - P* = s_c(W#1, 128)² · G(W#1) = 0.0020953.
  - s_fast = √(P*/G).
- **Prediction:** s_pred = min(s_fast, s_slow), with s_slow = max(s_i, s_ii) (S* and c* unchanged, from Gates 5/6).
- **Route:** "fast" if s_fast < s_slow, otherwise "slow".
- **Blind test (8b):** truth s_c at 96×48×48 for W#2, W#4, W#5, W#6, W#8, W#10, W#12 and R#14, plus X#1 (pure a8, reported alone).
  - Preset verdict: median |ln(s_pred / s_c(96))| ≤ 0.10 holds; ≥ 0.25 does not hold; otherwise partly.
- **Config and specification:** `code/configs/gate8_dns.yaml`. Code: `code/scripts/gate8_dns.py`, `code/src/committor/dns_gate8.py`.
- **Also in `results_8a/`:**
  - `posthoc.json`: the user's post-hoc observations 1–3 checked, and a8, G and the 64 route for the 52 rays (descriptive).
  - `g_all.npz`: G at 64 and 96 for the 52 rays and X#1.
