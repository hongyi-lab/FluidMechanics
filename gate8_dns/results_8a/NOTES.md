# Gate 8 (DNS) v1: blind test of the third condition (fast channel = oblique-wave route); 8d records; 8c completes Gate 7

- **Config:** `code/configs/gate8_dns.yaml`, written before any run; snapshot in `config_source.yaml`. The criteria and
  constants are the user's:
  - S* = 0.1655 (Gate 5)
  - c* = −2.82 (Gate 6)
  - P* from the W#1 128 truth (Gate 7)
- **Code:** `code/scripts/gate8_dns.py` and `code/src/committor/dns_gate8.py` (snapshots in this folder). It reuses the Gate 7
  `find_sc_local` and the Gate 5/6 theory code; dns.py, dns_theory.py and dns_local.py are not changed.
- **Order (h13):** one process at a time.
  1. posthoc
  2. predict (8a): frozen, then pushed to GitHub
  3. truth (8b)
  4. d8 (8d)
  5. c8_prepare, then the unchanged Gate 7 script in `8c_gate7_128/` (8c)
  6. analysis

## Log

- **~18:40, function checks before any run** (64 grid, short runs; scratch, not kept):
  - `theory_kz` bins sum to `run_delta_q` aa, ab, bb within 1e-16.
  - `gain_alpha` reproduces `run_gain` exactly (diff 0).
  - `diag_kz` reproduces the Gate 7 `run_diag` parts exactly (diff 0); the streak and roll bins sum to the totals within 4e-16.
  - The linear wave y stays at kx = α (M has kx = 0 only), so ‖y_α‖² = ‖y‖².
