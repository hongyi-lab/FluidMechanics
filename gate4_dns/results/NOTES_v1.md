# Gate 4a, first run (2026-11-17 00:22)

- Config: `code/configs/gate4_dns.yaml` (criteria as written before this run). Code: `code/src/committor/dns.py` (first version),
  `code/scripts/gate4_dns.py --stage 4a`. Grid 48x24x24.
- Result: items 1, 2, 4, 6 pass; item 3 fails for a1 - 1 (rel. err 1.8e-5) and a9 (8.2e-4) (a3 and both streak components pass);
  item 5 fails (no dt passes). The run stopped at the final log line (JSON of numpy bools); results.json holds all numbers
  but no "criteria" block.
- Item 3 cause: the nonlinear term was evaluated as P[(U + u') x (w_U + w')]; the O(1) product U x w_U is removed by the
  projection only to ~1e-16 absolute, which is large against a 1e-8 perturbation (a9 decays to 4e-11 by t = 100). Fixed in
  v2 by the algebraically identical split form (implementation issue, not a change of method).
- Item 5: see v2.
