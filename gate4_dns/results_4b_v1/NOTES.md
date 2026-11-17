# 20261117_gate4b_dns_v1

Gate 4 continued after the user's 4a decision (2026-11-17 ~07:40): revised item 5 (5a, 5b, 5c; post-hoc) and 4b (pilot
A/B/T_det with the Re ladder, resolution ladder, 5c). Config: `code/configs/gate4_dns.yaml` (history 07:40, e15-e21).
Script: `code/scripts/gate4b_dns.py`. Rule: only solver-correctness failures stop; parameter checks walk down the preset ladders.

    python code/scripts/gate4b_dns.py --config code/configs/gate4_dns.yaml --cfg5 code/configs/gate5_dns.yaml --run-dir experiments/20261117_gate4b_dns_v1

Smoke tests (short times, relaxed valley rule; scratchpad only) exercised every code path before this run.
