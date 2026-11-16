# Gate 6 (DNS): not run

Gate 6 was not started, because Gate 4a item 5 failed (see `../gate4_dns/README.md`).

The criteria were fixed before any run in `code/configs/gate6_dns.yaml`. That file holds:
- the seed-gain equation;
- the c* definition;
- composite criterion 6.1, with lines 0.10 / 0.25;
- slope-ratio criterion 6.2, with range [0.7, 1.3];
- criterion 6.3.
