#!/bin/bash
# Gate 7 driver (h14): four process chains on the one GPU; then analysis. Each stage is resumable (re-run this script).
cd /srv/hgao0864/project_main
PY=/srv/hgao0864/conda_envs/mfe_committor/bin/python
S="$PY code/scripts/gate7_dns.py --config code/configs/gate7_dns.yaml --run-dir experiments/20261118_gate7_dns_v1"
R=experiments/20261118_gate7_dns_v1
( $S --stage truth --level 96 --rays R#22,W#11 >> $R/stdout_P1.log 2>&1 && \
  $S --stage truth --level 128 --rays R#22,W#11 >> $R/stdout_P1.log 2>&1 && \
  $S --stage diag --level 128 --rays R#22,W#11 >> $R/stdout_P1.log 2>&1 ) &
( $S --stage truth --level 96 --rays W#3,W#7,W#9,R#15 >> $R/stdout_P2.log 2>&1 && \
  $S --stage diag --level 96 --rays W#3,W#7,W#9,R#15,R#13 >> $R/stdout_P2.log 2>&1 ) &
( $S --stage truth --level 128 --rays W#1 >> $R/stdout_P3.log 2>&1 && \
  $S --stage diag --level 128 --rays W#1 >> $R/stdout_P3.log 2>&1 ) &
( $S --stage theory --level 64 >> $R/stdout_P4.log 2>&1 && \
  $S --stage theory --level 96 >> $R/stdout_P4.log 2>&1 && \
  $S --stage theory --level 128 --rays W#1,W#3,W#7,W#9,R#22,W#11 >> $R/stdout_P4.log 2>&1 && \
  $S --stage diag --level 64 --rays W#1,W#3,W#7,W#9,R#22,R#13,R#15,W#11 >> $R/stdout_P4.log 2>&1 && \
  $S --stage scan --level 64 >> $R/stdout_P4.log 2>&1 && \
  $S --stage scan --level 96 >> $R/stdout_P4.log 2>&1 ) &
wait
$S --stage analysis >> $R/stdout_analysis.log 2>&1
echo "$(date +%H:%M:%S) all done" >> $R/stdout_driver.log
