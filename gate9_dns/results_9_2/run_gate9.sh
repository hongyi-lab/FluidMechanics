#!/bin/bash
# Gate 9 driver: 9.3 truth at 96 by class G, O, M, then V and 128 subject to the budget (h16), then 9.4 analysis.
# One process at a time. Every stage resumes from its checkpoint when this script is re-run. A failing stage stops the chain.
cd /srv/hgao0864/project_main
PY=/srv/hgao0864/conda_envs/mfe_committor/bin/python
R=experiments/20261119_gate9_dns_v1
S="$PY code/scripts/gate9_dns.py --config code/configs/gate9_dns.yaml --run-dir $R"
D=$R/stdout_driver.log
step() { echo "$(date +%H:%M:%S) start $1" >> $D; n=$1; shift; "$@" || { echo "$(date +%H:%M:%S) FAILED $n" >> $D; exit 1; }; echo "$(date +%H:%M:%S) done $n" >> $D; }
gated() {   # gated <name> <budget key> <command...>: run unless the budget stage says cut (exit 3)
    n=$1; key=$2; shift 2
    $S --stage budget --before $key >> $R/stdout_budget.log 2>&1; rc=$?
    if [ $rc -eq 3 ]; then echo "$(date +%H:%M:%S) CUT $n (budget, h16)" >> $D; return 0; fi
    if [ $rc -ne 0 ]; then echo "$(date +%H:%M:%S) FAILED budget before $n" >> $D; exit 1; fi
    step $n "$@"
}
[ -f $R/predictions_9.json ] && [ -f $R/PUSHED_9.txt ] || { echo "$(date +%H:%M:%S) FAILED: 9.2 not frozen and pushed" >> $D; exit 1; }
step truth_G bash -c "$S --stage truth --cls G >> $R/stdout_truth_G.log 2>&1"
step truth_O bash -c "$S --stage truth --cls O >> $R/stdout_truth_O.log 2>&1"
step truth_M bash -c "$S --stage truth --cls M >> $R/stdout_truth_M.log 2>&1"
gated truth_V V bash -c "$S --stage truth --cls V >> $R/stdout_truth_V.log 2>&1"
gated truth128 128 bash -c "$S --stage truth128 >> $R/stdout_truth128.log 2>&1"
step analysis bash -c "$S --stage analysis >> $R/stdout_analysis.log 2>&1"
echo "$(date +%H:%M:%S) all done" >> $D
