#!/bin/bash
# Gate 8 driver (h13): one process at a time, in the user's order 8b -> 8d -> 8c -> analysis (8a, frozen and pushed, is done
# before this script starts and is checked here). Every stage is resumable: re-running this script continues where it stopped.
# A failing stage stops the chain (logged as FAILED) so that no later stage runs on missing data.
cd /srv/hgao0864/project_main
PY=/srv/hgao0864/conda_envs/mfe_committor/bin/python
R=experiments/20261118_gate8_dns_v1
S="$PY code/scripts/gate8_dns.py --config code/configs/gate8_dns.yaml --run-dir $R"
S7="$PY code/scripts/gate7_dns.py --config code/configs/gate7_dns.yaml --run-dir $R/8c_gate7_128"
D=$R/stdout_driver.log
step() { echo "$(date +%H:%M:%S) start $1" >> $D; shift_name=$1; shift; "$@" || { echo "$(date +%H:%M:%S) FAILED $shift_name" >> $D; exit 1; }; echo "$(date +%H:%M:%S) done $shift_name" >> $D; }
[ -f $R/predictions_8a.json ] && [ -f $R/PUSHED_8a.txt ] || { echo "$(date +%H:%M:%S) FAILED: 8a not frozen and pushed" >> $D; exit 1; }
step 8b_truth bash -c "$S --stage truth >> $R/stdout_8b_truth.log 2>&1"
step 8d bash -c "$S --stage d8 >> $R/stdout_8d.log 2>&1"
step 8c_prepare bash -c "$S --stage c8_prepare >> $R/stdout_8c.log 2>&1"
step 8c_truth bash -c "$S7 --stage truth --level 128 --rays W#3,W#7,W#9 >> $R/stdout_8c.log 2>&1"
step 8c_analysis bash -c "$S7 --stage analysis >> $R/stdout_8c.log 2>&1"
step analysis bash -c "$S --stage analysis >> $R/stdout_analysis.log 2>&1"
echo "$(date +%H:%M:%S) all done" >> $D
