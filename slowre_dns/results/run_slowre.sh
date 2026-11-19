#!/bin/bash
# Driver: theory (Re 500) and a Re 400 theory check; then truth 64, truth 96 (control), gain and pilot at the same time; then analysis.
cd /srv/hgao0864/project_main
P=/srv/hgao0864/conda_envs/mfe_committor/bin/python
RUN=experiments/20261119_slowre500_dns_v1
CFG=code/configs/slowre_dns.yaml
SC=/srv/hgao0864/.claude_tmp/claude-2042693/-srv-hgao0864/3e584786-dffb-461a-bab9-58ca0a87ddcd/scratchpad/slowre_check400
S="$P code/scripts/slowre_dns.py --config $CFG --run-dir $RUN"
d() { echo "$(date +%H:%M:%S) $*" >> $RUN/stdout_driver.log; }
d "start theory"
$P code/scripts/slowre_dns.py --config $SC/check400.yaml --run-dir $SC/run --stage theory > $SC/stdout_theory400.log 2>&1 &
$S --stage theory > $RUN/stdout_theory.log 2>&1 || { d "theory FAILED"; exit 1; }
d "done theory"
$S --stage truth --level 64 > $RUN/stdout_truth64.log 2>&1 & p1=$!
$S --stage truth --level 96 > $RUN/stdout_truth96.log 2>&1 & p2=$!
$S --stage gain > $RUN/stdout_gain.log 2>&1 & p3=$!
$S --stage pilot > $RUN/stdout_pilot.log 2>&1 & p4=$!
wait $p1; d "done truth64 rc $?"
wait $p3; d "done gain rc $?"
wait $p4; d "done pilot rc $?"
$S --stage analysis > $RUN/stdout_analysis_pre96.log 2>&1; d "done analysis (before 96) rc $?"
wait $p2; d "done truth96 rc $?"
wait
$S --stage analysis > $RUN/stdout_analysis.log 2>&1; d "done analysis rc $?"
d "all done"
