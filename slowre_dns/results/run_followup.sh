#!/bin/bash
# Follow-up (16:20): the 96 controls were stopped to free the GPU for the main line; they run here after the driver ends.
cd /srv/hgao0864/project_main
P=/srv/hgao0864/conda_envs/mfe_committor/bin/python
RUN=experiments/20261119_slowre500_dns_v1
S="$P code/scripts/slowre_dns.py --config code/configs/slowre_dns.yaml --run-dir $RUN"
d() { echo "$(date +%H:%M:%S) $*" >> $RUN/stdout_driver.log; }
until grep -q "all done" $RUN/stdout_driver.log; do sleep 60; done
while ps -eo args | grep -v grep | grep -q "slowre_dns.py.*--stage pilot"; do sleep 60; done
$S --stage analysis > $RUN/stdout_analysis_main.log 2>&1; d "followup: analysis (main line, pilot) rc $?"
d "followup: start truth96 V#1,V#2"
$S --stage truth --level 96 --rays V#1,V#2 > $RUN/stdout_truth96.log 2>&1; d "followup: done truth96 rc $?"
$S --stage analysis > $RUN/stdout_analysis_final.log 2>&1; d "followup: analysis (final) rc $?"
d "followup all done"
