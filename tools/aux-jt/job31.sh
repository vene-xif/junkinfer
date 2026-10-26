#!/bin/bash
# job 31: verify-step cost (1..4 tokens, TP2) with KVarN on / off
while ! grep -q JOB30_DONE ~/jt/job30.log 2>/dev/null; do sleep 30; done
cd ~/jt
for v in 1 0; do echo "== JI_KVARN=$v"; JI_TP=2 JI_KVARN=$v ./vb.sh 2>&1 | tail -8; done
echo JOB31_DONE $(date +%T)
