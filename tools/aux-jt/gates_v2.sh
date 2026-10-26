#!/bin/bash
# KL gates on the v2 references: production TP2 (sparse defaults) and KVarN-6; one model process at a time
cd ~/jt
for cfg in "JI_TP=2" "JI_TP=2 JI_KVARN=1"; do
  for sr in L_code:0 L_code64:0 L_code98:0 L_code:1 L_code:2; do
    ./kl_v2.sh ${sr%:*} ${sr#*:} $cfg 2>&1 | tail -1 | cut -c1-150
  done
done
echo GATES_DONE $(date +%T)
