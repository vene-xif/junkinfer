#!/bin/bash
# spgate.sh "LABEL ENV..." ... : KL on the v2 refs, rows 64k / 100k / 130k / 261k (+ timing column)
cd ~/jt
for cfg in "$@"; do
  for sr in ${ROWS:-L_code64:0 L_code98:0 L_code:1 L_code:2}; do
    r=$(./kl_v2.sh ${sr%:*} ${sr#*:} JI_TP=2 $cfg 2>&1 | tail -1)
    echo "$cfg | $(echo "$r" | grep -oE "^L_code[0-9]* r[0-9]") $(echo "$r" | grep -oE "KL mean [0-9.]+ p99 [0-9.]+ max [0-9.]+ top1 [0-9.]+")"
  done
done
echo SPGATE_DONE $(date +%T)
