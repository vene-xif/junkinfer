#!/bin/bash
# job 40: is the KVarN 130k from-scratch state bad, or the in-process tail eval after the prefill? restore the saved state
while ! grep -q JOB39_DONE ~/jt/job39.log 2>/dev/null; do sleep 30; done
cd ~/jt
for v in "JI_KVARN=1" "JI_KVARN=0"; do
  r=$(./kl_v2.sh L_code 1 JI_TP=0 GGML_VK_VISIBLE_DEVICES=0 $v JR_STATE_DIR=/mnt/ssd/kvcache/tmp_j36 2>&1 | tail -1)
  echo "restored j36 state, $v: $(echo "$r" | grep -oE "KL mean [0-9.]+ p99 [0-9.]+ max [0-9.]+ top1 [0-9.]+") | $(grep -cE "restore|loaded" ~/jt/kl.log) restore lines"
done
echo JOB40_DONE $(date +%T)
