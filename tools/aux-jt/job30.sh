#!/bin/bash
# job 30: KVarN prefill-path correctness: 32k prefix prefilled from scratch, single card vs TP2
while ! grep -q JOB29_DONE ~/jt/job29.log 2>/dev/null; do sleep 30; done
cd ~/jt
for v in "JI_TP=0 GGML_VK_VISIBLE_DEVICES=0" "JI_TP=2"; do
  d=/mnt/ssd/kvcache/tmp_j30; rm -rf $d; mkdir -p $d
  r=$(./kl_v2.sh L_code 0 $v JI_JPF=1 JI_KVARN=1 JR_STATE_DIR=$d 2>&1 | tail -1); rm -rf $d
  echo "KVarN prefill-from-scratch 32k $v: $(echo "$r" | grep -oE "KL mean [0-9.]+ p99 [0-9.]+ max [0-9.]+ top1 [0-9.]+")"
done
echo JOB30_DONE $(date +%T)
