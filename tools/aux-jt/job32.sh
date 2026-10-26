#!/bin/bash
# job 32: KVarN prefill-path failure at 130k: isolate sparse prefill / depth / TP
while ! grep -q JOB31_DONE ~/jt/job31.log 2>/dev/null; do sleep 30; done
cd ~/jt
row() { S=$1 R=$2; shift 2; d=/mnt/ssd/kvcache/tmp_j32; rm -rf $d; mkdir -p $d
  r=$(./kl_v2.sh $S $R JI_JPF=1 JI_KVARN=1 JR_STATE_DIR=$d "$@" 2>&1 | tail -1); rm -rf $d
  echo "KVarN from scratch $S r$R $*: $(echo "$r" | grep -oE "KL mean [0-9.]+ p99 [0-9.]+ max [0-9.]+ top1 [0-9.]+")"; }
row L_code 1 JI_TP=2 JI_SPARSE_PF=0
row L_code64 0 JI_TP=2
row L_code 1 JI_TP=0 GGML_VK_VISIBLE_DEVICES=0
echo JOB32_DONE $(date +%T)
