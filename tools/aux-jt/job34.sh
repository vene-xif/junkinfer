#!/bin/bash
# job 34: KVarN prefill copy-buffer path: force the separate buffer at 32k (passes on the compute-buffer path)
while ! grep -q JOB33_DONE ~/jt/job33.log 2>/dev/null; do sleep 30; done
cd ~/junkinfer/engine && cmake --build build -j12 2>&1 | grep -E " error|rror:" | head -5
cd ~/jt
for v in "JI_KVR_PFT=1" ""; do
  d=/mnt/ssd/kvcache/tmp_j34; rm -rf $d; mkdir -p $d
  r=$(KLX="-lv 4" ./kl_v2.sh L_code 0 JI_TP=0 GGML_VK_VISIBLE_DEVICES=0 JI_JPF=1 JI_KVARN=1 JR_STATE_DIR=$d $v 2>&1 | tail -1); rm -rf $d
  echo "32k from scratch [$v]: $(echo "$r" | grep -oE "KL mean [0-9.]+ p99 [0-9.]+ max [0-9.]+ top1 [0-9.]+") | $(grep -oE "KVarN copy in its own buffer[^)]*\)" ~/jt/kl.log | head -1)"
done
echo JOB34_DONE $(date +%T)
