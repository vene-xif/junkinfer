#!/bin/bash
# job 36: KVarN 130k prefill from scratch (single card): keep the state, per-block K/V error vs the v2 reference state
while ! grep -q JOB38_DONE ~/jt/job38.log 2>/dev/null; do sleep 30; done
cd ~/jt
d=/mnt/ssd/kvcache/tmp_j36; rm -rf $d; mkdir -p $d
r=$(./kl_v2.sh L_code 1 JI_TP=0 GGML_VK_VISIBLE_DEVICES=0 JI_JPF=1 JI_KVARN=1 JR_STATE_DIR=$d 2>&1 | tail -1)
echo "130k from scratch: $r"
ls -la $d
python3 ~/jt/kvrblk.py $d/L_code.jtok_r1_a130047.state /mnt/ssd/kvcache/jref_v2/L_code.jtok_r1_a130047.state 8192
echo JOB36_DONE $(date +%T)
