#!/bin/bash
# job 24: KVarN decode-only KL, single card vs TP2, 32k (L_code r0) and 130k (r1)
cd ~/jt
ROWS="L_code:0 L_code:1" ~/jt/spgate.sh "JI_TP=0 GGML_VK_VISIBLE_DEVICES=0 JI_KVARN=1" "JI_TP=2 JI_KVARN=1"
echo JOB24B_DONE $(date +%T)
