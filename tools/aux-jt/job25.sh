#!/bin/bash
# job 25: KVarN TP2 after the helper-mirror fix, plus single card again, 32k / 130k / 261k
cd ~/jt
ROWS="L_code:0 L_code:1 L_code:2" ~/jt/spgate.sh "JI_TP=2 JI_KVARN=1" "JI_TP=0 GGML_VK_VISIBLE_DEVICES=0 JI_KVARN=1"
echo JOB25_DONE $(date +%T)
