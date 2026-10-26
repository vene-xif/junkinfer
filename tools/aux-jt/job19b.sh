#!/bin/bash
# job 19: ranking ceiling - exact-mass oracle (JI_SP_ORACLE=2) vs mass-estimate ranking (default SEL=1) at fixed budgets
cd ~/jt
S="JI_TP=0 GGML_VK_VISIBLE_DEVICES=0 JI_SPARSE_FRAC=0"
cfgs=(); for b in 16384 32768 65536; do cfgs+=("$S JI_SPARSE=$b JI_SP_ORACLE=2" "$S JI_SPARSE=$b"); done
ROWS="L_code:1 L_code:2" ~/jt/spgate.sh "${cfgs[@]}"
echo JOB19B_DONE $(date +%T)
