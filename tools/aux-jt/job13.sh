#!/bin/bash
# job 13: 16-bit block list check (pf 32768 must reproduce 0.00172), prefill tail modes, decode SEL=1 / FRAC
cd ~/jt
pfrow() { d=/mnt/ssd/kvcache/tmp_j13; rm -rf $d; mkdir -p $d; r=$(./kl_v2.sh L_code 1 JI_TP=2 JI_JPF=1 JI_SPARSE=0 JR_STATE_DIR=$d "$@" 2>&1 | tail -1); rm -rf $d
  echo "prefill $* | 130k: $(echo "$r" | grep -oE "KL mean [0-9.]+ p99 [0-9.]+ max [0-9.]+ top1 [0-9.]+")"; }
pfrow JI_SPARSE_PF=32768
pfrow JI_SPARSE_PF=32768 JI_SPF_MODE=0
pfrow JI_SPARSE_PF=32768 JI_SPF_MODE=2
ROWS="L_code98:0 L_code:1 L_code:2" ~/jt/spgate.sh "JI_SPARSE=65536 JI_SP_SEL=1" "JI_SPARSE_FRAC=0.3" "JI_SPARSE_FRAC=0.7"
echo "ring timeouts: $(echo "$SUDO_PW" | sudo -S dmesg 2>/dev/null | grep -c "ring .* timeout")"
echo JOB13_DONE $(date +%T)
