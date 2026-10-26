#!/bin/bash
# job 20: two-GPU final gate of the sparse defaults (after the helper card came back)
cd ~/jt
d=/mnt/ssd/kvcache/tmp_j20; rm -rf $d; mkdir -p $d
r=$(./kl_v2.sh L_code 2 JI_TP=2 JI_JPF=1 JR_STATE_DIR=$d 2>&1 | tail -1); rm -rf $d
echo "end-to-end defaults | L_code:2: $(echo "$r" | grep -oE "KL mean [0-9.]+ p99 [0-9.]+ max [0-9.]+ top1 [0-9.]+")"
NMAX=0 JI_CACHE_SWAPS=16 ./mtpsrv.sh >/dev/null && {
  for run in 1 2; do echo "decode 200k defaults run $run: $(cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache scripts/depth_kv.sh 200000 1 2>&1 | grep -oE "decode [0-9]+ tokens at [0-9.]+ t/s")"; done
  echo "prefill speed defaults: $(cd ~/junkinfer && timeout 2000 python3 scripts/depth_bench.py 130000 8 http://127.0.0.1:8080 x nocache | grep -oE "prompt [0-9]+ tokens: prefill [0-9.]+ t/s \([0-9]+ s\)")"; }
NMAX=1 PMIN=0.5 JI_CACHE_SWAPS=16 JI_MTP_VOCAB_FILE=$HOME/junkinfer/ref/mtp/hyperqwen_draft_vocab_40960.json JI_MTP_HEAD_TYPE=iq4_nl ./mtpsrv.sh >/dev/null && { echo "short spectest (MTP, 1024 tokens):"; python3 spectest.py | cut -c1-90; }
P=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk "{print \$1}"); [ -n "$P" ] && kill $P
echo "ring timeouts: $(echo "$SUDO_PW" | sudo -S dmesg 2>/dev/null | grep -c "ring .* timeout")"
echo JOB20_DONE $(date +%T)
