#!/bin/bash
# job 14: mass-estimate ranking for sparse prefill (JI_SPF_SEL) and decode (SEL=1 + FRAC); speed checks
cd ~/jt
pfrow() { d=/mnt/ssd/kvcache/tmp_j14; rm -rf $d; mkdir -p $d; r=$(./kl_v2.sh L_code 1 JI_TP=2 JI_JPF=1 JI_SPARSE=0 JR_STATE_DIR=$d "$@" 2>&1 | tail -1); rm -rf $d
  echo "prefill $* | 130k: $(echo "$r" | grep -oE "KL mean [0-9.]+ p99 [0-9.]+ max [0-9.]+ top1 [0-9.]+")"; }
pfrow JI_SPARSE_PF=32768 JI_SPF_SEL=1
pfrow JI_SPARSE_PF=65536 JI_SPF_SEL=1
ROWS="L_code98:0 L_code:1 L_code:2" ~/jt/spgate.sh "JI_SP_SEL=1 JI_SPARSE_FRAC=0.5" "JI_SP_SEL=1 JI_SPARSE_FRAC=0.35"
NMAX=0 JI_SPARSE_PF=32768 JI_SPF_SEL=1 JI_CACHE_SWAPS=16 ./mtpsrv.sh >/dev/null && echo "prefill speed pf32768 SEL: $(cd ~/junkinfer && timeout 1500 python3 scripts/depth_bench.py 130000 8 http://127.0.0.1:8080 x nocache | grep -oE "prompt [0-9]+ tokens: prefill [0-9.]+ t/s \([0-9]+ s\)")"
for v in "JI_SP_SEL=1 JI_SPARSE_FRAC=0.5" "JI_SPARSE_FRAC=0.5"; do
  NMAX=0 JI_CACHE_SWAPS=16 env $v ./mtpsrv.sh >/dev/null || continue
  for run in 1 2; do echo "decode 200k $v run $run: $(cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache scripts/depth_kv.sh 200000 1 2>&1 | grep -oE "decode [0-9]+ tokens at [0-9.]+ t/s")"; done
done
P=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk "{print \$1}"); [ -n "$P" ] && kill $P
echo "ring timeouts: $(echo "$SUDO_PW" | sudo -S dmesg 2>/dev/null | grep -c "ring .* timeout")"
echo JOB14_DONE $(date +%T)
