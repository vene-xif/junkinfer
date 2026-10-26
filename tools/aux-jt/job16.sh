#!/bin/bash
# job 16: sparse prefill variants at 261k (frac 0.7, local window 16k) + speed; end-to-end (sparse prefill + new
# default sparse decode: SEL=1 FRAC=0.5) at 130k / 261k
cd ~/jt
pfrow() { R=$1; shift; d=/mnt/ssd/kvcache/tmp_j16; rm -rf $d; mkdir -p $d; r=$(./kl_v2.sh L_code $R JI_TP=2 JI_JPF=1 JR_STATE_DIR=$d "$@" 2>&1 | tail -1); rm -rf $d
  echo "$* | row $R: $(echo "$r" | grep -oE "KL mean [0-9.]+ p99 [0-9.]+ max [0-9.]+ top1 [0-9.]+")"; }
B="JI_SPARSE_PF=32768 JI_SPF_SEL=1"
pfrow 2 JI_SPARSE=0 $B JI_SPARSE_PF_FRAC=0.7
pfrow 2 JI_SPARSE=0 $B JI_SPARSE_PF_FRAC=0.5 JI_SPF_LOCAL=16384
pfrow 1 $B JI_SPARSE_PF_FRAC=0.5
pfrow 2 $B JI_SPARSE_PF_FRAC=0.5
for v in "$B JI_SPARSE_PF_FRAC=0.7" "$B JI_SPARSE_PF_FRAC=0.5 JI_SPF_LOCAL=16384"; do
  NMAX=0 JI_CACHE_SWAPS=16 env $v ./mtpsrv.sh >/dev/null || continue
  echo "prefill speed $v: $(cd ~/junkinfer && timeout 2000 python3 scripts/depth_bench.py 130000 8 http://127.0.0.1:8080 x nocache | grep -oE "prompt [0-9]+ tokens: prefill [0-9.]+ t/s \([0-9]+ s\)")"
done
P=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk "{print \$1}"); [ -n "$P" ] && kill $P
echo "ring timeouts: $(echo "$SUDO_PW" | sudo -S dmesg 2>/dev/null | grep -c "ring .* timeout")"
echo JOB16_DONE $(date +%T)
