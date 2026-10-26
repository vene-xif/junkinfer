#!/bin/bash
# job 12: realistic sparse-PREFILL quality: the 130k prefix prefilled from scratch (JDE TP2 prefill) per budget, tail
# decoded dense; then prefill t/s on the 151k prompt for the 64k budget
cd ~/jt
for pf in 8192 32768 65536; do
  d=/mnt/ssd/kvcache/tmp_pf$pf; rm -rf $d; mkdir -p $d
  r=$(./kl_v2.sh L_code 1 JI_TP=2 JI_JPF=1 JI_SPARSE=0 JI_SPARSE_PF=$pf JR_STATE_DIR=$d 2>&1 | tail -1)
  echo "pf $pf prefix-from-scratch 130k: $(echo "$r" | grep -oE "KL mean [0-9.]+ p99 [0-9.]+ max [0-9.]+ top1 [0-9.]+")"
  rm -rf $d
done
NMAX=0 JI_SPARSE_PF=65536 JI_CACHE_SWAPS=16 ./mtpsrv.sh >/dev/null && echo "pf 65536 speed: $(cd ~/junkinfer && timeout 1500 python3 scripts/depth_bench.py 130000 8 http://127.0.0.1:8080 x nocache | grep -oE "prompt [0-9]+ tokens: prefill [0-9.]+ t/s \([0-9]+ s\)")"
P=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk "{print \$1}"); [ -n "$P" ] && kill $P
echo "ring timeouts: $(echo "$SUDO_PW" | sudo -S dmesg 2>/dev/null | grep -c "ring .* timeout")"
echo JOB12_DONE $(date +%T)
