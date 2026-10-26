#!/bin/bash
# decode t/s at 200k (restored, Qwen-Sharp fixture), TP2, no MTP, per sparse budget
cd ~/jt
for b in 0 16384 65536 131072; do
  NMAX=0 JI_SPARSE=$b JI_CACHE_SWAPS=16 ./mtpsrv.sh >/dev/null || { echo "budget $b: server failed"; continue; }
  for run in 1 2; do
    echo "budget $b run $run: $(cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache scripts/depth_kv.sh 200000 1 2>&1 | grep -E "decode" | grep -oE "decode [0-9]+ tokens at [0-9.]+ t/s")"
  done
done
P=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk "{print \$1}"); [ -n "$P" ] && kill $P   # idle server: safe
echo SPSPEED_DONE $(date +%T)
