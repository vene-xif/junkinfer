#!/bin/bash
# job 27: thermal test - TP2 decode at 200k, 3 x 1024 tokens, with tlog running
cd ~/jt
NMAX=0 JI_CACHE_SWAPS=16 ./mtpsrv.sh >/dev/null || { echo "server failed"; echo JOB27_DONE; exit; }
for run in 1 2 3; do echo "decode 200k run $run: $(cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache scripts/depth_kv.sh 200000 1 2>&1 | grep -oE "decode [0-9]+ tokens at [0-9.]+ t/s")"; done
P=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk "{print \$1}"); [ -n "$P" ] && kill $P
echo JOB27_DONE $(date +%T)
