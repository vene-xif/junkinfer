#!/bin/bash
# job 28: thermal test - TP2 prefill of the 151k prompt (sparse defaults), tlog guard at 88 C
cd ~/jt
NMAX=0 JI_CACHE_SWAPS=16 ./mtpsrv.sh >/dev/null || { echo "server failed"; echo JOB28_DONE; exit; }
echo "prefill 151k: $(cd ~/junkinfer && timeout 2000 python3 scripts/depth_bench.py 130000 8 http://127.0.0.1:8080 x nocache 2>&1 | grep -oE "prompt [0-9]+ tokens: prefill [0-9.]+ t/s \([0-9]+ s\)")"
P=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk "{print \$1}"); [ -n "$P" ] && kill $P
echo JOB28_DONE $(date +%T)
