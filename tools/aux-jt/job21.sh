#!/bin/bash
# job 21: reproduce the server crash: restore 200k + decode, then a fresh 151k prompt (sparse defaults)
cd ~/jt
NMAX=0 JI_CACHE_SWAPS=16 ./mtpsrv.sh >/dev/null || { echo "server failed"; echo JOB21_DONE; exit; }
echo "decode 200k: $(cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache JI_GEN=64 scripts/depth_kv.sh 200000 1 2>&1 | grep -oE "decode [0-9]+ tokens at [0-9.]+ t/s")"
echo "prefill 151k: $(cd ~/junkinfer && timeout 2000 python3 scripts/depth_bench.py 130000 8 http://127.0.0.1:8080 x nocache 2>&1 | tail -1 | cut -c1-200)"
cp ~/jt/mtp_srv.log ~/jt/crash21_srv.log
P=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk "{print \$1}"); [ -n "$P" ] && kill $P
echo "ring timeouts: $(echo "$SUDO_PW" | sudo -S dmesg 2>/dev/null | grep -c "ring .* timeout")"
echo JOB21_DONE $(date +%T)
