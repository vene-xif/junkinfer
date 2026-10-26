#!/bin/bash
# job 22: exact replay of job 20 server steps (2x restore 200k + 1024-token decode, then 151k prompt), log kept
cd ~/jt
NMAX=0 JI_CACHE_SWAPS=16 ./mtpsrv.sh >/dev/null || { echo "server failed"; echo JOB22_DONE; exit; }
for run in 1 2; do echo "decode 200k run $run: $(cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache scripts/depth_kv.sh 200000 1 2>&1 | grep -oE "decode [0-9]+ tokens at [0-9.]+ t/s")"; done
echo "prefill 151k: $(cd ~/junkinfer && timeout 2000 python3 scripts/depth_bench.py 130000 8 http://127.0.0.1:8080 x nocache 2>&1 | tail -1 | cut -c1-200)"
cp ~/jt/mtp_srv.log ~/jt/crash22_srv.log
P=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk "{print \$1}"); [ -n "$P" ] && kill $P
echo "ring timeouts: $(echo "$SUDO_PW" | sudo -S dmesg 2>/dev/null | grep -c "ring .* timeout")"
echo JOB22_DONE $(date +%T)
