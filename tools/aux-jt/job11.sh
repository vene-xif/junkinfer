#!/bin/bash
# job 11: reproduce the sparse-prefill hang (JI_SPARSE_PF=32768, 151k prompt): TP2, then single card
cd ~/jt
for cfg in "TP2" "TP0"; do
  if [ $cfg = TP0 ]; then export JI_TP=0; fi
  NMAX=0 JI_SPARSE_PF=32768 JI_CACHE_SWAPS=16 ./mtpsrv.sh >/dev/null || { echo "$cfg: server failed"; continue; }
  echo "$cfg: $(cd ~/junkinfer && timeout 1500 python3 scripts/depth_bench.py 130000 8 http://127.0.0.1:8080 x nocache | grep -oE "prompt [0-9]+ tokens: prefill [0-9.]+ t/s \([0-9]+ s\)")"
  dmesg_hang=$(echo "$SUDO_PW" | sudo -S dmesg 2>/dev/null | grep -c "ring .* timeout"); echo "$cfg: ring timeouts in dmesg: $dmesg_hang"
  [ "$dmesg_hang" != 0 ] && { echo "HANG REPRODUCED ($cfg)"; break; }
done
P=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk "{print \$1}"); [ -n "$P" ] && kill $P
echo JOB11_DONE $(date +%T)
