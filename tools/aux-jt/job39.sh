#!/bin/bash
# job 39: KVarN speed after the ring-in-tail fix (TP2, sparse defaults): decode 200k x2 (1024 tok), prefill 151k; VRAM / GTT
while ! grep -q JOB36_DONE ~/jt/job36.log 2>/dev/null; do sleep 30; done
cd ~/jt
mem() { for c in 0 1; do d=/sys/class/drm/card$c/device; echo -n "card$c vram $(( $(cat $d/mem_info_vram_used) >> 20 )) gtt $(( $(cat $d/mem_info_gtt_used) >> 20 )); "; done; echo; }
NMAX=0 JI_CACHE_SWAPS=16 SARGS="-lv 4" JI_KVARN=1 ./mtpsrv.sh >/dev/null || { echo "server failed"; echo JOB39_DONE; exit; }
echo "loaded: $(mem)"
for run in 1 2; do echo "KVarN decode 200k run $run: $(cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache scripts/depth_kv.sh 200000 1 2>&1 | grep -oE "decode [0-9]+ tokens at [0-9.]+ t/s")"; done
echo "after decode: $(mem)"
echo "KVarN prefill 151k: $(cd ~/junkinfer && timeout 2000 python3 scripts/depth_bench.py 130000 8 http://127.0.0.1:8080 x nocache 2>&1 | grep -oE "prompt [0-9]+ tokens: prefill [0-9.]+ t/s \([0-9]+ s\)")"
echo "after prefill: $(mem)"
P=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk "{print \$1}"); [ -n "$P" ] && kill $P
echo "ring timeouts: $(echo "$SUDO_PW" | sudo -S dmesg 2>/dev/null | grep -c "ring .* timeout")"
echo JOB39_DONE $(date +%T)
