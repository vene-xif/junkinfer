#!/bin/bash
# job 18 (single card while the helper GPU is off the bus): decode-only KL of the new sparse defaults; decode t/s at 200k defaults vs dense
cd ~/jt
~/jt/spgate.sh "JI_TP=0 GGML_VK_VISIBLE_DEVICES=0" 2>&1 | sed "s/^/decode-only defaults single /"
for v in "JI_TP=0 JI_GPU=0" "JI_TP=0 JI_GPU=0 JI_SPARSE=0"; do
  NMAX=0 JI_CACHE_SWAPS=16 env $v ./mtpsrv.sh >/dev/null || { echo "$v: server failed"; continue; }
  for run in 1 2; do echo "decode 200k $v run $run: $(cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache scripts/depth_kv.sh 200000 1 2>&1 | grep -oE "decode [0-9]+ tokens at [0-9.]+ t/s")"; done
done
P=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk "{print \$1}"); [ -n "$P" ] && kill $P
echo "ring timeouts: $(echo "$SUDO_PW" | sudo -S dmesg 2>/dev/null | grep -c "ring .* timeout")"
echo JOB18B_DONE $(date +%T)
