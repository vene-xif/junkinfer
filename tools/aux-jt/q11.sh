#!/bin/bash
cd ~/junkinfer/engine && cmake --build build -j12 2>&1 | grep -E "error" | head
for t in 0.02 0.04 0.06; do for row in "L_code98 0" "L_code 1"; do ~/jt/kl.sh $row JI_TOPK_TAU=$t; done; done
export JI_IGNORE_EOS=1
for e in "JI_X=0" "JI_TOPK_TAU=0.02" "JI_TOPK_TAU=0.04" "JI_TOPK_TAU=0.06"; do
  ~/jt/restart.sh JI_SLOT_DIR=/mnt/ssd/kvcache $e >/dev/null
  for N in 2000 245000; do echo -n "$e depth $N: "; (cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache scripts/depth_kv.sh $N 5 2>&1 | grep -o "decode 128 tokens at [0-9.]*" | awk '{print $5}' | tail -3 | tr '\n' ' '); echo; done
done
P=$(ps -eo pid,args | grep "[b]in/llama-server -m /mnt/ssd" | awk "{print \$1}"); kill $P
