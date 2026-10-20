#!/bin/bash
cd ~/junkinfer/engine && cmake --build build -j12 2>&1 | grep -E " error" | head
L8="15=8192,19=8192,23=8192,27=8192,31=8192,35=8192,39=8192"
L4="15=4096,19=4096,23=4096,27=4096,31=4096,35=4096,39=4096"
CFGS=("JI_SP_DENSE_LAYERS=3,7,11 JI_SP_LBUDGET=$L8" "JI_SP_LBUDGET=$L8" "JI_SP_DENSE_LAYERS=3,7,11 JI_SP_LBUDGET=$L4")
for c in "${CFGS[@]}"; do for row in "L_code98 0" "L_code 1" "L_code 2"; do ~/jt/kl.sh $row $c; done; done
export JI_IGNORE_EOS=1
for c in "${CFGS[@]}"; do
  ~/jt/restart.sh JI_SLOT_DIR=/mnt/ssd/kvcache $c >/dev/null
  for N in 100000 245000; do echo -n "${c:0:40} .. depth $N: "; (cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache scripts/depth_kv.sh $N 6 2>&1 | grep -o "decode 128 tokens at [0-9.]*" | awk '{print $5}' | tail -4 | tr '\n' ' '); echo; done
done
P=$(ps -eo pid,args | grep "[b]in/llama-server -m /mnt/ssd" | awk "{print \$1}"); kill $P
