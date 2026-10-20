#!/bin/bash
# regression of the generic EXL3 build on the current model + per-layer dense decode
echo "== regression"
~/jt/kl.sh L_code 1 JI_X=0
export JI_IGNORE_EOS=1
~/jt/restart.sh JI_SLOT_DIR=/mnt/ssd/kvcache >/dev/null
for N in 2000 245000; do echo -n "default depth $N: "; (cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache scripts/depth_kv.sh $N 4 2>&1 | grep -o "decode 128 tokens at [0-9.]*" | awk '{print $5}' | tail -3 | tr '\n' ' '); echo; done
echo -n "prefill 132k (tail layers 3,7,11): "; ~/jt/venv/bin/python ~/jt/srvpf.py ~/jt/docs_all.txt
echo "== dense decode layers"
for e in "JI_SP_DENSE_LAYERS=7,11" "JI_SP_DENSE_LAYERS=3,7,11"; do
  ~/jt/restart.sh JI_SLOT_DIR=/mnt/ssd/kvcache $e >/dev/null
  for N in 64000 245000; do echo -n "$e depth $N: "; (cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache scripts/depth_kv.sh $N 4 2>&1 | grep -o "decode 128 tokens at [0-9.]*" | awk '{print $5}' | tail -3 | tr '\n' ' '); echo; done
done
P=$(ps -eo pid,args | grep "[b]in/llama-server -m /mnt/ssd" | awk "{print \$1}"); kill $P; sleep 5
for e in "JI_SP_DENSE_LAYERS=7,11" "JI_SP_DENSE_LAYERS=3,7,11"; do for row in "L_code98 0" "L_code 1" "L_code 2"; do ~/jt/kl.sh $row $e; done; done
