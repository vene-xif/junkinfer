#!/bin/bash
export JI_IGNORE_EOS=1
for e in "JI_X=0" "JI_SP_NORM=2 JI_SP_BG=2" "JI_SP_BG=2" "JI_SPARSE=32768" "JI_SPARSE=24576"; do
  ~/jt/restart.sh JI_SLOT_DIR=/mnt/ssd/kvcache $e >/dev/null; echo "== $e"; (cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache scripts/depth_kv.sh 245000 3 2>&1 | grep decode | cut -c1-90)
done
P=$(ps -eo pid,args | grep "[b]in/llama-server -m /mnt/ssd" | awk "{print \$1}"); kill $P; sleep 5
for row in "L_code98 0" "L_code 1" "L_code 2"; do ~/jt/kl.sh $row JI_SPARSE=32768; done
