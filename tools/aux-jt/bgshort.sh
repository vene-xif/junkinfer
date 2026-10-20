#!/bin/bash
OUT=~/jt/bgshort.out; rm -f $OUT
for cfg in "JI_X=0" "JI_SPARSE=8192 JI_SP_BG=2 JI_SP_SEL=1" "JI_X=0"; do
  ~/jt/restart.sh JI_SLOT_DIR=/mnt/ssd/kvcache $cfg >/dev/null
  echo "== $cfg" >> $OUT
  (cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache scripts/depth_kv.sh 2000 3 2>&1 | grep -E "decode" | cut -c1-120) >> $OUT
done
P=$(ps -eo pid,args | grep "[b]in/llama-server -m /mnt/ssd" | awk "{print \$1}"); [ -n "$P" ] && kill $P; sleep 5
for B in 8192 16384 32768; do ~/jt/kl.sh L_code 1 JI_SPARSE=$B JI_SP_BG=2 JI_SP_SEL=1 >> $OUT; done
~/jt/kl.sh L_code 1 JI_SPARSE=16384 JI_SP_BG=2 >> $OUT
echo DONE >> $OUT
