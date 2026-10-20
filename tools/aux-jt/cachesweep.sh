#!/bin/bash
# 247k decode from the saved context per cache config: t/s of 3 runs + hit rate
for cfg in "$@"; do
  ~/jt/restart.sh JI_SLOT_DIR=/mnt/ssd/kvcache $cfg > /dev/null
  r=$(cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache scripts/depth_kv.sh 245000 3 2>&1 | grep -o "decode 95 tokens at [0-9.]*" | awk "{print \$5}" | tr "\n" " ")
  P=$(ps -eo pid,args | grep "[b]in/llama-server -m /mnt/ssd" | awk "{print \$1}"); kill -INT $P; sleep 8
  h=$(grep -o "hit rate [0-9.]*%" ~/jt/srv.log | tail -1)
  echo "$cfg | t/s $r | $h"
done
