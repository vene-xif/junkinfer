#!/bin/bash
# usage: depthsweep.sh "LABEL ENV..." ...   prefill at 64k (+4k), decode at 100k / 245k (128 tokens, T=0.7) from saved KV
for cfg in "$@"; do
  set -- $cfg; label=$1; shift
  env "$@" JI_TP_MARGIN_MIB=${MARGIN:-1900} SARGS="-lv 3" ./mtpsrv.sh >/dev/null || { echo "$label: server failed"; continue; }
  echo "== $label"
  python3 dpb.py 64000 4096
  (cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache scripts/depth_kv.sh 100000 1 | grep -E "decode|restored" | cut -c1-140)
  (cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache scripts/depth_kv.sh 245000 1 | grep -E "decode|restored" | cut -c1-140)
done
