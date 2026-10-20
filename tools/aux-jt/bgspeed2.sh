#!/bin/bash
OUT=~/jt/bgspeed2.out; rm -f $OUT
run() { D=$1; shift
  ~/jt/restart.sh JI_SLOT_DIR=/mnt/ssd/kvcache "$@" >/dev/null
  echo "== depth $D | $*" >> $OUT
  grep -E "jde: (device-local budget|sparse decode attention)" ~/jt/srv.log | cut -c16-200 >> $OUT
  (cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache scripts/depth_kv.sh $D 3 2>&1 | grep -E "decode" | cut -c1-120) >> $OUT
}
run 245000 JI_SPARSE=8192 JI_SP_BG=1
run 245000 JI_SPARSE=16384 JI_SP_BG=1
run 245000 JI_SPARSE=16384 JI_SP_BG=2
run 100000 JI_SPARSE=8192 JI_SP_BG=1
run 245000 JI_SPARSE=8192 JI_SP_BG=1 JI_PROFILE=32 JI_PROFILE_GPU=1
grep -E "jde_prof: per token|jde_prof: CPU|jde_gpu" ~/jt/srv.log | tail -22 | cut -c16-200 >> $OUT
P=$(ps -eo pid,args | grep "[b]in/llama-server -m /mnt/ssd" | awk "{print \$1}"); [ -n "$P" ] && kill $P
echo DONE >> $OUT
