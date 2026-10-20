#!/bin/bash
# decode t/s at depth: dense vs sparse + tail estimation; one server at a time
OUT=~/jt/bgspeed.out
run() { D=$1; shift
  ~/jt/restart.sh JI_SLOT_DIR=/mnt/ssd/kvcache "$@" >/dev/null
  echo "== depth $D | $*" >> $OUT
  grep -E "jde: (device-local budget|expert cache [0-9]|[0-9]+ elastic|sparse decode attention)" ~/jt/srv.log | cut -c16-200 >> $OUT
  (cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache scripts/depth_kv.sh $D 3 2>&1 | grep -E "decode" | cut -c1-120) >> $OUT
}
run 245000 JI_X=0
for B in 8192 16384 32768 61440; do run 245000 JI_SPARSE=$B JI_SP_BG=2 JI_SP_SEL=1; done
run 245000 JI_SPARSE=16384 JI_SP_BG=2
run 100000 JI_X=0
for B in 8192 25600; do run 100000 JI_SPARSE=$B JI_SP_BG=2 JI_SP_SEL=1; done
P=$(ps -eo pid,args | grep "[b]in/llama-server -m /mnt/ssd" | awk "{print \$1}"); [ -n "$P" ] && kill $P
echo DONE >> $OUT
