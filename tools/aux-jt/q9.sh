#!/bin/bash
# system knobs: decode at 2k and 245k (fixed 128 tokens, 3 runs: report runs 2-3)
while ps -eo args | grep -q "[q]8.sh"; do sleep 20; done
export JI_IGNORE_EOS=1
meas() {   # $1 label, rest env
  local lab=$1; shift
  ~/jt/restart.sh JI_SLOT_DIR=/mnt/ssd/kvcache "$@" >/dev/null
  for N in 2000 245000; do
    r=$(cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache scripts/depth_kv.sh $N 3 2>&1 | grep -o "decode 128 tokens at [0-9.]*" | awk '{print $5}' | tail -2 | tr '\n' ' ')
    echo "$lab depth $N: $r"
  done
}
B=06:00.0
meas "baseline (MRRS 512)" JI_X=0
echo kot | sudo -S -p "" setpci -s $B CAP_EXP+08.w=5000:7000
echo kot | sudo -S -p "" lspci -vv -s $B 2>/dev/null | grep -o "MaxReadReq [0-9]* bytes" | head -1
meas "MRRS 4096" JI_X=0
echo always | sudo -S -p "" true; echo kot | sudo -S -p "" sh -c 'echo always > /sys/kernel/mm/transparent_hugepage/enabled'; cat /sys/kernel/mm/transparent_hugepage/enabled
meas "MRRS 4096 + THP always" JI_X=0
for z in 7 6 5; do meas "+ JI_X3_ZC=$z" JI_X3_ZC=$z; done
P=$(ps -eo pid,args | grep "[b]in/llama-server -m /mnt/ssd" | awk "{print \$1}"); kill $P
