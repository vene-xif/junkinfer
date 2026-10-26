#!/bin/bash
# KVarN bisect: each step logged + synced (box may freeze)
L=~/jt/kvrbis.log; st(){ echo "$(date +%T) $*" >> $L; sync; }
cd ~/jt; : > $L
st start-server; NMAX=0 JI_KVARN=1 JI_CACHE_SWAPS=16 SARGS="-lv 4" ./mtpsrv.sh >/dev/null || { st server-fail; exit 1; }
for N in ${STEPS:-32000}; do
  st restore+decode $N
  (cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache timeout 900 scripts/depth_kv.sh $N 1 2>&1 | grep -E "decode|restored|error" | cut -c1-160 >> $L); sync
  st done $N; grep -E "KVarN|error|ERROR" ~/jt/mtp_srv.log | tail -4 >> $L; sync
done
st end
