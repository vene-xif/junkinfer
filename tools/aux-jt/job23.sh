#!/bin/bash
# job 23: KVarN round trip (single card): restore depth_32000 -> sealed records -> save via hook (materialize) -> diff
cd ~/jt
NMAX=0 JI_TP=0 JI_GPU=0 JI_KVARN=1 JI_KVR_TRACE=1 JI_CACHE_SWAPS=16 SARGS="-lv 4" ./mtpsrv.sh >/dev/null || { echo "server failed"; echo JOB23_DONE; exit; }
curl -s -X POST localhost:8080/slots/0?action=restore -H "Content-Type: application/json" -d "{\"filename\":\"depth_32000.bin\"}"; echo
curl -s -X POST localhost:8080/slots/0?action=save -H "Content-Type: application/json" -d "{\"filename\":\"kvr_rt.bin\"}"; echo
grep -E "KVarN|plain write" ~/jt/mtp_srv.log | head -6 | cut -c1-180
python3 ~/jt/kvrcmp.py /mnt/ssd/kvcache/depth_32000.bin /mnt/ssd/kvcache/kvr_rt.bin
P=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk "{print \$1}"); [ -n "$P" ] && kill $P
rm -f /mnt/ssd/kvcache/kvr_rt.bin
echo JOB23_DONE $(date +%T)
