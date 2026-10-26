#!/bin/bash
# job 29: KVarN short-context slowdown: no MTP vs MTP, with the speculative profiler; q8 MTP reference
while ! grep -q JOB26_DONE ~/jt/job26.log 2>/dev/null; do sleep 30; done
cd ~/jt
MTPENV="JI_MTP_VOCAB_FILE=$HOME/junkinfer/ref/mtp/hyperqwen_draft_vocab_40960.json JI_MTP_HEAD_TYPE=iq4_nl"
NMAX=0 JI_KVARN=1 JI_CACHE_SWAPS=16 ./mtpsrv.sh >/dev/null && { echo "KVarN no MTP:"; python3 spectest.py | cut -c1-70; }
for v in "JI_KVARN=1" "JI_KVARN=0"; do
  NMAX=1 PMIN=0.5 JI_CACHE_SWAPS=16 JI_SPEC_PROF=200 SARGS="-lv 4" env $v $MTPENV ./mtpsrv.sh >/dev/null || continue
  echo "$v MTP:"; python3 spectest.py | cut -c1-70
  cp ~/jt/mtp_srv.log ~/jt/prof_$v.log
  grep -E "spec_prof|jp_scope|step" ~/jt/mtp_srv.log | tail -6 | cut -c1-220
done
P=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk "{print \$1}"); [ -n "$P" ] && kill $P
echo JOB29_DONE $(date +%T)
