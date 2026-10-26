#!/bin/bash
# job 33: KVarN vs q8 with MTP, JI_PROFILE=200: per-step phases, expert misses per step size
while ! grep -q JOB32_DONE ~/jt/job32.log 2>/dev/null; do sleep 30; done
cd ~/jt
MTPENV="JI_MTP_VOCAB_FILE=$HOME/junkinfer/ref/mtp/hyperqwen_draft_vocab_40960.json JI_MTP_HEAD_TYPE=iq4_nl"
for v in "JI_KVARN=1" "JI_KVARN=0"; do
  NMAX=1 PMIN=0.5 JI_CACHE_SWAPS=16 JI_PROFILE=200 SARGS="-lv 4" env $v $MTPENV ./mtpsrv.sh >/dev/null || continue
  echo "== $v"; python3 spectest.py | cut -c1-60
  cp ~/jt/mtp_srv.log ~/jt/prof33_$v.log
done
P=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk "{print \$1}"); [ -n "$P" ] && kill $P
echo JOB33_DONE $(date +%T)
