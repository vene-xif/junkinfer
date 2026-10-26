#!/bin/bash
# job 37: VRAM / GTT use of the MTP server, KVarN vs q8 (spill hypothesis for the KVarN MTP slowdown)
true
cd ~/jt
MTPENV="JI_MTP_VOCAB_FILE=$HOME/junkinfer/ref/mtp/hyperqwen_draft_vocab_40960.json JI_MTP_HEAD_TYPE=iq4_nl"
mem() { for c in 0 1; do d=/sys/class/drm/card$c/device; echo -n "card$c vram $(( $(cat $d/mem_info_vram_used) >> 20 )) gtt $(( $(cat $d/mem_info_gtt_used) >> 20 )) MiB; "; done; echo; }
for v in "JI_KVARN=0" "JI_KVARN=1"; do
  echo "== $v"; mem
  NMAX=1 PMIN=0.5 JI_CACHE_SWAPS=16 env $v $MTPENV ./mtpsrv.sh >/dev/null || continue
  echo -n "loaded: "; mem
  python3 spectest.py | cut -c1-60
  echo -n "after: "; mem
  grep -E "budget available|VRAM budget|elastic expert slots" ~/jt/mtp_srv.log | cut -c14- | head -4
  P=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk "{print \$1}"); [ -n "$P" ] && kill $P; sleep 15
done
echo JOB37_DONE $(date +%T)
