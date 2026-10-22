#!/bin/bash
# Information-gathering campaign (2026-10-23). Resumable: steps run in order, state/<step>.done marks finished ones,
# state/<step>.tries counts attempts (a step that crashed / hung the box twice is skipped). @reboot cron resumes.
# Summary: ~/jt/campaign/summary.txt (one block per step); raw logs: out/<step>.*
C=~/jt/campaign; S=$C/state; O=$C/out; SUM=$C/summary.txt; J=~/jt; K=~/junkinfer
exec 9>$C/lock; flock -n 9 || { echo "already running"; exit 0; }
cd $J
V=$K/ref/mtp/hyperqwen_draft_vocab_40960.json
BASE="NMAX=0 JI_CACHE_SWAPS=16 JI_TP_MARGIN_MIB=1500 JI_PROFILE=100"
MTP="NMAX=1 PMIN=0.5 JI_MTP_VOCAB_FILE=$V JI_MTP_HEAD_TYPE=iq4_nl JI_CACHE_SWAPS=16 JI_TP_MARGIN_MIB=1900 JI_PROFILE=100"
log() { echo "$*" | tee -a $SUM; }
stop_all() {   # production supervisor + any server
  [ -f $K/logs/supervisor.pid ] && kill "$(cat $K/logs/supervisor.pid)" 2>/dev/null
  P=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk "{print \$1}"); [ -n "$P" ] && kill $P && sleep 8
  P=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk "{print \$1}"); [ -n "$P" ] && kill -9 $P && sleep 3
}
srv() { stop_all; env "$@" SARGS="-lv 4" ./mtpsrv.sh | grep -q up; }   # fresh server (mtpsrv.sh), env overrides
alive() { curl -sf -m 5 http://127.0.0.1:8080/health > /dev/null; }
story() { timeout 600 python3 q.py "Tell me a short story about a lighthouse keeper." 300 2>&1 | cut -c1-12; }
prompt() { (cd $K && timeout 1800 python3 scripts/depth_bench.py $1 2 http://127.0.0.1:8080 x nocache 2>&1 | tail -1 | cut -c1-90); }
events() { tr -d "\000" < $J/mtp_srv.log | grep -a -c -E "$1"; }
keep() { cp $J/mtp_srv.log $O/$1.server.log 2>/dev/null; }

# ---- steps (name: function) ------------------------------------------------------------------------
# A: decode slowdown after a prefill of N tokens: baseline, right after, after 120 s idle, after a slot erase
slow() { N=$1; srv $BASE || { log "  server failed"; return 1; }
  b=$(story); p=$(prompt $N); r1=$(story); sleep 120; r2=$(story)
  curl -s -X POST "http://127.0.0.1:8080/slots/0?action=erase" > /dev/null; r3=$(story)
  log "  N=$N baseline $b | prompt: $p | after $r1 | +120s idle $r2 | after erase $r3 | TP lost $(events "TP2 sync .* lost") slow-steps $(events "TP2 step") alive $(alive && echo y || echo n)"
  keep slow_$N; }
# A: hang reproduction: 24k fresh prompt + decode, 3 times, JDE prefill on / off
hang() { J0=$1; srv $BASE JI_JPF=$J0 || { log "  server failed"; return 1; }
  for i in 1 2 3; do p=$(prompt 24000); r=$(story); log "  JI_JPF=$J0 run $i: prompt: $p | after $r | TP lost $(events "TP2 sync .* lost") slow-steps $(events "TP2 step") alive $(alive && echo y || echo n)"
    alive || { srv $BASE JI_JPF=$J0; }; done
  keep hang_jpf$J0; }
# B: MTP garbage after a deep context: N-token prompt then the 3 spectest prompts (garbage = ~0 accepted drafts)
mtpb() { N=$1; shift; srv $MTP "$@" || { log "  server failed"; return 1; }
  p=$(prompt $N); log "  N=$N $* | prompt: $p"
  timeout 1500 python3 spectest.py 2>&1 | cut -c1-120 | sed "s/^/    /" | tee -a $SUM
  log "    TP lost $(events "TP2 sync .* lost") alive $(alive && echo y || echo n)"; keep mtp_${N}_$(echo "$*" | tr " =" "__"); }
# C: KL vs the full-attention reference (production env), ~31k and ~64k rows
kl() { stop_all; r=$(TP=2 JI_TP_HOT=$K/ref/tp/tp_hot_qwen36_exl3.bin timeout 3600 ./kl.sh $1 0 JI_JPF=1 JI_CACHE_SWAPS=16 JI_TP_MARGIN_MIB=1500 2>&1 | tail -2)
  log "  $1 r0: $r"; cp ~/jt/kl.log $O/kl_$1.log; }
# E: saved deep contexts for later tests
fixture() { srv $BASE || return 1; r=$(cd $K && JI_SLOT_DIR=/mnt/ssd/kvcache timeout 3600 scripts/depth_kv.sh $1 1 2>&1 | grep -E "decode|restored|saved|n_saved" | tail -2 | cut -c1-120)
  log "  depth $1: $r"; ls -la /mnt/ssd/kvcache/depth_$1.bin 2>&1 | tee -a $SUM; }
# D: soak: production config (supervisor), 60 min of mixed traffic
soak() { stop_all; (cd $K && scripts/serve_prod.sh > /dev/null); t0=$(date +%s); i=0; e=0
  while [ $(( $(date +%s) - t0 )) -lt 3600 ]; do i=$((i+1))
    tl=$(timeout 600 python3 tooltest.py 2>&1 | grep -c "tool_calls") || true
    s=$(story); p=$(prompt 8000)
    [ "$tl" -ge 2 ] || e=$((e+1)); echo "$i tools $tl story $s prompt $p" >> $O/soak.txt; done
  log "  $i iterations, $e tool-call failures, restarts: $(wc -l < $K/logs/restarts.log 2>/dev/null || echo 0), last: $(tail -1 $O/soak.txt)"; }
final() { stop_all; (cd $K && scripts/serve_prod.sh) | tee -a $SUM; }

STEPS=(
 "A1_slow_1k:slow 1000" "A2_slow_2k:slow 2000" "A3_slow_4k:slow 4000" "A4_slow_8k:slow 8000" "A5_slow_16k:slow 16000"
 "A6_hang_jpf1:hang 1" "A7_hang_jpf0:hang 0"
 "B01_mtp8k:mtpb 8000" "B02_mtp16k:mtpb 16000" "B03_mtp32k:mtpb 32000"
 "B04_mtp16k_jpf0:mtpb 16000 JI_JPF=0" "B05_mtp32k_jpf0:mtpb 32000 JI_JPF=0"
 "B06_mtp16k_win0:mtpb 16000 JI_MTP_PF_WIN=0" "B07_mtp16k_sw2:mtpb 16000 JI_CACHE_SWAPS=2"
 "B08_mtp16k_margin2200:mtpb 16000 JI_TP_MARGIN_MIB=2200" "B09_mtp4k:mtpb 4000"
 "C1_kl_code31:kl L_code" "C2_kl_code64:kl L_code64"
 "E1_fixture32k:fixture 32000" "E2_fixture150k:fixture 150000" "E3_fixture200k:fixture 200000"
 "D1_soak:soak" "Z_final:final")
log "=== campaign (re)start $(date)"
for st in "${STEPS[@]}"; do
  name=${st%%:*}; cmd=${st#*:}
  [ -f $S/$name.done ] && continue
  t=$(( $(cat $S/$name.tries 2>/dev/null || echo 0) + 1 )); echo $t > $S/$name.tries
  if [ $t -gt 2 ]; then log "== $name SKIPPED (failed / hung twice)"; touch $S/$name.done; continue; fi
  log "== $name (try $t) $(date +%H:%M)"
  $cmd; touch $S/$name.done
done
log "=== campaign done $(date)"
