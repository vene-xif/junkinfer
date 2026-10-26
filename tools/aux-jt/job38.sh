#!/bin/bash
# job 38: KVarN rings in the KV tensors tails: build, KL (decode from restored states), MTP speed vs job 33
while ! grep -q JOB37_DONE ~/jt/job37.log 2>/dev/null; do sleep 30; done
cd ~/junkinfer/engine && cmake --build build -j12 2>&1 | grep -E " error|rror:" | head -5
cd ~/jt
for cfg in "JI_TP=0 GGML_VK_VISIBLE_DEVICES=0" "JI_TP=2"; do
  for r in 0 1; do
    out=$(./kl_v2.sh L_code $r $cfg JI_KVARN=1 2>&1 | tail -1)
    echo "$cfg r$r: $(echo "$out" | grep -oE "KL mean [0-9.]+ p99 [0-9.]+ max [0-9.]+ top1 [0-9.]+") | $(grep -oE "window ring [^)]*\)" ~/jt/kl.log | head -1) | $(grep -oE "budget available [0-9]+ MiB" ~/jt/kl.log | head -1)"
  done
done
MTPENV="JI_MTP_VOCAB_FILE=$HOME/junkinfer/ref/mtp/hyperqwen_draft_vocab_40960.json JI_MTP_HEAD_TYPE=iq4_nl"
NMAX=1 PMIN=0.5 JI_CACHE_SWAPS=16 JI_KVARN=1 env $MTPENV ./mtpsrv.sh >/dev/null && { echo "== MTP JI_KVARN=1 (rings in tail)"; python3 spectest.py | cut -c1-60; grep -E "budget available|window ring" ~/jt/mtp_srv.log | cut -c14- | head -3; }
P=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk "{print \$1}"); [ -n "$P" ] && kill $P
echo JOB38_DONE $(date +%T)
