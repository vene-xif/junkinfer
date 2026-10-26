#!/bin/bash
# job 26: KVarN speed (TP2, sparse defaults): decode 200k (1024 tok x2), prefill 151k, short spectest (MTP, 1024 tok);
# expert cache size vs q8; end-to-end KL with KVarN prefill (130k from scratch)
cd ~/jt
for v in "JI_KVARN=1" "JI_KVARN=0"; do
  NMAX=0 JI_CACHE_SWAPS=16 SARGS="-lv 4" env $v ./mtpsrv.sh >/dev/null || { echo "$v server failed"; continue; }
  echo "$v cache: $(grep -oE "expert cache [0-9.]+ MiB|elastic[^,]*slots[^,]*" ~/jt/mtp_srv.log | head -2 | tr "\n" " ")"
  for run in 1 2; do echo "$v decode 200k run $run: $(cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache scripts/depth_kv.sh 200000 1 2>&1 | grep -oE "decode [0-9]+ tokens at [0-9.]+ t/s")"; done
  echo "$v prefill 151k: $(cd ~/junkinfer && timeout 2000 python3 scripts/depth_bench.py 130000 8 http://127.0.0.1:8080 x nocache 2>&1 | grep -oE "prompt [0-9]+ tokens: prefill [0-9.]+ t/s \([0-9]+ s\)")"
done
NMAX=1 PMIN=0.5 JI_KVARN=1 JI_CACHE_SWAPS=16 JI_MTP_VOCAB_FILE=$HOME/junkinfer/ref/mtp/hyperqwen_draft_vocab_40960.json JI_MTP_HEAD_TYPE=iq4_nl ./mtpsrv.sh >/dev/null && { echo "KVarN short spectest (MTP, 1024 tokens):"; python3 spectest.py | cut -c1-90; }
P=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk "{print \$1}"); [ -n "$P" ] && kill $P
d=/mnt/ssd/kvcache/tmp_j26; rm -rf $d; mkdir -p $d
r=$(./kl_v2.sh L_code 1 JI_TP=2 JI_JPF=1 JI_KVARN=1 JR_STATE_DIR=$d 2>&1 | tail -1); rm -rf $d
echo "KVarN end-to-end (prefill from scratch) 130k: $(echo "$r" | grep -oE "KL mean [0-9.]+ p99 [0-9.]+ max [0-9.]+ top1 [0-9.]+")"
echo "ring timeouts: $(echo "$SUDO_PW" | sudo -S dmesg 2>/dev/null | grep -c "ring .* timeout")"
echo JOB26_DONE $(date +%T)
