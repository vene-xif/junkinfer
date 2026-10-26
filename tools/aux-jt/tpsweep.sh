#!/bin/bash
# JI_TP sweep: short decode (256 tokens, ignore eos) per config
tg() { for i in 1 2 3; do curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Write a long story about a lighthouse keeper."}],"max_tokens":256,"temperature":0,"ignore_eos":true,"chat_template_kwargs":{"enable_thinking":false}}' | python3 -c 'import json,sys; d=json.load(sys.stdin); print(round(d["timings"]["predicted_per_second"],1), end=" ")'; done; echo; }
for cfg in "$@"; do
  ~/jt/restart.sh $cfg JI_TP_HOT=/home/vene/jt/route_counts.bin JI_PROFILE=256 >/dev/null 2>&1
  echo -n "$cfg: "; tg
  grep -a "TP helper\|TP step lost" ~/jt/srv.log | cut -c14-140
  grep -a "jde_prof: per token" ~/jt/srv.log | tail -1 | cut -c14-120
done
P=$(pgrep -f "[b]in/llama-server -m /mnt/ssd"); [ -n "$P" ] && kill $P
