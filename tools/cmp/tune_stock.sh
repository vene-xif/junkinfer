#!/bin/bash
# Tune stock llama.cpp expert placement: tune_stock.sh "1|2 ncmoe cachemib" ... ; results to ~/cmp/tune.log
cd ~/cmp
CT=$HOME/junkinfer/ref/templates/qwen_sharp.jinja
STOCK=~/llama.cpp-master/build/bin/llama-server
IQ4=~/models/ssd-overflow/Qwen3.6-35B-IQ4_NL.gguf
for cfg in "$@"; do
  set -- $cfg; CARDS=$1; NC=$2; MC=$3
  VIS=""; [ "$CARDS" = 1 ] && VIS="GGML_VK_VISIBLE_DEVICES=1"
  env $VIS RADV_PERFTEST=nogttspill setsid nohup $STOCK -m $IQ4 -ngl 99 -ncmoe $NC --moe-cache-mib $MC --fit off -fa on -c 40960 -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 -t 6 -np 1 --jinja --chat-template-file $CT --host 127.0.0.1 --port 8080 > srv_tune.log 2>&1 < /dev/null &
  P=$!; ok=0
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:8080/health >/dev/null && { ok=1; break; }; kill -0 $P 2>/dev/null || break; sleep 5; done
  if [ $ok = 1 ]; then
    r=$(python3 - <<'PY'
import json, urllib.request
def post(c, n):
    b = {"messages": [{"role": "user", "content": c}], "max_tokens": n, "ignore_eos": True, "temperature": 0, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
    q = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(q, timeout=3600))["timings"]
t = post("Tell me a short story about a lighthouse keeper.", 3072)
txt = open("/home/vene/tinyinfer/data/corpus_full.txt", encoding="utf-8", errors="ignore").read()[:14000]
u = post(txt + "\n\nSummarize.", 16)
print("decode %.2f t/s, prefill %d tok %.1f t/s" % (t["predicted_per_second"], u["prompt_n"], u["prompt_per_second"]))
PY
)
  else r="failed to start: $(grep -i -m1 -E "error|alloc|out of" srv_tune.log)"; fi
  echo "cards $CARDS ncmoe $NC cache $MC :: $r" >> tune.log
  kill $P 2>/dev/null; for i in $(seq 1 60); do kill -0 $P 2>/dev/null || break; sleep 1; done
  sleep 3
done
echo TUNE_DONE >> tune.log
