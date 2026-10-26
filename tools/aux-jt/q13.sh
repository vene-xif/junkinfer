#!/bin/bash
# single vs dual comparison: short decode, depth decode (saved KV), server prefill of a 68k prompt
export JI_IGNORE_EOS=1
short() { for i in 1 2 3; do curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Write a short poem about the sea."}],"max_tokens":128,"temperature":0,"ignore_eos":true,"chat_template_kwargs":{"enable_thinking":false}}' | python3 -c 'import json,sys; d=json.load(sys.stdin); print(round(d["timings"]["predicted_per_second"],1), end=" ")'; done; echo; }
pf() { python3 - <<'PY'
import json, urllib.request
p = open('/home/vene/jt/p68k.txt').read()
r = urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8080/completion', json.dumps({"prompt": p, "n_predict": 1, "cache_prompt": False}).encode(), {"Content-Type": "application/json"}), timeout=3000)
t = json.load(r)["timings"]; print(t["prompt_n"], "tokens", round(t["prompt_per_second"], 1), "t/s")
PY
}
depth() { for N in 100000 245000; do echo -n "  depth $N: "; (cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache timeout 500 scripts/depth_kv.sh $N 6 2>&1 | grep -o "decode 128 tokens at [0-9.]*" | awk '{print $5}' | tail -4 | tr '\n' ' '); echo; done; }
kills() { P=$(pgrep -f "[b]in/llama-server -m /mnt/ssd"); [ -n "$P" ] && kill $P; sleep 3; }
for cfg in "JI_DUAL=31 JI_UBATCH=1024" "" "JI_DUAL=30 JI_UBATCH=1024" "JI_DUAL=31 JI_UBATCH=1536"; do
  kills; ~/jt/restart.sh $cfg JI_SLOT_DIR=/mnt/ssd/kvcache >/dev/null 2>&1
  echo "== ${cfg:-single} | vram $(cat /sys/class/drm/card0/device/mem_info_vram_used) gtt0 $(cat /sys/class/drm/card0/device/mem_info_gtt_used)"
  echo -n "  short: "; short
  depth
  echo -n "  prefill: "; pf
  echo "  gtt0 after: $(cat /sys/class/drm/card0/device/mem_info_gtt_used)"
done
kills
