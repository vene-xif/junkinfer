#!/usr/bin/env bash
# Short decode/prefill smoke through llama-server: one ~1k-token prompt, 128 generated tokens.
# usage: scripts/smoke_decode.sh <label> [llama-server args...]     (env passes through)
# prints: label, prefill t/s, decode t/s, and any MoE-cache stats lines from the server log
set -uo pipefail
LABEL=$1; shift
ENGINE=${JUNKINFER_ENGINE:-$HOME/junkinfer/engine/build/bin}
MODEL=${JI_MODEL:-/mnt/ssd/models/Qwen3.6-35B-IQ4_NL.gguf}
PORT=${JI_PORT:-18090}
LOG=/tmp/ji-smoke-$LABEL.log
export RADV_PERFTEST=${RADV_PERFTEST-nogttspill}
"$ENGINE/llama-server" -m "$MODEL" -ngl 99 -fa on --fit off -np 1 --port $PORT "$@" > "$LOG" 2>&1 &
PID=$!
for i in $(seq 120); do
  sleep 2
  curl -sf localhost:$PORT/health >/dev/null && break
  kill -0 $PID 2>/dev/null || { echo "$LABEL: server died"; tail -5 "$LOG"; exit 1; }
done
python3 - "$LABEL" $PORT <<'EOF'
import json, sys, urllib.request
label, port = sys.argv[1], sys.argv[2]
text = ("Explain, step by step and with examples, how a CPU cache hierarchy works, why cache misses are "
        "expensive, and how software can be written to be cache friendly. ") * 12
body = {"prompt": text, "n_predict": 128, "temperature": 0, "cache_prompt": False}
r = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=900).read())
t = r["timings"]
print(f"{label}: prefill {t['prompt_per_second']:.1f} t/s ({t['prompt_n']} tok)  decode {t['predicted_per_second']:.2f} t/s ({t['predicted_n']} tok)")
EOF
kill $PID; wait $PID 2>/dev/null
grep -E "moe_cache|MoE cache" "$LOG" | sed "s/^/    /" | tail -6
