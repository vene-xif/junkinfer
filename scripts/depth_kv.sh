#!/usr/bin/env bash
# Decode at depth from a saved KV context (server started with JI_SLOT_DIR): restores <name>.bin into slot 0 if it
# exists, otherwise prefills the depth_bench prompt of ~N tokens once and saves it. Then decodes (prompt reuse).
# N > 110000: the prompt is built from the engine sources (big_bench.py), the docs corpus is ~116k tokens.
# usage: scripts/depth_kv.sh N [runs=2]     e.g. JI_SLOT_DIR=/mnt/ssd/kvcache scripts/serve.sh ... ; scripts/depth_kv.sh 100000
set -euo pipefail
N=${1:?tokens}; RUNS=${2:-2}; NAME="depth_${N}.bin"; URL=http://127.0.0.1:8080
cd "$(dirname "$0")/.."
r=$(curl -s -X POST "$URL/slots/0?action=restore" -H "Content-Type: application/json" -d "{\"filename\":\"$NAME\"}")
if echo "$r" | grep -q n_restored; then echo "restored $NAME: $r"; else
  echo "no saved context ($NAME): prefilling"
  if [ "$N" -gt 110000 ]; then python3 scripts/big_bench.py "$N" 8 cache; else python3 scripts/depth_bench.py "$N" 8 "$URL"; fi
  curl -s -X POST "$URL/slots/0?action=save" -H "Content-Type: application/json" -d "{\"filename\":\"$NAME\"}"; echo; fi
for i in $(seq 1 "$RUNS"); do
  if [ "$N" -gt 110000 ]; then python3 scripts/big_bench.py "$N" 128 cache; else python3 scripts/depth_bench.py "$N" 128 "$URL" show; fi
done
