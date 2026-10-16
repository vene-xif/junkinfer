#!/usr/bin/env bash
# CPU-only quant KLD on VENE (its GPU is off-limits since 2026-10-16): llama-perplexity full-logit KLD vs the
# BF16 base (data/refs/ppl_wiki_c2048.kldbase), 8 chunks x 2048 tokens, gentle on the shared box
# (nice 19, 6 of 12 threads, one quant at a time; the user's own llama-server runs here too).
# usage: ref/eval_cpu.sh [--dry-run]
set -euo pipefail
cd "$(dirname "$0")/.."
D=$PWD/data; GG=$D/gguf; EVAL=$D/eval; BASE=$D/refs/ppl_wiki_c2048.kldbase
LB=$PWD/src/llama.cpp/build/bin; IKB=$PWD/src/ik_llama.cpp/build/bin
DRY=0; [ "${1:-}" = --dry-run ] && DRY=1
NT=6; CH=8
# quant -> binary (ik quants need ik_llama)
QS="Q8_0:$LB IQ4_XS:$LB Q4_K_M:$LB IQ3_S:$LB IQ4_KT:$IKB IQ3_KT:$IKB"
fail=0
[ -f "$BASE" ] || { echo "MISSING $BASE"; fail=1; }
for qb in $QS; do
  q=${qb%%:*}; b=${qb#*:}
  [ -f "$GG/Qwen3.6-35B-A3B-$q.gguf" ] || { echo "MISSING gguf $q"; fail=1; }
  [ -x "$b/llama-perplexity" ] || { echo "MISSING $b/llama-perplexity"; fail=1; }
done
[ $fail = 0 ] || exit 1
if [ $DRY = 1 ]; then echo "dry run OK: $QS"; exit 0; fi
mkdir -p "$EVAL"
for qb in $QS; do
  q=${qb%%:*}; b=${qb#*:}
  out="$EVAL/cpu-kld-$q.txt"
  if grep -q "Mean.*KLD" "$out" 2>/dev/null; then echo "skip $q (done)"; continue; fi
  echo "[$(date '+%F %T')] >>> $q"
  CUDA_VISIBLE_DEVICES= nice -n 19 "$b/llama-perplexity" -m "$GG/Qwen3.6-35B-A3B-$q.gguf" -ngl 0 -t $NT \
     -c 2048 -b 2048 -ub 512 --chunks $CH --kl-divergence-base "$BASE" --kl-divergence > "$out.tmp" 2>&1 || { echo "FAILED $q"; tail -5 "$out.tmp"; continue; }
  mv "$out.tmp" "$out"
  grep -E "Mean +KLD|Same top|Mean +PPL\(Q\)" "$out" || true
done
echo "[$(date '+%F %T')] done"
