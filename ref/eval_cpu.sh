#!/usr/bin/env bash
# CPU-only quant KLD on VENE (its GPU is off-limits since 2026-10-16): llama-perplexity full-logit KLD vs the
# BF16 base (data/refs/ppl_wiki_c2048.kldbase), 8 chunks x 2048 tokens, gentle on the shared box
# (nice 19, 4 of 12 threads, one quant at a time; the user works on this box: its GPU is never touched).
# usage: ref/eval_cpu.sh [--dry-run]
set -euo pipefail
cd "$(dirname "$0")/.."
D=$PWD/data; GG=$D/gguf; EVAL=$D/eval; BASE=$D/refs/ppl_wiki_c2048.kldbase
LB=$PWD/src/llama.cpp/build/bin; IKB=$PWD/src/ik_llama.cpp/build/bin
DRY=0; [ "${1:-}" = --dry-run ] && DRY=1
NT=4; CH=8
# quant -> binary (ik quants need ik_llama)
QS="UD-IQ4_NL:$LB Q8_0:$LB IQ4_XS:$LB Q4_K_M:$LB IQ3_S:$LB IQ4_KT:$IKB IQ3_KT:$IKB"
ANCH=/mnt/chonker/ai/models   # UD-IQ4_NL = the model JunkInfer serves on AUX-1
gguf_of() { if [ "$1" = UD-IQ4_NL ]; then echo "$ANCH/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf"; else echo "$GG/Qwen3.6-35B-A3B-$1.gguf"; fi; }
fail=0
[ -f "$BASE" ] || { echo "MISSING $BASE"; fail=1; }
for qb in $QS; do
  q=${qb%%:*}; b=${qb#*:}
  [ -f "$(gguf_of $q)" ] || { echo "MISSING gguf $q"; fail=1; }
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
  CUDA_VISIBLE_DEVICES= nice -n 19 "$b/llama-perplexity" -m "$(gguf_of $q)" -ngl 0 -t $NT \
     -c 2048 -b 2048 -ub 512 --chunks $CH --kl-divergence-base "$BASE" --kl-divergence > "$out.tmp" 2>&1 || { echo "FAILED $q"; tail -5 "$out.tmp"; continue; }
  mv "$out.tmp" "$out"
  grep -E "Mean +KLD|Same top|Mean +PPL\(Q\)" "$out" || true
done
echo "[$(date '+%F %T')] done"
