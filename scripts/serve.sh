#!/usr/bin/env bash
# JunkInfer server on one RX 580 (MVP: tuned mainline llama.cpp Vulkan from the engine fork).
# usage: scripts/serve.sh [model.gguf] [extra llama-server args...]
set -euo pipefail
ENGINE=${JUNKINFER_ENGINE:-$HOME/junkinfer/engine/build/bin}
MODEL=${1:-/mnt/ssd/models/Qwen3.6-35B-IQ4_NL.gguf}
shift || true

# RADV: keep device-local allocations out of GTT when VRAM is nearly full (2.5x decode on this card, Aug 2026)
export RADV_PERFTEST=${RADV_PERFTEST-nogttspill}

if [ "${JI_ENGINE:-jde}" = jde ]; then
  # JunkInfer decode engine (JDE): custom Vulkan decode, experts in host RAM + an adaptive VRAM expert cache sized
  # from the VRAM left after weights, KV cache and compute buffers (JI_CACHE_MIB=auto). q8_0 KV (with llama.cpp's
  # Hadamard rotation, which JDE implements) at the full 262k context. Prefill stays on the ggml path: ubatch 1024
  # lets op-offload stream the experts once per 1024 tokens; larger ubatches cost too much VRAM at 262k
  # (the KQ mask/scores scale with ctx x ubatch) and starve the expert cache.
  export JI_DECODE=1 JI_CACHE_MIB=${JI_CACHE_MIB:-auto} JI_CACHE_SWAPS=${JI_CACHE_SWAPS:-2}
  UB=${JI_UBATCH:-1024}
  exec "$ENGINE/llama-server" -m "$MODEL" \
    -ngl 99 --cpu-moe --no-repack -fa on --fit off \
    -c "${JI_CTX:-262144}" -ctk q8_0 -ctv q8_0 \
    -np 1 --jinja -t "${JI_THREADS:-6}" -b "$UB" -ub "$UB" \
    --host "${JI_HOST:-127.0.0.1}" --port "${JI_PORT:-8080}" \
    "$@"
fi
# JI_ENGINE=ggml: tuned mainline llama.cpp Vulkan path (MVP v0), q8_0 KV, full 262k context
exec "$ENGINE/llama-server" -m "$MODEL" \
  -ngl 99 -ncmoe "${JI_NCMOE:-34}" -fa on --fit off \
  -c "${JI_CTX:-262144}" -ctk q8_0 -ctv q8_0 \
  -np 1 --jinja \
  --host "${JI_HOST:-127.0.0.1}" --port "${JI_PORT:-8080}" \
  "$@"
