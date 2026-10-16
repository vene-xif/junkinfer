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
  # JunkInfer decode engine (JDE): custom Vulkan decode, experts in host RAM + an adaptive VRAM expert cache.
  # Needs --cpu-moe --no-repack and an f16 KV cache (KVarN/q8 KV in JDE is a later step), so the context is
  # limited by VRAM: 32k f16 KV = 640 MiB. Prefill stays on the ggml path; ubatch 2048 lets op-offload stream
  # the experts to the GPU once per 2048 tokens (215 t/s vs 123 at ubatch 512).
  export JI_DECODE=1 JI_CACHE_MIB=${JI_CACHE_MIB:-4500} JI_CACHE_SWAPS=${JI_CACHE_SWAPS:-2}
  exec "$ENGINE/llama-server" -m "$MODEL" \
    -ngl 99 --cpu-moe --no-repack -fa on --fit off \
    -c "${JI_CTX:-32768}" -ctk f16 -ctv f16 \
    -np 1 --jinja -t "${JI_THREADS:-6}" -b 2048 -ub 2048 \
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
