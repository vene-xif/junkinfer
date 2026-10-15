#!/usr/bin/env bash
# JunkInfer server on one RX 580 (MVP: tuned mainline llama.cpp Vulkan from the engine fork).
# usage: scripts/serve.sh [model.gguf] [extra llama-server args...]
set -euo pipefail
ENGINE=${JUNKINFER_ENGINE:-$HOME/junkinfer/engine/build/bin}
MODEL=${1:-/mnt/ssd/models/Qwen3.6-35B-IQ4_NL.gguf}
shift || true

# RADV: keep device-local allocations out of GTT when VRAM is nearly full (2.5x decode on this card, Aug 2026)
export RADV_PERFTEST=${RADV_PERFTEST-nogttspill}

exec "$ENGINE/llama-server" -m "$MODEL" \
  -ngl 99 -ncmoe "${JI_NCMOE:-34}" -fa on --fit off \
  -c "${JI_CTX:-262144}" -ctk q8_0 -ctv q8_0 \
  -np 1 --jinja \
  --host "${JI_HOST:-127.0.0.1}" --port "${JI_PORT:-8080}" \
  "$@"
