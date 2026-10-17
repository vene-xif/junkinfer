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
  # from the VRAM left after weights, KV cache and compute buffers (JI_CACHE_MIB=auto), plus elastic expert slots
  # in the unused KV tail and in the ggml compute buffer (idle while JDE decodes). q8_0 KV (with llama.cpp's
  # Hadamard rotation, which JDE implements) at the full 262k context. Prefill stays on the ggml path.
  # ubatch 4096 (Oct 2026): fuller MoE GEMM tiles (EXL3 prefill 394 t/s on 6.6k vs 361 at 2048, 298 vs 278 at 37k);
  # its larger compute buffer (2.75 GiB, mostly the KQ mask) costs decode nothing because JDE uses it for expert
  # slots - which needs one buffer: ggml's 1 GiB suballocation blocks are raised to 3.75 GiB (RADV allows 4 GiB).
  export GGML_VK_SUBALLOCATION_BLOCK_SIZE=${GGML_VK_SUBALLOCATION_BLOCK_SIZE:-4026531840}
  # --load-mode none puts the CPU experts in pinned host memory, so op-offload uploads are async DMA instead of
  # staged synchronous copies (prefill +35%). Needs a GTT limit above the expert size: the kernel default is half
  # of RAM (16 GB on AUX-1); AUX-1 boots with ttm.pages_limit=6291456 amdgpu.gttsize=24576 (24 GB).
  export JI_DECODE=1 JI_CACHE_MIB=${JI_CACHE_MIB:-auto} JI_CACHE_SWAPS=${JI_CACHE_SWAPS:-2}
  UB=${JI_UBATCH:-4096}
  # JI_SLOT_DIR: enables slot save/restore (POST /slots/0?action=save|restore {"filename": ...}) - a prefilled deep
  # context can be kept on disk and restored in seconds instead of minutes of prefill
  SLOT_ARGS=()
  [ -n "${JI_SLOT_DIR:-}" ] && SLOT_ARGS=(--slot-save-path "$JI_SLOT_DIR")
  exec "$ENGINE/llama-server" -m "$MODEL" \
    -ngl 99 --cpu-moe --no-repack --load-mode none -fa on --fit off \
    -c "${JI_CTX:-262144}" -ctk q8_0 -ctv q8_0 \
    -np 1 --jinja -t "${JI_THREADS:-6}" -b "$UB" -ub "$UB" \
    --host "${JI_HOST:-127.0.0.1}" --port "${JI_PORT:-8080}" \
    "${SLOT_ARGS[@]}" "$@"
fi
# JI_ENGINE=ggml: tuned mainline llama.cpp Vulkan path (MVP v0), q8_0 KV, full 262k context
exec "$ENGINE/llama-server" -m "$MODEL" \
  -ngl 99 -ncmoe "${JI_NCMOE:-34}" -fa on --fit off \
  -c "${JI_CTX:-262144}" -ctk q8_0 -ctv q8_0 \
  -np 1 --jinja \
  --host "${JI_HOST:-127.0.0.1}" --port "${JI_PORT:-8080}" \
  "$@"
