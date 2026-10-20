#!/usr/bin/env bash
# JunkInfer server on one RX 580 (MVP: tuned mainline llama.cpp Vulkan from the engine fork).
# usage: scripts/serve.sh [model.gguf] [extra llama-server args...]
#
# Environment (all optional):
#   JI_HOST=0.0.0.0        listen address (default 127.0.0.1: local only)    JI_PORT=8080
#   JI_API_KEY=secret      require "Authorization: Bearer secret" (OpenAI) / "x-api-key" (Anthropic)
#   JI_ALIAS=name          model name reported to clients (default qwen3.6-35b-a3b)
#   JI_THINKING=off        default thinking off (clients can still enable it per request with
#                          chat_template_kwargs {"enable_thinking": true}); default: on
#   JI_CTX=262144          context length    JI_CACHE_RAM=8192  host-RAM prompt cache (MiB) for switching conversations
#   JI_SLOT_DIR=dir        enable slot save/restore to disk
#   Sparse attention (default on, a deliberate trade: not free, worth it): prefill from 32k context (JI_SPARSE_PF=8192,
#   the prompt's last 4096 positions exact; +30 % at 80k, ~2.4x at 262k) and decode from 32k (JI_SPARSE=16384, +8 % at
#   72k, +44 % at 245k); KL vs full attention ~0.00045 for both together at 130k (on top of the weight / KV
#   quantization error). JI_SPARSE_PF=0 / JI_SPARSE=0 turn them off.
#   Other qwen35moe fine-tunes (exl3_to_gguf.py): MiniCPM-V 4.7 EXL3 3.74 bpw (mul1, 6-8 bit dense layers) needs
#   JI_UBATCH=2048 at 262k context (its larger dense weights + the 4096-token compute buffer over-commit VRAM).
#   JI_SP_DENSE_LAYERS=3,7,11: max-quality decode (sparse decode KL ~0.00033 -> ~0.00005, the harness floor) for -6 % at
#   100k / -13 % decode at 245k.
# Endpoints: OpenAI /v1/chat/completions, /v1/completions, /v1/responses, /v1/models; Anthropic /v1/messages.
set -euo pipefail
ENGINE=${JUNKINFER_ENGINE:-$HOME/junkinfer/engine/build/bin}
DEFAULT_MODEL=/mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf
[ -f "$DEFAULT_MODEL" ] || DEFAULT_MODEL=/mnt/ssd/models/Qwen3.6-35B-IQ4_NL.gguf
MODEL=${1:-$DEFAULT_MODEL}
shift || true

# client-facing options shared by both engines
API_ARGS=(--alias "${JI_ALIAS:-qwen3.6-35b-a3b}" --cache-ram "${JI_CACHE_RAM:-8192}")
[ -n "${JI_API_KEY:-}" ] && API_ARGS+=(--api-key "$JI_API_KEY")
case "${JI_THINKING:-on}" in
  off|0|false|no) API_ARGS+=(--reasoning off) ;;
esac

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
    "${SLOT_ARGS[@]}" "${API_ARGS[@]}" "$@"
fi
# JI_ENGINE=ggml: tuned mainline llama.cpp Vulkan path (MVP v0), q8_0 KV, full 262k context
exec "$ENGINE/llama-server" -m "$MODEL" \
  -ngl 99 -ncmoe "${JI_NCMOE:-34}" -fa on --fit off \
  -c "${JI_CTX:-262144}" -ctk q8_0 -ctv q8_0 \
  -np 1 --jinja \
  --host "${JI_HOST:-127.0.0.1}" --port "${JI_PORT:-8080}" \
  "${API_ARGS[@]}" "$@"
