#!/usr/bin/env bash
# JunkInfer server on one RX 580 (MVP: tuned mainline llama.cpp Vulkan from the engine fork).
# usage: scripts/serve.sh [model.gguf] [extra llama-server args...]
#
# Environment (all optional):
#   JI_HOST=0.0.0.0        listen address (default 127.0.0.1: local only)    JI_PORT=8080
#   JI_API_KEY=secret      require "Authorization: Bearer secret" (OpenAI) / "x-api-key" (Anthropic)
#   JI_MODEL=path          model: a GGUF or an exllamav3 checkpoint directory (default: Sage-A1.5 EXL3 4.0 bpw directory,
#                          else its GGUF, else Qwen3.6-35B-A3B EXL3 4.0 bpw)
#   JI_ALIAS=name          model name reported to clients (default: sage-a1.5 / qwen3.6-35b-a3b by model file)
#   JI_THINKING=off        default thinking off (clients can still enable it per request with
#                          chat_template_kwargs {"enable_thinking": true}); default: on
#   JI_CTX=262144          context length    JI_CACHE_RAM=8192  host-RAM prompt cache (MiB) for switching conversations
#   JI_SLOT_DIR=dir        enable slot save/restore to disk
#   Sparse attention (default on, a deliberate trade: not free, worth it): prefill from 32k context (JI_SPARSE_PF=8192,
#   the prompt's last 4096 positions exact; +30 % at 80k, ~2.4x at 262k) and decode from 32k (JI_SPARSE=65536 keys since
#   2026-10-23: 200k decode 25.6 dense -> 31.5 t/s, decode KL vs full attention 0.0009 at 130k / 0.0036 at 261k; the old
#   16384 gave 0.0115 / 0.0177 for ~the same speed - measured on the corrected v2 references, engineering log 2026-10-23).
#   JI_SPARSE_PF=0 / JI_SPARSE=0 turn them off.
#   Other qwen35moe fine-tunes (exl3_to_gguf.py): MiniCPM-V 4.7 EXL3 3.74 bpw (mul1, 6-8 bit dense layers) needs
#   JI_UBATCH=2048 at 262k context (its larger dense weights + the 4096-token compute buffer over-commit VRAM).
#   JI_SP_DENSE_LAYERS=3,7,11: max-quality decode (sparse decode KL ~0.00033 -> ~0.00005, the harness floor) for -6 % at
#   100k / -13 % decode at 245k.
# Endpoints: OpenAI /v1/chat/completions, /v1/completions, /v1/responses, /v1/models; Anthropic /v1/messages.
set -euo pipefail
ENGINE=${JUNKINFER_ENGINE:-$HOME/junkinfer/engine/build/bin}
# default model (2026-10-26, user): Sage-A1.5 (InternScience, a Qwen3.5/3.6-MoE fine-tune, same architecture and
# tokenizer; jyohukuchan/Sage-A1.5-exl3-4.0bpw converted with tools/exl3/exl3_to_gguf.py), else Qwen3.6-35B-A3B
# A model may be an exllamav3 checkpoint directory: loaded natively, no GGUF conversion (engine llama-exl3.cpp, 2026-10-31)
DEFAULT_MODEL=/mnt/ssd/models/Sage-A1.5-exl3-4.0bpw
[ -e "$DEFAULT_MODEL" ] || DEFAULT_MODEL=/mnt/ssd/models/Sage-A1.5-exl3-4.00bpw.gguf
[ -e "$DEFAULT_MODEL" ] || DEFAULT_MODEL=/mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf
[ -e "$DEFAULT_MODEL" ] || DEFAULT_MODEL=/mnt/ssd/models/Qwen3.6-35B-IQ4_NL.gguf
MODEL=${1:-${JI_MODEL:-$DEFAULT_MODEL}}
case "$(basename "$MODEL")" in
  Sage-A1.5*) DEF_ALIAS=sage-a1.5 ;;
  MiniCPM-V-4.7*) DEF_ALIAS=minicpm-v-4.7 ;;
  *) DEF_ALIAS=qwen3.6-35b-a3b ;;
esac
shift || true

# client-facing options shared by both engines
API_ARGS=(--alias "${JI_ALIAS:-$DEF_ALIAS}" --cache-ram "${JI_CACHE_RAM:-8192}")
# chat template: JI_CHAT_TEMPLATE (default: Qwen-Sharp, ref/templates/qwen_sharp.jinja - peculiar-ragdoll/Qwen-Sharp-
# Chat-Templates; fetch: curl -L -o ref/templates/qwen_sharp.jinja https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates/resolve/main/chat_template.jinja
# - agents such as Hermes got tool calls as plain text with the GGUF's embedded template). JI_CHAT_TEMPLATE=embedded: the GGUF's own
CT=${JI_CHAT_TEMPLATE:-$(cd "$(dirname "$0")/.." && pwd)/ref/templates/qwen_sharp.jinja}
[ "$CT" != embedded ] && [ -f "$CT" ] && API_ARGS+=(--chat-template-file "$CT")
[ -n "${JI_API_KEY:-}" ] && API_ARGS+=(--api-key "$JI_API_KEY")
case "${JI_THINKING:-on}" in
  off|0|false|no) API_ARGS+=(--reasoning off) ;;
esac

# RADV: keep device-local allocations out of GTT when VRAM is nearly full (2.5x decode on this card, Aug 2026)
export RADV_PERFTEST=${RADV_PERFTEST-nogttspill}
# Which GPU(s): AUX-1 has the PCIe 3.0 x16 card at Vulkan index 1 and the chipset PCIe 2.0 x4 card at index 0
# (enumeration follows the PCI bus). Single-card mode must run on the x16 card: expert misses stream over its link.
# JI_GPU=1 (default) pins it; JI_GPU=all exposes both (dual-card modes).
JI_GPU=${JI_GPU:-1}
# JI_DUAL=S (dual card): layers 0..S-1 on the x16 card (Vulkan1, experts streamed from pinned host memory and cached),
# layers S..39 + the LM head on the x4 card (Vulkan0) with their experts resident in its VRAM (~290 MiB per layer).
# JDE runs one stage per card; pinned host memory belongs to the x16 card (GGML_VK_HOST_DEVICE).
# At 262k: JI_DUAL=32 (ub 4096) - decode 245k 30.0 -> 33.0 t/s, prefill = single card; JI_DUAL=31 needs
# JI_UBATCH=2048 (245k 33.8, prefill -7 %). The x4 card must never over-commit: evicted weights run at PCIe 2.0 x4
# speed (watch /sys/class/drm/card0/device/mem_info_gtt_used: ~130 MB is normal, GBs = over-committed).
EXPERT_ARGS=(--cpu-moe)
if [ -n "${JI_DUAL:-}" ]; then
  JI_GPU=all
  export GGML_VK_HOST_DEVICE=${GGML_VK_HOST_DEVICE:-1}
  EXPERT_ARGS=(-dev Vulkan1,Vulkan0 -ts "$JI_DUAL,$((41 - JI_DUAL))" --n-cpu-moe "$JI_DUAL")
elif [ "${JI_TP:-0}" != 0 ]; then
  # JI_TP=2 (recommended dual-card decode): full tensor parallel - both cards run every layer on half the heads
  # (Gated DeltaNet, shared expert, LM head), routed experts split by ownership (the x4 card holds a hot set),
  # attention on the x16 card. Decode short 46.3 -> 58.0, 100k 33.8 -> 43.8, 245k 30.0 -> 40.0 t/s (2026-10-21).
  # JI_TP=1: only the routed experts on the x4 card (short 49.9, 100k 41.0, 245k 37.9). The model runs on the x16
  # card as in single-card mode; prefill = single card. JI_TP_HOT = routing counts file, JI_TP_FRAC = share of
  # the routing traffic the x4 card takes (TP=2 default 0.7, TP=1 1.0), JI_TP_SLOTS = cap. JI_TP_DUP (TP=2, default
  # 0.25): share of the traffic held by both cards (per-token balance). Validated: API suite 45/45, 117k needle.
  # MTP speculative decoding: JI_MTP=1 (below).
  JI_GPU=all
  export JI_TP GGML_VK_HOST_DEVICE=${GGML_VK_HOST_DEVICE:-1}
  # hot set = routing counts of the served model (JI_ROUTE_DUMP + tools/aux-jt/routegen.py 400): Sage-A1.5 routes
  # differently from Qwen3.6 (top-80 per layer overlap 36 %)
  # 2026-11-01 Sage-A1.5: hot set from the routing traces of a 24-prompt corpus (tools/routes/mkhot.py, ~19k decoded
  # tokens; the old tp_hot_sage15_exl3.bin matched the native model's routing poorly: its 2269 helper experts carried
  # 27 % of the traced selections, this set 43 % on held-out traces) and no experts on both cards (JI_TP_DUP=0: the
  # 0.25 share took ~540 primary cache slots while the helper idled 7-10 ms per step). Primary misses per token
  # 24.6 -> 13.5 (trace simulation), JDE ms per step 25.3 -> 23.8, decode short 52.3 -> 54.2, 116k 37.9 -> 39.0 t/s
  case "$DEF_ALIAS" in
    sage-a1.5) HOTF=tp_hot_sage15_native_corpus24.bin; export JI_TP_DUP=${JI_TP_DUP:-0} ;;
    *) HOTF=tp_hot_qwen36_exl3.bin ;;
  esac
  export JI_TP_HOT=${JI_TP_HOT:-$(cd "$(dirname "$0")/.." && pwd)/ref/tp/$HOTF}
  EXPERT_ARGS=(-dev Vulkan1 --cpu-moe)
fi
[ "$JI_GPU" != all ] && export GGML_VK_VISIBLE_DEVICES=$JI_GPU

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
  # JI_CACHE_SWAPS: expert-cache evictions per token (2026-10-22: 2 -> 16, decode +4-14 %: the cache adapts to a
  # conversation's experts within a few tokens instead of hundreds)
  export JI_DECODE=1 JI_CACHE_MIB=${JI_CACHE_MIB:-auto} JI_CACHE_SWAPS=${JI_CACHE_SWAPS:-16}
  UB=${JI_UBATCH:-4096}
  # JI_SLOT_DIR: enables slot save/restore (POST /slots/0?action=save|restore {"filename": ...}) - a prefilled deep
  # context can be kept on disk and restored in seconds instead of minutes of prefill
  SLOT_ARGS=()
  [ -n "${JI_SLOT_DIR:-}" ] && SLOT_ARGS=(--slot-save-path "$JI_SLOT_DIR")
  # JI_MTP=1 (JI_TP=2; serve_prod.sh default): MTP speculative decoding - the sidecar's draft head runs in JDE on the
  # x4 card (CPU-loaded sidecar, copied to that card's VRAM), one draft token, drafted when its probability >=
  # JI_MTP_PMIN (0.5), HyperQwen's 40,960-token profiled draft vocabulary, IQ4_NL head, attention over the last 8192
  # cells, prompt catch-up of the last 128 positions per ubatch. Lossless (greedy outputs identical).
  # 2026-10-23 (governor performance), 600-token replies: rewrite 43.2 -> 49.4, story 49.6 -> 58.5, explain 48.0 ->
  # 58.9 t/s; prefill -2-4 % fresh, -6 % for a 7.5k prompt on a 72k context. Needs JI_TP_MARGIN_MIB=1900.
  MTP_ARGS=()
  if [ "${JI_MTP:-0}" != 0 ]; then
    REF=$(cd "$(dirname "$0")/.." && pwd)/ref/mtp
    export JI_MTP_VOCAB_FILE=${JI_MTP_VOCAB_FILE:-$REF/hyperqwen_draft_vocab_40960.json} JI_MTP_HEAD_TYPE=${JI_MTP_HEAD_TYPE:-iq4_nl}
    # an exllamav3 checkpoint directory brings its own MTP layer: "<dir>::mtp" = its MTP view (native loader; the EXL3
    # MTP block re-quantized to IQ4_NL / Q5_K / Q6_K / Q8_0 at load); a GGUF model uses the Qwen3.6 sidecar
    # draft p-min: Sage-A1.5's own head 0.8 (2026-10-31 sweep 0.5 / 0.65 / 0.8 / 0.9 / 0.95: rewrite / story / explain
    # 43.1 / 62.2 / 51.9 .. 47.6 / 63.7 / 55.9 .. 43.7 / 61.5 / 51.7 t/s), the Qwen3.6 sidecar 0.5
    if [ -d "$MODEL" ]; then DEF_MTP="$MODEL::mtp"; DEF_PMIN=0.8; else DEF_MTP="$(dirname "$MODEL")/mtp-IQ4_NL.v2.gguf"; DEF_PMIN=0.5; fi
    MTP_ARGS=(-md "${JI_MTP_MODEL:-$DEF_MTP}" --spec-type draft-mtp -ngld 0 -devd none
              --no-spec-draft-backend-sampling --spec-draft-n-max 1 --spec-draft-p-min "${JI_MTP_PMIN:-$DEF_PMIN}")
  fi
  exec "$ENGINE/llama-server" -m "$MODEL" \
    -ngl 99 "${EXPERT_ARGS[@]}" --no-repack --load-mode none -fa on --fit off \
    -c "${JI_CTX:-262144}" -ctk q8_0 -ctv q8_0 \
    -np 1 --jinja -t "${JI_THREADS:-6}" -b "$UB" -ub "$UB" \
    --host "${JI_HOST:-127.0.0.1}" --port "${JI_PORT:-8080}" \
    "${SLOT_ARGS[@]}" "${MTP_ARGS[@]}" "${API_ARGS[@]}" "$@"
fi
# JI_ENGINE=ggml: tuned mainline llama.cpp Vulkan path (MVP v0), q8_0 KV, full 262k context
exec "$ENGINE/llama-server" -m "$MODEL" \
  -ngl 99 -ncmoe "${JI_NCMOE:-34}" -fa on --fit off \
  -c "${JI_CTX:-262144}" -ctk q8_0 -ctv q8_0 \
  -np 1 --jinja \
  --host "${JI_HOST:-127.0.0.1}" --port "${JI_PORT:-8080}" \
  "${API_ARGS[@]}" "$@"
