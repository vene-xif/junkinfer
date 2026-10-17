# STATE — JunkInfer

> Live record. Update sections in place. History goes to `engineering-log.md`.

## Goal

Make Qwen3.6-35B-A3B (`qwen35moe` and fine-tunes) genuinely usable on RX 580 8 GB cards.
Targets (dev box AUX-1; see README for the table):

- single card: decode >= 50 t/s at any depth to 262k; prefill as high as physics allows
- dual card: decode >= 80 t/s; prefill >= 1000 t/s at zero context, >= 500 t/s at 262k
- weights quality >= EXL3 3.5 bpw, KV quality >= q8_0 KV (KLD vs BF16, `quality-gates.md`)
- MVP must include: OpenAI-compatible server, tool-call parsing, chat templates, prompt/prefix caching
- release runs on one card; Linux; RX 470/570/580 8 GB first

## Where we are (2026-10-16)

- **EXL3 served end to end** (community EXL3 4.0 bpw, `tools/exl3/exl3_to_gguf.py` -> 17.2 GiB GGUF on the SSD):
  `scripts/serve.sh /mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf` at 262k / q8_0 KV: prefill 310 t/s (6.6k),
  239 t/s (29k); decode 18-27 t/s (JDE: EXL3 mat-vec kernels, zero-copy GPU misses); smoke_api ALL PASS.
  IQ4_NL for comparison: prefill 353 / 267, decode 28-46. Quality vs BF16 not measured yet (VENE GPU).
  (2026-10-17: expert uploads now overlap compute - weight prefetcher with alias mode + narrow copy shader.)
  (2026-10-17: own GCN prefill flash-attention kernel, default on: FA -25%; EXL3 prefill 337 t/s at 6.6k,
  262 t/s at 37k with no prefix reuse (233 with ggml's FA).)
  (2026-10-17 night: prefill 361 t/s at 6.6k, 275 at 37k, 160 at 116k; decode 28-32 t/s short, 16 t/s at 116k
  after the EXL3 cache margin cut, 1830 MiB expert cache.)
  (2026-10-18: + compute-buffer expert slots, one-shot tokens: decode ~36 t/s short, ~21-23 t/s at 37k.)
- JDE elastic-slot NaN bug (garbage after a long prompt that follows an earlier request) fixed 2026-10-17.

- **MVP v1 = JDE in llama-server** (`scripts/serve.sh`, default `JI_ENGINE=jde`): full **262k context, q8_0 KV**
  (JDE implements llama.cpp's quantised-KV Hadamard rotation), expert cache auto-sized from free VRAM (2.8 GB),
  ubatch 2048, experts pinned (`--load-mode none`, needs AUX-1's 24 GB GTT boot params), elastic expert slots in
  the unused KV tail. Warm decode **45-46 t/s**, prefill **306 t/s** (6.7k tokens), **291 t/s at 33.6k** (115 s);
  decode at 33.6k depth 39 t/s. `smoke_api.py` ALL PASS.
  Shorter context buys speed: `JI_CTX=32768 JI_UBATCH=2048` -> decode 46 t/s, prefill 215 t/s.
- JDE in the KLD harness: 21.1 ms/token at short depth, 39 t/s over a 2048-token row, KL at the noise floor
  (engine e08c04714). See engineering-log 2026-10-16.
- **VENE GPU is off-limits** (3090 Xid 79 twice under exl3 load). Remaining reference work (EXL3 bars, KLD
  evals of the quants) needs a CPU plan. BF16 reference dumps (`data/refs`) are complete and usable.

### Earlier (2026-10-15)

- **Reference run on VENE in progress** (`ref/run_ref.sh`, log `ref-run.log`). Produces the BF16
  reference, routing traces, imatrix, GGUF quants, EXL3 3.0/3.5/4.0, and the KLD bars.
- **AUX-1 box fixed:** `amdgpu.dpm=0` removed from grub (it was left in from vLLM wave W45 and
  silently fixed the GPU at boot clocks). RAM upgraded to 32 GB.
- **Baseline (mainline llama.cpp 8bb3cf0, Vulkan, DPM on, 32 GB, IQ4_NL 19.3 GiB, -fa 1,
  RADV_PERFTEST=nogttspill, 1 repeat, smoke):**

  | config | pp512 | tg32 |
  |---|---|---|
  | `-ncmoe 40` (all experts on CPU) | 136.1 t/s | 19.9 t/s |
  | `-ncmoe 29` (11 layers of experts on GPU) | 155.3 t/s | 21.9 t/s |

- Bus: host-to-device 6.8 / 7.9 GB/s (pageable / pinned) through ggml-vulkan, on a PCIe 3.0 x16
  link (15.75 GB/s theoretical). Same as with DPM off: the bus limit is the copy path, not clocks.

## MVP (2026-10-15)

`scripts/serve.sh`: engine fork 942a676f0+ (mainline b76b13379, Vulkan), Qwen3.6-35B IQ4_NL, 262,144 context,
q8_0 KV, `-ncmoe 34`, flash attention, `--jinja`. `scripts/smoke_api.py` ALL PASS: chat via the model's
template, tool call parsed into `tool_calls` (finish_reason tool_calls), prefix cache reused
(6,689-token prompt -> 23 new tokens, 49.5 s -> 1.4 s). Prefill 137 t/s (6.7k tokens), decode 18.8-21.4 t/s
at short depth. Smoke numbers, one run each.

## Plan (one feature at a time, KLD gate before speed claims)

| # | step | status |
|---|---|---|
| 0 | MVP: pinned mainline llama.cpp Vulkan build + server on AUX-1, best mainline config, recorded | DONE 2026-10-15 (`scripts/serve.sh`, `scripts/smoke_api.py` ALL PASS) |
| 1 | Fast upload path: measured (11.2 GB/s batched DMA; BAR writes 7.9); ggml-vulkan async staging ring | DONE 2026-10-15 |
| 2a | Hybrid expert execution, serial (GPU hits + CPU misses, adaptive LFU cache) | DONE, correct, no speed gain yet (GPU half is the bottleneck) |
| 2b | gfx803 decode efficiency: measured (ggml: ~1,500 dependent dispatches x 7-10 us bubbles, matvecs at 10-108 GB/s); GCN matvec in LDS reaches 192-229 GB/s | DONE (diagnosis + kernel lab) |
| 3 | (IN PROGRESS) Decode engine JDE (`decode-engine.md`): native-layout kernels, CPU overlap, replayed command buffers | 21 -> 39 t/s, KL at floor; next: async cache update, LM head/router, MTP |
| 2c | Overlap CPU misses with GPU hits; tune swap churn | |
| 3 | Prefill expert streaming per layer with large ubatch (BELLS staging done right) | |
| 4 | MTP speculative decoding with rejection sampling | IMPLEMENTED (JDE verifies 1-4 tokens, server `--spec-type draft-mtp`), correct; not faster yet (MoE scaling, see log) - off by default |
| 5 | KVarN KV cache on gfx803 (Vulkan) | |
| 6 | Weight format by KLD gate: IQ4_XS / ik KT / EXL3 transcode | |
| 7 | Sparse attention for decode (Quest-style) and prefill (XAttention-style), KLD-gated | |
| 8 | Second RX 580: layer split with resident experts | waiting for hardware |
| 9 | CPU kernels for missed experts (ik_llama iqk / Strata AVX2) | |

## Decode cost model (measured 2026-10-15, b=1, IQ4_NL)

GPU op time ~25 ms/token with no experts on the GPU (should be ~8 ms). CPU experts (all 40 layers) + syncs
~25 ms. Hybrid at 76% GPU hits: same total, because the GPU half got slower by what the CPU half saved.

## Open questions

- Does mainline's new `moe_cache_size` context param (device expert cache) work on Vulkan? Test in step 2.
- Why does ggml-vulkan's host-to-device path stop at ~8 GB/s on gen3 x16? (step 1)

## Resume point (2026-10-16 ~01:00)

JDE milestones 1-3 DONE (v1 correct, KL 0.0007 vs llama.cpp); next: profile, hybrid GPU experts + overlap, replay (engine repo, branch junkinfer): accessor API in ggml-vulkan, 10 shaders in
`src/junkinfer/shaders` (all compile for gfx803), `src/junkinfer/jde.h`. Next: `jde.cpp` host engine (per-layer
GPU segments + CPU experts via ggml type traits, re-recorded per token), CMake shader build, hook in
`llama_context::decode` before `process_ubatch`; gate: llama-junk-ref -b 1 compare, KL at the noise floor.
Requirements found: experts must not be in CPU_REPACK buffers (`--no-repack`); KV f16 + flash attn for v1.
VENE reference run continues unattended (`ref-run.log`, `./ref/run_ref.sh --list`).
