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

## Current state (2026-10-26) — PROJECT PAUSED

Handoff for the next engineer / agent: `../../HANDOFF.md` (access, rules, workflow, open work in order).

- **Production:** boot service `junkinfer.service` on AUX-1 (`scripts/serve_boot.sh`): LAN `0.0.0.0:8080`, two cards
  -> TP2 decode + two-card JDE prefill + MTP (one card -> single-card JDE), q8_0 KV, 262k context, sparse attention,
  Qwen-Sharp chat template, thinking on, auto-restart. `ji-thermal-guard.service` logs GPU temperatures. Verified
  across two reboots; `smoke_api.py` ALL PASS (2026-10-26). Engine 4f5d347ee, project 3f44c65+.
- **Served model: Sage-A1.5** (since 2026-10-26, user; Qwen3.6 fine-tune, same architecture / tokenizer, EXL3 4.0
  bpw mul1 codebook): TP2, no MTP (the Qwen3.6 draft head doesn't help it), decode 39.6 / 45.0 / 41.7 t/s
  (rewrite / story / explain), prefill 541 t/s (6.8k); Qwen3.6 same day, no MTP: 43.2 / 51.2 / 48.0 (mul1 mat-vecs
  ~20 % slower). Details HANDOFF.md §2. Qwen3.6 numbers below.
- **Speed (100 W cap, >= 1024 tokens):** decode short 51.6 / 61.1 / 55.8 t/s (TP2 + MTP), 200k 32.4 (no MTP);
  prefill ~550 t/s (7k), 288-293 t/s (151k). Targets not met: decode 80, prefill 1000 / 500.
- **Quality:** sparse end to end KL 0.00125 (130k) / 0.00165 (261k) vs full attention (q8_0 KV itself ~0.001).
- **KVarN-6 (`JI_KVARN=1`): opt-in, not production-ready.** Decode from restored states KL 0.0005-0.0011; MTP
  short decode at q8 parity since the ring-in-tail fix (4f5d347ee: 32.9 / 34.1 / 34.5 -> 50.6 / 60.8 / 55.3 t/s).
  Open: prefill from scratch > 64k wrong (130k KL 0.27), decode at 200k 25-27 vs 32.4 t/s.
- **Hardware:** two power-offs without log entries under two-card load (2026-10-24, 2026-10-25), not thermal
  (GPUs <= 70 C): power delivery suspected (new PSU pending; 100 W cap stays). TP2 prefill ring hang S10 open.
- **Next (in order):** HANDOFF.md §9 — KVarN 130k prefill (job40 ready), KVarN 200k decode profile, K6V5/K6V4,
  S10 soak, API suite (needs pip permission), single-card release config, throughput levers (open-paths.md).

## Where we are (2026-10-19) — SUPERSEDED 2026-10-26 (see above)

- **MVP usable** (README "Using it"): `scripts/serve.sh` (EXL3 4.0 bpw default, alias qwen3.6-35b-a3b, JI_HOST /
  JI_API_KEY / JI_THINKING / JI_CACHE_RAM), `scripts/junkinfer.service`. `scripts/api_suite.py` (OpenAI + Anthropic
  SDKs): 45/45 - chat, thinking separation, streaming, tools (parallel, typed args, code arguments, 24-tool agent
  set, round trips, loops, tool_choice none/required/named), JSON / JSON schema, prompt caching across turns and tool
  loops, Completions, Responses, Anthropic Messages (tools, tool_choice, thinking, streaming, Claude-Code-style),
  disconnects, queued requests, long-output coherence; `--long`: 117k-token needle via the API (sparse attention).
- Dual card (2026-10-20): see resume point.
- Speed (2026-10-19/20): decode ~44 t/s short, ~35 at 72k, ~32 at 100k, 29.6 at 245k; prefill ~450 t/s (7k), 305-319
  (132k), 317 (117k needle). Sparse attention KL ~0.00045 at 130k. API suite 46/46.

## Resume point (2026-10-24) — SUPERSEDED 2026-10-26

Sparse attention re-tuned on corrected v2 references (features/sparse-attention.md): decode 200k 32.4 t/s at KL 0.0011,
prefill 151k ~290 t/s at KL 0.0014. KL references: ~/jt/*_v2.jref + jref_v2/ (kl_v2.sh) - the v1 ones are corrupt.
AUX1 boots with amdgpu.gpu_recovery=0. Production (serve_prod.sh) stopped since 2026-10-23 (user: not needed for now).
Next: KVarN-6 (sealed path broken: KL ~1.7; save/restore hooks fixed), incl. asymmetric K6V5 / K6V4 (user-approved);
then MVP hardening (S10 reliability soak, API suite, single-card release config, graph-path bug).

## Resume point (2026-10-23, later) — SUPERSEDED

Production: `scripts/serve_prod.sh` = TP2 + JDE prefill + MTP (JI_MTP=1, margin 1900) + JI_CACHE_SWAPS=16 + Qwen-Sharp
template + supervisor (auto-restart) + CPU governor performance (needs `scripts/cpu_performance.sh install` once
for reboots). Fresh decode 49-59 t/s (600-token replies), prefill ~530 t/s (6.8k), 375 t/s for 7.5k at 72k.
Fixed today: post-prefill slowdown (governor), MTP garbage after restore + prefill, TP-loss crash path.
Next: KVarN-6 integration (kernels kvr_seal / attn_hmr + harness written, JI_ATTN_TEST=n JI_ATTN_KVR2=1).

## Resume point (2026-10-21, two RX 580s) — SUPERSEDED

AUX-1: Vulkan1 = x16 card, Vulkan0 = chipset x4 card. Best dual mode: `JI_TP=2 scripts/serve.sh` (engine c4f8da270+):
decode short 59.1, 100k 45.2, 245k 41.1 t/s (single 46.3 / 33.8 / 30.0); code rewrite + n-gram spec 50.6 (single 32.6);
prefill = single (~348 t/s at 80k).
JI_TP=1 (MoE only) and JI_DUAL=32 (layer split) remain. kl.sh TP=2 / TP=1 / DUAL=S. Profiling: JI_TP_PROF=64.
MTP on JDE (helper GPU, opt-in): correct and lossless, no decode gain (verify cost); see the 2026-10-21 log.
Two-card prefill (2026-10-22, 100 W cap until the PSU swap): `JI_TP=2 JI_JPF=1 JI_TP_MARGIN_MIB=1500` (JDE TP
prefill, sparse attention with 128-key pages, exchange queue, BK16 GEMMs). 100 W: fresh 7.2k prompt 555 t/s warm
(507 first after start; graph path 484 at 145 W), 4096-token ubatch 630, restored 72k ~395, steady 247k ubatch 467.
KL unchanged (0.0003-0.0007 vs full attention at 130k / 261k). Fixed TP2 sparse decode metadata (KL 0.04 -> 0.0004).
Both GPUs are power-bound at 100 W (80 W: 488 t/s vs 629 - throughput ~ watts; see the 2026-10-22 list results
in the log: work/energy cuts only). Open toward 1000 t/s: EXL3 GEMM efficiency (~45 % of peak; 64 VGPR, 4 waves /
SIMD, no spills - barrier / LDS-latency bound: double-buffered LDS), TP overhead (~1.1 s of GEMM + ~1.8 s on the helper
not attributed per ubatch), JI_PF_NMB=4 fails. Re-measure everything at 145 W after the PSU swap. Prefill tests at
depth: `~/jt/dpb.py N [extra]` (saved KV; the first ubatch after a restore copies the helper's kv head).
Vision (mmproj approved) later.

## Open optimization paths

Full ordered list (effort tiers, status, expected gains, rejected items): `open-paths.md`.

## Earlier status (2026-10-16)

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
  (2026-10-18: serve.sh ub 4096 + 3.75 GiB suballocation blocks: prefill 394 t/s at 6.6k, 298 at 37k.)
  (2026-10-18 later: decode 44 t/s short (stripe-major EXL3 mat-vecs, fused output stage); 247k-token prompt:
  prefill 90.6 t/s, decode 13.7 t/s at that depth; 116k: 161 / 20.1 t/s. Long-context fixes: FA query chunks
  (2 s GPU job timeout), q8_0 KV read directly by the FA kernel (no context-sized scratch -> no KV eviction).)
  (2026-10-18 night: warm decode from saved contexts (scripts/depth_kv.sh): 37k 33.9, 116k 23.9, 247k 18.2 t/s.)
  (2026-10-18 late: prefill FA ji_fa2 (scalar-register query rows, 2 keys per lane): 247k-token prompt 90.6 ->
  97.3 t/s before the 2-keys-per-lane step (8k FA a further -22 %), 68k-token prompt 346 -> 290 s; decode attention
  attn_d2 + parallel attn_merge: 247k decode 17.9 -> ~19.7 t/s warm, attention 20.8 ms + MoE ~13 ms of ~51 ms
  GPU per token. Decode attention is bound by its KV load pattern (probes in the engineering log).)
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
| 7 | Sparse attention for decode (Quest-style) and prefill (XAttention-style), KLD-gated | DEFAULT ON: decode from 32k (raw-basis page statistics + variance), prefill from 32k (exact last 4096 positions); KL ~0.00045 together at 130k |
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
