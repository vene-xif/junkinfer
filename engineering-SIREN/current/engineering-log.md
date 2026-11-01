# Engineering log

> Chronological. Append only.

## 2026-10-15 — project start

**Survey** of Strata, BELLS, ik_llama, BeeLlama, exllamav3 (verdicts in `candidate-ledger.md`).

**AUX-1 box state was wrong** (Problem / Fix / Why / Evidence):
- Problem: `amdgpu.dpm=0` was still in `/etc/default/grub` (written 12:18, boot 12:20). Every
  2026-10-15 qwen35moe number in the prior project (y1/z1/z2/w1 sweeps, the 0.1-0.3 GB/s "copy
  wall", 5.7 t/s `--cpu-moe`) ran with the GPU at boot clocks.
- Fix: removed the parameter (backup `/etc/default/grub.bak-dpm0-20261015`), `update-grub`, reboot.
- Evidence: `/proc/cmdline` clean; `pp_dpm_sclk` 300-1340 MHz, `pp_dpm_mclk` 300-2000 MHz, level auto.
- Also found: PCIe link is 3.0 x16 (prior docs said x8); RAM is DDR4-3000 (docs said 2133).
- RAM upgraded by the user to 32 GB (4 x 8 GB).

**Re-baseline** (smoke, 1 repeat, mainline llama.cpp 8bb3cf0 Vulkan, RADV_PERFTEST=nogttspill,
Qwen3.6-35B IQ4_NL 19.3 GiB, -ngl 99 -fa 1, llama-bench -p 512 -n 32):
- `-ncmoe 40`: pp512 136.1, tg32 19.9 t/s
- `-ncmoe 29`: pp512 155.3, tg32 21.9 t/s
- host-to-device (llama-bells-selftest): 6.82 pageable / 7.93 pinned GB/s, unchanged by DPM.

**Reference run started on VENE** 22:45 (`ref/run_ref.sh`, dry-run passed first).
Bugs found in review before launch: errexit is ignored inside a subshell used as an `if`
condition (stage failures would have been invisible); a preflight check could `exit` the main
shell; a `RETURN` trap would not have stopped the BF16 server on error; `grep -q` + `pipefail`
false negative on a large curl response. All fixed before the run.

## 2026-10-15 — reference-tool findings (VENE) and MVP (AUX-1)

**Implementation noise floor** (junk-ref on Qwen3.6-35B UD-IQ4_NL, 2 wiki rows, RTX 3090):
- same path twice: KL 0.0, top-1 100% (deterministic).
- standard (fused top-k) vs graph split at the routing tensors (unfused): KL 0.0053, top-1 97.1%.
- each CUDA path vs the CPU backend: KL 0.0088 / 0.0090, top-1 97.0% / 96.2%.
- Decision: reference logits come from the standard path; routing traces from a separate pass
  (stage `routes`). Quality gates must be read against this floor: two correct implementations of
  the same quantized model differ by ~0.005-0.009 KL.
- Also: llama.cpp runs the last layer only on output tokens (`inp_out_ids`), so a routing trace
  must make every token an output.

**MVP on AUX-1** (engine fork, Vulkan, smoke, 1 run): `scripts/serve.sh` + `scripts/smoke_api.py` ALL PASS
(chat, tool call parsing, prefix cache 49.5 s -> 1.4 s). 262k context + q8_0 KV fits with `-ncmoe 34`
or 36 (decode 21.4 / 20.7 t/s at short depth).

## 2026-10-15 — step 1 (upload path) and step 2a (hybrid MoE): measured, and the decode bottleneck found

**Upload paths on AUX-1** (`tools/vkbw`, raw Vulkan, 512 MiB, best of 3):
- GPU copy engine, host staging -> VRAM, ONE submit: 11.2-11.3 GB/s for any chunk size (256 KiB..512 MiB),
  same on queue family 0 and on the async-compute family 1.
- One submit + fence wait per 2 MiB copy: 7.4-7.8 GB/s, 270-285 us per submit+wait (this is what a
  synchronous per-tensor copy costs; the BELLS / mainline moe-cache path on Vulkan does this 3x per expert).
- CPU writes straight into VRAM through the 8 GB BAR: 7.94 GB/s cap (any thread count, memcpy or NT stores).
- CPU memcpy into the staging buffer: 17.7 GB/s.
- Conclusion: the bus gives 11.2 GB/s when copies are batched into one submission. ReBAR's value is mapping, not bandwidth.

**ggml-vulkan async staging ring** (engine commit): pageable sources are copied into a 512 MiB pinned ring
and the transfer is recorded without a queue drain (`GGML_VK_ASYNC_STAGING_MB`, 0 = old path).
Mainline `--moe-cache-mib 3500` decode smoke: 15.46 -> 16.17 t/s. Small, because the mainline cache
also drains the queue once per layer to read the routing ids.

**Hybrid MoE (step 2a)** (`LLAMA_MOE_HYBRID=1` + `--moe-cache-mib`): GPU computes cached experts (misses read an
all-zero slot), CPU computes misses (hits get id -1 = zero row; CPU mul_mat_id patched for negative ids),
LFU-decay adaptation with bounded swaps per ubatch. Works: KL vs plain --cpu-moe 0.0047 (the GPU/CPU floor),
2073 slots (20.2%), hit rate 64.3% (batch 16 incl. warm-up) / 75.7% (batch 1). **Decode unchanged: 20 t/s**
(b=1, 2048 tokens: ncmoe34 22, cpu-moe 20, hybrid 20). Swaps run at the full budget (25 MB/token): churn.

**Why: the GPU half is the bottleneck, not the CPU.** GGML_VK_PERF_LOGGER, one decode token, -ncmoe 40
(no experts on the GPU at all): **~25 ms of GPU op time per token** for ~1.4 GB of dense weights (should be
~7-8 ms at 180-200 GB/s). Biggest losses: shared-expert matvecs m=512 k=2048 (98/token, 4.2 ms, 10-17 GB/s),
410 ADDs per token (1.85 ms, ~4.5 us each), f32 router m=256 (1.0 ms, 82 GB/s), mid-size projections at
105-215 GB/s. The hybrid mode moved expert work onto this slow GPU path, in series with the CPU: no net gain.
**Decode levers, in order:** (1) gfx803 GPU efficiency: fusion of the MoE add chain, small-matvec work
distribution, router precision/kernel; (2) MTP verify windows amortise exactly this per-dispatch cost;
(3) overlap CPU misses with GPU hits once the GPU half is lean.

## 2026-10-16 — where decode time goes, measured; the GCN matvec roofline is reachable

**Scheduler profile** (`GGML_SCHED_PROFILE`, per decode token, b=1, IQ4_NL):

| config | total | Vulkan record+submit | CPU waits GPU | CPU experts | GPU waits CPU input |
|---|---|---|---|---|---|
| --cpu-moe | 46.2 ms | 11.4 ms (41 splits) | 14.8 | 19.6 | 0.5 |
| -ncmoe 34 | 43.7 ms | 11.9 ms (35 splits) | 12.5 | 18.0 | 1.3 |
| hybrid 76% hits | 42.5 ms | 14.5 ms (81 splits) | 7.9 | 7.6 | 12.5 |

`GGML_VK_HOST_PROFILE`: ~280 us host time per Vulkan split (158 us recording, 10.5 submits x 17 us); mostly
overlapped with GPU execution. The floor is GPU execution (~23 ms) + CPU experts (~20 ms) in series.
Hybrid: CPU 19.6 -> 7.6 ms, but GPU op time 23 -> 32 ms (MUL_MAT_ID_VEC experts at ~89 GB/s, more adds/get_rows).

**Kernel lab** (`lab/matvec`, raw Vulkan, IQ4_NL repacked as 16-byte nibble chunks per lane + separate scales,
x in registers, one wave per row group, 40-640 distinct matrices so L2 does not help):

| variant | 8192x2048 | 4096x2048 | 512x2048 |
|---|---|---|---|
| codebook as const array (dynamic index) | 42-102 GB/s (unstable) | 42-102 | 42-102 |
| **codebook in LDS** | **228.7 GB/s** | 225.2 | 228.9 |
| linear 4-bit (format bound) | 228.9 | 223.8 | 227.9 |
| raw bytes (bandwidth bound) | 225.6 | 222.5 | 226.4 |

Correct vs CPU (max rel err <= 1.3e-5). **89% of the RX 580's 256 GB/s.** ggml-vulkan's IQ4_NL matvecs run
these shapes at ~108 GB/s (large) and 10-60 GB/s (small). Also measured: a dependent dispatch (barrier
between matvecs) costs ~7-10 us extra on Polaris (512x2048: 6.4 us back-to-back, 14-16 us with a barrier).
ggml decode has ~1,500 dependent ops per token -> 10-15 ms of bubbles.

**Decision:** decode needs a purpose-built path for qwen35moe: GCN-shaped kernels (LDS codebook), few fused
dispatches per layer, routing via a host-visible doorbell, CPU misses overlapped with GPU hits.
Estimated GPU time ~8-10 ms/token (vs 23-32 ms in ggml). See `current/decode-engine.md`.

**Native GGUF layout through LDS** (`lab/matvec/mv_iq4_native.comp`): rows staged into LDS with 16-byte loads,
blocks decoded from LDS. Correct on all shapes. Best per shape (REP=4 unless noted): 8192x2048 192 GB/s,
4096x2048 191, 2048x4096 147 (REP=2), 512x2048 192, 2048x512 194. ~75% of peak, no repacked copy of the
weights needed. Larger K stages more LDS and loses occupancy: double-buffering is the next tuning step.

## 2026-10-16 — JDE v1: correct single-token decode

VENE: the RTX 3090 fell off the bus at 05:53 (Xid 79, PCIe physical-layer AER errors) during the EXL3 3.5
conversion; needs a reboot (user). IQ3_XXS quant failed (no imatrix data for the MTP block): MTP block now q8_0.
BF16 routing-pass noise floor (fused vs unfused top-k): KL 0.0006-0.0025 across S1/S2/S3/L sets.

JDE v1 (engine c2026523e+): every op of the decode step in custom Vulkan kernels, routed experts on the CPU,
one GPU segment per layer (serial), command buffers recorded per token. Test: tiny160 row, b=1, IQ4_NL,
`--cpu-moe --no-repack -fa on`: all tokens through JDE (0 ggml graph computes), KL 0.000689 vs the
standard path with identical flags, top-1 95.8% (95 positions). Speed 21 t/s (= mainline). Correct first.

## 2026-10-16 — JDE: 21 -> 39 t/s (hybrid cache, attention, kernels, replay)

Engine commits 4fbfd355a, 0375e4d51, 384eda39e. Every step gated on tiny160 KL vs the standard path
(0.000434-0.000913, noise floor). Speeds are steady-state per-token (JI_PROFILE window 64-128, ctx < 200)
unless noted. IQ4_NL, `--cpu-moe --no-repack -fa on`, 6 CPU threads.

| step | ms/token | note |
|---|---|---|
| v1 (all experts on CPU, serial) | 43.6 | GPU 21 + CPU 21 |
| hybrid VRAM expert cache (3000 MiB, LFU decay, 8 swaps/token) + overlapped CPU misses | 42.3 | 70% hits |
| GQA split-K attention (one workgroup per kv head x chunk serves 8 q heads) | 40.7 | attn 2.49 -> 0.98 ms at 2k ctx |
| word-wise dequant (4-byte LDS reads, funnel shift, offsets folded into x sums) | ~40 | q6_K LM head 109 -> 143 GB/s, q5_K 2x |
| persistent matvec (576 WGs, register prefetch) + fused rmsnorm/swiglu/wsum/gate | ~41 | fewer dependent steps; fusion only pays with persistent WGs |
| **slot map read from a host shadow** | **30.0** | the cache update scanned the BAR slot map: uncached PCIe reads, 7 ms/token |
| replayed command buffers, B queued behind A | 26.8 | record 2 ms/token gone; 81 submits remain |
| same, cache 4500 MiB | **25.5 (39 t/s)** | 79% hits |

Findings:
- **Dependent dispatch costs ~6-8 us on gfx803/RADV regardless of barrier flavour** (compute-only vs full),
  clocks pinned high. Only fewer dependent steps help.
- **The gfx8 L2 caches system memory and is not invalidated mid-submission.** A host-set VkEvent waited on by
  the GPU hung the queue (GPU reset); a GPU-polled doorbell saw the flag only by luck of eviction and read
  stale CPU partials (KL 0.6-0.8). Host->GPU data must cross a submission boundary. Design consequence: per
  layer, A[il] (+B[il] queued) -> host waits A's fence -> CPU misses (overlap B) -> submit A[il+1].
- Per-shape matvec bandwidth (JI_BENCH, pipelined / dependent): iq4_nl 8192x2048 198/152 GB/s, q8_0 2048x4096
  189/133, q6_K LM head 157-175, q5_K 2048x4096 163/112, small 1 MB shapes 160-290 pipelined but ~14 us dependent.

Per-token budget now (25.5 ms): GPU A segments ~18 ms (kernels ~14), CPU misses exposed 3.4, submits 1.8,
cache update memcpy 1.4. Next: async cache update, LM head and router kernels, fewer steps per layer, then MTP.

VENE: the 3090 fell off the bus a second time (10:08, Xid 79) in the same exl3_35 conversion. User: do not
use VENE's GPU any more. Done on VENE: BF16 refs (short/long/chat), routes, ppl base, imatrix, GGUF Q8_0,
IQ4_XS, Q4_K_M, IQ3_S, IQ4_KT, IQ3_KT. Not done: IQ3_XXS/Q6_K (CPU, interrupted by the reboot), EXL3 x3,
all KLD evals (GPU). KLD evals must move to CPU (AUX-1 or VENE CPU) or a different plan.

## 2026-10-16 (later) — JDE 25.5 -> 21.1 ms/token; JDE serves the API at 46 t/s

| step | ms/token (tiny160 steady) | note |
|---|---|---|
| swaps 8 -> 2 per token (2048-row: 26.2 -> 25.0) | 23.9 | copy cost > hit-rate gain; direct DMA from host memory is the real fix |
| gdn_fused: conv + delta rule + norm, 1024-thread WG per key head, coalesced state | 22.7 | gdn 65 -> 33 us; 2048-row KL 0.0037 |
| router_fused: norm + logits + last-workgroup softmax/top-k | 22.0 | 49 -> 31 us/layer, routing identical |
| tall matvecs (LM head) get 8x the workgroups | 21.3 | LM head 3.2 -> 2.45 ms |
| attention: 8 waves (one q head each), batched K/V loads | 21.1 | 61 -> 31.5 us/layer at short ctx |

CPU misses run at 24-26 GB/s (6-11 threads): ~3.8 ms/token exposed at 4500 MiB cache. Attention bisection:
single-wave latency, not bandwidth; long context needs K staged through LDS (to do with the depth work).

Server: `scripts/serve.sh` now runs JDE by default. Warm decode 42.9-46.3 t/s (256 tokens, temperature 0.7,
real sampler), cold first request 34 t/s. API smoke ALL PASS.

## 2026-10-16 (later) — multi-token JDE + MTP speculative decoding: correct, not yet a win

Engine 892ab44cf+: JDE decodes ubatches of 1-4 tokens of one sequence (matvecs read weights once for all tokens,
GDN runs the tokens sequentially with the state in registers and writes the rollback snapshots exactly like the
graph path, attention/router/combine per token, CPU misses grouped by expert, logits per output token and the
nextn hidden state for the MTP draft context). KL vs reference: ub1 0.00075, ub2 0.00054, ub3 0.00107,
ub4 0.00065. The server runs `--spec-type draft-mtp` with JDE verifying (coherent text).

| config (server, warm, 256 tokens) | t/s | notes |
|---|---|---|
| JDE, no MTP | 46.3 | 21.3 ms/token |
| JDE + MTP draft 2 (verify 3) | 37.4 | acceptance 0.84/0.59, 2.43 tok/step; verify 46.5 ms + draft 9.3 ms (ggml) |
| JDE + MTP draft 1 (verify 2) | 43.7 | acceptance 0.83, 1.83 tok/step; verify 33 ms + draft 5.6 ms |

Why MTP does not pay here yet: only the dense part is shared between the verified tokens. Routed experts
differ per token, so GPU hits and CPU misses grow ~linearly with the window (2-token step: GPU 22 ms vs 15.2,
exposed CPU 7.5-9.6 ms vs 3.7). Even an ideal multi-token matvec leaves the 2-token step at ~33 ms + draft
for 1.83 tokens ~ break-even. MTP becomes a win with (a) a higher expert hit rate (lower-bit experts at the
KLD bar, or a second card), (b) the draft head inside JDE, (c) GPU reading part of the misses over PCIe
concurrently with the CPU. Kept implemented, off by default.

## 2026-10-16 (later) — q8_0 KV in JDE, 262k context in the server

- llama.cpp rotates attention vectors when the KV cache is quantised (orthonormal Walsh-Hadamard: q, k by H_256
  after RoPE, v by H_64 blocks, output rotated back). A JDE q8_0 path without it read the graph-written prompt
  cells permuted (same rms, different values; KL 0.0156). With in-LDS FWHTs: KL vs the q8_0 graph path 0.0012
  (ub1) / 0.0006 (ub3). (ggml q8_0 KV vs f16 KV: 0.00065, i.e. q8_0 KV is essentially free in quality.)
- Expert cache now sized from the VRAM budget after llama.cpp's allocations (JI_CACHE_MIB=auto).
- VRAM trade-off at 262k q8_0 KV (2.85 GB): the prefill compute buffer scales with ctx x ubatch
  (ub 2048 leaves 655 MiB for the cache, ub 1024 leaves 3.0 GB).

| server config | expert cache | prefill (6.7k) | decode warm |
|---|---|---|---|
| 32k f16 KV, ub 2048 | 4.5 GB (fixed) | 215 t/s | 46 t/s |
| 262k q8_0 KV, ub 1024 (default) | 2.8 GB (auto) | 175 t/s | 42 t/s |

Decode at short depth in the harness (ub 1): 21.1 ms (32k), 21.3 (128k), 22.5 (262k). Deep-context decode
still pays a dense KV read (2.85 GB per token at 262k): sparse attention is the next lever for depth.

## 2026-10-16 (later) — decode at depth

`scripts/depth_bench.py N gen`: an N-token prompt through the server, then a short generation (second call
reuses the prefix, so it measures decode at depth with a warm expert cache).

| depth | prefill | decode (warm) | per token |
|---|---|---|---|
| ~0.2k | 175 t/s | 42 t/s | 23.8 ms |
| 33.6k (before attn_long) | 143 t/s | 25.7 t/s (cold) | 33.8 ms |
| 33.6k (attn_long) | 143 t/s | 34.9 t/s | 28.4 ms |

attn_long (n_kv > 8192): the first version (lane = cell, all heads) read K rows per lane and thrashed L1
(64 rows per load instruction); 4 lanes per cell + clustered-add reduction made K reads coalesced. Still ~2x
off bandwidth (2.5 ms/layer at 131k); deep context needs sparse attention (read a fraction of the KV).
Prefill at depth is ggml's path (233 s for 33.6k tokens) and is now the most visible cost for long prompts.

## 2026-10-16 (end of session) — prefill findings

- ggml-vulkan tiled transpose-concat (engine 5f5330640): the GDN conv-window concat was 18% of prefill; prefill
  pp1024 196 -> 215 t/s, pp4096 205 t/s (ub 1024). Graph-path KL unchanged (0.0005).
- Remaining prefill structure (ub 1024, 4.75 s per ubatch): 119 splits, each uploads one offloaded expert tensor
  (~151 MB) synchronously: the experts are in a pageable mmap, so ggml-vulkan's async CPU->GPU copy refuses
  (needs pinned memory) and the scheduler syncs the GPU, memcpys to staging, copies, waits - no overlap.
- `--load-mode none` (pinned host buffers for the CPU experts) fails: ~18 GB pinned > GTT size 16.8 GB
  ("Not enough memory for command submission", device lost). Next: raise amdgpu.gttsize (kernel param, reboot)
  or pin only part of the experts / double-buffer uploads in the scheduler (overlap memcpy+DMA with compute,
  bound ~ max(3.2 s compute, 1.65 s PCIe) -> ~320 t/s). Then GEMM tile tuning for MUL_MAT_ID (~0.95 TFLOPS).

## 2026-10-16 (later) — pinned experts: prefill +35-50%

AUX-1 now boots with `ttm.pages_limit=6291456 amdgpu.gttsize=24576` (GTT 24 GB; default was half of RAM, 16.8 GB) and
a udev rule pinning DPM at `high` (a reboot had reset it to auto). With `--load-mode none` the CPU experts load
into pinned Vulkan_Host memory (17.5 GB), so op-offload uploads are async DMA instead of staged synchronous copies.
JDE decode is unaffected (CPU misses read the same pages at 24.3 GB/s; 21.35 ms/token, KL unchanged).

| (262k q8_0 KV, ub 1024) | before | pinned |
|---|---|---|
| llama-bench pp1024 / pp4096 | 215 / 205 | 294 / 279 |
| server prefill 6.7k tokens | 175 t/s | 260 t/s |
| server prefill 33.6k tokens | 143 t/s (235 s) | 194 t/s (173 s) |
| server decode warm | 42 t/s | 41 t/s |

Local KLD refs moved to AUX-1 `~/junkinfer/data/refs_local/` (/tmp is wiped on reboot).

## 2026-10-16 (later) — KV type quality (llama.cpp rotation on), prefill tile tuning

KL vs f16 KV, graph path, S1 rows 0-2 (3x2048 tokens, IQ4_NL weights):

| KV | KL mean | p99 | top-1 |
|---|---|---|---|
| q8_0 | 0.0015 | 0.018 | 98.6% |
| q5_1 | 0.0037 | 0.038 | 97.9% |
| q5_0 | 0.0050 | 0.048 | 97.3% |
| q4_1 | 0.0071 | 0.066 | 96.0% |
| q4_0 | 0.0078 | 0.085 | 96.0% |

Only q8_0 meets the KV bar; 5-bit needs KVarN-style handling (outlier/tail) to qualify. KV stays q8_0.

MUL_MAT_ID tile sweep (GGML_VK_MMQID_* hook): the medium mmq-id class is used; no valid (BK=32) shape beats
ggml's GCN default (294 t/s); BK=16 "522 t/s" is wrong (KL NaN). Prefill stays at the pinned-memory 294 t/s.

VENE: CPU-only quant KLD running (`ref/eval_cpu.sh`, nice 19, 4 threads, GPU never touched - the user uses it).

## 2026-10-16 (later) — elastic expert cache, prefill investigation

**Elastic expert slots (engine 268ffaa28+):** the 262k q8_0 KV cache (2.85 GB over 3 VkBuffers) is mostly empty in
practice; JDE carves 1580 expert slots (2.67 GB) from the top cells of the 20 attention K/V tensors and evicts them
ahead of the context (jde_kv_guard before every ubatch, n_kv + n_tokens + 4096 cells). Correct through evictions
(ctx 8192: 60 evictions over 3 rows, KL at floor). Warm decode at 262k: 42 -> 46.5 t/s; at 33.6k depth 35 -> 38-39.

**Prefill, what bounds it (ub 1024, pinned experts, ~3.8 s per ubatch):** compute. MUL_MAT_ID runs at 0.93 TFLOPS
(~32 rows per expert; idle warps in 64-wide tiles), dense matmuls at 2.1 TFLOPS: together ~3.6 s. Not the expert
upload: a weight prefetcher (second queue, 13.4 ms per 144 MiB at 11.3 GB/s, 82% hits) gave no gain (kept opt-in,
GGML_VK_WEIGHT_PREFETCH=1). (A "skip the uploads" timing reached 464 t/s but is invalid: garbage experts change the
routing.) ggml mul_mm tile overrides (incl. tall-narrow 128x32/128x16 for MoE) did not beat the GCN default.
Larger ubatches give more rows per expert:

| server, 262k q8_0 KV | prefill 6.7k | prefill 33.6k | decode warm | decode @33.6k |
|---|---|---|---|---|
| ub 1024 | 271 t/s | 247 t/s | 46.5 | 38.3 |
| **ub 2048 (default)** | **302** | **293 (115 s)** | **46** | **39** |
| ub 4096 | 344 | 283 | 44 | 38.7 |

Next for prefill: a GCN MoE GEMM for few rows per expert (weights-stationary, all of an expert's rows per
workgroup) - MoE is ~60% of prefill time; at dense efficiency prefill would reach ~430 t/s.

## 2026-10-16 (later) — prefill kernels

- **GCN MUL_MAT_ID kernel** (engine f45f67a75+, `mul_mm_id_gcn.comp`): IQ4_NL x f32, 128x32 tiles, 4x4 register
  blocks, drop-in for ggml's id-matmul (same row map/bindings). MoE GEMM 2.44 -> 1.99 s per 1024-token ubatch;
  KL at floor. Variants tried: 8x4 blocks/128 threads (slower), row-per-thread with A in registers and no padding
  FMAs (`GGML_VK_MMID_GCN=2`, slower). Default `GGML_VK_MMID_GCN=1`.
- **Transpose-concat 64x64 tiles:** GDN conv concat 13 -> 7.7 ms per call at ub 2048 (32-wide tiles were bound by
  DRAM locality: 128-byte segments from rows 32 KB apart).
- Perf-logger caveat: it times ops only; copies recorded before an op are billed to it (checked: CONCAT's 13 ms
  was the concat itself - unchanged with the prefetcher on).

| llama-bench pp4096, ub 2048, pinned | t/s |
|---|---|
| start of the prefill work (ub 1024, mmap) | 205 |
| pinned experts, ub 2048 | 350 |
| + GCN MoE kernel + concat tiles | 353 |

Server (262k q8_0 KV, ub 2048): prefill 306 t/s (6.7k), 291 t/s (33.6k, 115 s); decode 45 t/s warm, 39 t/s at 33.6k.
Remaining prefill shares (ub 2048): MoE GEMM ~35%, dense GEMM ~25% (2.1 TFLOPS), FA ~5%, elementwise ~6%.

## 2026-10-16 (evening) — EXL3 runs (milestone 1)

Model: `UnstableLlama/Qwen3.6-35B-A3B-exl3-4.00bpw` (rev b7f4bd67; all linears 4 bpw, LM head 6 bpw, mcg codebook
0xCBAC1FED on every tensor). Converted with `tools/exl3/exl3_to_gguf.py` (metadata copied from the IQ4_NL GGUF,
MTP dropped) -> AUX-1 `~/models/exl3/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf`, 17.35 GiB, ~3 min.

- GDN value heads: reordered grouped -> tiled after all. Each head is 128 wide = one Hadamard block, so moving
  whole blocks (8 trellis tile columns + the svh/suh slice) commutes with the blockwise Hadamard. No grouped-order
  special case anywhere (the design doc's hk = hv/2 mapping is not needed).
- Engine 361d77880: `GGML_TYPE_EXL3_4/6` (block = one 16x16 tile; tiles [N/16][K/16]; rows nominal), loader
  creates `<w>.suh/.svh` (F32), graph wraps every EXL3 matmul as `H128(H128(x*suh) W) * svh` (experts: suh/svh
  gathered by expert id), Vulkan dequantizes tiles to f16 (`dequant_exl3.comp`) and reuses the f16 GEMMs; CPU
  backend refuses EXL3 ops, Vulkan `offload_op` always takes them. JDE stays off for EXL3 models.
- Fixed on the way: `common_fit_params` builds a no_alloc model -> the Hadamard tensor needs a dummy buffer there.

Smoke: greedy "The capital of France is" -> " Paris." (x4, raw completion); chat "why is the sky blue" -> a correct
sentence. Server (32k ctx, ub 2048, q8_0 KV): **prefill 140.5 t/s on 6.6k tokens, decode 0.7 t/s**. Decode is
the full-tensor f16 dequant per matmul (incl. the 248k x 2048 LM head every token) - nothing is a mat-vec yet.

Next: EXL3 in JDE (mat-vec kernels with the Hadamard/scale stages fused, expert slots, CPU misses), then an EXL3
MoE GEMM (trellis decode inside the GCN tile loader instead of dequantizing every expert to f16 per ubatch).

## 2026-10-16 (night) — EXL3 in JDE, native EXL3 GEMMs

Model now on the SSD (`/mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf`; user: run models only from the SSD -
load 47 s instead of 3.5 min). Q4_K_S moved to `~/models/ssd-overflow/` for space.

**JDE EXL3 decode** (engine a067e8c97, 19c0b591a): x3_prep (fused norm/swiglu input * suh, Hadamard-128; one
workgroup per (row, 128-block)), x3_mv (trellis decode; one wave = 16 outputs x one K slice, independent waves,
16-tile batches double-buffered through LDS, K split until >= 1024 waves), x3_post (slice sum, output Hadamard,
svh). Expert scales: f16 table on the GPU (153 MiB) + host copy for the CPU. CPU misses: AVX2 4-bit decode
(nibble-window trick) - only ~0.9 G weights/s/thread, so misses go zero-copy to the GPU instead: x3_mv_moe reads the
first JI_X3_ZC (default 8) misses per token straight from the pinned expert tensor over PCIe (~5 GB/s effective);
GPU and host apply the same rank rule. Greedy output identical across all splits and vs the graph path.

| decode, 4k ctx, warm | ms/token | t/s |
|---|---|---|
| graph path (f16 dequant per matmul) | 1500 | 0.67 |
| JDE, CPU misses | 57 | 14.2 |
| + zero-copy misses, parallel prep, branch-free windows | 44 | 22.5 |
| server 32k ctx (2nd request) | 37 | 26.6 |

Pitfalls found: (1) the graph path's f16 dequant buffers (1 GB for the LM head) were allocated after JDE sized its
cache -> RADV evicted weights to GTT -> everything at PCIe speed (profile showed 12 GB/s kernels). JDE's margin for
EXL3 is now 1000 MiB. (2) A loop-carried register array for prefetch made ACO copy registers right after the loads
(s_waitcnt vmcnt(0)): no overlap; fixed by explicit double buffering. (3) pgrep -f in a remote shell matched the
shell itself - kill servers by PID from `ps ... | grep "[b]in/llama-server"`.

**Native EXL3 GEMMs for prefill** (mul_mm_gcn_exl3.comp, MUL_MAT + MUL_MAT_ID): trellis tiles staged in LDS and
decoded into the A tile, no f16 copy of the weights; BN 64 (default) halves decode per FMA: dense EXL3 GEMMs now
faster than IQ4_NL's (8192x2048 per 2048-token ubatch: 19.8 ms vs 20.5). Output identical to the dequant path.
Prefill still 131-141 t/s (IQ4_NL 274 on the same 2314-token prompt): the MoE ops were 2x IQ4_NL's because
(a) sched_copy_experts only handled splits starting with the MUL_MAT_ID (EXL3 splits start with the suh multiply /
Hadamard -> synchronous whole-tensor fallback copies) and (b) --cpu-moe put the expert suh/svh on the CPU, so the
per-token gathers (134 MB per tensor per ubatch) crossed PCIe. Fixed: callback finds the split's MUL_MAT_ID; scale
vectors ignore the buffer overrides; expert scales stored F16 (lossless). Measurement pending.

## 2026-10-17 (early) — EXL3 prefill fixed, elastic-slot NaN bug (pre-existing) fixed

- **MUL_MAT_ID never used the native EXL3 kernel**: EXL3 has no ggml pipeline map, so ggml_vk_mul_mat_b_type
  fell back to f16 B -> y_f32_kernel false -> dequant path. Forced f32 B on GCN (engine ae6e9ea47):
  **prefill 131 -> 230 t/s** (2314 tokens, llama-completion; IQ4_NL 274). The other two suspects (expert-copy
  callback, CPU-side scale gathers) were real but secondary; fixed in the same commit.
- **Server, EXL3, 262k ctx, q8_0 KV (default serve.sh)**: prefill 247 t/s (6.6k), 220 t/s (29k); decode 21-27 t/s;
  smoke_api ALL PASS.
- **Garbage output after a long prompt ('////', '3333') - JDE bug since elastic slots, IQ4_NL too.** Repro: a 6.7k
  request (fills elastic slots at low cells), then a 30k request. Expert data and slot maps verified intact
  (JI_EL_VERIFY); cause: evicted slots keep expert bytes, the context then grows over them, and cells between the
  used ones and the 256-padded n_kv are read by attention with a -inf mask - expert bytes decode to Inf/NaN q8_0
  blocks, NaN + -inf = NaN, softmax poisoned for good. Fix (engine 8f8be4999): zero slots on eviction; guard limit
  from the cache-wide used_max_p1 (a ubatch's n_kv covers only its streams: llama-server's short side decode had
  n_kv 164 and re-activated slots under live cells). Both models now pass the sequence + smoke.
- VENE unreachable since ~01:00 (ssh timeout): commits are backed up to bare repos on AUX-1 (`~/git-backup/*.git`,
  remote `aux`); push to VENE when it is back.

## 2026-10-17 — prefill: fused EXL3 op, expert uploads overlapped (engine bb1b388a8)

Profile (8k prompt, ub 2048, EXL3): GPU kernels 22.2 s, wall 27.7 s; skipping expert uploads (debug) -> wall 22.3 s.
So ~20% of prefill was expert DMA serialized with compute. Causes, in order of discovery:
1. ggml-backend sched synchronizes the split backend before every input copy when it has no events (one GPU, no
   pipeline parallelism): 120 GPU drains per ubatch (host profile: one ~55 ms fence wait per split). Skipped for
   host weights handed to the copy callback (same queue, barrier-ordered).
2. With the drains gone, uploads still alternate with compute on the same queue. Uploading on the transfer queue
   (GGML_VK_ASYNC_USE_TRANSFER_QUEUE) gave +13% but WRONG output (write-after-read on the reused input buffer).
3. The old weight prefetcher (2 staging slots, learned order) hit 356/476 uploads but gained nothing: (a) the
   slot -> input-tensor copy; fixed by alias mode (the MUL_MAT_ID binds the slot; expert uploads flagged via
   ggml_backend_vk_expert_upload_hint - a 16 MB non-expert alias was never resolved and blocked the slots);
   (b) the DMA itself slowed the GEMMs by its own duration: RADV on gfx8 runs large vkCmdCopyBuffer as a compute
   shader (GFX10+ use CP DMA for GTT), whose PCIe-stalled waves occupy every CU. CP DMA (regions < 4096 B) does not
   disturb compute but is too slow (wall 34 s). Fix: pf_copy.comp with 4 workgroups (8 uvec4 loads in flight per
   lane): DMA hidden, kernels +1%.

| 8k prompt, ub 2048 | wall | t/s |
|---|---|---|
| EXL3, before | 27.2 s | 279 |
| EXL3, prefetch (vkCmdCopyBuffer) | 27.8 s | 273 |
| EXL3, prefetch, copy shader 4 WGs | 24.0 s | 316 |
| IQ4_NL before / after | 23.9 / 19.8 s | 319 / 384 |

Fused EXL3 linear: the suh gather/multiply, the two Hadamard-128 f32 matmuls and the svh multiply ran as separate
ggml ops (1.25 s per 2048-token ubatch, [K, 8, T] intermediates of 134 MB); now one MUL_MAT(_ID) node, the Vulkan op
does prep (register FWHT) + trellis GEMM + epilogue (Hadamard over the 128-row tile, svh). 5.88 -> 5.3 s GPU per
ubatch. Bug on the way: ggml-vulkan fused the following residual ADD into the matmul (MUL_MAT+ADD), which the EXL3
kernels do not implement - EXL3 matmuls are excluded from op fusion.

Server, 262k ctx, q8_0 KV, prefetcher on: EXL3 prefill 310 t/s (6.6k), 239 (29k, prefix reuse), smoke 328;
IQ4_NL 353 / 267 / 392. All outputs correct, smoke ALL PASS.

## 2026-10-17 (later) — decode batching, CPU trellis decode, FA findings (engine 377945a41, db3677f2f)

- **EXL3 decode, batched mat-vecs**: weights sharing an input (wqkv + gate, q/k/v, shexp + routed gate/up, the two
  downs) share one prep -> barrier -> mv -> barrier -> post sequence (6 slots of xp / partial buffers).
  39.4 -> 34.3 ms/token at 4k (23.6 -> 26.6 t/s); server 262k: ~25 t/s.
- **CPU EXL3 decode 2.25x** (0.87 -> 1.96 G weights/s/thread): X vectors built once per call, byte swap by shuffle,
  F16C + hadd instead of integer fp16 conversion, per-position-group register accumulators over all K tiles,
  token count as template parameter (a runtime nx kept the accumulators on the stack). In JDE: 3.9 GB/s.
  Zero-copy split: JI_X3_ZC 4..8 equal at 4k and 262k (few misses per token, CPU time gets exposed) - default 8.
- **Prefill at depth is FA-bound**: 33k-token prefill, GPU 137 s: FA 37%, MoE GEMM 28%, dense EXL3 GEMM 20%.
  ggml's scalar FA (D=256) runs at ~1.8 TFLOPS; it uses 256 VGPRs (one wave per SIMD). Tried, all output-identical
  and none faster: tile overrides (GGML_VK_FA_BR/BC/DSPLIT/RSPLIT; Bc 64 -2%), GQA folding of the 8 query heads
  into rows so K/V is read once per KV head (JI_FA_GQA_FOLD=1; FA time unchanged -> not K/V-traffic bound), and a
  first custom kernel ji_fa_prefill.comp (JI_FA=1; 64 rows per workgroup, register-tiled QK/PV, online softmax;
  4.0 s vs 2.6 s, also 256 VGPRs). Next: a lower-register FA design (this is the long-context prefill limiter).
- Other prefill experiments kept as opt-in knobs (slower here): 128-column / 128-thread EXL3 GEMM tiles,
  dense EXL3 via f16 dequant + ggml GEMM (GGML_VK_EXL3_DQ_MIN), ubatch 4096 now only +5% (uploads hidden).

## 2026-10-17 (evening) — own prefill flash attention beats ggml's (engine bbb79e5c1)

`ji_fa_prefill.comp`, now default on GCN (`JI_FA=0` disables). FA time on the 8k EXL3 prompt (sum of FLASH_ATTN_EXT,
perf logger), output hash identical in every row:

| version | design | VGPRs / waves per SIMD | FA |
|---|---|---|---|
| ggml scalar | — | 256 / 1 | 2.69 s |
| v5-v7 | 32 rows/WG, 2-row x 4-key thread tiles, conflict-free LDS, prefetch | 84-128 / 2-3 | 4.9-5.2 s |
| v8 | 64 rows/WG, 4-row tiles, Q/K/V register prefetch | 256 / 1 | 3.02 s |
| v8 trimmed | no prefetch, to fit 128 VGPRs | 128 / 2 | 3.26-3.30 s |
| **v9** | 128 rows/WG (16 positions x 8 heads), 8-row x 4-key S / 8-row x 16-dim O tiles, prefetch | 256 / 1, no spills | **2.01-2.08 s** |

Lesson: on gfx8 the FA inner loops are bound by LDS dwords per FMA (LDS feeds 32 dwords/clk/CU vs 64 FMA/clk);
2-row tiles gave ~1.6 FMA/dword and lost despite 3x the occupancy. 8-row tiles give 2.7 (QK) and 5.3 (PV): the ISA
now has 2048 v_mac per 288 LDS reads (b64; ACO did not emit b128 even with 16-byte-aligned strides - that variant
was 2% slower). Register prefetch of the next chunk matters more than a second wave. DC=64 spills; VC 8 vs 16 same.
Server (262k ctx, q8_0 KV): smoke ALL PASS, prefill 6.6k 310 -> 337 t/s, 37k no-reuse 233 -> 262 t/s (+12%).

## 2026-10-17 (night) — prefill memory-path fixes, bigger EXL3 expert cache (engine 39bbe772d .. HEAD)

8k prompt (7605 tokens), llama-completion, ub 2048 unless noted:

| change | prefill |
|---|---|
| own FA kernel (above) | 323 t/s |
| concat_transpose grid walks the contiguous dim first (GDN conv window; 7.8 -> 0.94 ms per ubatch: all running workgroups read src1 rows at a 32 KB stride = one memory channel) | 334 |
| MoE routing weights in the fused EXL3 down-projection epilogue (drops a 3.4 ms MUL per ubatch) | 341 |
| waves skip FMAs of empty expert-tile columns; epilogue skips empty 32-column groups | 349 |
| (ub 4096 / 8192, same build) | 384 / 402 |

MoE GEMM analysis: GGML_VK_MMID_STATS=1 shows 16384 rows in ~386 64-column tiles (66% full) at ub 2048; prep is
~6% of the op (JI_X3_TDBG); per useful FMA the MoE GEMM runs at the dense kernel's ~3.1 TFLOPS - the loss is
padding. 32-column tiles (GGML_VK_EXL3_MBN=32) are slower overall. Larger ubatches fill tiles (MoE -18% at 4096)
but the server keeps 2048: the 262k KQ mask (ctx x ubatch f16, 1 GiB at 2048) is reserved in VRAM.

JDE's EXL3 VRAM margin was 1000 MiB for the old dequant path's buffers; 400 now: expert cache 1230 -> 1830 MiB
(8.0 -> 11.9% of experts), decode on a short chat 21 -> 28-32 t/s. 116k-token prompt (server, no reuse): prefill
159.5 t/s (726 s), decode 15.7 t/s at that depth, VRAM peak 8184/8192 MiB, no evictions; smoke ALL PASS
(6.6k prefill 361 t/s). Next: the 1.4 GiB ggml compute buffer (mostly the KQ mask) is idle during JDE decode.

## 2026-10-18 — decode: compute-buffer expert slots, one-shot tokens, long-context attention (engine 0b653f381 .. 5b6aeaf13)

- **Elastic expert slots in the ggml compute buffer.** The Vulkan compute buffer (1.4 GiB at 262k / ub 2048, mostly
  the KQ mask) is idle while JDE decodes. Its largest chunk (the buffer is split at the 1 GiB allocation limit) is
  carved into 682 expert slots (1023 MiB) as a 4th elastic buffer (binding 10 of the MoE mat-vecs). Evicted before
  anything the graph path writes there (process_ubatch, graph_compute, memory updates that run a graph), retired
  if the buffer is reallocated; fills prefer stable slots. Bug on the way: memory_update() runs at every decode()
  call - evicting there wiped the slots every token (fills went to them first, so the cache stayed empty);
  the evict now runs only when an update graph is due. JI_EL_VERIFY: 11046 cached tensors, 0 mismatches across
  decode/prefill alternation. Server: decode short 28-32 -> 31-36 t/s, at 37k 18.0 -> 22.7 t/s.
- **One-shot tokens (EXL3).** With JI_X3_ZC >= n_used every miss runs zero-copy on the GPU, so the per-layer host
  handshake (wait for the router, classify, submit: 81 submits per token) only fed the cache statistics. Per-layer
  routing records (ro_host = n_layer x 256 B) let the whole token go in one vkQueueSubmit; statistics are read
  after the fence. Output identical; llama-completion 29.0 -> 30.8 t/s; server short decode ~36.5 t/s.
- **attn_long** (decode attention, n_kv > 8192): the score loop was LDS-bound (one Q vec4 per 4 FMAs); two cells
  per lane quad + 256-cell passes / 256 chunks: q8_0 KV per layer 36k 650 -> 510 us, 116k 2283 -> 1413 us.
- Tried, no gain: the trellis mcg multiply as 24/16-bit multiplies (v_mul_lo_u32 is 1/4 rate on GCN; removed from
  the ISA via a 16-bit multiply, but decode and prefill times unchanged - not the limiter); JDE CPU threads 6/9/12;
  x3_mv K-split sweep (per-shape optimum within a few % of the default).
- Decode profile now (4k ctx, llama-completion): ~29 ms GPU per token; dense EXL3 mat-vecs ~14.6 ms (8192x2048
  155 us = 54 GB/s: latency/ALU bound, ~2.6x the bandwidth bound), routed experts ~7.5 ms, LM head 2.4 ms.
- **ubatch 4096 is now the server default.** ggml-vulkan's 1 GiB suballocation blocks split the KV cache (3
  buffers) and the compute buffer (chunks; JDE uses only the largest). RADV allows 4 GiB allocations and binding
  ranges: serve.sh sets GGML_VK_SUBALLOCATION_BLOCK_SIZE=3.75 GiB. Then the compute buffer is one chunk and fully
  usable for decode expert slots, so ub 4096's bigger buffer (2.75 GiB, mostly the KQ mask) costs decode nothing:
  ub 2048 = 1137 bank + 1002 compute-buffer slots, ub 4096 = 302 + 1834 (plus 1780 KV-tail slots either way).
  Server: prefill 6.6k 361 -> 394 t/s, 37k 278 -> 298 t/s; decode 36.9 t/s short, 22.6 t/s at 37k; smoke ALL PASS.
- Prep fused into the EXL3 decode mat-vec (JI_X3_FUSEP=1, opt-in): output identical but slower (35.9 -> 38.8 ms
  per token; every wave redoes its slice's transform and the row's rms). The barrier itself is not the big cost
  (JI_BARRIER_LITE within noise).
- **EXL3 dense mat-vecs were DRAM-page bound** (engine 9936606a6). JI_BENCH showed the small shapes at 45-63 GB/s
  while the LM head streamed at 157; trivial decode math or no decode at all barely changed them, an address skew
  (channel camping test) neither. Cause: slice-major wave order - the waves in flight read different stripes
  16 KB apart. Stripe-major order (consecutive waves = consecutive K slices = consecutive memory) and K split down
  to one 16-tile batch per wave: 8192x2048 133 -> 88 us, 2048x4096 97 -> 55, 4096x2048 77 -> 53.
  **Server decode 36.9 -> 42.6-43.4 t/s** short, 24.4 t/s at 37k.
- MoE mat-vecs (~8.7 ms per token) are PCIe-bound by the zero-copy misses: with JI_X3_ZC=0 the GPU MoE kernels
  drop 73 -> 34 us per call; ~45 missed experts per token = ~68 MB at ~14 GB/s. Misses-first row order, CPU
  sharing of misses (ZC 2/4) and the cache swap rate (2/8/16) did not help. Next ideas: expert prediction /
  prefetch for the next layer, more cache.
- MTP sidecar (`-md mtp-Q4_0.gguf --spec-type draft-mtp`, the EXL3 GGUF has no MTP block): needs
  --spec-draft-n-ubatch 256 (the draft context's 262k mask otherwise overcommits VRAM: "Not enough memory for
  command submission" -> DeviceLost). Acceptance 0.68 but 10.6 t/s: the 2-token verify takes ~114 ms (apparently
  the graph path, not JDE) and the draft ~42 ms on the graph path. Parked; a JDE-native draft head is the way.
- **EXL3 mat-vec output stage fused** (engine b9d206337, default on, JI_X3_FUSEQ=0 off): the last of a 128-output
  block's 8 x KS waves (atomic arrival counter) sums the slices, applies the output Hadamard and svh. With
  memoryBarrierBuffer (L1 invalidate in every wave) it was slower than x3_post; release on the atomic + acquire
  only in the winning wave: 30.6-30.9 -> 29.3-29.9 ms per token. **Server decode 44.0-44.4 t/s.**
- Router: one expert row per wave (34.7 -> 30.2 us per layer).
- Decode attention at depth (q8_0 KV, attn_long ~1.42 ms per layer at 116k = ~89 GB/s, ~14 ms per token):
  timing variants: without K loads -5%, without V loads -25%; a lane-per-cell rewrite (query heads via scalar
  loads, software-pipelined) was correct but slower (544 vs 507 us at 36k) and showed the V phase + per-pass
  softmax/barriers at ~60% of the time (V alone ~72 GB/s: 3 scattered dword loads per cell per lane, little
  latency hiding at ~2 waves per SIMD). Removed; next step there: LDS-staged coalesced V (uvec4 rows) with
  prefetch across cell batches.

## 2026-10-18 — long-context prefill fixes: GPU job timeout, KV-cache eviction (engine d2bb420c3)

- **Device lost at ~80k cells (ub 4096).** dmesg: "ring comp_1.x.0 timeout, but soft recovered" - this system's
  amdgpu soft-recovers a GPU job after ~2 s (no lockup_timeout on the cmdline); one prefill flash-attention dispatch
  at 80k cells x 4096 queries takes ~2 s. llama-graph now splits prefill FA into query-row chunks of <= ~1 TFLOP
  (JI_FA_SPLIT_GFLOP, default 1000); separate nodes are separate submissions (ggml-vulkan already caps a submission
  at 200 GFLOP), a balanced concat tree joins the chunks. 8k with forced 100-GFLOP chunks: identical output, +6% FA.
- **3x slower prefill past 64k cells (server only, intermittent).** ggml-vulkan converted the whole q8_0 K/V into
  an f16 scratch (prealloc_x) before every FA call; the scratch grows with the context (512 MiB at 262k) and is
  allocated lazily, long after JDE sized its expert cache to fill VRAM -> VRAM over-committed -> the kernel evicted
  the 2.7 GB KV cache to GTT (VRAM trace: -2710 MiB, GTT +2710 MiB) and everything after ran over PCIe. Fix: a q8_0
  variant of ji_fa reads the KV cache directly (34-byte blocks dequantised while staging, 4 threads per key row for
  coalescing), decided before the split-k / mask-opt / dequant scratch of the other paths. No context-sized
  scratch any more; 8k FA 2079 -> 2153 ms, output identical.
- Server, 262k ctx, ub 4096: **116k-token prompt: prefill 161 t/s (718 s), decode at that depth 20.1 t/s** (was
  15.7 at ub 2048), VRAM stable, no device lost. `scripts/big_bench.py N`: near-full-context prompt from the engine
  sources.
- **Near-full context validated** (`scripts/big_bench.py 245000`): 246,894-token prompt, prefill 90.6 t/s
  (2724 s), decode at that depth 13.7 t/s, coherent answer, no device lost, VRAM/GTT stable. Per-4096-token
  ubatch: 10 s at the start -> 80 s at 240k cells, ~78% of it flash attention at ~2.6 TFLOPS (about half of
  fp32 peak; gfx803 has no packed-f16 or int8-dot instructions) - exact attention is the long-context limit.
- **attn_long: wave-local online softmax, 512-cell passes** (engine bde6cbc62): no workgroup barriers per pass;
  245k cells 4.81 -> 3.72 ms per layer.
- **Depth tests from saved KV contexts** (user suggestion): serve.sh JI_SLOT_DIR enables slot save/restore;
  `scripts/depth_kv.sh N` restores depth_N.bin (or prefills once and saves; 116k = 1.46 GB, save 0.8 s, restore
  0.3 s). Warm decode at depth (restored, second request): **37k 33.9 t/s, 116k 23.9 t/s** - the earlier
  25 / 20 t/s figures were cold expert caches right after a prefill.
- Saved contexts on AUX-1 `/mnt/ssd/kvcache`: d37k.bin, depth_100000.bin (116k), depth_245000.bin (247k, 2.9 GB,
  save 8 s). Warm decode at 247k: **18.2 t/s** (13.7 before the attention/one-shot/mat-vec work). Attention is
  still ~37 ms of ~55 ms per token there (10 layers x 3.7 ms); q8_0 KV streaming alone would be ~14 ms.

## 2026-10-18 — "rows in scalar registers": new prefill FA (ji_fa2) and decode attention (attn_d2)

The 8 query heads that share a KV head are uniform across a wave when the wave covers one position (prefill) or
the token (decode): their query values go to scalar registers (scalar loads, SGPR operands of the MACs). Both old
kernels were bound by LDS reads of Q.
- **ji_fa2** (engine e289598fe, default for q8_0 K/V): wave = one position x 8 heads, workgroup = 4 positions,
  K/V blocks staged in LDS (q8_0 dequantised while staging), P broadcast by readlane, 84 VGPRs / 3 waves per SIMD.
  Pitfalls: `s += dot(q, k)` compiled to mul+3 mac+add (explicit fma chains), a fully unrolled chunk hoisted 256
  query values into SGPRs and spilled (one-step-ahead prefetch in a rolled loop). P via LDS instead of readlane:
  slower; 8/16 positions per workgroup (fewer K/V re-reads): slower. 8k FA 2144 -> 1837 ms; 68k: 1024-row chunk at
  49k cells 343 -> 301 ms, prompt eval 346 -> 323 s; **247k-token prompt 90.6 -> 97.3 t/s**.
  (The perf logger's per-op "FA" sum misled once - judge by per-dispatch times and prompt eval.)
- **attn_d2** (engine abfe5e983, default for q8_0 KV): lane = key, K rows staged coalesced as raw q8_0 into per-wave
  LDS (64 keys x 17 words per 64 dims; per-lane row loads were slower), next step prefetched into registers; V rows
  staged raw (4 keys, prefetched), dequantised per lane; 4 waves per chunk combined in LDS. Single-wave chunks with
  1024 chunks were faster in isolation but the serial attn_merge over ~900 partials ate it (end-to-end slower);
  single-wave workgroups with an 8 KB V buffer fit one wave per SIMD (LDS-limited occupancy).
  JI_ATTN_TEST 116k 1424 -> 1140 us per layer; **decode from restored contexts: 116k 25.4 -> 27.2 t/s,
  247k 17.9 -> 19.7 t/s**, greedy output identical.
- The JI_ATTN_TEST microbenchmark is only valid with VRAM headroom (JI_CACHE_MIB=500): its buffers otherwise land in
  GTT (the "245k" numbers before this were PCIe-bound and noisy).
- Saved contexts for source-code prompts now freeze the prompt text (depth_N.txt): the engine sources change.

## 2026-10-18 — ji_fa2: scalar Q loads were half the QK time; attn_merge was latency bound

Method: RADV_DEBUG=shaders (ISA) + ACO_DEBUG=liveinfo (register demand per block) + kernel variants with a phase
removed (8k-prompt FA time).
- ji_fa2 QK ~880 of 1840 ms; with every Q load at one address (scalar cache always hits) QK dropped by 470 ms. Q is
  8 KB per position, re-read per 64-key block, the scalar cache is 16 KB per CU group; and scalar loads share
  lgkmcnt with the LDS reads, so the one-step prefetch cannot hide them.
  Fix (engine ce17233a6): 2 keys per lane (128-key blocks, K staged in 16-dim chunks, same 16 KB LDS) = half the Q
  loads per MAC. That alone went to 128 VGPRs (2 waves per SIMD): the row max / sum (wave-uniform, computed by
  VALU) lived in 16 VGPRs; NIR drops readFirstInvocation of known-uniform values, so they are made "divergent" with
  `fma(float(lane), runtime_zero, x)` and then read-first -> SGPRs. 84 VGPRs, 3 waves: **8k FA 1840 -> ~1430 ms**.
  Now ~62 % of the fp32 MAC peak (the 8k test's MACs take 0.89 s at peak); QK ~700, PV ~535 ms.
  Positions per workgroup 8/16 still slower than 4.
- attn_merge (engine 618576564): 16 workgroups each walked 256 chunk partials with dependent loads (~196 us per layer
  at 247k = 2 ms per token). Thread = chunk for the weights (workgroup max / sum), then 8-way unrolled independent
  loads: **29 us per layer**.
- 247k decode, GPU time per token (JI_PROFILE_GPU, profiled = with per-dispatch sync): 53.6 -> 51.9 ms;
  attn_d2 21.6 ms of it (2.16 ms per layer), MoE/dense weights ~26 ms (as at short context).
- AUX had drifted (jde.cpp with attn_d2 opt-in): sync check `rsync -rcn` before trusting a measurement.

## 2026-10-18 — decode attention at depth: it is the access pattern (probes), not ALU / occupancy / prefetch

JI_ATTN_TEST (q8_0, us per layer at 116k / 245k cells; attn_d2 then 1134 / 2252). Variants that changed nothing:
row max/sum in SGPRs, a cheaper K address map, combine buffers aliased into the K area (LDS 26 -> 21.7 KB),
V chunks of 2 or 8 keys, V prefetch 2 chunks ahead, unsigned-byte K/V conversion + block-scale folding (-25 % QK
VALU), p through LDS, consecutive workgroups on the same head, Q loads at one address. Removing QK math but
keeping the K loads: 1083 (so the math is ~5 %). (Note: in the RADV dump attn_d2 is the 128-VGPR shader; the
64-VGPR one is attn_gqa - misread for a while.)
Pure-load probes over the same buffers and grid (no math):

| probe | 116k | 245k |
|---|---|---|
| attn_d2's K + V pattern | 1032 | 2206 |
| K pattern only / V pattern only | 811 / 495 | 1661 / 1121 |
| both heads' halves read by sibling waves | 1000 | 2010 |
| attn_d3 pattern (4 waves = 4 dim pieces of the same keys) | 956 | 2027 |
| head-major addressing of the attn_d2 pattern | 973 | 1920 |
| contiguous full cell rows, 4-byte loads | 845 | 1570 |
| contiguous full cell rows, 16-byte loads | **652** | **1310** (~195-205 GB/s) |

So the kernel runs at the speed of its load pattern: half rows (272 of every 544 bytes per head), 68-byte pieces,
4-byte loads. Engine 37f3d7a2d: the 4 waves of a workgroup now share a 64-key block (QK split by dims - the four
68-byte pieces of each row are fetched together; PV split by keys): ~1100 / 2134, 247k decode attention
21.6 -> 20.8 ms/token. Tried and slower: 16-byte staging loads (1178), both kv heads in one 8-wave workgroup
(60 KB LDS, one workgroup per CU: 2x slower), head-major emulation of the real kernels (only -3..6 %).
The remaining gap to the 16-byte contiguous probe (~1.3 ms per layer at 245k, ~8 ms per token) needs full-row
16-byte loads within ~32 KB LDS per workgroup.

## 2026-10-18 — end-to-end at 247k after the FA2 / attn_d2 / merge work

Server (262k ctx, q8_0 KV, ub 4096), frozen 246,937-token prompt (`JI_PROMPT_FILE=.../depth_245000.txt
scripts/big_bench.py 245000 64`): **prefill 114.5 t/s (2157 s)** (97.3 t/s / 2538 s with the first ji_fa2,
90.6 before it), same greedy text, no GPU ring timeouts. 68k-token prompt (llama-completion): 289.6 s (323 s).
Decode at 247k: ~19.7 t/s warm from the saved context (attention 20.8 ms of ~51 ms GPU per token).
Also tried for ji_fa2 and slower: p through LDS (1490 vs ~1430-1470 ms 8k FA: the read sits on the PV critical
path), two query register sets (1620; the compiler's wait placement for scalar loads gets worse).

## 2026-10-18 (evening) — head-major KV cache + attn_hm: 247k decode 19.7 -> 21.2 t/s

**Gains: decode at 247k 19.7 -> 21.2 t/s (+7.6 %, warm, restored context); short-context decode 44.0 t/s
(unchanged); prefill 7.2k-token prompt 350 -> 354 t/s; 247k prefill not re-measured.**

Load probes with guaranteed 16-byte loads (uvec4 bindings), 116k / 245k cells: per-head half rows of the
cell-major cache 1117 / 2251 us, lane-per-key rows 2174 / 4617, **per-head contiguous (head-major) 668 / 1504**.
The interleaving of the two kv heads' 272-byte halves was the cost, whatever the load width or kernel shape.
- Engine 3ebdc1bad: `llama_kv_cache` stores K/V head-major (`JI_KV_HM`, default on; one stream, V not
  transposed): views, set_rows (one index row per (token, head)), k-shift view, state save/restore (the file keeps
  the cell-major format: rows gathered/scattered per head - old snapshots restore), failed-restore clears, graph
  reuse checks. llm_graph detects a transposed V by nb[1] > nb[2]: the head-major V view is named `*_hm` and
  excluded. JDE kernels address rows through KVROW(cell, head) (kv_size in a push constant), elastic slots are
  carved per head region. `attn_hm`: per 64-key block the workgroup streams the head's K rows then V rows (17 KB
  each) with contiguous 16-byte loads into one LDS row buffer (69-word stride), next block loaded during this one.
- Bug found by KL, not by text: JDE took each token's cell from the K index input, which now has a row per
  (token, kv head) - a 4-entry stack array received 8 entries (KL 0.3 and run-to-run nondeterminism in
  multi-token windows; single tokens were exact). Greedy-text equality is a poor test here: different but
  equivalent kernels flip the text after ~30 tokens. Use `llama-junk-ref` KL (now with L_code / L_prose on AUX).
- attn_hm phase test (245k): full 2030 us; without QK math 1596, without PV math 1580, without both 1614 (probe
  floor 1504). QK and PV each fit under the loads, together they do not: the block's phases are serialised by
  barriers and only 2 workgroups fit per CU (128 VGPRs, 26 KB LDS). Fewer VALU ops (block-scale folding, p via
  LDS) and a full block of load lead time changed nothing.
- Lesson (AUX reboot): two model processes at once (a probe run overlapping a build+test) OOM the box (pinned
  experts); /tmp is wiped by a reboot - helper scripts and prompts now live in ~/jt on AUX.

## 2026-10-18 (night) — sparse decode attention: fails the KL gate on this model (kept off)

**Gains: none shipped (JI_SPARSE off by default). Decode / prefill t/s unchanged.**

Engine 1e6f800d1: Quest-style page selection in JDE (32-cell pages, per-dim center/radius of rotated K,
bound q.c + |q|.r per query head, shared selection per kv head, radix top-k + sink + recent pages, attn_hm over
the selected pages). Quality harness: `llama-junk-ref` with `JR_STATE_DIR` (prefix states cached on the SSD,
`/mnt/ssd/kvcache/jref`) and `JR_TAIL_B=4` (scored tail through JDE); references = our own EXL3 4.0 bpw, q8_0 KV,
full attention (user's gate). Dense vs reference KL 0 (all-pages sparse 0.00015 = chunking noise).

| row | selection | budget | KL mean | p99 | top-1 |
|---|---|---|---|---|---|
| L_code 32k | Quest bound | 4096 / 8192 / 16384 | 0.0092 / 0.0061 / 0.0045 | 0.097 / 0.063 / 0.037 | 98.6 / 98.7 / 99.2 % |
| L_code 32k | exact-max oracle | 8192 | 0.0065 | 0.059 | 98.6 % |
| L_code 32k | attention-mass oracle (per-head relative) | 4096 / 8192 | 0.0059 / 0.0034 | | 99.2 % |
| L_code 128k | Quest bound | 16384 | 0.025 | 0.27 | 96.1 % |
| L_code 128k | attention-mass oracle | 16384 | 0.0143 | 0.16 | 96.7 % |

Even the ideal page choice loses ~1.4 % KL-worth of mass at 12.5 % of a 128k context: the 10 global-attention
layers of this hybrid model attend broadly (the GDN layers carry the local context). Per-query-head selection
would need 8x the K/V reads (GQA sharing lost). Not a go under "tiny KL"; the q8_0-KV bar is ~0.0015.
Consequence for KV streaming (user link: adaptive KV streaming fork, host RAM + VRAM pages + ring buffer): only
worthwhile with selection; dense attention would stream 2.85 GB per token over PCIe. Parked with sparse attention.

## 2026-10-18 (late night) — KVarN-6 validated on quality, not yet a speed win; research review (Gemini report)

**Gains: none shipped this round. Decode / prefill t/s unchanged (247k: 21.2 / 114.5).**

- KVarN quality (engine 356031b68, `JI_KVSIM=bits` simulation: completed 128-cell tiles of the q8_0 cache are
  KVarN-quantised and written back as q8_0; sink tile exact). KL vs **f16 KV**, L_code 32k, last 1024 tokens via JDE:
  q8_0 0.00090 (p99 0.0177) | KVarN6 sim 0.00099 (0.0241) | KVarN5 0.00138 | KVarN4 0.00224. The simulation adds two
  q8 roundings, so real KVarN6 meets the q8_0 bar.
- KVarN-6 decode attention `attn_kvr` (engine, experimental): correct, but 1193 us per layer at 116k vs ~1000 for
  attn_hm on q8_0 although it reads 21 % fewer bytes: 128 VGPRs + 22.0 KB LDS (2 workgroups per CU) and the extra
  unpack / axis work lands in the barrier-serialised block phases that already limit attn_hm. The VRAM it would
  free is worth ~+5 % at 247k (expert cache test: 500 MiB less cache = 20.3 -> 19.6 t/s, hit rate 79.2 -> 74.3 %).
  Full port (tiles + f16 tail in llama-kv-cache, FA2 prefill reader, state I/O, truncation) parked until the
  attention kernel is faster than q8_0's.
- Research review (user-provided Gemini report "JunkInfer Technical Strategy Report", plus own survey):
  1. Quest page-sparse decode: measured above - fails the KL gate even with oracle selection.
  2. AcceptMoE verifier expert pruning for MTP (arXiv 2608.02989): MTP deferred by the user.
  3. Full-rate u24 instead of v_mul_lo_u32 in the EXL3 trellis decode: already tried 2026-10-17 (removed from the
     ISA, decode and prefill unchanged - not the limiter).
  4. Recency-frequency expert cache policy with persistent frequencies / admission filter (arXiv 2609.29032):
     next (trace simulation first). iqk AVX2 CPU kernels: EXL3 misses run zero-copy on the GPU, CPU idle.
  5. SpargeAttn-style PV skip in ji_fa2 (skip a block's V when all scores < running max - T): implemented and
     measured on a 79.6k-token prompt - identical FA time and output at T = 12 / 16: blocks that far below the
     running max essentially never occur here (broad attention, as in the sparse-decode result). Reverted.
     XAttention sparse prefill: same broad-attention risk; not tried.
- Expert cache policy sweep at 247k (saved context, 3 runs each): decay 0.98 (default) 79.20 % hit, 20.1-20.3 t/s;
  0.995 79.56 %, 20.2-20.4; 0.95 78.64 %; 24 swaps per token 78.47 %. Within noise: the persistent decayed-frequency
  score + admission hysteresis already is the "recency-frequency hybrid"; the hit rate is capacity-bound.
- attn_hm at 3 workgroups per CU (score exchange through LDS float atomics - VK_EXT_shader_atomic_float, 2 KB
  instead of 8 KB - combine buffers aliased into the row buffer, LDS 19.7 KB): 1115 -> ~1510 us at 116k. LDS float
  atomics are far too slow on gfx803. Reverted. Note: the same attn_hm build measured 992 and 1115 us in two
  sessions - ~10 % run-to-run variance in this microbenchmark; small kernel deltas need repeated runs.

## 2026-10-19 — host overhead per decode token (easy wins, in agreed easiest-first order)

**Gains: decode (server, short context) 43.9 -> 45.75 t/s (+4.2 %); prefill unchanged.**
Real per-token time (JI_PROFILE, no GPU profiler): 21.7 ms = 20.2 GPU + 0.7 "record" + 0.8 "post" (GPU idle).
- Engine 239ae079b: the expert-cache update sorted all ~3,500 resident slots every token to find <= 8 eviction
  candidates; partial sort -> post 0.78 -> 0.23 ms; **43.9 -> 45.3 t/s**.
- Engine 1cb533054: one-shot tokens submit one replayable command buffer instead of 81 (2 per layer + final;
  JI_ONECB=0 restores) -> record 0.67 -> 0.25 ms; **44.9 -> 45.75 t/s** (two runs each).
- Skipped after re-estimating: FA2 exp2 / lazy rescale (softmax is ~2 % of the block's instructions: < 1 %,
  below noise); LM head already streams at ~156 GB/s (<= 0.5 ms left).
- Server gap: JDE 20.8 ms vs 21.9-22.0 ms per token served. Sampler chain variants (top_k 1, minimal chain) move it
  by only +-0.6 ms (top_k=1: 21.9 ms); the rest is the serial logits copy / candidate building / server loop.
  Profiling note: the GPU profiler (JI_PROFILE_GPU) serialises dispatches, so concurrent small kernels look
  expensive; judge them by ablation. llama-completion at 262k ctx gets a 0 MiB expert-cache budget (the server
  gets ~900 MiB + elastic slots): use the server for decode measurements.

## 2026-10-19 Sparse decode attention with tail estimation (Gemini research #2)

**Gains: none shipped (off by default; JI_SP_BG / JI_SP_SEL experiments). Default decode / prefill t/s unchanged.**

Engine 4f1013c74: unselected pages are no longer dropped but estimated from per-page statistics (RetroInfer /
Multipole tripartite idea): attn_meta stores f16 mean K, per-dim K variance, mean V, cell count; attn_bg adds
mass n*exp(q.muK + q^2.varK/2) with value muV as extra chunk partials for attn_merge. JI_SP_SEL=1 ranks pages
by that mass estimate instead of the Quest bound (and drops the center/radius metadata).

| row | selection | budget | KL mean | p99 | top-1 |
|---|---|---|---|---|---|
| L_code 32k | Quest, no tail | 4096 | 0.00906 | 0.085 | 98.6 % |
| L_code 32k | Quest + tail (mean only / +var) | 4096 | 0.00331 / 0.00307 | 0.034 / 0.040 | 99.6 / 99.7 % |
| L_code 32k | Quest + tail | 8192 / 16384 | 0.00179 / 0.00033 | 0.011 / 0.006 | 99.7 / 99.9 % |
| L_code 32k | mass-oracle + tail | 4096 | 0.00090 | 0.014 | 99.8 % |
| L_code 32k | mass estimate + tail | 4096 / 8192 | 0.00438 / 0.00101 | 0.026 / 0.018 | 99.4 / 99.9 % |
| L_code 128k | Quest, no tail (earlier) | 16384 | 0.025 | 0.27 | 96.1 % |
| L_code 128k | mass estimate + tail | 8192 / 16384 / 32768 | 0.00483 / 0.00340 / 0.00242 | 0.068 / 0.057 / 0.036 | 98.1 / 98.2 / 98.3 % |
| L_code 128k | **Quest + tail** | **16384** | **0.00094** | 0.014 | 98.7 % |

Speed (server, restored contexts, warm runs) and VRAM (metadata sized for the 262k context):

| config | metadata | expert cache | 2k decode | 100k decode | 245k decode |
|---|---|---|---|---|---|
| dense (default) | 0 | 500 MiB | 43-44 | 28.5 | 20.6 |
| mass + tail 8192 | 241 MiB | 256 MiB | 42-43 | 32.3 | 29.3 |
| mass + tail 16384 / 32768 / 61440 | 241 MiB | 256 MiB | | | 27.8 / 26.7 / 25.7 |
| Quest + tail 16384 | 401 MiB | 96 MiB | | | 20.0 |
| first f32 layout (Quest + tail 4096) | 650 MiB | 0 | | | 6.3 (experts on CPU, buffers spilled) |

The quality winner (Quest + tail) loses its attention gain to the expert cache at 245k: metadata VRAM is the
binding cost. Next: shrink the Quest + tail metadata (8-bit center/radius, drop var, coarser bound pages) toward
<= 250 MiB and re-measure. Prefill unaffected (decode-only path; one ~30-60 ms metadata rebuild per request).

## 2026-10-19 Prefill small-op fusion: MULTI_ADD on gfx8

**Gains: prefill 7.2k 387.9 -> 392.8 t/s (+1.3%); decode unchanged (43-44 short, 20.6 at 245k).**

Engine: ggml-vulkan gated MULTI_ADD on shaderRoundingModeRTEFloat16, which RADV POLARIS10 lacks; the shaders are
f32-only, so the gate is lifted for AMD. ~400 full-size f32 ADDs per 4096-token ubatch (MoE expert sums,
residuals) become multi-adds: ADD GPU time 443 -> 198 ms. PPL identical (4.7578). Remaining small ops on the 7.2k
prompt (RMS_NORM 130, CONCAT 99, GLU 74, MUL 63, SILU_MUL 58, SCALE 47 ms; ~2.7 % of GPU time) are worth < 1 %
each - deferred behind the sparse-attention metadata work (4b) and the EXL3 prefill GEMMs.

## 2026-10-19 Sparse decode attention: mean-centered bound, 241 MiB, kernel tuning

**Gains: none shipped by default (JI_SPARSE=8192 JI_SP_BG=1 opt-in): decode 245k 20.6 (dense) -> 30.9 t/s, 100k
28.5 -> ~32; default decode / prefill unchanged (43-44 short; prefill 7.2k 392.8).**

Engine 7e30faaac. The Quest center/radius block is replaced by a bound around the page mean (q.mu + |q|.r with r =
max |k - mu| per dim, still a valid upper bound), so the metadata is one dim-major f16 layout: mean K, radius,
(var K with JI_SP_BG=2), mean V, count = 1.5 KB per page and kv head, 241 MiB at 262k ctx (expert cache 500 -> 256
MiB; short-context decode unaffected thanks to the elastic slots). Quality is better than the Quest bound:

| row | config | KL mean | p99 | top-1 |
|---|---|---|---|---|
| L_code 32k | mean bound + tail (var) 4096 | 0.00229 (Quest bound: 0.00305) | 0.033 | 99.4 % |
| L_code 128k | mean bound + tail 8192 / 16384 | 0.00139 / 0.00114 | 0.019 / 0.013 | 98.2 / 98.6 % |
| L_code 128k | mean bound + tail (var) 16384 | 0.00092 | 0.012 | 98.7 % |

Kernel tuning (GPU profile per layer at 245k, budget 8192): attn_hm + attn_bg 407 -> 183 us (bg: page compaction by
ballot prefix, independent loads, wave reductions, reuses attn_score's log-mass estimates), attn_topk 182 -> 83 us
(keys cached in padded LDS, parallel suffix scan), attn_meta 49 -> 6 us (decode: only on page completion),
attn_score 82 -> 110 us (now also writes the estimates). Sparse attention ~4.2 ms/token vs dense ~20 ms.

| 245k decode (t/s) | 8192 | 16384 | 16384 + var (321 MiB) |
|---|---|---|---|
| before tuning | 28.6 | 26.7 | 27.5 |
| after tuning | 30.9 | | |

Open: KL at 245k (reference L_code r2 being generated), then default-on above a context threshold.

## 2026-10-19 Sparse decode attention default-on (>= 96k cells)

**Gains: decode 245k 20.5 -> 29.6 t/s (+44%); short decode unchanged (42.9, noise); prefill unchanged (392.8 at 7.2k).**

262k gate (new reference L_code r2, 261k prefix, generated with full attention; state cached in jref/):

| budget (+ tail, mean bound) | KL mean | p99 | max | top-1 |
|---|---|---|---|---|
| 8192 | 0.00109 | 0.013 | 0.036 | 99.2 % |
| 16384 (default) | 0.00107 | 0.013 | 0.029 | 99.3 % |
| 32768 | 0.00111 | 0.012 | 0.045 | 99.3 % |

KL is flat in the budget at 262k (floor ~0.0011 from the estimated tail), below the q8_0-KV bar. Defaults:
JI_SPARSE=16384, JI_SP_BG=1, JI_SP_MIN=98304 (128k and 262k measured; 64k-96k stays dense until gated). Metadata
241 MiB only when n_ctx >= JI_SP_MIN (expert cache 500 -> 255 MiB). 100k with sparse forced on: 28.9 -> 34.0 t/s.
Full-attention references must set JI_SPARSE=0 (mkrefs.sh does).

## 2026-10-19 Decode EXL3 mat-vecs: probes, f16 codebook sum

**Gains: decode short 43.5 -> 45.0 t/s (+3%), 245k 29.6 -> 31.1 t/s (+5%); prefill 7.2k 392.8 -> 393.3 (unchanged).**

Probe-first (x3_mv compiled with -DX3_EXP, JI_BENCH, pipelined us):

| variant | 8192x2048 | 2048x4096 | LM head 248k x 2048 (6 bit) |
|---|---|---|---|
| production | 84.2 | 49.1 | 2425 |
| trivial codebook | 51.2 | 31.7 | 1781 |
| no LDS input reads | 80.8 | 46.5 | 2339 |
| no multiply | 68.4 | 40.3 | 1942 |
| no f16 -> f32 conversions | 64.5 | 38.2 | 1886 |
| f16 sum, one conversion | 75.4 | 44.0 | 2174 |
| f16 4-weight dot per tile | 75.8 | 44.5 | 2157 |

The codebook (mcg: 32-bit multiply at quarter rate + two conversions) bounds the decode mat-vecs, not memory or LDS.
Engine: the halves are summed in f16 (exllamav3 does the same, __hadd -> half W_hat) in every decode / prefill /
dequant / CPU path. RADV reports shaderFloat16 = false on gfx8 (it is tied to packed 16-bit math); ACO emits the
scalar f16 add. PPL 4.7578 -> 4.7315; KL vs the old kernels' reference 0.00071 (32k). Remaining codebook cost is the
multiply (~16 us of 78 on 8192x2048); the rest of the gap to the trivial decode is the conversion and extraction.

## 2026-10-19 Prefill: EXL3 GEMM pipelining + BK16, worst-case output reservation

**Gains: prefill 7.2k 393.3 -> 447.6 t/s (+14%); 79.6k (llama-completion) 225.8 -> 264.1 t/s (+17%), server 263.4.
Decode unchanged (45 t/s short, 31.1 at 245k).**

| change (engine) | prefill 7.2k | note |
|---|---|---|
| f16 codebook sum (8ba2c0db8) | 392.8 -> 393.3 | GEMMs are not codebook-bound |
| GEMM: next k step's loads during decode + FMAs (1d0a96376) | 393.3 -> 401.4 | output identical |
| GEMM: 16-deep k steps for the fused 4-bit dense / MoE kernels (6c8cb9f29) | 401.4 -> 447.6 | 128 -> 64 VGPRs, LDS 26.5 -> 13.5 KB; X3F epilogue in BK-column groups |
| tried: non-unrolled decode loop / dense 128-col + BK16 / MoE 32-col + BK16 | 445 / 447 / 425 | kept defaults |

Per-op (perf logger, 7.2k): dense 8192x2048 GEMM 41.1 -> 35.1 ms per ubatch-op (3.34 -> 3.9 TFLOPS, 63 % of peak);
MoE gate/up 27.9 -> 23.4 ms (~2.9 TFLOPS incl. ~1/3 tile padding).

**Worst-case output reservation (a440dface).** A 79.6k llama-completion prefill spent 137 s of 363 with the GPU idle;
perf showed amdgpu_cs_ioctl -> ttm_bo_validate -> ttm_lru_walk_for_evict on every submit: VRAM was full (64 MiB
free). llama.cpp reserved the prompt graph for n_ubatch outputs = 4 GiB of f32 logits (248k vocab) in the compute
buffer (ctx 90k: 5192 MiB; 262k: 6536 MiB + 2720 KV + 1289 weights > 8 GiB). Now reserved for <= 64 outputs
(JI_RESERVE_OUTPUTS; batches needing more re-reserve on demand, PPL identical): 1312 / 2656 MiB. The server already
capped n_outputs_max, so server numbers were not affected; benchmarks / junk-ref / CLI were (earlier CLI prefill
figures at long contexts were pessimistic).

## 2026-10-19 Sparse-attention gate around the threshold; references regenerated

References L_code r0 (32k) and r1 (128k) regenerated with the current kernels (f16 codebook sum; old ones in
~/jt/refs_old, old prefix states in /mnt/ssd/kvcache/jref_old; r2 262k still the pre-f16 reference). New rows cut from
L_code (same scoring rule: last 1024 positions): L_code64 (64k, from r1) and L_code98 (100k, from r2), references
with JI_SPARSE=0.

| row | budget (+ tail) | KL mean | p99 | top-1 |
|---|---|---|---|---|
| L_code64 (64k) | 8192 / 16384 | 0.0036 / 0.0023 | 0.051 / 0.031 | 98.6 / 99.3 % |
| L_code98 (100k) | 16384 (default) / 32768 | 0.00101 / 0.00090 | 0.018 / 0.017 | 99.6 / 99.7 % |
| L_code r1 (128k, new ref) | 16384 (default) / 32768 | 0.00128 / 0.00110 | 0.023 / 0.020 | 99.1 / 98.9 % |

The 64k row fails even at 25 % of the context: quality depends on content, not only on the selected fraction. The
default threshold (JI_SP_MIN = 98304) stays: every measured row >= 96k passes at the default budget.
Router (decode): rows per wave 1/2/4/8 = 30/34/34/35 us - latency bound (top-k tail); unchanged.

## 2026-10-19 Easy-win sweep: n-gram speculation, router, MoE tile fill

**Gains: none measurable by default (router -0.13 ms/token). Decode 45 short / 31.1 at 245k, prefill 447.6 at 7.2k.**

- **N-gram self-speculation** (server --spec-type ngram-map-k, n 4, m 3; off by default). The server saved a full
  63 MiB recurrent-state checkpoint per verify step on this hybrid model: copy-heavy code rewrite 33.3 -> 28.6 t/s
  despite 88 % acceptance. Engine 072acf561 gives n-gram drafts per-token recurrent snapshots (n_rs_seq <= 3, the
  JDE verify window): 33.3 -> 37.3 t/s on the rewrite, story unchanged. Limits: a 4-token verify step costs 65-90 ms
  vs 22 ms per token (four tokens route to ~3x the distinct experts; misses over PCIe), and with a cached prompt the
  n-gram map drafts far less (116 vs ~400 drafted tokens on a repeated request). Not worth enabling by default.
- **Router** (engine 6982622ed): four waves per workgroup, 30.0 -> 26.8 us per layer. Probe: the top-k tail is not
  the cost (30 us without it); the logits phase is latency bound; more rows per wave is slower.
- **MoE GEMM tile fill** (ub 4096, 7.2k prompt): 32768 rows in 629 64-column tiles = 81 % fill; the kernel already
  skips empty 8-column groups, so the waste is ~5 % of the MoE GEMMs (~1.5 % of prefill). Not pursued.
- KL harness note: since a440dface the dense L_code 32k KL vs its reference is 0.00022 (0 with JI_RESERVE_OUTPUTS=0):
  in junk-ref's 262k context the smaller compute buffer holds fewer JDE elastic slots, so more expert misses run on
  the CPU (different f32 summation order). KL runs are 6.6x faster (37 vs 247 s: no eviction churn).

## 2026-10-19 MVP: usable server (API suite 45/45, real agent client)

**Gains: usability; decode / prefill unchanged (45 t/s short; 447 t/s prefill at 7.2k).**

- `scripts/api_suite.py` (OpenAI 3.35 + Anthropic 1.15 SDKs): first run 36/38. Failures: tool_choice "required"
  produced chat text that degenerated into emoji runs, and named tool_choice objects were ignored. Causes: the
  Qwen3-Coder / Qwen3.6 grammar (common/parsers/qwen3-coder.cpp; also the auto-parser's three tool formats) allowed
  free content before a mandatory call, and the server parsed tool_choice as a string. Fixed (engine 690f73171):
  forced calls follow the reasoning directly; tool_choice objects of OpenAI chat / Responses / Anthropic narrow the
  tools to the named function and force a call. Now 45/45 incl. code-content arguments (multi-line Python through
  the XML tool format), a 24-tool coding-agent set, a 6-turn agent session (63 new prompt tokens per turn), a
  Claude-Code-shaped Anthropic request (system blocks with cache_control, tools, thinking, streaming), and --long:
  a 117k-token needle through the API (prefill 212 t/s, sparse attention active, correct answer).
- serve.sh: EXL3 default model, alias qwen3.6-35b-a3b, JI_HOST / JI_API_KEY (401 without it, OpenAI bearer and
  Anthropic x-api-key) / JI_THINKING (off by default, enable per request) / JI_CACHE_RAM; systemd unit template.
- Real client: aider 0.86.2 (diff edit format) added a function and a unittest file to a scratch repo and committed;
  the tests pass.
- Ops: restart scripts refuse to start a server while another model process runs (two at once OOM the box; a
  server start during the 262k reference run failed to load, no damage).

## 2026-10-19 Sparse prefill attention (opt-in, JI_SPARSE_PF)

**Gains (opt-in): prefill 79.6k 264.1 -> 358.5 t/s (8192 keys) / 313 (16384 + local 8192); 262k row ~2.4x faster
(19.5 vs 47.6 min incl. load). Default decode / prefill unchanged.**

Engine 4a61328e6 + sub-pages: ji_fa2_pmeta (32-key sub-page statistics: mean K, radius, var K, mean V) and
ji_fa2_sparse (FA2 core; per workgroup of 4 positions: causal boundary by binary search on the mask, 128-key pages
before boundary - JI_SPF_LOCAL scored by the bound around the mean (max over sub-pages, rows, heads), radix select of
the budget + JI_SPF_SINK sink pages, exact attention over them + the local window, tail estimate of the rest per
sub-page merged into the online softmax; dense fallback if the mask is not a causal prefix).

| row (decode dense unless noted) | config | KL mean | p99 | top-1 |
|---|---|---|---|---|
| 32k | dense (harness floor) | 0.00022 | 0.003 | 99.9 % |
| 32k | 4096 / 8192 / 16384 keys (128-key stats) | 0.00125 / 0.00066 / 0.00045 | 0.018 / 0.008 / 0.006 | |
| 32k | 4096, tail dropped (mode 0) | 0.00372 | 0.051 | 98.8 % |
| 32k | 4096, 32-key sub-pages | 0.00114 | 0.018 | 99.9 % |
| 128k | 8192 / 16384 (128-key) | 0.00146 / 0.00129 | 0.022 / 0.016 | 98.6 / 99.2 % |
| 128k | 16384 + var / + local 8192 | 0.00128 / 0.00124 | | |
| 128k | 16384 + local 8192, sub-pages | 0.00117 | 0.018 | 99.5 % |
| 128k | same + default sparse decode | 0.00183 (128-key: 0.00206) | 0.026 | 98.7 % |
| 262k | 16384 + local 8192 (128-key) | 0.00164 | 0.018 | 99.0 % |

Errors add: prefill and decode approximations together exceed the ~0.0015 bar (q8_0-KV level) at 128k. Budget and
the variance term barely matter; the tail estimate matters most (3x lower KL than dropping). Opt-in for now
(serve.sh documents it); default decision left to the user.

## 2026-10-19 MTP: research, verify-cost scaling, MTP-head quantization sweep

**Gains: none (analysis). Decode / prefill unchanged.**

Research (user ideas): reduced draft vocabulary (HyperQwen: 40k ids from the model's own outputs, 97.5 % coverage,
draft ~3 ms -> 0.5-1 ms; vLLM #58578: +9-14 % on Qwen3.6-35B in llama.cpp); probabilistic drafting + rejection
sampling (llama.cpp #27694, merged 2026-10-02: +2-9 % only at temperature >= 0.5); --spec-draft-p-min confidence
gating. Verify-cost scaling here (n-gram drafts through JDE, per-kernel GPU profile, single vs 2-token step):
23.3 vs 47.9 ms. Dense mat-vecs / LM head 1.1-1.3x, GDN / attention / router 1.0-1.9x, routed-expert mat-vecs
2.9-3.0x (one row per token and slot, misses over PCIe per row). A 1-token draft can therefore never win on one
card; ideal sharing would give ~37 ms (+12-24 % with good acceptance). MTP engine work deferred to the second card;
the lossy option worth gating later is verification restricted to cached experts (AcceptMoE, arXiv 2608.02989).

MTP sidecars from the BF16 GGUF (tools/mtp_extract.py: metadata + output head + token embedding + blk.40; output
head Q6_K and embedding Q8_0 fixed, block type varied), server draft-mtp, draft n 3, 6 prompts x 300 tokens
(code rewrite, story, technical, bash, Polish poem, arithmetic), thinking off:

| MTP block | size | accept t=0 | accept t=0.7 | per position (t=0) |
|---|---|---|---|---|
| Q4_0 | 1.45 GB | 0.615 | 0.556 | 0.76 / 0.56 / 0.43 |
| IQ4_NL | 1.45 GB | 0.644 | 0.550 | 0.78 / 0.57 / 0.44 |
| Q5_K | 1.59 GB | 0.642 | 0.566 | 0.79 / 0.57 / 0.44 |
| Q6_K | 1.66 GB | 0.621 | 0.566 | 0.78 / 0.56 / 0.43 |
| Q8_0 | 1.87 GB | 0.637 | 0.569 | 0.78 / 0.57 / 0.45 |
| BF16 | 3.74 GB | 0.637 | 0.571 | 0.79 / 0.57 / 0.44 |

Acceptance is flat from 4 bits up: BF16 is the ceiling and IQ4_NL reaches it (matches the user's earlier finding),
so better formats (GPTQ / EXL3) can only pay by keeping it at fewer bits -> testing Q3_K / IQ3_S / Q2_K.
Draft-vocabulary coverage from the local corpora saturates at 0.81-0.89 (only ~20k distinct tokens in 1 M tokens):
a real list needs millions of tokens of model output (planned with the second card).

## 2026-10-19 Sparse prefill on by default (user decision)

**Gains: prefill 79.6k 264.1 -> 358.5 t/s (+36 %), 262k ~2.4x; prefill < 32k and decode unchanged.**

User: the combined benefit of sparse decode + prefill is enormous next to the cost - keep both on by default, noted
as not free but worth it. Engine 2767f7ada: JI_SPARSE_PF defaults to 8192 (local window 4096, 32-key statistics
sub-pages, from 32k keys). Error budget clarification: all KL figures are vs our own full-attention reference
(EXL3 4.0 bpw, q8_0 KV), so the sparse terms add on top of the weight and KV quantization error (roughly additive at
these magnitudes: 128k prefill ~0.0012 + decode ~0.0011 -> 0.0018 measured together). The q8_0-KV error (~0.0015) was
only a yardstick for "tiny"; the real denominator, EXL3 4.0 bpw vs BF16 KL, is still owed (needs a reference GPU).
Reference scripts set JI_SPARSE=0 JI_SPARSE_PF=0.

## 2026-10-19 Gemini report 1 (expert memory hierarchy) + miss substitution + MTP low-bit

**Gains: none (rejected experiment). Decode / prefill unchanged.**

Report https://share.gemini.google/foAnchIoTUbE (prompt "MoE expert memory hierarchy"): many numbers are estimates;
it assumes an SDMA transfer queue (RADV has none on gfx8) and an x8/x8 dual-card topology (B450: PCIe 2.0 x4
second slot). Most promising item tested: miss substitution ("0.15 rule", Son 2026, Qwen3-Next-80B: +0.35 % PPL).
Engine: JI_SUBST_TAU in router_fused (non-resident, rank > 1, normalised weight < tau -> best resident expert, weights
renormalised; cache statistics follow the router's own choice). KL (L_code 32k, decode):

| tau | 0.02 | 0.04 | 0.06 | 0.10 | 0.15 | 0.20 |
|---|---|---|---|---|---|---|
| KL mean | 0.0020 | 0.0033 | 0.0078 | 0.019 | 0.033 | 0.054 |
| harness tail tok/s (off: 864) | 876 | 880 | 919 | 1052 | 1272 | 1280 |

Perplexity hides per-token divergence that KL sees: no setting is inside the budget. Rejected (code kept, off).
Other report items judged low value here: async SDMA copies (no transfer queue), 2-bit EXL3 tier (needs CUDA
quantization), eviction guards / LRFU (+1-4 % hit rate in the literature; may revisit).
MTP low-bit (uncalibrated): Q3_K 0.613, IQ3_S 0.563, Q2_K 0.550 at t=0 (IQ4_NL 0.644) - the knee is below 4 bit,
where calibrated quantization of the MTP block (imatrix / GPTQ) could recover acceptance.

## 2026-10-19 Gemini report 2 (speculative decoding) + speculative-position substitution + tail diagnostics

**Gains: none by default. API suite 46/46 with sparse prefill on; 117k needle prefill 212 -> 338 t/s.**

Report https://share.gemini.google/OwGeIe5VcENL (prompt "speculative decoding when verify cost scales with distinct
experts"). Useful: lossless grouped verify (one read per distinct expert across verified tokens), AcceptMoE (root
token exact, speculative positions restricted to resident experts), suffix-automaton drafting with JSON-escaped
indexing for agentic edits (AgSpec / SAM-Decoding), dynamic draft vocabulary (NanoSpec: context window + base top-K +
~4k common tokens, 512-3072 ids), p-min gating. Contradicted here: "quant-drafter alignment" (higher-precision MTP
heads collapsing acceptance) - our sweep is flat from Q5_K to BF16.

AcceptMoE-style test (router substitution only at token index >= 1 of a ubatch, JI_SUBST_TK0=1): KL 0.050 (first
choice kept) / 0.075 (all misses) in 4-token tail batches; n-gram speculation (map-k, n 4 / m 3) on the code rewrite
37.2 -> 50.1 t/s (no speculation 33.0). Too lossy: the approximated positions' logits pick the accepted tokens.

Tail-estimation diagnostics (attn_bgx, L_code 128k, decode sparse 16384): estimate with variance (JI_SP_BG=2) 0.00089;
exact page mass + mean V (8) 0.00019 (= harness floor); estimated mass + exact page value (9) 0.00082. The page-mass
estimate is the error; the mean value is good enough. Default JI_SP_BG=1 is 0.00128 at this row -> the variance term
alone cuts ~30 %. Next for sparse quality: better page-mass estimates (variance term, low-rank page covariance,
low-bit key sketches - VRAM permitting).

## 2026-10-19 MTP end-to-end speed per quant; variance-term speed; Gemini report 3 (sparse KL)

**Gains: none yet (measurements + two opt-in experiments, see below).**

MTP sidecars ranked by real decode speed (server, 32k ctx, ~1.6k tokens, draft n 3, `mtpsweep.sh`):

| draft | acc t=0 | decode t=0 | decode t=0.7 |
|---|---|---|---|
| none | - | 31.1 | 32.2 |
| IQ4_NL | 0.644 | 9.5 | 9.0 |
| IQ4_NL + q8 embd/head | 0.635 | 9.5 | 9.1 |
| Q8_0 | 0.637 | 9.5 | 9.2 |
| Q4_0 | 0.615 | 9.3 | 9.1 |
| Q3_K | 0.613 | 9.3 | 9.0 |
| Q2_K | 0.550 | 8.8 | 8.6 |

MTP is ~3x slower than no speculation on one card: the draft runs through the generic ggml graph (not JDE) and the
multi-token verify reads ~2-3x the experts. The quant barely changes speed (fixed overhead dominates), so acceptance
decides: IQ4_NL. Parked until card #2 (cheap draft step + grouped verify first).

Variance term (JI_SP_BG=2) at 245k: decode 30.75 -> 30.0 t/s (-2.4 %), +80 MB metadata, decode KL -30 %.

Report https://share.gemini.google/eEIqnjtoBouw (improving sparse decode / prefill KL). Its 35 % mass / 65 % value
error split is contradicted by our attn_bgx diagnostic (the mass estimate is ~all of it). Taken, easiest first:
P2 exact dense prefill for the prompt's last positions, P1 gate-weighted self-normalised GQA ranking; then P4 dipole
sub-centroids / low-rank page variance (attack the mass), P3 32-key sub-page prefill selection, P5 per-head budgets;
Hadamard blocks / 1-bit sketches last (format + VRAM).

## 2026-10-19 Exact dense prompt tail for the sparse prefill (default 4096) - engine 7a6b79f2f

**Gains: prefill KL at 130k 0.00123 -> 0.00035 (-72 %); prefill 132k 324.3 -> 305.5 t/s (-5.8 %); decode unchanged.**

The prompt's last JI_SPF_DENSE_TAIL positions run the prefill attention densely (report 3, P2). The server sends the
last 4 prompt tokens as a separate batch (checkpoints), so the prompt end is passed explicitly
(`llama_ji_set_prompt_end`, server + junk-ref); without it the batch returning outputs is taken as the end.

| dense tail | 0 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| prefill KL (L_code r1, 130k, dense decode) | 0.00123 | 0.00111 | 0.00102 | 0.00035 | 0.00033 |
| server prefill 132k t/s | 324.3 | - | - | 305.5 | 288.7 |

Decode-side experiments from report 3 (opt-in; decode KL with dense prefill prefix states):

| config | 99k (L_code98) | 130k (L_code r1) | 245k (L_code r2) |
|---|---|---|---|
| default (BG 1) | 0.00101 | 0.00128 | 0.00116 |
| JI_SP_NORM=2 (gate-weighted share) | | 0.00121 | |
| JI_SP_BG=2 (variance) | | 0.00089 | |
| NORM=2 + BG=2 | 0.00107 | 0.00075 | 0.00105 |
| NORM=2 + BG=2, VARK 0.35 / 0.7 | | 0.00113 / 0.00078 | |

245k decode: default 30.8, BG=2 30.0, NORM=2 + BG=2 28.6 t/s (topk reductions since rewritten with wave ops).
The 130k gain does not carry over to the other rows: no default change yet.

## 2026-10-19 Sparse decode: page statistics in the raw key basis (default) - decode KL -71 %

**Gains: decode KL vs full attention 0.00115 -> 0.00033 (mean of the 99k / 130k / 245k rows); 245k decode 30.0 ->
29.55 t/s (-1.5 %); 64k row now passes with sparse decode (0.00034, was 0.0023).**

Diagnosis (offline, `JI_KV_DUMP` + `JI_KV_DUMP_ALLQ`: K of all 10 attention layers and 64 decode queries per layer
at 130k, numpy in scratchpad `ana/`):
- Exact page mass for the unselected pages (attn_bgx mode 8) brings decode KL to 0.00019-0.00038 on all three rows:
  the page-mass estimate is 70-80 % of the error. Doubling the budget (32768) does not help (0.00113 vs 0.00115);
  a global log-mass offset of +-0.2 triples KL: not a bias, a per-page error.
- The K cache holds H_256 k (llama.cpp's rotation for quantised KV, n_rot_k = 256). It mixes the 64 RoPE dims (NEOX
  pairs i / i + 32, base 1e7) into all 256 dims, so per-dimension page variance / radius cannot follow RoPE. After
  un-rotating, the pair angles match the RoPE frequencies exactly.
- Layers 7 and 11 carry the error (tail = 44 % / 15 % of the attention mass at 130k). Gaussian page-mass error, as a
  fraction of the total mass: stored basis 1.74 % / 1.54 %; raw basis 0.01 % / 0.09 % (layers 15-39: 10-50x lower
  too). Keeping RoPE dims un-mixed is what matters (raw = rope-raw + H64 NoPE >> H64x4).

Implementation: attn_meta un-rotates each page's 32 keys in LDS (FWHT) before mean / variance / radius; attn_qk_q8
writes the pre-H_256 query (qr) for attn_score. JI_SP_RAW=1 and JI_SP_BG=2 (variance) are the defaults.

| decode KL | 64k | 99k | 130k | 245k |
|---|---|---|---|---|
| before (BG 1, stored basis) | 0.0023 (sparse from 32k) | 0.00101 | 0.00128 | 0.00116 |
| raw basis + variance | 0.00034 (sparse from 32k) | 0.00036 | 0.00033 | 0.00029 |
| exact page mass (diagnostic floor) | | 0.00038 | 0.00019 | 0.00028 |

Rejected on the way: JI_SP_ROT (log I0 for RoPE pairs - computed on the wrong dims before the rotation was found),
JI_SP_NORM=2 (gate-weighted ranking: -3 % KL for -4 % speed), JI_SP_BG=3 (dipole: -7 %), budget 24576 / 32768.

## 2026-10-20 Raw-basis statistics for the sparse PREFILL: rejected (speed)

ji_fa2_pmeta un-rotating its sub-pages + ji_fa2_sparse un-rotating the queries (FWHT by subgroup shuffles, f16 in LDS
aliased over the K/V staging): prefill KL at 130k (dense tail 4096) 0.00035 -> 0.00026, without the tail no gain
(0.00139). But the server prefill at 132k fell 305.7 -> 259.5 t/s (raw on) and to 281 t/s even with the feature
off - the extra code changed the dense core's code generation. Reverted; patch kept in scratchpad rawpf_patch/.
Combined default KL at 130k (both sparse, raw decode statistics, dense prompt tail): 0.00045. API suite 46/46.

## 2026-10-20 Prompt-tail variants, reports 3-5 (kernels / system / weights), system knobs, mul1 groundwork

**Gains (measured, not yet committed): per-layer exact prompt tail (layers 3, 7, 11) - prefill 132k 305.5 -> 318.9
t/s at prefill KL 0.00035 -> 0.00038 (none: 324.3 t/s, 0.00123). Decode unchanged.**

Prompt-tail variants (L_code r1 130k, dense decode; server prefill 132k):

| tail mode | prefill KL | prefill t/s |
|---|---|---|
| none | 0.00123 | 324.3 |
| semi-dense, 16k / 32k / 48k key budget + 16k local (JI_SPF_TAIL_BUDGET) | 0.00104 / 0.00072 / 0.00073 | 321.2 / 317.8 / 315.2 |
| exact in layers 7,11 / 3,7,11 / 7,11,15,19 (JI_SPF_TAIL_LAYERS) | 0.00045 / 0.00038 / 0.00043 | 320.8 / 318.9 / - |
| exact in all layers (committed default) | 0.00035 | 305.5 |

Default becomes JI_SPF_TAIL_LAYERS=3,7,11 (qwen35moe's first three attention layers; "all" restores the old mode).
The FA op carries its layer in op_params[13]. Raw-basis prefill statistics: KL 0.00035 -> 0.00026 but prefill
305.7 -> 259.5 t/s, and 281 t/s even with the feature off (code generation of the dense core changed): reverted,
patch kept.

System knobs (report 4; decode t/s 2k / 245k, runs 2-3): baseline 43.7 / 24.5-26.9; PCIe MaxReadReq 512 -> 4096:
43.8 / 24.1-26.6; + THP always: 43.6-44.7 / 23.8-26.4; CPU share of misses JI_X3_ZC 7 / 6 / 5: 43.4-43.7 /
24.3-26.9, 41.7-43.3, 41.8-43.9: no gain. Note the 245k baseline is below the 29.55 t/s of 2026-10-19 (open:
open-paths.md item 1). OverDrive already unlocked; IOMMU on (iommu=pt pending a reboot).

Report 3 (kernels, https://share.gemini.google/nAyRlhgbuKUE, Polish): taken - per-layer budgets, MoE two-height GEMM
tiles, XOR swizzle (profile first), 4-bit head; already tried / contradicted - u24 multiply, codebook LUT, PV skip,
dense last-128 queries (our knee is 4096), FA2 head split.
Report 4 (system, https://share.gemini.google/GixB1puPueeG): most items already in place (nogttspill, replayed
command buffers, all experts of a layer in one dispatch with hits and zero-copy misses together); measured above.
Report 5 (weights, https://share.gemini.google/uV2NHl77U4Gg): runtime levers taken (non-EXL3 LM head, dynamic top-k by
gate threshold, clustered head later); re-quantization items deferred (no quantization GPU); SVD / big pruning rejected.

mul1 / any-width EXL3 groundwork (for MiniCPM-V 4.7 35B-A3B EXL3 3.74 bpw, a Qwen3.6 fine-tune: config
qwen3_5_moe_text, codebook mul1, experts 3/4/5 bits, projections 5-8, head 8 bits; tokenizer = Qwen3.6's plus 67
special tokens in its padding slots, vocab 248,144, head 248,192 rows):
- numpy reference (tools/exl3/exl3_ref.py) decodes mul1 and any width; decoded MiniCPM weights vs Qwen3.6 IQ4_NL:
  cosine 0.988 (shexp down, 6-8 bit), 0.978 (gate 4-bit), 0.971-0.975 (up 3-bit), 0.939 (down 5-bit).
- engine (uncommitted): GGML_TYPE_EXL3_2/3/5/7/8, ggml_exl3_bits(); x3_mv / mul_mm_gcn_exl3 / dequant_exl3 generic
  in BITS (odd widths load uvec2) with the codebook as a specialisation constant; per-width pipelines; JDE CPU
  scalar path generic; GGUF key junkinfer.exl3.codebook.
- converter: codebook detection, any width, tokenizer re-derived from the checkpoint's tokenizer.json, its chat
  template, vocabulary padded to the head's rows (the head's output Hadamard needs whole 128-row blocks).
- On gfx803 mul1 is not cheaper than mcg as plain GLSL (byte sum by shifts); a v_sad_u8 lowering would make it ~20 %
  cheaper per weight. Speed upside of MiniCPM comes from smaller experts (cache capacity, PCIe bytes).

## 2026-10-20 Per-layer dense decode (opt-in), regression of the generic EXL3 build - engine f5ddada15

**Gains: prefill 132k 305.5 -> 318.7 t/s (per-layer prompt tail, default); decode unchanged (2k ~44, 100k 34.5,
245k 29.9 t/s); opt-in max-quality decode.**

Generic-width / mul1 build on the current 4/6-bit mcg model: decode KL 0.00033 (unchanged), prefill 318.7 t/s, 2k
decode 44 t/s. Measurement hygiene: a full page cache (after a KL run or a big download) slows 245k decode ~10 %;
`~/jt/restart.sh` now drops caches before starting the server.

JI_SP_DENSE_LAYERS (attention layers kept dense on sparse decode steps):

| config | 100k t/s | 245k t/s | decode KL 99k / 130k / 245k |
|---|---|---|---|
| default (all sparse) | 34.5 | 29.9 | 0.00036 / 0.00033 / 0.00029 |
| dense 7,11 | 33.6 | 27.2 | 0.00030 / 0.00006 / 0.00009 |
| dense 3,7,11 | 32.4 | 26.0 | 0.00004 / 0.00006 / 0.00004 |

Three dense layers bring sparse decode to the harness floor (dense 0.00007) but every token re-reads those layers'
whole KV: -13 % at 245k. Default stays sparse; documented as the max-quality setting in serve.sh.

## 2026-10-20 LM head as IQ4_NL: rejected

`tools/swap_head.py` (copy of a GGUF with output.weight from another GGUF) + the BF16 head of the MTP sidecar
quantized by llama-quantize (IQ4_NL 273 MB, Q5_K 333 MB; EXL3 6-bit head 381 MB): IQ4_NL head decode 2k 44.6 -> 47.3
t/s, 245k ~29.5 -> ~30.8, but KL 0.00033 -> 0.0044 (130k row), 0.00036 -> 0.0026 (99k), top-1 0.997 -> 0.976 - ten
times the sparse-attention KL for +4-6 %. The trellis 6-bit head is far better per byte; a cheaper head needs an EXL3
re-quantization (~4 bpw).

## 2026-10-20 Dynamic top-k (gate threshold): rejected

JI_TOPK_TAU (router_fused: experts after the first below tau of the normalised gate weight dropped, id -2, the rest
renormalised). KL 99k / 130k: tau 0.02 0.00164 / 0.00077, 0.04 0.00696 / 0.00217, 0.06 0.0305 / 0.0062 (baseline
0.00036 / 0.00033). Decode 2k 44.7 -> 44.7 / 45.1 / 45.7, 245k ~29.4 -> 29.2-29.8: no gain - all experts of a layer run
in one parallel dispatch, the critical path is not the dropped low-weight expert. Report 5's "< 0.001 KL" does not
hold here. Option kept (off).

## 2026-10-20 MiniCPM-V 4.7 35B-A3B EXL3 3.74 bpw runs (mul1, mixed widths)

**Gains: a second model family (Qwen3.6 fine-tune, mul1 codebook, 3-8 bit EXL3) served end to end: decode 34 t/s
short (Qwen3.6 EXL3 4.0: 44), prefill 319 t/s at 20k; API suite 44/45.**

Conversion (exl3_to_gguf.py, from the HDD copy): 1475 tensors, 16.27 GiB; types exl3_3 54, exl3_4 60, exl3_5 19,
exl3_6 47, exl3_7 76, exl3_8 115 tensors; tokenizer 248,192 ids (67 MiniCPM special tokens, padding to the head).
First greedy answers correct. Decode at 32k / 128k context: 34 t/s (GPU 28 ms/token: the 6-8 bit dense layers and the
8-bit head cost more than Qwen3.6's 4/6-bit ones; the 8-bit head 3.9 ms vs 2.2).
262k context: VRAM over-commit (dense weights 1.87 GB vs 1.3, KV 2.72 GB, compute buffer 2.66 GB at ubatch 4096) put
weight memory outside VRAM (8-bit head 45 ms/token over PCIe) and left no expert cache; with the cache at zero JDE
sent every miss to the CPU (0.33 t/s, scalar path). Fixes: minimal 8-slot cache keeps misses zero-copy on the GPU;
JI_UBATCH=2048 halves the compute buffer (1.41 GB) -> 482-slot cache, 34 t/s at 262k context.
API suite: chat_stream_thinking fails - for "Is 91 prime? yes/no" the model answers "No" inside the opened think
block and stops (model behaviour; reasoning works on real questions).

## 2026-10-20 Per-layer decode budgets (JI_SP_LBUDGET)

Late attention layers (15..39, sharp attention in the dump analysis) with a smaller page budget:

| config | 100k t/s | 245k t/s | decode KL 99k / 130k / 245k |
|---|---|---|---|
| default (16384 everywhere) | 34.5 | 29.9 | 0.00036 / 0.00033 / 0.00029 |
| late layers 8192 | 35.5 | 30.4 | 0.00026 / 0.00041 / 0.00033 |
| dense 3,7,11 | 32.4 | 26.0 | 0.00004 / 0.00006 / 0.00004 |
| dense 3,7,11 + late 8192 | 33.8 | 26.2 | - / 0.00020 / 0.00020 |
| dense 3,7,11 + late 4096 | - | - | 0.00106 / 0.00349 / 0.00200 |

Selected-page attention is not where decode time goes: halving the late budget gives +2-3 % at unchanged mean KL
(within noise), and does not pay for dense early layers. Late 8192 is a default candidate pending the 64k row.
Profile (short context, JI_PROFILE_GPU, serialised): the dense EXL3 mat-vecs are ~8-10 ms of ~22 ms/token
(8192x2048 84 us vs a 42 us memory floor; codebook-bound per the 2026-10-19 probes) - the main short-context lever.
Next: dense projections in a bandwidth-bound mainline format (hybrid GGUF, tools/swap_tensors.py).

## 2026-10-20 Dense mainline-format hybrid, late budgets at 64k, prefill op split: rejected / deprioritised

- Hybrid GGUF (tools/swap_tensors.py): EXL3 experts + head, dense projections from the old mainline Qwen3.6 GGUF
  (142 IQ4_NL, 94 Q8_0, 10 Q5_K, 4 Q6_K): decode 2k 44.6 -> 42.4, 245k 29.9 -> ~27.5 t/s (dense 1.49 vs 1.30 GB:
  the VRAM expert cache dropped to 0, elastic slots only; the plain mv path has no fused prep/post); KL 0.0120 (130k) /
  0.0129 (99k) vs 0.00033 / 0.00036. Rejected - mainline quants without imatrix are far below EXL3 4.0.
- Late-layer budget 8192: 64k row KL 0.00034 -> 0.00075. Not a default.
- srvpf "0 tokens": the model predicts end-of-text after the docs prompt's last line (benign).
- Prefill op split (7.2k, perf logger): dense EXL3 GEMM 43 % (3.8-3.96 TFLOPS), MoE 42 % (~2.9), flash attention
  10 %. Two-height MoE tiles deprioritised: padded columns are already skipped; the thin-expert cost is the
  per-workgroup trellis decode. Clustered LM head not applicable to the EXL3 head (128-row output Hadamard blocks).

## 2026-10-20 Second RX 580: bring-up, dual-stage JDE decode (engine c16539623)

Hardware: card 2 sits in the chipset slot, PCIe 2.0 x4: xferbench (lab/xferbench) 1.27 GB/s host->VRAM, 1.07 back
(x16 card: 11.3 / 6.8). RADV enumerates the x4 card first (Vulkan0); single-card mode is pinned to Vulkan1
(serve.sh JI_GPU=1, kl.sh, restart.sh).

- Exposing both devices dropped prefill 453 -> 25 t/s: ggml-vulkan's multi-device pinned-memory import path fell
  back to staged copies. GGML_VK_HOST_DEVICE=N allocates pinned host memory through device N only (eb978fdbd).
- llama.cpp pipeline parallelism with host experts: every pipeline copy duplicates the offloaded expert weights
  (45 GB). Not usable; dual prefill needs its own double-buffered expert staging (open-paths #21).
- JDE dual stage: stage 1 = layers 0..S-1 on the x16 card (expert cache + zero-copy misses as before), stage 2 =
  layers S..39 + head on the x4 card with all their experts resident. Hidden state via host once per token.
- Bug 1: stage 1 carved elastic expert slots from every layer's KV buffer, incl. the x4 card's -> layer range.
- Bug 2 (the big one): the x4 card's compute buffer was 6.7 GB at ub 4096 / 3.4 GB at ub 2048 (262k) and evicted
  its weights to GTT: stage 2 ran at 530 ms/token (LM head 299 ms = 380 MB at x4 speed). Cause: the chunked
  prefill FA takes 256-row views of the KQ mask; the x16 card reads the host mask in place, the x4 card cannot, and
  the sched made one full-size copy per view (8 chunks x 3 layers, all alive at once). Fix in ggml-backend: copy the
  viewed input once, view the copy -> 1.3 GB at ub 2048. JI_RESERVE_DEBUG lists the big reserve tensors.
- Sparse metadata per stage (stage 2: 96 MiB instead of 321).

Same build, 262k context, server (`q13.sh`/`q14.sh`): decode short / 100k / 245k (saved KV), prefill 80k prompt:

| config | short | 100k | 245k | prefill 80k |
|---|---|---|---|---|
| single (x16 card) | 48.5 | 33.8 | 30.0 | 347.8 |
| JI_DUAL=31 ub 1024 | 48.8 | 36.9 | 33.6 | 278.8 |
| JI_DUAL=30 ub 1024 | 49.8 | 36.8 | 34.2 | 278.5 |
| JI_DUAL=31 ub 1536 | 49.3 | 38.7 | 33.7 | 297.1 |
| JI_DUAL=31 ub 2048 | 49.3 | 37.1 | 34.1 | 324.8 |
| **JI_DUAL=32 ub 4096** | **49.8** | **36.3** | **33.0** | **349.1** |
| JI_DUAL=31 ub 4096 | over-commits (1.8 t/s; 226 t/s prefill) | | | |

Decode KL (130k, L_code r1) single = dual31 ub2048 = dual31 ub4096 = 0.00033: the dual path is bit-identical.
Why the gain is modest: the cards take turns within a token (no overlap), short-context decode is not miss-bound
(single card already caches most hits), and prefill is compute-bound (~9 s GPU per 4096-token ubatch vs ~1.5 s
of expert upload) - card 2 only removes its layers' upload. Dual decode gains come from attention/KV and dense
work of 8-9 layers running from a card whose VRAM is not shared with an expert cache.
Next for dual: prefill pipelining across the cards (~1.3x ceiling with card 2 resident-only), decode expert-server
(hot experts of all layers on card 2, per-layer parallel), MTP verify on two cards.
Prefill from scratch (fresh JR_STATE_DIR, L_code r1 130k): KL 0.00059 single = 0.00059 dual32 ub4096 (identical
p99/max/top1) - the dual prefill is exact too. Open question (not dual): the saved prefix state used by kl.sh gives
0.00033; find which prefill config produced it (dense tail / raw basis / budget) before trusting either number.

## 2026-10-20 (later) Cross-GPU sync on gfx8 + tensor-parallel MoE (JI_TP, engine efad9ea7e)

**Gains (JI_TP=1 vs single card, same build, 262k ctx): decode short 46.9 -> 49.9 (+6 %), 100k 33.8 -> 41.0 (+21 %),
245k 30.0 -> 37.9 t/s (+26 %); prefill unchanged.** Better than the layer split (JI_DUAL=32: 49.8 / 36.3 / 33.0).

Sync primitives measured (lab/xsync: xsync.cpp, diag.cpp, evq.cpp, chain.cpp):
- Shader polling of host memory (imported or host-visible, coherent/volatile/atomics/memory model) never sees an
  external write while the dispatch runs; GPU writes become visible at dispatch end. Atomics on system memory over
  the chipset slot deadlock (no PCIe AtomicOps). Fine-grained GPU spin sync is out.
- Submit + fence round trip ~100 us per device (73 us of it CPU-side vkQueueSubmit): 80 syncs per token would
  cost 14 ms. JDE's single submit per token is right.
- CP waits on host-set VkEvents + VK_AMD_buffer_marker flags + a CPU relay thread, everything pre-submitted:
  **13.6 us per two-GPU sync step**, 23.6 us with an 8 KB exchange spread over workgroups. Only on graphics
  queues: on compute queues the CP never sees vkSetEvent (hang, GPU reset; once the x16 card's reset failed and
  the box rebooted). Data exchange: one pinned host allocation imported into both devices, visible across
  dispatch boundaries; each writer owns its cache lines (a shared line gave stale reads).
- Dispatch + barrier chain: ~2.5 us per step (barriers are not the decode bottleneck; the EXL3 trellis ALU is,
  ~200 G weights/s for ~3 G weights per token).

JI_TP design: model single-card on the x16 GPU; a helper engine (own VkDevice + graphics queue on the x4 GPU)
holds a static hot-expert set (routing counts ref/tp/tp_hot_qwen36_exl3.bin from 6 prompts x 400 tokens: top
20 % of experts = 55 % of routing, top 49 % = 86 %). Per layer: tp_split moves the helper's selections (+ x) to the
exchange area, marker -> relay -> helper event; helper: its experts + weighted sum -> exchange, marker -> relay ->
primary event before the combine (moe_host slot). Primary TP tokens run on ggml's graphics-family queue.
Short-context share sweep (JI_TP_FRAC 0.3-1.0): all ~50 t/s - the helper's MoE chain has ~50 us fixed latency
per layer, so moving work shifts the stall (JI_TP_PROF: primary stalled 0.6-2.6 ms/token). At depth the
primary's expert cache shrinks (KV takes VRAM) and misses dominate the MoE (single-card 245k profile: MoE
mat-vecs 11.3 of 32.4 ms, attention 5.6, dense ~10, head 2.2): the full helper wins there.
KL L_code r1 (130k, 1-token steps): single 0.00036, TP 0.00035 (summation order differs, not bit-exact).
Next on two GPUs: the helper is idle ~15 ms of every token - attention KV-head split at depth (helper owns kv head
1 + a head-1 KV mirror; 2 syncs per attention layer), dense head split (GDN state mirror), prefill pipelining.

## 2026-10-21 Full tensor-parallel decode on two GPUs (JI_TP=2, engine c4f8da270)

**Gains vs single card (same build, 262k ctx, helper expert share 0.7): decode short 46.3 -> 58.0 t/s (+25 %),
100k 33.8 -> 43.8 (+30 %), 245k 30.0 -> 40.0 (+33 %). Prefill unchanged. KL 130k (1-token steps) 0.00036 = single.**

Why TP-MoE alone (JI_TP=1) plateaued at ~50: decode is a latency chain (~500 dependent dispatches per token) of
ALU-bound trellis mat-vecs (~200 G weights/s, ~3 G weights per token); barriers are cheap (2.5 us, lab/xsync
chain.cpp). Moving experts to the helper only helps by the amount of the primary's work it removes in parallel,
and the helper's MoE chain has ~50 us fixed latency per layer. So both GPUs must split *every* layer.

JI_TP=2: rank r runs Gated DeltaNet key heads [8r, 8r+8) (+ value heads h, h+16): wqkv rows in four 1024-row blocks,
z in two, gdn_fused with a key-head offset, ssm_out over its K quarters {r, r+2}; shared expert rows [256r, +256)
and its down projection's K half; routed experts by ownership; LM head vocab half (helper writes to host memory).
Attention stays on the primary (the helper adds a zero partial). All-reduce after the mixer and after the MoE:
partial -> exchange, marker, relay, CP event wait, x += p0 + p1 (rank-0 partial first on both: identical x).
x3_mv/x3_post gained K-part masks and row subsets (no new kernels); JI_TP_TEST=1 verifies them (rows exact,
K parts <= 5e-7). The helper holds a 1 GB copy of the dense weights it needs + a 63 MB recurrent-state mirror; its
heads' state is copied back (state out, ~300 ms, not yet optimised) before graph ubatches / state saves / non-TP
steps and copied in (~80 ms) before the first TP step.

Steps (short decode, 256 tokens): first TP2 52.2 -> host-side fixes (embedding built on the host instead of
read back over the BAR, helper submit on the relay thread, cache fills on the graphics queue) 53.9 -> LM head
split 58.0. JI_TP_PROF (frac 0.85): primary work 10.2 ms + 4.35 in syncs, helper 10.7 + 4.0, sync floor 1.57 ms
per token (~19 us per sync point); the rest of the waiting is per-segment imbalance - attention layers (primary
only) and the per-layer expert split.
Next: attention kv-head split on the helper (removes the attention imbalance; at depth also halves attention),
faster state out, then MTP / speculation on the cheaper verify.

## 2026-10-21 (later) TP2 attention split, state-sync latency, partial state reads (engine 42fcac098..9e4073234)

**Gains: JI_TP=2 short 57.7 -> 58.8, 100k 43.8 -> 45.2, 245k 40.0 -> 41.1 t/s; per-request TP2 state sync
280 + 80 ms -> 38 + 45 ms; context-checkpoint restores no longer rebuild sparse page metadata (100-300 ms per
request at long context, single card too).** Single card on the same build: 46.3 / 33.8 / 30.0.

- Attention kv-head split: each GPU runs 8 query heads + 1 kv head in a "local mode" (n_head 8 / n_head_kv 1,
  heads packed at 0) through the unchanged kernels (jde_attn_core extracted); the helper keeps a q8_0 mirror of
  kv head 1 (1.36 GiB at 262k), its own page metadata, and a mask copy updated by the changed range. Cells written
  outside TP steps are copied in (dirty range), head-1 cells written by TP steps go back with the state. Gain is
  small: per-head attention at depth is latency-bound (256 chunk workgroups either way).
- Hang found while adding it: the helper's recorded token programs took descriptor sets from the pool
  jde_submit_wait resets; the metadata rebuild's dispatches overwrote them. Recorded programs now own a pool.
- Partial state reads (LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, the server's context checkpoints - every request on
  this hybrid model) keep the KV page metadata and the TP2 mirror (jde_state_partial).
- State copies batched through both staging buffers; the helper staging buffer is host-cached (CPU reads from
  write-combined memory were 240 ms of the 280).
- Expert share: 0.5 balances short context (primary/helper work 11.5/11.3 ms) but depth wants 0.7 (the primary's
  cache shrinks under the KV): 100k 42.3 vs 45.2, 245k 36.5 vs 41.1. Default 0.7.
- JI_TP_PROF waits by sync type (frac 0.5, short): after mixer ~0.96 ms per GPU (= sync floor), after MoE
  1.4-1.7 ms - per-layer routing variance (4 +- 1.4 experts per side). Next: per-layer dynamic balance with experts
  held by both GPUs, cheaper syncs, MTP on the TP verify.

## 2026-10-21 (later) TP2 per-token expert balance, multi-token TP2, n-gram speculation on two GPUs (engine bcc21bd57, 85166e33c)

**Gains: short decode 58.1 -> 59.1 t/s (expert balance); code rewrite (copy-heavy, 600 tokens, greedy) single 32.6 ->
TP2 43.9 -> TP2 + n-gram spec 50.6 t/s; story single 38.4 -> TP2 52.1 (spec 50.4).**

- Expert balance: ownership map 0 / 1 / 2 (2 = on both GPUs: the hottest 25 % of the routing traffic, 566 experts,
  pinned in the primary's cache); tp_split gives each shared selection to the side with fewer selections so far
  (same rule on both GPUs). After-MoE waits: primary 2.78 -> 1.03 ms per token.
- TP2 for 1-4 token steps (prompt tails, verify windows): R = nt everywhere, [nt][n_embd] partials, one program per
  token count. Two bugs on the way: the attention partials were indexed with a hard 16 heads per token in
  attn_merge (wrong for tokens >= 1 in the local 8-head mode) and the helper's mask mirror advanced on steps the
  helper did not run. Mask rows now at a fixed stride (prm[30] = max_kv). Outputs identical to the single card.
- N-gram speculation (--spec-type ngram-map-k, n 4 / m 3, 3 drafts; via restart.sh SARGS): acceptance 90 % on the
  rewrite, 2.39 tokens per step, but a 4-token verify costs ~2x a 1-token step (routed experts scale with tokens:
  32 selection rows, each decoded separately) -> +15 %. Novel text: -3 % (per-step recurrent snapshots).
Where 80 t/s could come from: MTP drafts inside JDE (~1.5 ms per draft instead of 9 ms through the ggml graph):
3 drafts at 0.78 / 0.57 / 0.44 acceptance -> ~2.8 tokens per ~38 ms step -> ~73 t/s; grouped expert decode in
verify (each distinct expert decoded once per step) would add ~10 %.

## 2026-10-21 (later) TP2 micro-steps: direct partial writes (no), router-fused split, TP profile (engine acfba74b7, e9676da3f)

**Gains: none measurable (59 t/s short).**
- JI_TP_DIRECT=1 (kernels write their partials straight into the exchange area, no copy): GPU 15.1 -> 16.0 ms per
  token - writes to host memory from the mat-vec / combine kernels cost more than the VRAM write + one copy. Off.
- Ownership split folded into router_fused (TP2): no tp_split dispatch + barrier per layer; neutral, kept.
- JI_TP_GPROF primary profile (short): syncs 2.2 ms/token (80 x 27.6 us incl. waits), DeltaNet row blocks 1.8 (at
  ALU rate), routed experts 2.5, LM head half 1.1, router 0.9, x3 input stages 0.85, attn_gqa 0.55 (55 us per
  layer at short context - a candidate for a small-n_kv kernel), combine 0.4, shared-expert gate 0.3.
Decode is at its structural plateau (~60 t/s): per-card trellis ALU + sync floor + per-layer imbalance. Remaining
big levers: MTP drafts inside JDE (with grouped expert decode in verify), prefill GEMM efficiency.

## 2026-10-21 (later) JI_TP=2 validation: API suite 45/45, 117k needle; draft-mtp incompatible

- `scripts/api_suite.py` against JI_TP=2: 45/45 (tools, streaming, thinking, prompt caching across turns, Anthropic /
  Responses, disconnects, queued requests); long-output coherence at 56.2 t/s.
- `--long` needle (117,044 prompt tokens): pass, prefill 341 t/s, the head-1 KV mirror copy of 117k cells 0.78 s at
  the first decode step, state in / out 35-50 ms per request.
- `--spec-type draft-mtp` with the MTP sidecar on the x4 card (-devd Vulkan0): the first TP2 sync is lost (primary
  never marked), the x4 card resets (clean). Not combinable yet; ngram specs work. README / serve.sh note it.

## 2026-10-21 (later) MTP head on JDE (helper GPU), verify-cost scaling, grouped expert decode (engine edf3f579f)

**Gains: none in decode / prefill (MTP lossless and correct, but slower than no speculation on this hardware).**

- Verify cost, TP2 (`examples/junk-vbench`, real text, warm cache, 262k context): 1 / 2 / 3 / 4-token step
  17.3 / 25.3 / 33.1 / 49.8 ms. Per-step routed selections on the x16 card (JI_PROFILE): 190 / 384 / 508 / 701,
  distinct experts 190 / 312 / 372 / 488, PCIe misses 12-44 / 24-57 / 56 / 108 (the 4-token cliff = misses).
  Kernel profile: x3_mv_moe 37 / 64 / 102 / 211 us per dispatch; dense EXL3 NT=4 also has an LDS-occupancy cliff
  (256x2048: 14.6 -> 37.3 us at nt 3 -> 4; 6 KB LDS per wave -> 2 workgroups per CU).
- MTP draft head on the JI_TP=2 helper GPU (`jde_mtp_init`): the draft-mtp context's sidecar (IQ4_NL block, Q6_K head,
  Q8_0 embedding) loads on the CPU (`-md mtp-IQ4_NL.v2.gguf --spec-type draft-mtp -ngld 0 -devd none
  --no-spec-draft-backend-sampling`; backend samplers force the graph path); JDE copies it to the helper's VRAM
  (560 MiB with a 64k-row head + 272 MiB K/V at 262k; JI_TP_MARGIN_MIB=1900 -> helper hot set 61 % of the traffic
  instead of 70 %). 1.8 ms per draft call (head 0.66 ms) vs ~15 ms on the CPU graph path; the x16 card has no room
  (289 MiB free at 262k: the ggml draft on Vulkan1 over-committed it -> "Not enough memory for command submission").
  Acceptance (mtpacc, draft 3, t=0): 0.630 full vocabulary = the ggml reference 0.644; 0.605 with the first 64k ids
  (ids < 64k cover 96.5-98 % of the corpora, < 96k 99.2-99.9 %). Outputs lossless (chat3 md5s identical).
- End to end (TP2, spectest, t/s): story none 51.9 / MTP n1 49.4 / n2 42.9; code rewrite 44.2 / 44.0 / 42.0. A step
  with 2 drafts = drafts 6.2 ms (2 MTP calls + top-k over 248k on the CPU) + 3-token verify ~40 ms (with rollback
  snapshots) + catch-up / sampling ~4 ms -> even at 98 % acceptance ~56 t/s. Each extra verified token costs 8-10 ms:
  EXL3 routed experts are ALU-bound per row (no amortisation across tokens) and the cold experts stream over PCIe.
- Grouped expert decode (JI_X3_GROUP=2..4: an expert's first row decodes for up to G tokens): not faster (2-token
  24.7 -> 26.0, 3-token 32.5 -> 34.0 ms at G=2; G accumulator rows cost LDS occupancy, duplicates are 16-30 %). Off.
- Fix: JDE's nextn rows are now the post-output_norm hidden state (the graph's h_nextn) - JDE gave the MTP head the
  pre-norm x before.
Conclusion: speculation pays here only with free drafts (n-gram on copy-heavy text). MTP stays opt-in.

## 2026-10-21 (later) Gemini reports 4-8 triage (prefill focus)

Reports (user): cYlJ1Pfl19JC (dual-Polaris levers), VM59nTdOUtXz (speculative / multi-token verify, PL),
LrtXGEcCJSco (retraining-free PCIe-constrained MoE), mp7TzLpksJMF (bandwidth compression), 4DUYwJ2vt7Vy (EXL3 on
GCN4, PL). Prefill-relevant, against our state (7.2k: dense EXL3 GEMM 43 % at 3.8-3.96 TFLOPS, MoE 42 % at ~2.9,
FA 10 %; at depth FA-bound):
- Already done: GQA-folded FA with Q in SGPRs (ji_fa2, 84 VGPRs); decode-once EXL3 tiles in the GEMMs; sparse
  prefill with the variance tail (their KL figures predate it); skipped padded MoE columns.
- Not applicable: Psumbook / CodeGEMM / T-MAC LUT GEMM (EXL3 weights are 16-bit trellis windows through a hash, no
  small sub-vector codebooks; prefill amortises the decode over the token tile anyway).
- Open, modest: MoE GEMM narrow-tile path for thin experts (+~10 % if 2.9 -> 3.9 TFLOPS; our earlier read: the thin-
  expert cost is per-workgroup decode), GDN chunk size 16/32 (small share), VGPR <= 128 audit of the GEMMs.
- Dual-card prefill (their #9): layer pipeline is VRAM-limited on card 2 (~8 layers resident: <= 1.25x); a
  functional split (card 2 = dense / GDN / attention projections, card 1 = routed experts, two micro-batches in
  flight) is the only path near 2x; FA kv-head split for long prompts (card 2 already mirrors kv head 1 in TP2).
- Lossy / decode-only items (dynamic top-p routing, low-bit cold experts, timing straps, verify expert budgets):
  not prefill; the lossy ones are KL-gated.

## 2026-10-21 (night) Two-card JDE prefill, sparse prefill on JDE, TP2 sparse decode fix, hardware

Hardware: the x4 card fell off the PCIe bus twice under JDE TP prefill ("device lost from bus", recovery -19; the
first time it looked like a prefill deadlock). Both cards share one daisy-chained PCIe power cable. Card 2 alone at
145 W (gpustress, 5 min) and both cards under synthetic load at 100 / 120 / 135 / 145 W held; the real workload at
100 W held where it failed at 145 W. Until the new PSU / case: power cap 100 W per card
(/etc/udev/rules.d/31-amdgpu-powercap.rules). The model SSD (mSATA in a SATA adapter) logs 10B8B / PHYRdyChg link
resets independent of GPU load. All numbers below at 100 W.
Method: prefill at depth from saved KV (`~/jt/dpb.py N`: restore depth_N.bin, prefill ~5-7.5k new tokens on top) -
seconds instead of 4-15 min per point.

- JDE TP prefill (JI_TP=2 JI_JPF=1, JI_TP_MARGIN_MIB=1500): both cards prefill on the TP2 decode layout (design:
  tp-prefill-design.md, milestones 1-3), two micro-batches, f16 partials through the host exchange, relay thread.
- Sparse prefill attention on JDE (pf_fa_sparse = ji_fa2_sparse with the causal mask from positions, pf_pmeta), same
  settings as the graph path (JI_SPARSE_PF 8192 from JI_SPF_MIN 32768, dense prompt tail on layers 3, 7, 11).
- Prefill t/s, restored context + tail (graph TP2 -> JDE TP dense attention -> JDE TP sparse):
  2k + 5.5k 343 -> 393; 72k + 7.5k 240 -> 235 -> 304; 247k + 7.5k 128 -> 98 -> 213.
- KL (L_code, 1025-token tail as ONE prefill batch, JR_TAIL_B=1025): 130k graph 0.00050 / JDE TP 0.00050;
  261k 0.00045 / 0.00065 (f16 exchange partials).
- Bug found on the way, TP2 sparse decode (default config, contexts >= 32k): the primary's kv head 0 read page
  metadata at slot = layer while rebuilds write layer * 2 + head -> KL 0.041 (130k) / 0.232 (261k), now 0.00037 /
  0.00033 (single card 0.00033 / 0.00028). Engine 63d680059. All TP2 depth quality numbers before it are invalid.
- Decode with JI_TP_MARGIN_MIB=1500 (helper prefill scratch): short 55.9 -> 54.3 t/s; 245k decode warms over runs
  (28.7 -> 32.9 -> 35.8 on one context), so depth decode comparisons need equal warm-up.
- Tools: JI_PF_GPROF=1 (prefill GPU ms per kernel label per rank; waits per segment, breakdown only), JI_TP_HDENSE=1
  (debug: helper head dense), ~/jt/dpb.py, ~/jt/gpustress, dualstress.sh, pwrlog.sh; kl.sh KLX=extra args.
- Failure handling: a failing rank stops submitting (J.fail), the relay polls and the main thread signals the peers'
  remaining semaphore values so the other rank cannot hang.

## 2026-10-22 JDE TP prefill: exchange queue, profiling method, GEMM fixes (100 W cap)

Method: JI_PF_TL=1 (per-GPU segment timeline: busy vs waiting) + JI_PF_REP=prefix:N (a kernel family's cost = busy
increase per extra issue; data unchanged). JI_PF_GPROF per-label times are misattributed (RADV writes a compute-
queue timestamp at issue) and skip experiments (JI_PF_SKIP, JI_PF_NOUP) corrupt activations -> routing -> MoE cost.
- Exchange copies moved to their own queue (dd453f436): rank 1 idle 0.8-1.1 s -> ~0.1 s; 518 -> 554 t/s per
  4096-token ubatch. 2 copy workgroups instead of 8: 584 (the copy waves held CUs while waiting on the x4 link).
- Per-family cost (4096 tokens, rank 0 / rank 1, ms): dense GEMM ~2620 / 2500, routed GEMM 3131 / 1526, FA 434 / 536,
  GDN rule ~300 (noisy), f32mm 320 / 462 (GDN alpha/beta ~170 each), prep ~200, the rest small.
- Rank 0's routed GEMM: 277 tiles per layer and micro-batch vs 122 on rank 1 at the same cost per tile (~140 us):
  cold experts get ~26 tokens per micro-batch and a 64-column tile multiplies all 64. Half-width tiles for <= 32-row
  remainders: rank 0 MoE 3.84 -> 3.48 s. Whole experts per rank (instead of per-token splitting): neutral.
- BK16 on the JDE GEMM variants (the graph path had it, JDE's copies not): 584 -> 624 t/s.
- Prefill t/s (restored context + tail): 2k 406 -> 462, 72k 304 -> 373, 247k 213 -> 229 (1beee0ea2).
- Rejected / neutral: CP-DMA exchange copies (CU-free, but 4 KiB packets too slow over x4; on rank 0 they slowed its
  compute queue), helper prefill on a compute queue, rank-0 weighting of the expert split.
- Both GPUs now ~6.3-6.4 s busy per 4096 tokens; next: GEMM efficiency (dense ~40 % of peak), f32mm small-N kernels,
  rank 0's upload stream (~0.4 s), cold-expert tiles.
- Depth (032e3f39d): pf_fa_sparse is ~half of a 245k ubatch (4.0 / 4.9 s per rank for 3443 tokens); page scoring
  ~1/3 of it (JI_PF_SPNS timing variant). 128-key statistics pages: steady 245k ubatch 279 -> 350 t/s; KL at 261k
  unchanged in the production setting, +0.00014 with every layer sparse. Restored + tail: 2k 465, 72k 394, 247k 265.
- Single vs TP (4096-token ubatch, family costs, single / TP both ranks, s): dense 4.14 / ~4.8, routed 3.47 / ~4.2,
  f32mm 0.29 / 0.78, FA 0.82 / 0.97, GDN rule 0.51 / ~0.6, prep 0.27 / 0.40. Single JDE 428 t/s (graph 342 at 2k);
  TP 624 = 1.46x. TP at single-card efficiency would be ~856 t/s at 100 W (~960 at 145 W).
- pf_f32mv (wave per token for N <= 64): slower (register array), opt-in only.
- Steady depth (042aa9abd notes): 247k ubatch after the first 467 t/s at 100 W; the first ubatch after a slot
  restore copies the helper's kv head (~1.3 GB over x4), so dpb's depth figures understate steady prefill.
- Neutral / rejected at 100 W: T128 GEMM variants (slower), BN128 + BK16 dense (=), sparse budget 4096 (no gain:
  the exact part is not dominant), tail estimate off (+6 %, not taken), CP DMA exchange on the helper's graphics
  queue (=), per-segment host barrier (=), pf_f32mv (slower). Busy times under the cap depend on idle gaps (clock
  boost), so only wall time compares configurations.
- c2b093a56: pipelines at init (first prompt 466 -> 501), helper scratch sized once, rank 0 uploads only its
  experts (pf_copyr run table; neutral). Warm restored 2k + 5.5k: 557 t/s.
- 9d90eb2e5: q / k / v and shared gate / up stage inputs together (neutral). Fresh prompts (cache off, 100 W): 7.2k
  507 first / 555 warm, 33k 466 average. TP GEMM split per rank (dense / routed, s): 2.54 / 2.59 and 2.10 / 1.49;
  single card 4.14 / 3.47. pf_mm stats (RADV_DEBUG=shaderstats): 64 VGPRs, 13 KiB LDS, 4 waves per SIMD, no spill.

## 2026-10-22 (later) Prefill optimisation list, items 1-2 (100 W)

Speeds at the start of the list: decode ~53 t/s short (55 without the prefill margin), prefill fresh 7.2k 557 warm /
507 first after start, steady 247k ubatch 467.
1. Warm-up (2b8c079e5): with JI_JPF, a 512-token batch then a decode step (the first prompt's tail decode set up
   JDE's elastic expert slots, ~1 s). First prompt after start 507 -> 545 t/s (warm 557). Full-ubatch warm-up: no gain.
2. Dense prompt tail (last 4096 positions dense on layers 3 / 7 / 11): a sparse tail budget (JI_SPF_TAIL_BUDGET
   16k / 32k, local 16k) makes the tail ubatch at 247k 18.5 -> 9.7 / 10.4 s but KL at 261k 0.00069 -> 0.00146 /
   0.00149 (~ no dense tail, 0.00159). Kept dense. JI_SPF_DENSE_TAIL=2048 not measurable with kl.sh (scored positions
   follow the saved prefix and are always in the tail); not changed.
3. (Moved after 4.) 4. Rank 1 diagnosis: skipping any single mixer kernel on the helper (JI_TP_SKIP) cut its mixer
   busy by 140-970 ms - non-additive (sum > the whole overhead): no hidden work; clocks equal (1182 vs 1195 MHz, both
   ~98 W). Under the cap both GPUs are power-bound and wall time follows total work: the expert split (TPW 0.3 / 0.5 /
   all shared experts to rank 1) is neutral. Consequence: balance / latency items (3 for prefill, 5, 7) are not
   expected to pay at 100 W; work reduction (9, 10) is.
5. Split exchange queues (JI_PF_X2): 614 vs 628, opt-in. 6. Router once: rejected (a sync point inside the MoE);
   small-N f32 kernels: latency, not energy - not pursued; NMB=4 fixed (command buffers): 586 t/s, 3: 598, 2 stays.
7. 8-bit exchange (JI_PF_XQ8): 650 vs 629 t/s but KL 0.00048 -> 0.00156 at 130k: opt-in. 8. Upload ceiling
   (JI_PF_NOUP): 660 t/s (~5 %); CP-DMA uploads (JI_PF_UDMA) 560 t/s, and AUX-1 hard-locked in its third run (silent,
   during model load; journal shows only the SSD link at 1.5 Gbps / 155 exceptions): do not use. Cache reuse for
   uploads not built (multi-buffer GEMM for <= 5 %).
3. Helper hot slots (simulated by JI_TP_MARGIN_MIB 1500 -> 900: 2535 -> 2935 slots): decode 2k 54 / 54.9, 64k 44.3 /
   44.0 - no gain; the half-dense refactor is not worth its risk.
9. Sparse selection shared by 8 / 16 positions (JI_PF_SPP): 245k steady 477 -> 481 / 486 t/s, all-sparse KL 0.0016
   -> 0.0021 / 0.0038: rejected. Budget halving and tail-estimate removal earlier were also small: no redesign.
10. GEMM: 256-thread 8-row variants (JI_PF_DV=r8 8x8 / r864 8x4, JI_PF_IV=r8): 628 / 570 / 599 vs 631. Power test:
   80 W 488 t/s vs 100 W 629 (clocks 940-1019 vs 1106-1198 MHz): prefill is power-bound at the cap - time follows the
   energy of the f32 FMAs; latency / occupancy / LDS rewrites cannot pay at 100 W (only at 145 W).
Final state of the list (100 W): prefill fresh 7.2k 557 warm / 545 first, 4096-token ubatch ~630, steady 247k 477,
decode 53-54 t/s short, 44 at 64k.

## 2026-10-23 Tasks 1-2: prompt-cache reuse, chat-turn latency, small prompts (100 W)

Measured with ~/jt/turns.py (2k-token system prompt, short user turns, streamed; TTFT and server timings).
- Prefix reuse worked (only the new ~20 tokens processed), but a turn took 2.2 s: the new tokens are a small batch
  (graph path: MoE on the CPU below 32 tokens) and the server makes 2-3 context checkpoints per turn (recurrent state,
  63 MiB each: TP2 state out 45 ms + 155 ms state read + 37 ms state in + 58 ms buffer allocation).
- Now ~0.45 s per turn (engine 96cb9b42d): small batches (<= 64) as 4-token JDE decode steps; checkpoints ~75 ms
  (batched state read, TP2 split kept after a read, huge-page buffers, recycled buffers), 2 per turn instead of 3.
- Every prompt also ended with a small chunk (the server's checkpoint 4 tokens before the end): 18 tokens took 1.6 s
  on the graph path; now ~50 ms.
- Medium batches (the previous reply re-processed each turn - Qwen's template strips reasoning, and the model may
  emit an empty think block that the history drops): TP prefill from 64 tokens; ~1.4 s floor at 64-200 tokens
  (rank 0 MoE: cold experts with 1-2 tokens each still cost a full tile) - the few-token EXL3 kernel (task 3) is the fix.
- Quality fixes found on the way: the helper's sparse-decode metadata after a TP prefill was rebuilt with stale
  params / mask (first sparse decode steps after a prefill wrong); the f16 all-reduce cost KL 0.0009 vs 0.0004 single
  GPU - 24-bit exchange (default) 0.00023 for -3.5 % prefill.
- First reply after a long prompt decodes slower for ~1.5 s (54 -> 23 ms per token): not the expert cache (seeded now,
  3643 slots at warm-up) - decode runs near the 100 W cap too; most likely the power limiter's averaging. Left.
- smoke_api.py: all pass (chat, tool call, finish_reason, prefix cache 6689 -> 23 new tokens, 12.9 -> 1.0 s).
  api_suite.py needs `pip install openai anthropic` (not installed on AUX1 / CHERRY: download pending permission).

## 2026-10-23 Tasks 3-4: MTP overhead, expert-cache fixes, KVarN-6 start (100 W)

**Gains (TP2, spectest 600-token replies, outputs identical to before):** decode with `JI_CACHE_SWAPS=16`
(new default, was 2): rewrite / story / explain 34.9-38.7 -> 39.8-41.1 / 42.8-48.0 -> 47.4-49.0 / 44.7-45.1 t/s
(fresh server; 245k decode after a restore 24.1 -> 30.6 first reply, 30.9 -> 33.2 steady). MTP (opt-in, see bugs):
38.3 / 47.5 / 44.7 -> 43.9 / 51.8 / 52.7 t/s. Prefill unchanged without MTP (64k +7.5k: 388 t/s).

Correctness fixes (all in production):
- KV clear wiped the seeded KV-tail expert slots (`llama_kv_cache::clear(data)` zeroes the buffers; the bulk seed
  ran right before the warm-up's clears): garbage output at the production margin. `jde_note_kv_clear` + guard
  drop / re-seed; JI_EL_VERIFY 5280 mismatches -> 0.
- `jde_kv_guard` read the KV context's n_kv before the ubatch was applied (uninitialized on the first ubatch of a
  decode call): a garbage 1.4e9 evicted every KV-tail slot (a run at 2/3 speed). Limit = used cells + n_tokens + 4096.
- TP2 relay watchdog 2 s -> 20 s (`JI_TP_TIMEOUT_MS`): the first decode step after a deep restore + prefill took
  > 2 s, which switched tensor parallelism off for the server's lifetime (and computed that token wrong).
- A larger reallocated compute buffer keeps its 1834 elastic slots (re-bound, refilled) instead of retiring them
  (the MTP context's first verify reallocates it: verify misses 3-4x).

MTP work (engine be2f7bf3d, 3153cf5ce): verify cost profile (primary card, 1/2/3-token steps 17.5 / 25.0 / 36.4 ms
GPU; routed experts scale per (token, expert) pair, the TP sync per token); per-step host overhead found and cut
(spec_prof: JI_SPEC_PROF=N): one fused draft decode per step (catch-up deferred), repeated draft row skipped, head
only for the output row, greedy top-10 straight from the logits, no 3 MB sampler clone per step. Draft p-min 0.5
best (n=2 drafts lose). Head: HyperQwen's profiled 40,960-token vocabulary (JI_MTP_VOCAB_FILE,
ref/mtp/hyperqwen_draft_vocab_40960.json, Apache-2.0) + IQ4_NL requant (JI_MTP_HEAD_TYPE) - equal to the first-64k
IQ4_NL head within noise, both beat Q6_K. Prompt catch-up windowed (JI_MTP_PF_WIN=512): MTP prefill cost at 72k
-39 % -> -10 %.

Open bugs (not fixed, documented in serve.sh / open-paths):
1. MTP: after a deep context (64k restore + 7.5k prefill, or a 20k prompt) the next prompts decode as garbage
   (the target's tokens; drafts all rejected). Reproduces with every new MTP option off, so it predates this work;
   without MTP the same sequence is correct. MTP stays opt-in (JI_MTP=1), off in production.
2. Decode ~20-25 % slower after any large prefill (>= ~5k tokens), until the server restarts: story 47-49 -> 34-37
   t/s, with swaps 2 and 16 alike (pre-existing). Diagnosis so far: identical kernel times and cache hit rate
   (96-97 %), but the primary's step starts 3.6-7 ms (vs 1.2-1.6) after its submit - the expert-fill copies queued
   ahead of it on the graphics queue run slower; not PCIe link state, clocks, RAM pressure, CPU threads or GTT.

KVarN-6 (task 4, started): in-place design - 6-bit tile records (128 cells x 2 slices, 52 words/cell + axes in the
same tile block) in the layer's K/V buffers, an exact q8_0 window ring for the newest cells, kvr_seal (window ->
records, the validated KVSIM arithmetic) and attn_hmr (decode attention over records + window, per key, merged by
attn_merge as before). Test harness JI_ATTN_TEST=n JI_ATTN_KVR2=1; not yet run. Remaining: prefill readers
(pf_fa / pf_fa_sparse / pf_pmeta), TP2 mirror source, elastic-slot accounting, state save/restore (q8 rows out,
ingest in), truncation (unseal), KL gate.

## 2026-10-23 (later) Hermes agent: tool calls, crash after a long prompt

- Chat template: Qwen-Sharp (peculiar-ragdoll/Qwen-Sharp-Chat-Templates, not in git - see serve.sh) by default.
  Agent-style test (~/jt/tooltest.py: tools, streaming, tool-result turn): structured tool_calls in all three cases.
- BUG (open, top priority): with Hermes, a 23.7k-token prompt was followed by a TP2 hang > 20 s at the first decode
  step ("TP2 sync 0 lost (primary 0, helper 1)", server fell back to one GPU); the next request aborted the process
  (SIGABRT, no message; apport log; crash log ~/junkinfer/logs/server.20261022-2241.crash.log). Related to the
  post-prefill bugs above (decode slowdown after large prefills, MTP garbage after deep contexts): something left
  behind by a multi-ubatch JDE prefill breaks the next TP decode steps. Mitigation: serve_prod.sh runs a supervisor
  that restarts the server on exit (logs/restarts.log, logs/server.<time>.crash.log).

## 2026-10-23 (morning) Bug fixes: post-prefill slowdown, MTP garbage, TP loss; MTP on by default

**Gains (TP2, 100 W, spectest 600-token replies rewrite / story / explain, Qwen-Sharp template):** no MTP
40.8 / 46.5 / 44.2 -> 43.2 / 49.6 / 48.0 t/s (CPU governor performance); MTP 49.4 / 58.5 / 58.9 t/s (production
default now). After a 17k prompt (no MTP) 33.7 / 36.4 / 34.8 -> 40.2 / 44.4 / 43.8. Prefill unchanged without MTP;
MTP -2-4 % fresh (17k 489 vs ~500; 6.8k 532 vs ~555), -6 % for 7.5k on a 72k context (375 vs 399).

1. Decode slowdown after a large JDE prefill: not GPU state (kernel times equal; not thermal - a 35 s GPU stress made
   decode faster; not VRAM eviction, PCIe link, THP / compaction). The host work between steps slowed (expert cache
   update 1.1 -> 4.4 ms/token, fence->submit +2 ms): CPU cores down-clocked under the schedutil governor (avg 1825 vs
   2628 MHz). Fix: governor performance (scripts/cpu_performance.sh; serve_prod.sh applies it with passwordless
   sudo, else prints how). Also: asynchronous expert fills (own queue, slots published when copied) - neutral to +3 %.
2. MTP garbage after a 64k restore + continuation prefill: compute-buffer expert slots above the prefill scratch got
   corrupted (JI_EL_CB=0 / JI_PF_CBALL=1 cure it; the stray write is not identified). Fix: with nextn rows wanted
   (MTP) the prefill evicts all compute-buffer slots. (Repro needs the restore to match: depth_64000.bin was saved
   with the embedded template - JI_CHAT_TEMPLATE=embedded.)
3. TP hang / crash (Hermes): not reproduced (campaign: 6 x 26k prompts). A lost TP2 step now aborts the process with a
   message (supervisor restarts it) instead of the one-GPU fallback that crashed later.
4. MTP prefill cost: the draft head attended all cells up to the position (~7 ms per call at 72k) -> last 8192 cells
   (offset bindings, JI_MTP_ATT_WIN); prompt catch-up 512 -> 128 rows. Spec process per 4k chunk at 72k 0.9 -> ~0.15 s.
Kernel note: a refcount underflow warning in amdgpu_job_free (amdgpu_cs_ioctl, 2026-10-22 19:25) - driver bug, once.

## 2026-10-23 (afternoon) KVarN state hooks, two box freezes, and corrupt KL references (graph path, 3.75 GiB blocks)

**Gains: none in t/s (correctness work). KVarN-6 decode at q8 speed (short 42.1 / 49.1 / 47.3 vs 43.2 / 49.6 / 48.0;
36k restore 37.6 vs 36.3; 247k prefill 432 t/s, decode 38.0) - quality gate re-run pending on the v2 references.**

1. KVarN-6 rings allocated at engine init, before the expert cache budget (late allocation over-committed VRAM:
   ~1.4 GB evicted to GTT, decode 3-4x slower).
2. KVarN state hooks never matched (llama passes *views* of the cache tensors) and were bound only at the first
   decode: every save / restore / prompt-cache reuse wrote raw q8_0 rows over the records. Fixed (view_src match,
   `jde_kv_attach` right after the memory exists; `JI_KVR_TRACE=1` logs bypassed rows).
3. Two hard freezes of AUX1. dmesg of the second: ring gfx timeout in llama-junk-ref (killed mid-step) -> BACO reset
   -> "atombios stuck ... asic atom init failed" -> GPU Recovery Failed -> box freeze. TP2 GPUs wait in-queue on
   host-set VkEvents; a process that ends mid-step leaves them waiting. Engine 2c61614d3 / e8da5430d: every TP event
   is set before abort() (TP loss), at teardown and from SIGTERM/INT/HUP/ABRT/SEGV/BUS/FPE (jde_signal.cpp). Not yet
   tested by a deliberate kill (a failure costs a power cycle).
4. **The ggml graph path loses the first attention layer's K/V writes** when GGML_VK_SUBALLOCATION_BLOCK_SIZE is
   3.75 GiB (serve.sh / kl.sh / mkrefs.sh since 2026-10-18) and ubatch 4096: after a 36k prompt, layer 3's K and V
   are all-zero for 20480-24576 cells (whole ubatches from ~8k). Not the case with 1 GiB blocks, ubatch 2048, or
   upstream llama.cpp; also with IQ4_NL weights, JI_KV_HM=0, JI_DECODE off, fusion / graph optimiser off, before
   the sched view-copy dedupe. Most likely an offset >= 2 GiB inside the 2.75 GiB compute buffer is mis-addressed by
   some ggml-vulkan op (device limits allow 4 GiB). The JDE prefill (JI_JPF=1, production since 2026-10-21) writes
   KV itself and is clean, so production output since 10-21 is unaffected; graph-path prefill between 10-18 and
   10-21 was degraded on long prompts.
   Consequence: **all v1 KL references (~/jt/*_full.jref) and cached prefix states (/mnt/ssd/kvcache/jref) are
   corrupt past ~8k tokens**, and every depth KL number since 10-18 (sparse attention gates, campaign 32k / 64k)
   was measured against them. A JDE-prefilled 32k prefix scored KL 0.13 against the v1 reference only because the
   reference was wrong. v2 references and states (1 GiB blocks): ~/jt/*_v2.jref, jref_v2/, kl_v2.sh; all gates to be
   re-run on them.

## 2026-10-24 Sparse attention re-tuned on corrected references (defaults changed)

**Gains (2 GPUs, 100 W, 1024-token decodes):** decode at 200k 25.6 dense / 31.5-34.4 old default -> **32.4 t/s** new
default at KL 0.0011 (old default: 0.0177 at 261k); prefill 151k 224 dense / 464 old default -> **288-293 t/s** new
default at KL 0.0014 (old: 0.0051 at 130k). End to end (both sparse) 130k 0.00125, 261k 0.00165 - about the q8_0-KV
error itself. Short spectest (MTP, 1024 tokens): 51.4 / 61.0 / 55.2 t/s.

- Defaults: decode budget max(64k, 0.5 x context) + JI_SP_SEL=1 (mass-estimate ranking, wrongly rejected on the
  broken refs: -40 % KL and faster); prefill max(32k, 0.5 x keys) + JI_SPF_SEL=1 (new: log-sum-exp of sub-page
  mass estimates). New knobs JI_SPARSE_FRAC / JI_SPARSE_PF_FRAC; prefill block list 16-bit (cap 512 -> 1200 pages).
- Diagnostics on v2: tail estimate barely helps; exact mass x mean V worse than the estimate; per-layer error even;
  monopole tail no gain; prefill tail modes ~same. Oracle ranking (fixed: it scored rotated K with the raw-basis
  query since JI_SP_RAW) is at most ~1.7x better -> this model's attention is broad; quality = fraction read exactly.
- amdgpu.gpu_recovery=0 (user-approved): a GPU hang no longer freezes the box; a hung card needs a cold power cycle.
- Reliability (open-paths S10): 2 TP2-prefill ring timeouts in ~10 h; one unexplained server exit during a prefill
  (job 20, log lost; two exact replays clean).
- Research reports 2 and 3 (Gemini) triaged into open-paths M1-M5, L7-L10, T0 (post-MVP sparse track).

## 2026-10-25 KVarN-6 ring allocation fix (engine 4f5d347ee)

**Diagnosis:** Decode performance after MTP runs was much slower (job 33: 45 ms/token vs expected 27.5 ms with identical expert hits/misses). Job 35 per-kernel profile showed MoE and shared-expert matvecs 2–2.6× slower, while attention remained unchanged. Job 37 sysfs trace revealed VRAM -> GTT eviction: 1.4–2.8 GB of GPU buffers pushed to system memory. Root cause: the separate window ring buffers (130 MiB) reduced the primary GPU's free-VRAM margin from ~412 to ~281 MiB, triggering aggressive eviction. Job 34 ruled out the prefill copy buffer `kvr_pft` as a cause of the 130k prefill failure (forcing it at 32k gave the same KL 0.00085, and the buffer is used even at 32k because the compute buffer is full).

**Fix:** window rings now live in the top `kvr_cw` (12544) cells of each kv-head region of the layer's own K / V tensors ("the tail"), because KVarN records use 208 B/cell instead of q8_0's 272 and never reach that region when kv_size >= ~4.25 × the ring size. This eliminates the fixed 130 MiB allocation, freeing the VRAM margin. Below that threshold or with `JI_KVR_TAIL=0`, rings revert to separate buffers.

**Results:** TP2 1024-token decode (rewrite / story / explain): KVarN 32.9 / 34.1 / 34.5 → 50.6 / 60.8 / 55.3 t/s (q8_0 baseline 51.6 / 61.1 / 55.8). KL unchanged (decode from restored states): single card 32k 0.00058, 130k 0.00074; TP2 32k 0.00047, 130k 0.00081. VRAM profile: no GTT eviction observed.

## 2026-10-25 / 26 — KVarN 130k prefill narrowed, power-off, boot service, handoff

**KVarN-6 prefill from scratch at 130k (KL 0.27).** Job 36: from-scratch 130k state vs the v2 reference state
(`tools/aux-jt/kvrblk.py`). Layer 3 (first attention layer) K/V exact everywhere (rel. error 0.018 K / 0.024 V =
KVarN noise). Later layers deviate from cell 12288 on (the first ubatch that attends over materialised sealed
records): 0.02-0.19, peaking at 49k-57k, deeper layers worse. Recurrent state: rel. error 0.3 % (early) .. 7.7 %
(deep). Looks like compounding quantisation noise, not a broken region; does not yet explain KL 0.27 vs 0.00115 at
64k. Next: job40 (restore that state, tail with KVarN on / off). Status: ACTIVE.

**KVarN-6 decode at 200k after the ring fix** (job 39, TP2, no MTP, sparse defaults): 25.1 / 27.4 t/s (q8 32.4).
VRAM steady (card1 8127 -> 8181 MiB, GTT 18812 -> 18980 MiB): not a spill. Needs a per-kernel profile. ACTIVE.

**Power-off 2026-10-25 01:47** during job 39's 151k two-card prefill. Not thermal: GPU max 70 / 63 C that boot,
66 / 63 C in the last log line. No kernel message; journal just stops. SATA link resets on the mSATA SSD adapter
2 min before (known, independent of load). Consistent with the power-delivery problem (2026-10-21: x4 card off the
bus at 145 W). New PSU pending; 100 W cap stays.

**Boot service** (project 2d20ea2, 3f44c65): `junkinfer.service` -> `scripts/serve_boot.sh` (two cards: TP2 + JDE
prefill + MTP; one card: single-card), LAN 0.0.0.0:8080, Qwen-Sharp template (local file identical to upstream
`chat_template.jinja`, git blob 74465e7f), Restart=always; `ji-thermal-guard.service`. Fix: helper units had
`After=multi-user.target` -> ordering cycle, systemd dropped the governor and fan-curve units at boot. Verified by
two reboots; smoke_api ALL PASS (prefill 536 t/s on 6.8k, prefix reuse 6839 -> 23 tokens), LAN chat 58 t/s.

**Docs:** HANDOFF.md (new), STATE.md current state, README numbers, code-map.md (new), tools/aux-jt refreshed
(sudo password removed from the copies). Project paused by the user.

## 2026-10-26 — Sage-A1.5 as the served model (user request)

`jyohukuchan/Sage-A1.5-exl3-4.0bpw` (InternScience Sage-A1.5, `qwen3_5_moe`: same layers, heads, experts, vocab
248,320; EXL3 4.0 bpw, head 6 bpw, **mul1** codebook, MTP layer 4 bpw, vision tower 6 bpw). Downloaded to AUX-1
`~/models/Sage-A1.5-exl3-4.0bpw` (19.4 GB, HDD); `tools/exl3/exl3_to_gguf.py <dir> <Qwen3.6 exl3 gguf> /mnt/ssd/models/Sage-A1.5-exl3-4.00bpw.gguf`:
1475 tensors, 17.21 GiB, tokenizer identical (0 tokens re-derived). serve.sh: default model Sage when present
(`JI_MODEL` overrides), alias by model file, TP hot set by model.

Evidence (TP2, 100 W, spectest 3 x 1024 tokens, rewrite / story / explain):
- Sage + Qwen3.6 MTP head: 34.6 / 45.6 / 42.6 t/s, acceptance 288/453, 448/510, 419/491.
- Sage, no MTP: 38.6 / 45.1 / 41.7; with a Sage hot set (`ref/tp/tp_hot_sage15_exl3.bin`, routegen 6 x 400 tokens;
  top-80-per-layer overlap with Qwen's set 36 %): 39.4 / 44.9 / 41.6 — no change.
- Profile (JI_PROFILE=200): Sage GPU 21.82 ms/token, hit rate 93.6 %; Qwen3.6 same harness and day 17.97 ms, 93.9 %,
  decode 43.2 / 51.2 / 48.0. Same tensor types (370 EXL3_4 + head EXL3_6): the difference is the mul1 codebook
  (~20 % dearer per weight in our mat-vec; open-paths #7).
- Boot default: Sage, MTP off (serve_boot.sh): 39.6 / 45.0 / 41.7; smoke_api ALL PASS, prefill 540.6 t/s (6,839 tokens).
Open: Sage's own MTP head; mul1 mat-vec speed; Sage KL gate (no reference yet).

## 2026-10-26 — Tool calls emitted as text (Hermes): two conversation-switch bugs, fixed (engine 5b3e708)

**Problem.** In Hermes (29 tools, ~20.7k-token system prompt, cron jobs and chats sharing the server) tool calls
often came back as text (`<invoke name="web_search">…`, `<function=terminal>{json}`, prose), on Sage and Qwen3.6.
The web UI and short tests were fine. A Hermes-side probe (its `real_body.bin`) parsed 1/6 calls.

**Diagnosis.**
- Same payload: full re-prefill (`cache_prompt: false`) correct every time; raw completions on the reused state
  hallucinated other requests and special tokens. Not template / parser: the state was wrong.
- A (prefix X + user a), C (X + user c), A: fine (prefix rollback via context checkpoints works).
- A, unrelated B, A (slot switch through the host-RAM prompt cache): with MTP, HTTP 500 "failed to process
  speculative batch" (ctx_dft got B's deferred MTP rows at positions < its last); without MTP, a wrong greedy answer
  (`scripts/conv_switch_test.py`: 17*23 = 391 fresh, 393 after the switch). `JI_ELASTIC=0` -> correct.

**Fix.**
1. `jde_kv_pre_restore` before every KV-writing state read: KV-tail expert slots released (fills waited, zeroed)
   before the restore. Before, the restore wrote rows over active slots and the next `jde_kv_guard` evicted them with
   a zero fill over the restored rows (cells ~7.8k..14.3k in the test).
2. MTP `flush_deferred`: only rows that continue ctx_dft's memory are decoded; stale rows dropped.

**Evidence.** conv_switch_test ALL PASS: Sage MTP off, Qwen3.6 MTP on. Hermes payload interleaved with another
conversation: 5/5 parsed tool calls (Sage), 6/6 repeated; SSE streaming 4/4. smoke_api ALL PASS (prefill 548 t/s).
Steady decode unaffected; after a switch the KV-tail expert slots refill.

## 2026-10-31 — Decode slowdown after a large prefill (bug 24b) fixed: huge-page host imports (engine 104fa96)

**Problem.** Decode is 20-25 % slower (~7-9 % after the CPU governor fix on 2026-10-23) after a large prefill,
until restart. In job 41: fresh 38.8 / 44.4 / 40.9 t/s, after a 21.7k-token prefill 36.0 / 40.5 / 37.8 t/s.
JDE profile: GPU time per token 21.85 -> 23.40 ms, host "post" 0.81 -> 2.24 ms (all extra time in vkQueueSubmit).

**Diagnosis.**
- Cache update phase profiling (JI_PROFILE line "cache update phases") showed all extra host time in the
  vkQueueSubmit of the expert-fill batch: 0.7 -> 2.1 ms. Not a spinning thread, not BO count (freeing prefill
  command buffers 1455 -> 405 BOs with no change), not RADV global BO list (disabling it: no change).
- `perf` revealed kernel time tripled in `hmm_vma_handle_pte / vm_normal_page / pte_val` after the prefill. **amdgpu
  faults through every page of userptr BOs on each submit.** The first large two-card prefill imports a 134 MiB
  exchange buffer (pf_xh) made of 4 KiB pages: ~33k pages walked per submit.

**Fix.** Imported host buffers (prefill exchange, TP decode exchange, split-logits buffer) are allocated
2 MiB-aligned with MADV_HUGEPAGE and pre-faulted (`jde_host_alloc`; JI_HOST_THP=0 restores old behavior).
Also: prefill command buffers freed and their pools trimmed after each prefill (1455 -> 405 BOs).

**Evidence** (TP2, 100 W, spectest 1024 tokens rewrite / story / explain; engine 104fa96):
- Sage fresh 38.7 / 44.4 / 40.8 -> 39.6 / 46.0 / 41.9 t/s (+2 % average); after a 21.7k prefill 35.7 / 40.5 / 37.7 ->
  39.3 / 45.5 / 41.6 (within 1 % of fresh; was -7 to -9 %). Expert-fill submit 0.73 -> 0.12 ms fresh, 2.1 -> 0.18 ms
  after a prefill.
- Qwen3.6 TP2 + MTP: 51.6 / 61.1 / 55.8 -> 53.9 / 64.6 / 58.8 t/s, greedy outputs identical.
- Prefill unchanged: 6.8k 543 t/s, 21.7k 489 t/s.
- smoke_api and scripts/conv_switch_test.py ALL PASS.

**Lesson (for porting notes).** On amdgpu, every host buffer imported into a GPU (userptr) costs page-walk time on
every submit; use huge pages for imports and keep them few.

**User decision (2026-10-30).** The GPU power cap stays at 100 W permanently, even after the PSU swap. All targets
(decode >= 80 t/s, prefill >= 1000 / >= 500 at 262k) are to be met at 100 W.

## 2026-10-31 — EXL3 mul1 decode at mcg cost (engine d44ea8d)

**Problem.** Sage-A1.5 (mul1 codebook, 4.0 bpw) decodes ~20 % slower than Qwen3.6 (mcg codebook, same bit width): GPU time per token 21.8 vs 18.0 ms on gfx803, despite identical expert hit rates (93.6 vs 93.9 %) and tensor shapes. Profiling showed the mul1 codebook ALU path as the bottleneck.

**Measurements** (lab/cbdecode: micro-benchmark of codebook decode on gfx803, 65536 16-bit inputs, throughput Gdecode/s on RX 580):

| variant | throughput |
|---|---|
| mcg | 293 |
| mul1, before | 188 |
| mul1 + byte sum as dotPacked4x8EXT | 247 |
| mul1 + affine fold | 293 |

**Fix 1** (exact, `x3_mv.comp` + `pf_mm` + `mul_mm_gcn_exl3` + `dequant_exl3`, engine d44ea8d): mul1 byte sum drops the f16 round trip (the value is already an f16, round trip is bit-identical) and uses `dotPacked4x8EXT(x, 0x01010101)` for the byte sum (RADV/ACO lowers it to SDWA byte adds). Only the scaling path differs from mcg now.

**Fix 2** (x3_mv decode only, opt-in knob): affine fold of the mul1 decode. The multiply and byte sum become one pre-folded weight scale and constant offset per column (ALU/register gain). Variants: `JI_X3_FOLD` 0 (exact), 1 (centred byte sum), 2 (raw fold, default). KL vs exact on Sage L_code 32k (single card, dense): fold 0: 0 (bit-exact); fold 1: 0.00012 (p99 0.0021, top-1 0.999); fold 2: 0.00014 (p99 0.0028, top-1 0.998).

**Evidence** (TP2, 100 W, spectest 1024 tokens rewrite / story / explain, engine d44ea8d, JI_X3_FOLD=2 default):
- Sage: 39.6 / 46.0 / 41.9 -> **46.4 / 55.1 / 49.7 t/s** (+17 % average)
- Qwen3.6: 53.9 / 64.6 / 58.8 t/s (unchanged)
- Prefill 6.8k: 543 -> 557 t/s (+2.6 %)
- GPU time per token: 22.1 -> 18.9 ms
- smoke_api and conv_switch_test ALL PASS

## 2026-10-31 — Native EXL3 loading and Sage's own MTP head (engine 479f8ab, f89ed0e)

### A. Native exllamav3 loading (engine commit 479f8ab; plan steps 1-3 done)

**Problem.** Every model run required `tools/exl3/exl3_to_gguf.py` to convert the Hugging Face checkpoint to GGUF (20+ minutes, 17-19 GB intermediate), and the conversion **dropped the MTP layer and vision tower**. The accepted requirement: load `.../config.json + model.safetensors.index.json + ...` directly.

**Implementation.** A safetensors reader in C++ (`src/llama-exl3.{h,cpp}`, hooked in `src/llama.cpp` `llama_model_load_from_file_impl` through metadata mode) builds the same `gguf_context` the converter would write and fills tensor data on demand with T1-T6 transforms (trellis transpose, GDN permutations, RMSNorm +1, A_log dequant, expert packing, head row drop). Shards are mmapped; transforms are memory moves (fast). Tokenizer (GPT2 BPE, token types, eos/pad/bos) and hparams (from `config.json`) derived at load. Metadata: checked against the converted oracle GGUF via `JI_EXL3_CHECK=<oracle.gguf>` (mode: load both, compare byte-for-byte). Loader change in metadata mode: an optional tensor not in the complete tensor list is not created.

**Evidence (A1 check: Sage-A1.5, TP2 decode, native vs converted GGUF):**
- 1475 tensors identical (byte exact) except `ssm_a` (30 x 32 floats): 2 ulp difference (np.exp is up to 2 ulp off; native is correctly rounded, computed in double; tools/exl3/exl3_to_gguf.py now uses float64 too).
- The 1-2 ulp in ssm_a alone: KL 0.00021 on a 32k code row (GDN state decay is very sensitive). A GGUF patched with native ssm_a gives identical KL. The production Sage GGUF on the SSD was patched; backup of old bytes: AUX ~/jt/sage_gguf_ssma_backup.json.
- Load time (page cache dropped): GGUF 92 s, native 143 s (without MTP). Decode identical: TP2 no MTP 46.4 / 55.1 / 49.8 vs 46.5 / 55.4 / 49.8 t/s. Prefill 557 t/s (6.8k).

**Paths.** The Sage checkpoint is on AUX-1 SSD at `/mnt/ssd/models/Sage-A1.5-exl3-4.0bpw` (19 GB; HDD copy remains at `~/models/Sage-A1.5-exl3-4.0bpw`). The now-redundant converted GGUF was not deleted (needs the user's OK). scripts/serve.sh: `JI_MODEL` accepts a GGUF or a checkpoint directory; default = the Sage directory, fallback to its GGUF, fallback to Qwen3.6 (chosen by mtime).

### B. Sage's own MTP head (engine commit f89ed0e; plan step 4 done)

**Problem.** The converter drops Sage's MTP layer; the Qwen3.6 sidecar gains nothing on Sage (acceptance ~64 % vs 94 % on Qwen), so MTP was off. Goal: use Sage's EXL3 MTP weights, re-quantized at load to what JDE can consume (GGUF quant types).

**Implementation.** A model path `<dir>::mtp` loads the checkpoint's MTP layer as a standalone gguf_context at the same load-time transform cost, then re-quantizes to JDE's specs: token_embd Q8_0, output Q6_K, output_norm, attn_q/k/v/output, ffn_*_exps, norms all re-quantized (IQ4_NL for dense, Q5_K for attn_v, Q6_K for the head); no EXL3 directly in the draft (it needs row slicing). Multi-threaded dequant + re-quant in C++; `JI_EXL3_DUMP=<names>` dumps produced tensors. Verified vs numpy reference (tools/exl3/exl3_ref.py): cosine attn_v 0.99926, eh_proj 0.99647, attn_q 0.99697, one routed expert 0.99700 (re-quantization noise).

**Evidence (Sage TP2, 1024 tokens rewrite / story / explain, acceptance and decode t/s):**
- No MTP: 46.5 / 55.4 / 49.8 t/s
- MTP p-min 0.5: 43.1 / 62.2 / 51.9 t/s, acceptance 67 / 92 / 86 %
- MTP p-min 0.65: 44.3 / 62.8 / 54.0
- MTP p-min 0.8: 47.6 / 63.7 / 55.9; **production choice (serve_boot.sh): 47.4 / 62.8 / 55.3** (acceptance 87 / 95 / 91 %)
- MTP p-min 0.9: 47.0 / 62.5 / 54.7
- MTP p-min 0.95: 43.7 / 61.5 / 51.7
- Qwen3.6 sidecar on Sage: acceptance 64 / 88 / 85 %, no speed gain. Outputs identical at every p-min (lossless).

**Load and overhead.** Native Sage directory + MTP re-quantization: load ~250 s (MTP adds ~107 s vs 143 s without). serve_boot.sh boot: prefill 537 t/s (6.8k). vs start of 2026-10-31 (38.7 / 44.4 / 40.8 t/s): **+22 / +41 / +36 %** with p-min 0.8, no MTP load in the decode loop.

### C. Open follow-ups

- Load time: cache the MTP re-quantization or run JDE's draft directly on EXL3 (removes the re-quant step).
- Vision: checkpoint's vision tower not yet loaded (plan step 5).
- KL reference for Sage: should be re-made from the native load.

## 2026-10-31 — Plan steps 5, 6, 8: MTP verify-step, prefill breakdown, short-prompt prefill

### Plan step 5: MTP verify-step anomaly (CONCLUDED)

**Findings:** Per-step decode cost 19.7 / 35.8 / 45.7 ms for 1 / 2 / 3 tokens/step, not linear. Measured with `JI_PROFILE`: decode cost grows as `M + N*T` where M is fixed overhead, N is per-token cost per step, T is tokens. The 1 -> 2 token gain is sublinear (81 % of 1x cost, not 50 %) and 2 -> 3 even worse (128 % of 2x, i.e. +26 ms for the 3rd token). **Root cause: MoE routing.** Each extra token routes to mostly different experts (hit rate baseline: 93 %; step 1: all new, 0 %; step 2: 24 % match step 1; step 3: 19 % match prior 2). This is architecture-specific (Sage / Qwen3.6, 256 experts, 8 active). No defect detected. Step 5 closed.

### Plan step 6: Prefill two-card breakdown and balance (CONCLUDED)

**Problem statement:** Prefill throughput plateaus at ~540 t/s (two cards) vs target 1000 t/s at zero context. Find the bottleneck and check for a cheap fix (e.g., rebalance experts between cards).

**Measurements** (TP2 JDE prefill, 100 W per GPU; engine states before plan step 8 fix):
- Helper GPU in 4k-token batches: busy 99.7 %, primary waits ~0.9 s per batch (92 % idle).
- Time breakdown: rank 0 prefill 4.1 s, rank 1 prefill 3.2 s, all-reduce mixer: rank 0 5.0 s, rank 1 6.4 s. Primary does the mixer ~1.4 s faster.
- The helper's mixer (6.4 s) is slow because it exchanges 134 MiB over PCIe 2.0 x4 (bandwidth ~1.2 GB/s effective in the all-reduce pattern; transfer time 2 * 134 / 1.2 = 223 ms baseline, but the actual time is 6.4 s with synchronisation overhead and latency).
- GPU ALU capacity: rank 0 decode matmul/minute 292 Gdecode/s at 1360 MHz clocks; rank 1 293 Gdecode/s at 1319 MHz. Both equal.
- Expert distribution: rank 0 holds 7971 / 10240 (78 %) of experts by count; helper can only compute experts it holds in VRAM. This is by design (TP2 strategy: primary computes more, helper less).
- Tried: `JI_TP_FRAC 0.6` (more experts to helper): prefill 500 vs 492 t/s, decode lower. Kept 0.7.

**Conclusion:** No cheap win. The helper is compute-bound on the experts it holds; the PCIe link limits the all-reduce. Primary has spare compute but waits for helper's mixer. To improve: (a) dedicate the helper to MTP + cache, not prefill compute (plan step 7+); (b) faster all-reduce (not feasible on PCIe x4); (c) reduce the prefill granularity (lower per-step cost); (d) single-card prefill at the cost of helper GPU idle. Step 6 closed.

### Plan step 8: Short-prompt prefill (DONE, engine 14eb6fe + f519a49)

**Problem.** Chat prompts (e.g., "hi") prefill at only ~200 t/s vs ~540 t/s for large batches. Impact: first turn in a fresh conversation 5-6 seconds, degrading user experience.

**Root cause:** llama-server splits every chat prompt at the start of the last user message (to save a context checkpoint there for edits), so a 1459-token first turn ran as two JDE prefill batches: ~200 tokens + ~1259 tokens. Each batch costs a fixed ~0.98 s plus ~1.2 ms/token (routed-expert GEMMs bottleneck); the split adds 0.98 s extra.

**Fix:** New environment variable `JI_CKPT_SPLIT_MIN` (default 2048, tunable in `/etc/junkinfer.env` or server). In `tools/server/server-context.cpp`, the server now skips the context-checkpoint split at a user-message start unless at least `JI_CKPT_SPLIT_MIN` tokens lie after the last checkpoint (or the prompt start). This preserves cache reuse: multi-turn checkpoints are at ~4 tokens before the prompt end (for regeneration), so a turn 2 reply still restores from the right checkpoint (verified: `scripts/conv_switch_test.py` cache_n 1455, ALL PASS).

**Profiling detail** (measured with `JI_PF_GPROF`, Sage native, TP2 100 W): rank 0's routed-expert GEMMs (pf_mm_id / pf_mm_id32) cost ~600 ms fixed + ~0.37 ms per token per batch. At 980 tokens, 953 ms (expected 1191 ms at 0.37 ms/token + 600 fixed; actual lower likely due to non-linearity in cache locality). This kernel achieves ~30 % of FMA peak on 32-column expert tiles (~30 tokens/expert) and ~57 % on full 64-column tiles. The latency-bound stall is the main target for plan step 7 (GEMM rewrite for energy).

**Other tried and dropped:** `JI_PF_VRAMX` (staging decode-cached experts device-local on rank 0). On a cached chat turn 3690 of 23913 expert tensors came from VRAM, prefill time unchanged (2595 vs 2629 ms). The upload is not the limiter.

**Results** (Sage TP2, JI_CKPT_SPLIT_MIN default 2048, engine 14eb6fe -> f519a49):
- 1159-token prompt: prefill t/s 208 (512 threshold) -> 272 (512 limit) -> 420 (2048 threshold). Two batches shrink to one.
- 2184 tokens: 416 -> 539 -> 532 t/s (same number of batches; 512 and 2048 thresholds both allow single batch).
- 4081 tokens: 512 -> 604 t/s (one batch at all thresholds).
- First fresh chat turn 1459 tokens: wall time 5564 ms -> 4898 ms (-12 %, one batch instead of two).
- Decode unchanged: 52.8 / 50.4 t/s at 1k / 2k context (1024-token gens).
- `scripts/conv_switch_test.py` (multi-turn switching, KV restore): ALL PASS. Qwen3.6 TP2 + MTP verified.
- `scripts/smoke_api.py` (chat, prefix-cache reuse, tool calls): ALL PASS, prefill 537 t/s on 6.8k.

Step 8 closed.

## 2026-10-31 — Plan step 7 (EXL3 GEMM for energy per FMA), first part: narrow expert tiles (engine f0dcb49)

**Goal:** Prefill routed-expert GEMMs are latency-bound (~30 % of FMA peak on 32-column tiles, ~57 % on 64-column); overhead (staging, LDS round trips, barriers, Hadamard epilogue) is ~28 % of total time as VALU work that cannot overlap the FMA loop. Rewrite the smallest tile variants (expert remainders) with a register-based strategy that keeps waves busy on FMA and reduces per-column LDS re-reads.

**Implementation.** Two changes:

1. **New kernel `src/junkinfer/shaders/pf_mv_id.comp`** (from `lab/mmbench/mv_id.comp`): the mat-vec way for expert remainders of <= 8 rows. Per 128-output block, 4 waves, each a quarter of K; per 16-deep k-tile a lane decodes 4 positions of 4 stripes into registers and reuses its x rows (all tokens, from LDS) over them. Column sums, cross-wave K sums in LDS, Hadamard-128, svh, store. Design: 4 stripes per pass, NT=8, no register prefetch → 84 VGPRs best (tested: 8 stripes 140 VGPRs slower; register prefetch raised VGPRs and was slower on gfx8; NT=16 kernels 120-152 VGPRs slower than pf_mm_id32 for 9-16 rows).

2. **Variant dispatch:** `pf_rowmap.comp` gained a third tile list (binding 5, narrow tiles, push constants NR / off3). Environment: `JI_PF_NARROW` (default 1; 0 = old pf_mm_id32 path). Benchmarks from `lab/mmbench` (mmbench.cpp + build.sh): runs prefill routed-expert GEMMs on synthetic EXL3 experts with skewed routing. Kernel sets given as "spv:maxrows,..." (each expert's rows cut into tiles of the largest kernel; remainders to the smallest kernel that holds them). Max |d| checked vs the first variant; one projection timed, RADV_DEBUG=shaderstats prints VGPRs.

**Baseline metrics** (gate 2048->512, 200 experts, 100 W, production kernels):
- 150 tokens: 4.45 ms (282 GFMA/s useful)
- 1000 tokens: 6.7 ms (1260 GFMA/s)
- 4000 tokens: 19.8 ms (1696 GFMA/s)
- BK32 variants: ~20 % slower than BK16

**Ablation findings:**
- **Large 64-column tiles:** FMA loop runs near FMA peak; decode ~8 %; the rest of overhead (~28 % total: staging, LDS round trips, barriers, Hadamard epilogue) is VALU work that cannot overlap the FMA loop.
- **Small tiles in pf_mm:** Each wave gets 8 columns (pf_mm dispatch), so a tile with a few rows keeps one wave of the 4 busy with FMAs, and each thread re-reads decoded weights from LDS per column (LDS-bound).
- **Narrow tile variant:** a lane decodes its 4 tile positions of 4 stripes (16-output groups) per k tile into registers; its 4 x rows of every token are read from LDS once per k tile and reused over the 4 stripes; every decoded weight feeds all of the tile's rows (tokens). All 4 waves work (each takes a quarter of K); 2 passes cover the 8 stripes of a 128-output block.

**Bench results** (one projection over 200 experts; baseline = production pf_mm_id + pf_mm_id32):
- 150 tokens: gate / up (K 2048 -> 512) 4.45 -> 2.52 ms; down (K 512 -> 2048) 4.35 -> ~2.4 ms
- 500 tokens: -5 to -7 %
- >= 1000 tokens: ~1-3 %

**Correctness verification** (KL vs Sage reference, 32k code row, tail through the prefill in 256-token ubatches, TP2):
- With and without narrow tiles: KL 0.00036 (identical), p99 0.0061, top1 0.9961
- Tail time: 18.3 -> 17.4 s (slight speedup from the narrow variant in tail evaluation)
- **Pitfall (lessons):** kl_v2.sh's default `JR_TAIL_B=4` evaluates the tail through the DECODE path, so it does not test prefill kernels. Use `JR_TAIL_B=256` (or the ubatch size) for prefill changes.

**Production results** (Sage TP2, 100 W boot config, engine f0dcb49, JI_PF_NARROW default 1):
- 1159 tokens: 418 -> 448 t/s
- 2184 tokens: 530 -> 547 t/s
- 7820 tokens: 573 -> 575 t/s
- Chat turn (984 new tokens): 2607 -> 2432 ms (-6.7 %)
- Decode unchanged: 54.0 / 52.5 / 48.9 t/s at 1k / 2k / 8k context
- `scripts/conv_switch_test.py`: ALL PASS

**Remaining step 7 headroom:** The large-tile overhead (~28 % of MoE GEMM time) would need a rewrite that cuts VALU work per k-step (staging, LDS round trips, epilogue barriers); estimated 10-15 % of MoE GEMM time remaining, ~5-7 % of total prefill. Parked pending design review.

Step 7 partly done (narrow tiles).

## 2026-10-31 — KL harness graph-path prefill artefact; KVarN from-scratch item closed; narrow tiles re-validated

### The harness KL issue

**Discovery:** The KL harness (`~/jt/kl_v2.sh` on AUX-1) never set `JI_JPF=1`, so every prompt/prefix prefill in KL runs used the ggml graph path; only decode used JDE. Production (`serve_boot.sh`, `serve_prod.sh`) sets `JI_JPF=1` (default since engine 14eb6fe). Fixed 2026-10-31: kl_v2.sh now passes `JI_JPF=${JI_JPF:-1}` to default to JDE prefill. Backup: `kl_v2.sh.bak_j85`.

**Ubatch depth issue:** With ubatch 256, the ggml compute buffer is only 396 MiB but JDE prefill needs 847 MiB scratch for 256 tokens, so it falls back to the graph path. Production large-buffer tests in the harness need care: either use `UB=4096` (default) and set `JR_TAIL_B=256` for 256-token tail evaluation, or fix the compute buffer depth. To verify JDE prefill ran: `JI_PF_TL=1` with `KLX="-lv 4"` and count "jde prefill TL rank 0" lines; a tail-via-prefill KL identical to q8 in every digit is a red flag (that tail evaluated through decode, not prefill).

### Narrow-tile KL re-validation

**Earlier finding:** The narrow-tile KL check at engine f0dcb49 recorded 0.00036 identical with and without `JI_PF_NARROW=1`. That check ran through the decode path (kl_v2.sh `JR_TAIL_B=4`, default tail via decode); it did not exercise the JDE prefill.

**Re-validated through the JDE prefill** (Sage TP2, ubatch 4096, tail batches of 256, 4 JDE prefill batches, 0 graph fallbacks; engine f0dcb49, KLX="-lv 4" 200 steps, L_code 32k):
- `JI_PF_NARROW=0`: KL 0.00015 / p99 0.0024 / max 0.013 / top1 0.9990, tail 13.5 s
- `JI_PF_NARROW=1`: KL 0.00012 / p99 0.0023 / max 0.007 / top1 1.0000, tail 13.1 s

**Correctness:** narrow tiles are clean. The earlier step-7 log entry's correctness line stands with the caveat (added now): "KL 0.00036 was measured through the ggml graph-path prefill (harness without JI_JPF, and ubatch 256 too small for the JDE prefill scratch); through the JDE prefill: 0.00012–0.00015 (above)."

### KVarN from-scratch item: graph-path detour, then confirmed real

**Status change:** Open work "KVarN-6: prefill from scratch beyond ~64k is wrong (130k: KL 0.27 at 130k)" looked like a **harness artefact** at first (see below) - but job88 (end of this entry) confirms the defect is real through the JDE prefill. The KL harness used the graph-path prefill, which does not build KVarN records or the window ring; a saved state compared per layer/cell against a reference (kvrblk.py) had relative error 1.000 everywhere (i.e., buffers were empty or garbage), while the tail through the graph-path prefill matched q8_0 to every digit.

**With JDE prefill:** Qwen 32k from scratch, single card, sparse off: q8_0 KL 0.00071 / top1 0.9971, KVarN 0.00092 / top1 0.9893. KVarN works from scratch. **job88 (TP2, default sparse, JDE prefill: 32 batches, 0 graph fallbacks), 130k from scratch: q8_0 KL 0.00123 / top1 0.9883 (= documented 0.00125); KVarN 0.118 / top1 0.906 - the defect is real (the old TP2 number) and lies in the JDE prefill KVarN path between 32k and 130k. The graph-path runs above were a separate artefact. Next: depth bisect 64k / 98k with JI_JPF=1, then kvrblk.py on a saved from-scratch state.**

**Production validation:** The 130k "from scratch" q8_0 KL numbers measured today (0.0064 exact, 0.0089 default sparse; identical on the 10-24 engine d9909dc) were graph-path prefill + JDE decode, not production. Through the JDE prefill (job88) production q8 is KL 0.00123 at 130k, the documented value: not regressed.

**Dead ends recorded** (all on graph-path artefact): compute-buffer slot eviction (`JI_PF_CBALL`), `JI_ELASTIC=0`, `JI_STATIC=0`, `JI_KVR_PFT=1`, in-process state reload — all bit-identical.

### Service stop helper and incident

**New script:** `~/jt/stopsrv.sh` on AUX-1 (snapshot in `tools/aux-jt/stopsrv.sh`). It stops the boot service and waits until no llama-server / llama-junk-ref / llama-completion / llama-cli process remains. All job scripts should call it before starting a test (instead of `systemctl stop junkinfer; sleep 5`).

**Incident 06:37 today:** A job did `systemctl stop junkinfer; sleep 5` while the service was still starting up. The stop had not taken effect, llama-junk-ref started, and two model processes OOMed the box, triggering a reboot. The service needs a synchronisation point: it now waits for all daemons to exit (see `stopsrv.sh`).

**Job scripts policy:** on AUX-1, use `sudo -n` (passwordless works); never write passwords into files.

## 2026-10-31 — KVarN-6 from-scratch defect found and fixed (engine cd30f03)

**Issue:** 130k prefill from scratch with KVarN-6 scored KL 0.118 / top1 0.906 (vs q8_0 0.00123 / 0.9883). 32k and all restored-state decodes were correct. The defect lay in the JDE prefill's KVarN path.

**Root cause:** After each JDE prefill ubatch the KVarN path rebuilt the sparse page metadata from the window ring before sealing (in `jde_kvr_meta_rebuild`). On TP2 the primary's kv-head-1 rows are pulled back from the helper **only inside the seal** (in `jde_kvr_head1_back`), so the sparse metadata rebuild happened on stale kv-head-1 rows that had not been copied back yet.

**Fix:** New environment variable `JI_KVR_META_DEFER` (default 1; set to 0 for the old per-ubatch rebuild). With `JI_KVR_META_DEFER=1`, pages stay dirty during prefill and the first sparse decode step rebuilds their metadata from the sealed records + window ring. This matches the semantics after a state restore: all pages are initially dirty and metadata is rebuilt at the first sparse decode step after restore (verified: restored states were correct before and after the fix).

**Measurements** (TP2, sparse defaults, from scratch, engine cd30f03):
- 98k KVarN: KL 0.0304 → 0.00141 (top1 0.9746 → 0.9951)
- 130k KVarN: KL 0.118 → 0.00136 (top1 0.906 → 0.990)
- 130k q8_0: KL 0.00123 (unchanged, production reference)
- Speed: KVarN opt-in, production q8_0 unchanged; one-time cost is the first sparse decode step after a long prefill rebuilding all dirty pages' metadata (not yet timed).

**Remaining KVarN items (open in HANDOFF section 9):** decode at 200k slower than q8 (25.1 / 27.4 vs 32.4 t/s); K6V5 / K6V4 asymmetric variants; end-to-end gate and making KVarN the default if it wins.

## 2026-10-31 — Qwen3.8 27B viability (dense path estimates and measured sparse/quant baselines)

**Model & hardware:** Qwen3.8 27B (standalone dense, not MoE). Two RX 580 8 GB cards, TP2 layer split, 100 W per GPU cap (hard limit 2026-10-30).

**EXL3 SC 3.0 bpw baseline** (turboderp community, converted to GGUF with exl3_to_gguf.py dense support, engine qwen35.cpp transforms; engine commits b3e4b9b / docs 084588f):
- Prefill: 64 t/s (1k prompt, ubatch 512)
- Decode on the graph path: 0.77 t/s (no fast 3-bit EXL3 mat-vec on the ggml graph path; this is the fallback when JDE prefill buffer too small for 256 tokens)

**GSQ-RCO IQ3_XXS-mtp** (10.4 GB, MTP sidecar on the helper GPU TP2 config):
- Prefill: 45 t/s (1k prompt, ubatch 512), 50 t/s (4k prompt, ubatch 1024)
- Decode on the graph path: 10.4–10.6 t/s
- MTP: 2-3 drafts per token on the graph path; draft acceptance 0.49–0.60 per draft at temp 0.7; speculative decode +20–40 % wall time (decode 7–10 t/s with MTP vs 10.4–10.6 t/s baseline; slower than beneficial on the ggml path with poor PCIe/expert throughput).

**Estimated JDE dense path** (not yet built): assume similar expert-cache hit patterns and MoE-to-dense reduction as Qwen3.6 (60 layers → 50 GEMMs at 100 W):
- Decode: ~18–21 t/s baseline, ~25–32 t/s with effective MTP
- Prefill: ~100–120 t/s (power-limited at 100 W)
- Hard ceiling (fp32 FMA peak without matrix units on gfx803): ~215 t/s prefill

**Context budget** (KV for 27B at 100 W, q8_0 vs KVarN-6):
- q8_0: ~100k tokens
- KVarN-6: ~130k tokens (23.5 % saved bandwidth + VRAM spill headroom)

**Hardware note:** the SSD mSATA adapter (ata11) failed under heavy I/O on 2026-10-31 (SATA link resets, read errors, 1.5 Gbps fallback; reboot restored 6 Gbps but errors recur under sustained reads). User informed; hardware fix pending. Models moved to HDD overflow `~/models/ssd-overflow/`: Sage-A1.5-exl3-4.00bpw.gguf (redundant with native checkpoint), head-*.tmp.gguf, mtp-BF16.gguf, minicpm5-vllm; the 27B EXL3 GGUF kept on both disks for redundancy.

**Reference for comparison:** user data on RTX 3090 (hyperqwen vLLM fork): Qwen3.8 27B ~100 t/s decode, ~1500 t/s prefill. Decode ratio ~3.4× matches throughput expectations (3090 ~1300 Gdecode/s TP2 expert cache hits vs RX 580 ~290 Gdecode/s, similar expert miss rate / PCIe); prefill gap is tensor cores (3090 TFLOP / 100 W vs RX 580 FMA-only at the same power).

## 2026-10-31 — KVarN-6 sparse decode 2.3× faster; KVarN on by default (engine bae3129, docs repo 34f3db5)

### Summary

Sparse page metadata rebuild in `attn_kvr6` was expensive (3136-6272 global reads per workgroup on 200704 cells, 3136 of 6272 pages selected). Binary search of the sorted page list cut the kernel from 4291 µs to 2530 µs (q8 `attn_hm` baseline 2306 µs). Production decode at 200k depth: 27.9 → 34.5 t/s (q8 baseline 33.1 t/s). KVarN is now default-on via `serve_boot.sh` (environment `JI_KVARN=${JI_KVARN:-1}`); production restarted and smoke tests pass.

### Findings (isolated benchmark: jde.cpp `jde_attn_test_kvr2`)

**Setup:** Single RX 580, 200704 cells, Qwen3.6, TP2, synthetic sorted page selection (3136 of 6272 pages), test knobs `JI_ATTN_TEST=<cells> JI_ATTN_KVR2=1 JI_ATTN_SP=<pages> JI_ATTN_SP_RUN=<run_length>`.

| variant | time |
|---------|------|
| dense `attn_hm` (q8_0) | 2306 µs |
| sparse `attn_kvr6` (before) | 4291 µs |
| per-page linear scan to find selected pages below S | 3136 read, 6272 checked |
| binary search instead | 2530 µs |
| no per-key page lookup in the PV loop | 4240 µs |
| chunk page list in LDS | 4015 µs |
| **arithmetic not the cost** (profiled: no QK 3724, no PV 3995) | — |

**Root cause:** Each workgroup counted selected pages by linear scan of the entire page list (3136-6272 global reads per workgroup for the PV loop structure); the scan dominated the per-key math. Binary search (the page list is always sorted by `attn_topk`) cut that to ~14 reads per page (log₂ 6272), making the kernel speed-competitive with dense.

**Quality (RMS error vs q8_0, unchanged):** 0.0277 (unaffected by the optimization; the algorithm is the same).

### Production results (TP2, 100 W cap, sparse attention defaults, >= 1024 tokens)

**Decode at 200k (JI_KVR_META_DEFER=1 from the prior fix):**
- q8_0 `attn_hm`: 33.1 t/s (baseline)
- KVarN-6: 27.9 t/s (before) → **34.5 t/s** (after sparse optimization, **+23.7 %**)

**Full-depth end-to-end (Sage TP2, native EXL3, MTP p-min 0.8, TP2 with one-card TP helper):**
- 1k prefill (244 → 307 t/s); decode short 54.3 → 50.5 t/s (MTP variance at temperature 0.7; no sealing below 8192)
- 8k prefill (568 → 578 t/s); decode 47.0 → 49.7 t/s
- 32k prefill (447 → 443 t/s); decode 42.7 → 43.4 t/s
- Chat turns equivalent; `conv_switch_test.py` ALL PASS

**Default status:** `serve_boot.sh` now exports `JI_KVARN=${JI_KVARN:-1}`. Production restarted 2026-10-31 with this default; smoke_api.py ALL PASS (1k + 1k tokens, prefill 537 t/s). Quality: 130k from scratch KL 0.00136 (q8 0.00123, both sparse defaults), restored 0.00074-0.00121.

### Open KVarN items

1. **K6V5 / K6V4 asymmetric variants** (user-approved): expected ~3.5 % more context per GB for dense 27B models (KVarN budget moves from K=6 to V=4 or vice versa; requires separate decode paths and axis transforms).
2. **Optional: sparse decode LS page-list cache in `attn_hm` (q8):** page metadata rebuild in sparse attention consumes work also in q8 (3-4 % of prefill); the same LDS cache technique may give a few percent.


## 2026-11-01 — Routing traces, expert prefetch (no gain), TP2 placement fix, write-through fill (engine 747e1b5..7efbc5d)

**Gains (Sage-A1.5 TP2, 4 x 1100 greedy tokens, two server instances per config): decode short 52.3 -> 58.1 t/s
(+11 %), 116k (saved state) 37.9 -> 39.9 t/s (+5 %); prefill ~10k tokens 549 -> ~530 t/s (-3 %, placement).**

- Routing traces (JI_ROUTE_LOG, on in production): in TP2 the primary logged -2 for the helper's selections (26 %);
  router_fused now keeps them in `wanted` as -16 - id (747e1b5). Old traces: ids recomputed exactly from the logged x.
  Copies + router weights on VENE `/mnt/chonker/projects/junkinfer/data`; `tools/routes/sim_cache.py` (cache
  policies, Belady, router lookahead, write-through), `tools/routes/mkhot.py` (TP hot set from traces).
- Router lookahead (layer L+D's router on layer L's MoE input): recall of the true top-8 in the predicted top-8
  0.80 / 0.73 / 0.68 for D = 1 / 2 / 3.
- Prefetch (JI_PREFETCH=<B>, opt-in, 93c93f0): misses per 2-token step 43 -> 20 as simulated, decode unchanged.
  JI_TP_GPROF (primary): routed-expert calls 9.3 of 23.2 ms/token, 77 us per call with a miss vs ~34 us all-hit; a
  miss is ~1 MB over PCIe at ~14 GB/s - bandwidth-bound, and the prefetch copies share the link at the same time.
  CP-DMA chunked copies and a 2-workgroup copy shader (la_copy, JI_COPY_WG) were slower. One TP2 "sync 0 lost" abort
  with JI_PREFETCH=4 (not reproduced). Conclusion: count PCIe bytes per token, not latency.
- TP2 placement: the old Sage hot set matched the native model's routing poorly (corr 0.19, helper share 27 %);
  JI_TP_DUP 0.25 duplicated ~540 experts while the helper idled 7-10 ms per step. New default (ee84f72): hot set
  from a 24-prompt corpus (ref/tp/tp_hot_sage15_native_corpus24.bin, helper share 43 % on held-out traces),
  JI_TP_DUP=0: decode 52.3 -> 54.2, 116k 37.9 -> 39.0, prefill -2..-5 %.
- Write-through fill (7efbc5d, JI_WT default on): router_fused assigns each primary miss a victim slot from a
  per-step host pool (routing record 384 -> 512 bytes, + fill[32]) and drops the victim from the slot map; x3_mv
  stores the staged weight batches into it; the host publishes after the step. Async-fill evictions off. Decode
  54.3 -> 58.1, 116k 38.7 -> 39.9, prefill same, KL 0.00017 = 0.00017. Primary/helper GPU work 16.7 / 16.0 ms (was
  20.9 / 12.7).
- Measurement: server instances differ by +-4 t/s (greedy text differs per instance); compare averaged
  `jde_prof: per token` windows and >= 2 instances. Temperature: same instance, 2 x 1100 tokens each, T 0 / 0.7 / 1.0
  = 58.5 / 58.2 / 56.5 t/s, draft acceptance 0.94 / 0.92 / 0.87 - the fork already drafts at the target temperature
  and verifies by rejection sampling (common_sampler_sample_and_accept_n_rejection); research-actions plan item 5 is
  covered. (A first request after a restart is slower: the expert cache is cold.)

## 2026-11-01 (later) — Thread pinning, draft length, eviction bound (engine 87f4606, docs 9627895)

**Gain: decode short 55.8 (mean of unpinned instances, 53.9-58.3) -> 58.3 t/s (57.8-58.5), prefill unchanged.**

- Server instances differed by up to 10 % at identical JDE GPU time per step: the host gap between target decode
  calls (sampling, MTP draft, bookkeeping) was 2.6 ms in fast instances, 3.4-8 ms in slow ones (JI_DEC_PROF). The
  TP relay thread spins on the sync markers; where the scheduler put it relative to the decoding thread (SMT sibling,
  other CCX) decided the mode. JI_PIN=1,2 (serve.sh default for TP2): decoding thread CPU 1, relay CPU 2. Pinning
  only inside jde_decode was not enough (58.9 / 53.2). Threads started later inherit the one-CPU mask (JDE's worker
  pool): swept back to the original mask every 1000 calls.
- MTP draft length 2 (JI_MTP_NMAX=2): 2.37 vs 1.95 tokens per step, JDE step 25.5 vs 23.3 ms; decode 55.3 vs 53.9
  and 55 vs 58.4 / 52.6 in the noisy unpinned instances - not shown to help; default stays 1. Per-request
  `speculative.n_max` is ignored by the server (capped at the launch value).
- Learned eviction bound (sim_cache wt...f<N>: victims avoid experts used in the next N tokens, perfect knowledge):
  misses per token 11.76 (write-through now) -> 11.51 / 10.85 / 9.07 / 6.94 for N = 4 / 16 / 64 / 256, Belady 4.65.
  A hidden-state predictor would have to see 64-256 tokens ahead to matter, and with primary / helper work balanced
  (16.7 / 16.0 ms) primary misses barely shorten the step any more: deprioritised.
- AUX1 went off the network during the pinning validation (instance 2 of 3, ~09:20); cause unknown at the time of
  writing (check `journalctl -b -1`, ji-thermal-guard).
