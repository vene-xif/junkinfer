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
