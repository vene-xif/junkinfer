# JunkInfer Decode Engine (JDE)

Status: **in production** (default on for decode ubatches). Engine commits: 2026-10-16 through 2026-10-23, see engineering-SIREN/current/engineering-log.md. Built for Qwen3.6-35B-A3B hybrid (30 GatedDeltaNet + 10 full-attention, 256 experts, 8 routed + shared) in EXL3 4.0 bpw on RX 580 gfx803.

## Why

ggml-vulkan decode spends ~23 ms of GPU time per token on ~1.4 GB of weights (should be ~6 ms at roofline), because (1) dependent dispatch bubbles: each of ~1,500 dispatches per token pays ~7 us on gfx803, (2) small matvecs at 10–60 GB/s instead of the achievable 225 GB/s (codebook in LDS reachable), and (3) a CPU → GPU → CPU → GPU round trip per MoE layer prevents overlap. Decode at 245k context reads 2.85 GB of q8_0 K/V per token: dense attention alone was ~20 ms. Dense decode was ~42 ms total; the engine targets purpose-built kernels, fused operations, and few dispatches per layer.

## How it works

**Shared state with llama.cpp:** weights (native GGUF block layouts: IQ4_NL, Q8_0, Q5_K, Q6_K, F32, EXL3; staged into LDS), expert cache (hybrid bank and elastic slots), KV cache and recurrent state, logits output buffer, device/queue obtained through ggml-vulkan accessor API.

**One-shot token (single submit per ubatch):** replayed command buffers recorded per token count; per-token parameters (positions, KV cell, KV length) live in a device buffer that kernels read. With EXL3 zero-copy misses (GPU reads host experts directly), per-layer routing records (ro_host) feed expert cache statistics after the fence, eliminating 81 per-layer host handshakes.

**Kernel stages per token (40 layers):**
- **Norms:** rmsnorm (fused into the next matvec where possible), l2norm per attention head.
- **GatedDeltaNet layers (30):** conv1d(k=4)+silu, l2norm q/k, gates, gated delta rule on recurrent state (32×128×128 f32), output norm + silu(z).
- **Attention decode (10 layers):** q/k norm, partial RoPE (64 of 256 dims), head-major q8_0 KV write, attn_hm (split-K over KV length with 4 waves per block), attn_merge (parallel suffix scan), optional sparse pages with tail estimation (attn_score, attn_topk, attn_bg).
- **Router + top-8:** softmax, top-k, normalise, one expert row per wave (30.2 us/layer).
- **Shared expert + routed experts (8 hits or misses):** MoE mat-vecs (stripe-major, one dispatch per 16-expert tile), zero-copy misses read host experts on GPU, fused Hadamard output stage (EXL3).
- **LM head:** Q6_K 248,320 rows; logits to host.

**Host side between tokens:** expert cache update (partial sort to find ≤ 8 swaps, ~0.23 ms), async DMA fills (copies on own queue, slots published when done), multi-token decode steps (1–4 tokens per ubatch, state rollback snapshots), small-prompt path (batch <= 64: JDE decode instead of graph path MoE on CPU).

## Measured

Rows up to 2026-10-16 (later) used IQ4_NL weights; from 2026-10-16 (night) EXL3 4.0 bpw (slower per weight: trellis decode), so the series restarts lower. Single card unless noted.

| date | config | decode t/s | notes |
|---|---|---|---|
| 2026-10-16 | v1: all experts CPU, serial | 21 | GPU 21 + CPU 21 |
| 2026-10-16 | hybrid cache 70% hits | 23.7 | 42.3 ms/token |
| 2026-10-16 | attention, kernels, replay | 39 | 25.5 ms/token, 4500 MiB cache |
| 2026-10-16 (later) | GDN + router fusion, LM head WGs | 46.3 | 21.3 ms/token, server warm 256 tokens |
| 2026-10-16 (later) | q8_0 KV at 262k | 42 | short depth 21.1 ms, dense KV reads still 20 ms |
| 2026-10-18 | one-shot tokens, compute-buffer slots | 36.5 | short; elastic slots 1834 routed experts |
| 2026-10-18 | EXL3 stripe-major mat-vecs | 42.6–43.4 | short decode; fused output stage 44.0 t/s |
| 2026-10-18 (evening) | head-major KV cache + attn_hm | 21.2 | at 247k (restored context) |
| 2026-10-19 | expert cache partial sort, one-shot record | 45.75 | +4.2 % short decode |
| 2026-10-19 | sparse attention default (>= 96k) | 29.6 | at 245k (+44 %); 20.5 dense |
| 2026-10-19 | EXL3 f16 codebook sum | 45.0 | short (+3 %); 245k 31.1 (+5 %) |
| 2026-10-20 | per-layer sparse budgets (opt-in) | 30.4 | 245k, late layers at 8192 keys (+1.6 %) |
| 2026-10-21 | TP2 (two GPUs, full power) | 59.1 | short; see tensor-parallel-two-gpus.md |
| 2026-10-23 | TP2 at 100 W, governor performance | 43.2 / 49.6 / 48.0 | rewrite / story / explain, no MTP |
| 2026-10-23 | same + MTP | 49.4 / 58.5 / 58.9 | see mtp-speculative-decoding.md |

## gfx803 lessons

- **Dependent dispatch cost ~6–8 us regardless of barrier flavour.** Clocks pinned high; only fewer dependent steps help. Per-token budget: ~1,500 dispatches with barriers would be 10–15 ms wasted.
- **L2 caches system memory and is not invalidated mid-submission.** A host-set VkEvent waited on by a *compute* queue hung it (the CP never saw the host write; GPU reset) - on *graphics* queues it works, which is what TP2 uses; a GPU-polled doorbell read stale CPU partials. Host→GPU data must cross a submission boundary; design consequence: per layer A[il] → host waits fence → CPU misses (overlap B[il]) → submit A[il+1].
- **EXL3 dense mat-vecs DRAM-page-bound.** Small shapes reached only 45–63 GB/s (vs 157 GB/s on LM head). Cause: slice-major wave order; consecutive waves read K slices 16 KB apart. Fix: stripe-major (consecutive waves = consecutive K slices, consecutive memory), one 16-tile batch per wave → 8192×2048 133 → 88 us (server decode 36.9 → 42.6 t/s).
- **LDS float atomics (VK_EXT_shader_atomic_float) far too slow on gfx8.** attn_merge tried 3 workgroups per CU with float atomic combines in LDS: 1115 → ~1510 us at 116k; reverted.
- **Occupancy cliffs from register pressure.** attn_hm with 128 VGPRs fits 2 workgroups per CU; 84 VGPRs fits 3 (but float atomics slower). ji_fa2 with 84 VGPRs / 3 waves per SIMD is the ceiling for prefill Q-broadcast.
- **Scalar cache sharing with LDS reads.** Q loads (re-read per 64-key block) and LDS reads share lgkmcnt; one-step prefetch cannot hide them. Fix: 2 keys per lane (128-key blocks, K staged in 16-dim chunks) → half the Q loads.

## Rejected

- Quest page drop without tail estimation: ~1.4 % mass lost at 128k depth even with oracle selection (global layers attend broadly).
- Per-query-head selection (8× the K/V reads; GQA sharing lost).
- JI_SP_SEL=1 (rank pages by estimated mass): mass-oracle bound better.
- JI_SP_BG=3 (dipole term): no gain.
- Dynamic expert top-k (gate threshold): decode and prefill unchanged, KL degraded.
- LM head as IQ4_NL (vs EXL3 6-bit): KL 0.0044 vs 0.00033 at 130k.
- Hybrid GGUF (EXL3 experts + head, dense projections IQ4_NL): KL 0.0120 (130k) vs 0.00033; dense imatrix required.

## Porting notes (RDNA2 / other engines)

- Stripe-major wave order is not GCN-specific; any GPU with memory-bound wave-local reads (coalescing unit size, cache line size) benefits from locality.
- Host -> GPU handshakes inside one submission: on gfx8 only host-set VkEvents waited on graphics queues work (compute queues miss the host write, GPU-polled doorbells read stale L2). Re-test both on RDNA2 before designing around them.
- Metadata VRAM (241 MiB at 262k for sparse attention) competes with expert cache. On a 16 GB card, this no longer binds; adjust JI_SP_MIN (dense/sparse crossover) by re-measuring at the target VRAM bandwidth.
- Codebook (EXL3 mcg: 32-bit multiply at quarter rate) bounds small mat-vecs; the trellis decode is not a generic path, so porting is model-specific. Shared-expert / routed-expert mat-vec fusion depends on the model's expert layout.
