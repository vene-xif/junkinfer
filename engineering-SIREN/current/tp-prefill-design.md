# Two-card tensor-parallel prefill on JDE (design, 2026-10-21)

Goal: prefill >= 1000 t/s at short context and >= 500 t/s at 262k on two RX 580s, without changing the decode
layout (JI_TP=2). Baseline (server, JI_TP=2, single-card prefill): 8k 484, 80k 361, 245k 277 t/s.

## Why not the layer pipeline

Measured (engine 108a61d57): layers 27-39 resident on card 2 + llama.cpp pipeline parallelism (fixed: shared host-
weight copies, no per-ubatch realloc sync, layer-aware offload) gives 20k prefill 393 -> 486 t/s. More layers need
their experts streamed over card 2's PCIe 2.0 x4 link (0.32 s per layer per ubatch, slower than compute), and whole
layers on card 2 conflict with the TP2 decode set on that card (hot experts of every layer + dense halves + KV
head-1 mirror): decode would lose 15-20 %.

## Tensor parallel on the TP2 decode layout

Card 2 already holds, for decode, exactly rank 1's share: its GDN key heads / value heads, half of the shared expert,
attention kv head 1 (mirror), its recurrent-state half, hot routed experts (61-70 % of the routing traffic, no
uploads). Prefill reuses it: rank r runs, for a ubatch of T tokens, the same split as TP2 decode with GEMM kernels:
- GDN: wqkv rows of its heads (4 blocks of 1024), z (2 blocks), alpha / beta, conv + gated delta rule on its heads,
  ssm_out over its K quarters (partial) -> all-reduce 1.
- attention: q heads [8r, 8r+8) + kv head r, attention, wo over its K half (partial) -> all-reduce 1.
- MoE: router on both (identical), shared expert half rows + down K half, routed experts by ownership (balanced per
  token over the shared set) -> all-reduce 2.
- Rank 0 streams its (cold) experts from host per layer (DMA ring into VRAM staging, the idle ggml compute buffer).
Per rank ~55 % of the single-card work; all-reduces: 2 per layer, T x 2048 f16 partials (16 MB at T = 4096) through
host memory (card 2: 1.07 / 1.27 GB/s) ~20 ms each, 1.6 s per 4096-token ubatch: they must overlap compute ->
two micro-batches per ubatch, GPU-side waits on host-set events (graphics queues), relay thread as in decode.

## Milestones
1. Single-card JDE prefill (rank 0 of 1): EXL3 GEMM (x3g_prep + mul_mm_gcn_exl3 X3F, + K-part window), routed-expert
   GEMM (row map + MMID X3F), GDN prefill (conv + gated delta rule), RoPE / q,k norm / KV write for T tokens,
   ji_fa2 prefill (causal mask from positions), router / norms / combine for T rows, expert DMA ring. KL = graph path.
2. Rank split on both cards + all-reduce through the exchange area (no overlap).
3. Two interleaved micro-batches: the exchange of one overlaps the other's compute.
4. Long context: sparse prefill (ji_fa2_sparse + pmeta) on the kv-head split.
