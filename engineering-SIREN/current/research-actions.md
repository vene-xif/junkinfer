# Research actions (from the 2026-10-31 deep-research reports)

> Actionable results from external deep-research reports (Gemini), checked against our code and measurements.
> Keep the verdicts honest: anything not yet measured on AUX-1 is a hypothesis. Update in place; results go to
> `engineering-log.md`, live priorities to `open-paths.md`.
> Sources: report A = "Low-bit decode / GEMM kernels on GCN4" (research prompt 1, in Polish);
> report B = "Long-context KV compression, sparse attention, Qwen4 readiness" (prompt 3);
> report C = "Two-GPU tensor parallelism over weak PCIe without P2P" (prompt 2; written for a dense 27B, out of scope).

## Plan order (user-agreed 2026-10-31)

1. **Routing traces** (in progress): JDE `JI_ROUTE_LOG=<dir>` logs every decoded token's router top-8 ids, weights and
   the per-layer MoE input x (fp16) - enabled on the production server so real workloads accumulate on the HDD
   (`~/traces`, cap `JI_ROUTE_LOG_MAX_GB`, default 150). Reader: `tools/routes/read_trace.py`. Corpus traces through
   the prefill path later, when the GPUs are free. First datum: 34 % of a token's (layer, expert) selections repeat
   the previous token's.
2. Decode-kernel lab bench + contiguous activation reads in `x3_mv` (action #2 below).
3. **Trace simulation** (offline, VENE CPU): our cache policy vs perfect-knowledge Belady vs router lookahead
   (layer L+1 / L+2 routers applied to layer L's hidden state) vs MTP-draft prefetch -> the real upper bound of
   expert prefetch, before engine work.
4. **Router-lookahead expert prefetch**: predicted misses copied into cache slots by the async fill queue (DMA)
   1-2 layers ahead (a layer ~0.5 ms; an expert ~0.13 ms over x16), plus MTP-draft token prefetch. Estimated
   +15-30 % decode (bounded by miss bytes / PCIe bandwidth; decode is ~16 ms experts of ~20 ms per token).
5. MTP speculative sampling at temperature > 0 (action #3).
6. **Learned lookahead predictor on the hidden state** (not the KV cache): a small network trained on the traces to
   predict experts several layers / tokens ahead, compared against the routers' own lookahead on the same traces.
   (User: the router is optimized for the current token only - a predictor trained for multi-step lookahead may win.)
7. Integer accumulate / `v_sad_u8` in decode (actions #4-5), then the register-decoded prefill GEMM (action #6).

## Ranked actions (kernel / sampling details)

| # | Action | Source | Expected gain | Effort | Status |
|---|---|---|---|---|---|
| 1 | **Decode-kernel lab bench** (`x3_mv` on Sage shapes, real layout) before any ISA trick | A + our data | enables 2-5 | 0.5 day | open |
| 2 | **Contiguous activation reads in `x3_mv`**: write the Hadamard-prepped input in the tile-row order a lane uses (rows r0, r0+1, r0+8, r0+9 -> 4 adjacent floats) so the 4 LDS reads per tile per token become one `ds_read_b128`. Bit-identical. | A | ~0.75 op/weight saved (~7 %), more with MTP multi-token steps | 1 day | open |
| 3 | **MTP speculative sampling at temperature > 0**: our fork accepts a draft only if it equals the target's sampled token (`common_sampler_sample_and_accept_n`, exact match). Proper speculative sampling: sample the draft from p_draft, accept with min(1, p_t/p_d), on reject resample from max(0, p_t - p_d). Output distribution unchanged. Our MTP head gives p_draft (top 65k vocab). | B | acceptance at temp 0.6-0.7 up substantially (27B graph test: 0.49-0.60 per draft); est. +10-25 % decode in sampled chat; zero at temp 0 | 2-3 days | open - measure Sage acceptance at temp 0.7 first |
| 4 | **Integer accumulate in decode**: quantise rotated activations to 13-bit ints per 128 block; byte sum t (<= 1005) x a with `v_mad_i32_i24` (no `v_cvt_f32_u32` per weight); affine A*sum + C*sum(x) once per output (fold). | A | ~1 op/weight | 2 days | open - needs KL gate |
| 5 | **`v_sad_u8` byte sum** (1 full-rate instr; today 2 `v_add_u32_sdwa` per weight). Not reachable from GLSL: needs a NIR algebraic rule (4-byte sum -> `sad_u8x4`, ACO emits `v_sad_u8`) in a side-built Mesa loaded via `VK_ICD_FILENAMES`. The one case where a driver patch pays. | A | ~1 op/weight in decode and prefill | 2-3 days + keeping a Mesa build | open - only if #1 shows decode is still ALU-bound |
| 6 | **Register-decoded prefill GEMM ("quad-shared DPP")**: a wave decodes a 16x16 tile once (4 weights/lane), shares weights across the 4 lanes of an output column by DPP `quad_perm`, each lane owns 16 token columns (64 per quad), B from LDS; no LDS round trip for weights. Their efficiency model eta <= R/(R + D + L) (R columns reused per decoded weight, D decode ops, L activation loads) reproduces our measured 30 % (narrow) and 57 % (64-col) FMA peak exactly. | A | 20-40 % on expert GEMMs -> ~10-20 % prefill (their 80-88 % target is optimistic) | ~1 week | open (replaces the parked "large-tile rewrite") |
| 7 | **Fold in the prefill GEMM**: pf_mm still uses exact per-weight mul1 (`A*s + B` per weight); decode already folds (`JI_X3_FOLD=2`). | A | removes 1 FMA/weight in GEMM decode stage | 1 day | open - KL gate (decode fold cost KL 0.00014) |
| 8 | **Smarter sparse page scorer**: score pages mainly on the 192 non-rotary channels ([64:256), stationary) by centroid + spread; damp the 64 rotary channels whose min/max bounds RoPE oscillation inflates (Quest bound degenerates to an L1 bound there). If ranking improves, a smaller page budget may hold KL. | B | faster long-context decode / prefill IF a smaller budget holds | 3-5 days | open - KL-gated; our data: 16k-key budget = KL 0.0115 at 130k, so be skeptical |
| 9 | **Integer 6-bit QK in `attn_kvr6`**: `v_bfe_u32` codes x 12-bit pre-scaled query in `v_mad_i32_i24`, one float convert per token. | B | few % (QK loop ~20 % of `attn_kvr6`) | 2 days | low priority |
| 10 | **Page-list LDS cache in `attn_hm`** (as done for `attn_kvr6`, engine bae3129) | ours | few % of q8 sparse decode | hours | low priority (KVarN is default) |
| 11 | DPP / `ds_swizzle` instead of generic `subgroupShuffleXor` in the Hadamard stages (x3_prep / x3_post / FUSEP / FUSEQ) | A | few % of prep/post kernels | 1 day | low priority |

## Measured facts that bound these ideas

- `lab/cbdecode` (single RX 580): mcg, mul1 raw (fold), and the 24-bit-split variants of both all run at exactly
  293.1 G decodes/s; exact mul1 247, reference 188. Replacing the quarter-rate `v_mul_lo_u32` with two full-rate
  24-bit multiplies (report A's first claim) changed nothing **in that bench** - it saturates on something common.
  Hence action #1: measure inside the real `x3_mv` before trusting per-instruction accounting.
- mul1 raw ISA today: per weight ~1 `v_mul_lo_u32` (quarter rate), 2 `v_add_u32_sdwa` (byte sum), `v_cvt_f32_u32`,
  `v_mac_f32`, plus window extraction (`v_bfe_u32` / shifts).
- Decode budget (Sage, TP2, 100 W): ~3.6 ms dense + ~16 ms experts per token; both run `x3_mv`, so a 1.3-1.5x faster
  decode loop would be worth roughly +15-30 % decode t/s.
- Expert GEMM (prefill) ablation (`lab/mmbench`): 64-col tiles - FMA loop near peak, ~28 % overhead from staging /
  LDS round trips / barriers / Hadamard epilogue; narrow tiles LDS-bound (fixed by `pf_mv_id`).

## Report claims we checked and rejected

- "Absorb QK-RMSNorm gamma into Wq / Wk" (B): invalid - the norm is computed before gamma multiplies.
- "2,048-token sparse budget for existing (untrained) checkpoints" (B): contradicted by our tuning (16k keys = KL
  0.0115 at 130k; default max(64k, 0.5 x context)). 2k only works for models trained with sparse attention (QSA).
- Partial-RoPE / Hadamard pathologies, V-rotation vs output gate, tile split at the RoPE boundary (B): apply to
  Hadamard-rotated KV formats; our KVarN-6 does no rotation (per-tile Sinkhorn with per-channel and per-token
  scales) and measures at q8 quality (KL 0.00136 vs 0.00123 at 130k). Not needed unless quality ever regresses.
- "~80 ms/token of KV reads at 262k" (B): we use sparse decode; not applicable.

## Hardware / architecture notes worth keeping

- gfx803 has no `v_dot2/4`, no packed FP16, no matrix units; full-rate tools: `v_mul_u32_u24`, `v_mad_u32_u24`,
  `v_mad_i32_i24`, `v_bfe_u32`, `v_alignbit_b32` (funnel shift), `v_perm_b32`, `v_sad_u8`, SDWA byte/word selects,
  DPP (`quad_perm`, `row_shr/ror`, `row_mirror`), `ds_swizzle_b32`. 32-bit `v_mul_lo_u32` is quarter rate.
- 64 VGPRs per lane = 4 waves/SIMD; our measured cliffs: pf_mv_id 140 VGPR (1 wave) slow, 84 VGPR fine.
- Qwen gated attention (sigmoid output gate) collapses attention sinks (first-token mass ~47 % -> ~5 % per report B):
  sink-based KV / sparse heuristics are unreliable on Qwen3.5+; keep the exact local window and avoid sink-only
  shortcuts.

## Two-card exchange and P2P (report C + AUX-1 topology, 2026-10-31)

AUX-1 facts: card A 06:00.0 on a CPU root port (PCIe 3.0 x16); card B 05:00.0 behind the chipset (bridge shared
with the Ethernet NIC), PCIe 2.0 x4. Both cards have resizable BAR at 8 GB (full VRAM mappable); kernel has
CONFIG_PCI_P2PDMA=y; CPU Ryzen 5 2600 (Zen 1).

- **P2P is probably possible** (export a VRAM buffer as dma-buf from one device, import on the other; amdgpu uses a
  real peer mapping when pci_p2pdma_distance allows it, else silently migrates the buffer to GTT). **It buys little
  here:** every byte between the cards still crosses card B's PCIe 2.0 x4 link, host-staged or P2P. Prefill exchange
  (~1.4 s per 4096-token batch, 24-bit wire) is bandwidth on that link - unchanged by P2P. Decode syncs are latency:
  P2P would let the reader poll its own VRAM instead of host memory over PCIe, maybe a few us per sync, ~0.1-0.3 ms
  per token. Not worth the driver risk (hang-prone paths, see aux1-tp2-kill-freeze).
- **The real fix is lanes:** card B on CPU lanes (x8/x8 bifurcation on a board that supports it, PCIe 3.0) gives ~5x
  the bandwidth: prefill exchange ~1.4 s -> ~0.3 s per batch (est. +15-20 % two-card prefill) and lower decode sync
  latency. A 6650 XT pair would still be limited by the same x4 slot.
- Report C's playbook (single pre-recorded command buffer per step, exchange through pinned host memory with
  coherent / uncached mailbox + in-shader spin-polling, asymmetric split, compact wire formats) is essentially JDE's
  existing design (replayed command buffers, host exchange area, JI_TP_FRAC 0.7, 24-bit / q8 prefill wire).
- Small checks worth doing: (a) measure decode TP2 sync overhead per token (JI_TP_PROF) - report C's target is < 5 ms
  total; if ours is ~1-2 ms there is nothing to win; (b) q8 wire for MTP verify steps (K = 2-4 tokens) if those
  exchanges show up in the profile; (c) full-duplex overlap of the two directions on card B's link; (d) PCIe Max
  Payload Size 256 B on card B's path (lspci -vv DevCtl MaxPayload); (e) host-side waits: /dev/cpu_dma_latency = 0
  while serving if any CPU thread is on the per-layer path.

## Qwen4 readiness (report B; post-cutoff, unverified - check against the real release)

Reported for Qwen3.8-Flash-Next (the "Qwen4 preview") and expected in Qwen4:
- **QSA** (Qwen Sparse Attention) in the 1-in-4 attention layers: an MQA indexer (4 query heads, 1 key head, dim 128,
  over 4-token compressed micro-blocks, "c4") selects top 512 micro-blocks = 2,048 tokens per query. Needs: an indexer
  key pool (~100 MB at 262k in int8), a 4-token pending ring, micro-block-addressable KV. KVarN tiles (128 tokens,
  per-token scales) can serve 4-token micro-blocks without reading the whole tile.
- **N-gram ("Engram") embedding** table in host RAM, gathered per token by hashed n-grams, injected at layer 2: fits
  our host-RAM design, but Flash-Next's table is ~51B params - far beyond AUX's 32 GB RAM; the size of the 27B / 35B
  tiers' tables decides feasibility.
- **4-branch gated residual** (hyper-connections): wider residual stream, fused mix / combine kernels, int8 storage
  suggested on GCN.
- **MTP with IndexShare** (draft steps reuse the target's selected blocks; <= 4 draft tokens).
- None of this changes dense-27B prefill FLOPs (FFN-bound): QSA makes long-context attention cheap, not the FFN.
  An MoE tier (e.g. a 35B-A3B successor) remains the right target for this box.
