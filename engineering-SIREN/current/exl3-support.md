# EXL3 support — design

Status: ACTIVE, 2026-10-16. Milestone 1 (runs at all, graph path) DONE - see engineering-log 2026-10-16 evening. Goal (user): get EXL3-class quants of qwen35moe running in JunkInfer at all; then fast.
First model: community `UnstableLlama/Qwen3.6-35B-A3B-exl3-4.00bpw` (exllamav3 0.0.29, codebook mcg, 4 bpw,
LM head 6 bpw), on AUX-1 at `~/models/exl3/Qwen3.6-35B-A3B-exl3-4.00bpw`. (Making our own EXL3 quants and
the KLD bars wait until VENE's GPU is available again — user, 2026-10-16.)

## The format (verified)

Per linear (K inputs, N outputs): `trellis` int16 [K/16, N/16, 16*bits] (one 16x16 tile = 256 weights), `suh` fp16
[K], `svh` fp16 [N], `mcg` multiplier (0xCBAC1FED). Decode of tile position p: the 16-bit window ending at bit
(p+257)*bits of the tile's circular bitstream (MSB-first in uint32 words); value = mcg(window) =
fp16(lo) + fp16(hi) of ((window * mult) & 0x8fff8fff) ^ 0x3b603b60. Position -> (row, col) in the tile follows the
tensor-core fragment shuffle of exllamav3's reconstruct_tile. Original weight W (K x N) =
diag(suh) . H . W_hat . H . diag(svh), H = blockwise Sylvester Hadamard(128)/sqrt(128); so
y = H128( H128(x * suh) . W_hat ) * svh. Reference: `tools/exl3/exl3_ref.py` - cosine 0.997 vs the IQ4_NL
GGUF's weights (both ~4.5-bit approximations of the same BF16 tensor).

Non-EXL3 tensors: embeddings/norms bf16; in_proj_a/b, router (mlp.gate), shared_expert_gate fp16; A_log,
dt_bias, conv1d plain. No MTP layer, no vision use.

## Plan (milestone 1: running at all)

1. **GGUF converter** (`tools/exl3/exl3_to_gguf.py`): metadata from the existing qwen35moe GGUF (no MTP),
   llama.cpp's tensor transforms (A_log -> -exp, dt_bias rename, conv1d squeeze, RMSNorm +1). GDN stays in HF's
   *grouped* V-head order ... SUPERSEDED: heads are 128 wide = one Hadamard block, so whole heads are moved and
   the GGUF uses llama.cpp's tiled order like every other qwen35moe GGUF. EXL3 linears -> new ggml types EXL3_4 / EXL3_6
   (block 256 weights = 32*bits bytes; data kept in exllamav3's tile order; "rows" are nominal), plus F16
   `<name>.suh` / `.svh` tensors (experts: [E, K] / [E, N]).
2. **ggml**: register the types (sizes only; no generic kernels). The graph path never computes them.
3. **JDE**: EXL3 matvec (trellis decode in GLSL; x -> H(x*suh) fused into the input stage, H(y)*svh fused into
   128-row output groups), EXL3 expert slots + mv_moe, CPU misses (scalar first, then AVX2), grouped GDN,
   zeroing of a fresh sequence's recurrent state, all ubatches <= 4 tokens (prompts too, slow prefill at first).
4. Correctness: KL vs the BF16 reference dumps (copied from VENE), then speed.

Later: a prefill path (dequant-to-f16 Vulkan op for the graph, or JDE batched kernels), 3.5 bpw quants.
