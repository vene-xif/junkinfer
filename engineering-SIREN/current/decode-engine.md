# Decode engine (JDE) — design

Status: ACTIVE. 2026-10-16. Supersedes "tune ggml decode" as the decode plan (see engineering-log, 2026-10-16).

## Why

Measured on AUX-1 (engineering-log 2026-10-15/16): ggml decode spends ~23-32 ms of GPU time per token on ~1.4 GB
of dense weights (should be ~6 ms at the roofline), because (1) its gfx803 matvecs reach ~108 GB/s on large
and 10-60 GB/s on small shapes, (2) ~1,500 dependent dispatches per token each pay a ~7-10 us bubble, and
(3) a strict GPU -> CPU -> GPU round trip per MoE layer prevents CPU/GPU overlap. A GCN-shaped matvec with
an LDS codebook reaches 225-229 GB/s (89% of peak) in `lab/matvec`.

## Scope

Decode (ubatches of 1..~8 tokens: single-token decode and MTP verify windows) for `qwen35moe` on one Vulkan
GPU (gfx803 first). Prefill, the server, templates, tool parsing, prefix caching and sampling stay in llama.cpp.
The engine runs inside the llama.cpp fork: `llama_context` hands decode ubatches to JDE when `JI_DECODE=1`.

## Shared state with llama.cpp (no copies)

- Weights: the GGUF tensors as loaded by llama.cpp (dense in Vulkan buffers via `-ngl`, experts in host
  memory via `--cpu-moe`). JDE kernels read the **native GGUF block layouts** (IQ4_NL 18 B/32, Q8_0 34 B/32,
  Q5_K, Q6_K, F32) by staging whole rows into LDS with 16-byte loads (rows are 16-byte aligned), then
  decoding from LDS. No repacked copy: VRAM is the scarcest resource.
- Expert cache: the hybrid `llama_moe_cache` bank and slot maps (native layout, one zero slot).
- KV cache and recurrent state: llama.cpp's `k_l`/`v_l` and conv/ssm state tensors. Cell allocation and the
  attention mask come from llama.cpp's memory module (find_slot per ubatch, mask built on the host).
- Outputs: logits written into llama.cpp's output buffer.
- Device/queue: obtained from ggml-vulkan through a small accessor API added to the fork. JDE submits only
  when ggml's queue is idle (after `ggml_backend_synchronize`).

## Execution model (v1)

One pre-recorded command buffer per (window size, KV-length bucket), replayed every token. Per-token values
(positions, KV cell, KV length) live in a small device buffer that the shaders read, so the command buffer
does not change.

Per MoE layer, the CPU half runs concurrently through a **doorbell**:
1. GPU: norm, mixer (GDN or attention), residual, post-norm, router, top-8.
2. GPU writes the routing (ids, weights) and the normed hidden state to host-visible memory, then a flag.
3. GPU continues: shared expert + the routed experts present in the cache (zero slot for misses).
4. CPU (spinning on the flag) computes the missed experts from host RAM, writes the partial sum to
   host-visible memory and sets a Vulkan event (`vkSetEvent`).
5. GPU: `vkCmdWaitEvents` on that event, adds the CPU partial, residual, next layer.
The GPU never idles waiting for the host to *start* work; the host never pays a queue drain.

## Kernels (gfx803, wave64)

| kernel | notes |
|---|---|
| matvec (IQ4_NL / Q8_0 / Q5_K / Q6_K / F32) | rows staged in LDS, codebook in LDS, x in registers, subgroup reductions; epilogues: +residual, swiglu, *weight |
| matvec-id (experts) | the 8 selected experts in one dispatch, slot lookup from the slot map, zero slot skipped by flag |
| rmsnorm (+weight), l2norm per head | fused into the next matvec where possible |
| GDN step | conv1d(k=4)+silu, l2norm q/k, gates, gated delta rule on the 32x128x128 f32 state, norm*silu(z) |
| attention decode | q/k norm, partial RoPE (64 of 256 dims, MRoPE sections), KV write, split-K over the KV length |
| router + top-8 + norm | softmax, top-k, normalise, doorbell write |
| lm_head + sampling support | Q6_K 248,320 rows; logits to host |

## Milestones (each gated: per-op comparison against llama.cpp, then KLD vs the BF16 reference)

1. Accessor API in ggml-vulkan; JDE context; one kernel (rmsnorm) replayed; compared with ggml.
2. Native-layout matvec kernels for every type the model uses, verified per tensor; bandwidth per shape.
3. Full decode with ALL experts on the CPU (no cache), serial doorbell. Logits match llama.cpp (KLD ~ floor).
4. Hybrid: cached experts on the GPU + CPU misses concurrently (doorbell + event). Hit-rate, overlap measured.
5. Replayed command buffers, KV-length buckets, MTP windows (2-4 tokens).
6. Fusion / persistent-kernel work where the profile says the bubbles still dominate.

## Expected (to be measured, not claimed)

Dense weights ~6 ms + cached experts ~2 ms + GDN state ~0.5 ms + ~500 dispatch bubbles ~4 ms; CPU misses
(~80 experts, ~140 MB at ~22 GB/s ~ 6 ms) overlapped. ~12-16 ms/token -> 60-80 t/s at short depth, before MTP.

## The decode step, exactly (from llama.cpp `src/models/qwen35moe.cpp`, `delta-net-base.cpp`, ggml-cpu `gated_delta_net`)

Token t at position p. eps = 1e-6. `rms(x, w) = x / sqrt(mean(x^2) + eps) * w`.

```
x = dequant(tok_embd[t])                                    (tok_embd stays in host memory)
for il in 0..39:
  h = rms(x, attn_norm[il])
  if il % 4 != 3:                                           # Gated DeltaNet (30 layers)
    qkv = W_qkv h   (8192)  ;  z = W_gate h (4096)
    beta = sigmoid(W_beta h) (32)  ;  g = softplus(W_alpha h + dt_bias) * ssm_a     (32, ssm_a < 0)
    conv: window = [state_c (3 x 8192, oldest first), qkv]; out[c] = sum_k w[c*4 + k] * window[k][c]
          new state_c = last 3 columns of window; u = silu(out)
    q = u[0:2048], k = u[2048:4096] as 16 heads x 128, each l2-normalised: x / sqrt(sum x^2 + eps)
    v = u[4096:8192] as 32 heads x 128
    for v-head hv in 0..31:  hk = hv % 16                   # NOT hv / 2
      S = state_s[hv] (stored transposed: s[j*128 + i] = S[i][j])
      S *= exp(g[hv]);  d[j] = (v[j] - sum_i S[i][j] k[i]) * beta[hv];  S[i][j] += k[i] d[j]
      o[j] = sum_i S[i][j] q[i] / sqrt(128)
    o = rms(o per head, ssm_norm) * silu(z per head)
    y = W_ssm_out o (2048)
  else:                                                     # gated attention (10 layers)
    qg = W_q h (8192): head i = [q (256) | gate (256)] at qg[i*512 ...]
    k = W_k h, v = W_v h (2 heads x 256)
    q = rms(q per head, q_norm); k = rms(k per head, k_norm)
    IMRoPE on the first 64 of 256 dims (text: all sections share p), freq_base 1e7
    write k, v to the KV cell; attention per q head i with kv head i / 8, scale 1/16, mask from llama.cpp
    y = W_o (attn * sigmoid(gate))
  x = x + y ; r = x
  hn = rms(x, post_norm)
  p = softmax(W_router hn) (256) ; top-8 ids ; w = p[ids] / sum(p[ids])   (* expert_weights_scale if set)
  moe = sum_e w_e * down_e(silu(gate_e hn) * up_e hn)
  sh  = down_s(silu(gate_s hn) * up_s hn) * sigmoid(w_gs . hn)
  x = r + moe + sh
logits = W_out rms(x, output_norm)
```
