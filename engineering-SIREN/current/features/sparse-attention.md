# Sparse attention (decode + prefill)

Status: **on by default**.
- Decode: budget = max(64k keys, 50 % of the context), mass-estimate ranking, from 32k cells.
- Prefill: budget = max(32k keys, 50 % of the row's keys), mass-estimate ranking, from 32k keys, with the prompt's
  last 4096 positions dense.

Defaults set 2026-10-24 (engine daa518a65 + defaults commit), measured on the corrected v2 references. History:
engineering log 2026-10-18 .. 2026-10-24. Earlier commits: 4f1013c74 (tail estimate), 7e30faaac (mean bound),
7a6b79f2f (dense prompt tail), raw-basis statistics, 63d680059 (TP2 metadata slot fix).

> **All quality numbers in this doc before 2026-10-23 were wrong.** The KL references (and cached prefix states)
> were built on the ggml graph path, which with 3.75 GiB Vulkan blocks lost layer 3's KV writes for whole 4096-token
> ubatches beyond ~8k. The references were missing the same context the sparse path approximated, so sparse looked
> near-lossless (0.0003) when it was 0.003-0.018. v2 references: ~/jt/*_v2.jref, prefix states jref_v2/ (deep rows
> built with the JDE prefill). See engineering log 2026-10-23 (afternoon).

## Why

Qwen3.6-35B-A3B has 10 full-attention layers (3, 7, ..., 39) between 30 GatedDeltaNet layers. At 200k+ context,
dense decode attention reads ~2.85 GB of q8_0 K/V per token (~10 ms of a ~39 ms token on two cards at 100 W); dense
prefill attention dominates long prompts.

**Property that shapes everything: this model's global attention is broad.** The recurrent layers carry local
context, so the attention layers spread their weight across the context. Measured consequences:
- Quality depends on the fraction of the context read exactly, not on clever ranking. An oracle that ranks pages
  by their exact attention mass is at most ~1.7x better than our estimate at small budgets, and no better at 64k keys.
- Every attention layer contributes about equally to the error (single-layer dense ablation).
- The estimated "tail" (unselected pages) helps only a little: both its mass and its value matter, and per-page
  mean summaries carry little information.

## Decode path (JDE, per attention layer)

**Pages and metadata.** Pages are 32 cells. When a page completes, `attn_meta` stores per page and KV head
(dim-major f16, ~1.5 KB; 241 MiB at 262k):
- mean K, per-dim radius, per-dim variance, mean V, count;
- statistics in the **raw basis**: keys un-rotated (FWHT) from llama.cpp's H_256 rotation, so the RoPE pairs stay
  unmixed. This made the estimates ~3x better.

**Per step:**
1. `attn_qk_q8` writes the query in the stored basis (for exact scores) and in the raw basis (qr, for statistics).
2. `attn_score` computes a per-page estimated log-mass: log n + q.mu/16 + q^2.var/512. With `JI_SP_SEL=1`
   (default) this is the ranking; the alternative is the bound q.mu + |q|.r. Oracle mode `JI_SP_ORACLE=2` ranks by
   the exact page mass. It uses its own stored-basis query binding; before 2026-10-24 it scored with the raw query,
   so oracle results from 10-19..10-21 are void.
3. `attn_topk` keeps the budget's pages, plus sink and recent pages, per KV head (shared by its 8 query heads).
4. `attn_hm` runs exact attention over the selected pages.
5. `attn_bg` adds each unselected page as estimated mass x mean V.
6. `attn_merge` combines everything with online-softmax partials.

**Budget:** `prm[18] = max(JI_SPARSE, JI_SPARSE_FRAC x n_kv) / 32`, capped at the page count.

**TP2:** each GPU owns one KV head (metadata slot = layer x 2 + head).

## Prefill path

`pf_fa_sparse` (JDE TP prefill) is the FA2 core with sparse selection.
- Per workgroup of positions: 128-key pages scored from 32-key sub-page statistics (`pf_pmeta`).
  - `JI_SPF_SEL=1` (default) scores by estimated log-mass: log-sum-exp over sub-pages of q.mu s + q^2.var s^2 / 2.
  - The alternative is the bound around the mean.
- Selection: radix select of the budget, plus sink pages and the local window (`JI_SPF_LOCAL`, 4096).
- The block list is 16-bit and packed (1280 entries), so budgets up to ~1200 pages fit in the same LDS.
- Exact attention over the selected blocks; tail estimate per sub-page.
- The prompt's last 4096 positions are dense (`JI_SPF_DENSE_TAIL`).
- Budget: `max(JI_SPARSE_PF, JI_SPARSE_PF_FRAC x keys)`.

## Measured (v2 references; KL vs our dense full-attention output, code text, last 1024 positions)

For scale: q8_0 KV vs f16 KV costs ~0.001-0.0015.

**Decode only** (prefix dense, from restored states):

| config | 64k | 100k | 130k | 261k | decode t/s at 200k (2 GPUs, 1024 tok) |
|---|---|---|---|---|---|
| dense | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 25.6 |
| old default: 16k keys, bound ranking | 0.0028 | 0.0066 | 0.0115 | 0.0177 | 31.5-34.4 |
| 64k keys, bound | 0.00015 | 0.0014 | 0.0009 | 0.0036 | 31.2-31.7 |
| **default: SEL=1 + FRAC 0.5** | 0.0001 | 0.0009 | 0.0008 | 0.0011 | **32.4** |
| FRAC 0.7 (bound) | - | 0.0009 | 0.0009 | 0.00074 | - |

Single card at 200k: dense 20.7-20.9, default 23.7-25.1 t/s.

**Prefill only** (prefix prefilled from scratch, tail decoded densely):

| config | 130k | 261k | prefill t/s, 151k prompt (2 GPUs) |
|---|---|---|---|
| dense | 0.0001 | 0.0002 | 224 |
| old default: 8k keys | 0.0051 | - | 464 |
| 32k + SEL | 0.0015 | - | 336 |
| 64k + SEL | 0.0007 | 0.0018 | 261 |
| **default: 32k + FRAC 0.5 + SEL** | 0.0009 | 0.0014 | **288-289** |
| 32k + FRAC 0.7 + SEL | - | 0.00085 | 251 |

**End to end** (sparse prefill + sparse decode, defaults): 64k 0.0010, 100k 0.0010, 130k 0.00125, 261k 0.00165 -
roughly the size of the q8_0 KV error itself.

**Ranking ceiling** (fixed budgets; exact-mass oracle vs our estimate, 130k / 261k):

| budget | oracle | ours |
|---|---|---|
| 16k | 0.0026 / 0.0047 | 0.0044 / 0.0080 |
| 32k | 0.0014 / 0.0044 | 0.0023 / 0.0042 |
| 64k | 0.0012 / 0.0014 | 0.0009 / 0.0022 |

## Rejected (measured on v2)

- Variance-free ("monopole") tail.
- Tail variants: no tail, exact page mass with mean V (worse), estimated mass with exact page values (~same).
- Per-layer budgets: the error is spread evenly.
- Uniform 4-bit two-tier tail: it would read ~76 % of the dense bytes.
- Prefill tail modes 0 / 2: ~same.
- Layers 3/7/11 dense with sparse elsewhere: only -25 % KL.

## Known issues

- TP2 JDE prefill: intermittent ring timeout on the exchange queues (2 in ~10 h; open-paths S10). With
  `amdgpu.gpu_recovery=0` the box stays up, but the hung card needs a cold power cycle.
- The ggml graph path (non-JDE prefill) loses layer-3 KV writes with large Vulkan blocks: never use it for
  references.

## Knobs

- Decode: `JI_SPARSE` (64k floor), `JI_SPARSE_FRAC` (0.5), `JI_SP_SEL` (1), `JI_SP_MIN`, `JI_SP_BG`, `JI_SP_RAW`,
  `JI_SP_ORACLE` (diagnostic).
- Prefill: `JI_SPARSE_PF` (32768), `JI_SPARSE_PF_FRAC` (0.5), `JI_SPF_SEL` (1), `JI_SPF_LOCAL`, `JI_SPF_MODE`,
  `JI_SPF_DENSE_TAIL`.
- Quality-first setting: FRAC 0.7 on both.
- References: `JI_SPARSE=0 JI_SPARSE_PF=0`, JDE prefill, 1 GiB Vulkan blocks.

## Next (post-MVP; open-paths "sparse rework" / "post-MVP track")

Training-free first:
1. 8-token sub-centroids replacing the page summary (M1) and micro-block selection (M2). Finer selection and
   query-dependent tail values in the same ~1.5 KB. Breaks the page-oracle ceiling.
2. Value skipping inside the selected set (M3).
3. Value-aware ranking (L9) and pre-RoPE statistics for the RoPE dims (L10).

Then the data-efficient learned corrector:
4. L7/L8: a dump hook, ridge recalibration, an offline CPU proxy, and a zero-init residual MLP trained on CPU. Its
   first iteration needs ~4 h of prefill.

Throughput:
5. T0: a per-kernel profile first. Decode at 200k is capped near ~37 t/s by non-attention work.

## Porting notes

- The broadness is a property of hybrid (linear + softmax) models. Measure the oracle ceiling before investing in
  ranking. On a dense transformer, attention is usually much sparser.
- Keep references on a known-good path and check them for zero rows (this cost two weeks of misleading numbers).
- Statistics must be taken in a basis where RoPE pairs are unmixed (un-rotate llama.cpp's H_256).
