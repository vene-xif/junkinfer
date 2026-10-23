# Sparse attention (decode + prefill)

Status: **on by default** (decode from 96k cells, prefill from 32k keys). Engine commits: 4f1013c74 (tail
estimate), 7e30faaac (mean bound, 241 MiB), 7a6b79f2f (dense prompt tail), raw-basis statistics (2026-10-19),
2767f7ada (prefill default), 63d680059 (TP2 metadata slot fix). History: engineering log 2026-10-18 .. 2026-10-21.

## Why

Qwen3.6-35B-A3B has 10 full-attention layers (3, 7, 11, ... 39) between GatedDeltaNet layers. At 245k cells, dense
decode attention reads ~2.85 GB of q8_0 K/V per token: ~20 ms of a ~48 ms token on an RX 580. Sparse attention
reads a fixed budget of keys exactly and *estimates* the rest from per-page statistics, so its cost stops growing
with context.

The model's global layers attend broadly. Dropping the pages that aren't selected (Quest) fails the quality gate
even with oracle selection: 1.4 % of the mass is lost at 128k. **The win is the tail estimate**, not the selection.

## Decode path (JDE, per attention layer)

Data: K/V are q8_0, head-major (`JI_KV_HM`), with llama.cpp's H_256 rotation applied to K. Pages are 32 cells.
Per page and kv head, `attn_meta` stores (dim-major, f16, 1.5 KB per page and head; 241 MiB at 262k context):

- mean K, per-dim radius r = max |k − μ|, per-dim variance of K (`JI_SP_BG=2`), mean V, cell count.
- **Raw basis** (`JI_SP_RAW=1`): each page's 32 keys are un-rotated (FWHT in LDS) before the statistics are taken.
  The H_256 rotation mixes the 64 RoPE dims into all 256. Per-dim variance in the stored basis can't follow RoPE,
  which made the page-mass estimate 70–80 % of the error. The raw basis cut decode KL by 71 %.
- Metadata is rebuilt only when a page completes (decode), or for the dirty page range after a prefill or restore
  (`jde_*_meta_rebuild`, ~60 ms at 36k, ~530 ms at 200k).

Kernels per step:

1. `attn_qk_q8` writes the query in both bases: the stored-basis query for exact scores, and the pre-H_256 query
   (qr) for the statistics.
2. `attn_score`: a page's upper bound is q·μ + |q|·r (a valid bound, tighter than Quest's centre/radius). It also
   writes a per-page log-mass estimate: log n + q·μK + ½ q²·varK.
3. `attn_topk`: radix select of the budget (`JI_SPARSE`, default 16384 keys = 512 pages), plus sink pages
   (`JI_SP_SINK`) and recent pages (`JI_SP_RECENT`). Selection is shared per kv head (GQA: 8 query heads read the same
   pages). Keys are cached in padded LDS, with a parallel suffix scan.
4. `attn_hm` runs exact attention over the selected pages, writing chunk partials (max, sum, acc).
5. `attn_bg` estimates each unselected page as mass n·exp(q·μK + q²·varK/2) with value μV, written as extra chunk
   partials. Pages are compacted by a ballot prefix.
6. `attn_merge` combines all partials (online-softmax merge).

Cost at 245k: ~4.2 ms per token for attention vs ~20 ms dense. Decode 245k went from 20.5 to 29.6 t/s (+44 %, single
card, 2026-10-19). Today, with TP2 and the 100 W cap, it is ~38 t/s at 247k.

TP2: each GPU handles one kv head; the helper keeps its own metadata for head 1. Metadata slot = layer × 2 + head.
A slot-indexing bug (head 0 read slot = layer) made all earlier TP2 depth KL numbers invalid; fixed in 63d680059.

## Prefill path

`ji_fa2_sparse` / `pf_fa_sparse` (the JDE TP prefill) is the FA2 core plus sparse selection:

- Per workgroup of 4 positions: the causal boundary comes from a binary search on the mask (or from positions in
  JDE).
- 128-key pages before the local window (`JI_SPF_LOCAL`, 4096) are scored by the bound around the mean (max over
  32-key sub-pages, rows and heads). A radix select takes the budget (`JI_SPARSE_PF`, 8192) plus sink pages.
- Exact attention covers the selected pages and the local window. The rest gets a tail estimate per 32-key sub-page
  (`ji_fa2_pmeta` / `pf_pmeta`), merged into the online softmax.
- **Dense prompt tail** (`JI_SPF_DENSE_TAIL`, 4096): the prompt's last 4096 positions attend densely, because they
  determine the first generated tokens. That cut prefill KL at 130k from 0.00123 to 0.00035 for −5.8 % prefill speed.
  The prompt end is passed explicitly (`llama_ji_set_prompt_end`), because the server sends the last 4 tokens as a
  separate batch.
- Falls back to dense when the mask isn't a causal prefix. On by default from 32k keys (`JI_SPF_MIN`).

Prefill gains: 79.6k 264 → 358 t/s (+36 %); 262k ~2.4× faster; JDE TP2 247k+7.5k 98 → 213 t/s.

## Quality (KL vs our own full-attention reference: EXL3 4.0 bpw, q8_0 KV)

| row | decode KL (default) | notes |
|---|---|---|
| 64k | 0.00034 | the old stored-basis statistics failed here (0.0023) |
| 99k / 130k / 245k | 0.00036 / 0.00033 / 0.00029 | exact-page-mass floor 0.00038 / 0.00019 / 0.00028 |
| 130k, prefill + decode both sparse | 0.00045 | |

For scale: q8_0 KV vs f16 is ~0.0015. Errors from sparse prefill and sparse decode add roughly linearly.

## Rejected (measured)

- Quest page drop without a tail; per-query-head selection (8× the reads).
- `JI_SP_SEL=1` (rank pages by estimated mass).
- `JI_SP_NORM=2` (−3 % KL for −4 % speed).
- `JI_SP_BG=3` (dipole term).
- Raw basis for the *prefill* statistics: KL 0.00035 → 0.00026, but prefill −15 %, because the dense core's code
  generation changed. Patch kept in the scratchpad.
- Larger budgets (KL is flat in the budget above 8192 at 262k).
- Dynamic expert top-k.
- Per-layer budgets: late layers at 8192 gain +2–3 % at noise-level KL; still a default candidate.

## Knobs

`JI_SPARSE` (budget keys, 0 = off), `JI_SP_MIN` (98304), `JI_SP_BG` (2), `JI_SP_RAW` (1), `JI_SP_SINK`,
`JI_SP_RECENT`, `JI_SP_LBUDGET`, `JI_SP_DENSE_LAYERS`; prefill: `JI_SPARSE_PF` (8192), `JI_SPF_MIN` (32768),
`JI_SPF_LOCAL` (4096), `JI_SPF_DENSE_TAIL` (4096), `JI_SPF_TAIL_LAYERS`. Reference runs must set `JI_SPARSE=0
JI_SPARSE_PF=0`.

## Porting notes (RDNA2 / other engines)

- Nothing here depends on GCN. It needs per-page statistics in a basis where RoPE pairs stay un-mixed, and a merge
  of partial-softmax chunks. Any engine that rotates K before quantizing (llama.cpp's H_256) must un-rotate for the
  statistics.
- On a card with more VRAM bandwidth, the dense/sparse crossover moves to a deeper context. Re-measure `JI_SP_MIN`.
- Metadata VRAM (241 MiB at 262k) competes with the expert cache. On a 16 GB card that no longer binds.
