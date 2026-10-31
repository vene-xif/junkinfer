# KVarN-6 KV cache (in place)

Status: **opt-in (`JI_KVARN=1`), quality gate pending**. Engine: 356031b68 (simulation), 21a180646 (`attn_kvr6`),
ac2eed747 (integration), plus 2026-10-23 fixes (rings at init, state-hook view tensors, attach before state load).
History: engineering log 2026-10-18 (late night), 2026-10-23, 2026-10-31 (from-scratch defect confirmed through the JDE prefill: 130k KL 0.118).

## What it is

KVarN (BeeLlama; arXiv 2606.03458) is "dual-axis variance normalisation". Each 128-token × 128-channel slice of K or
V is balanced Sinkhorn-style: row and column standard deviations are equalised alternately. The result is then
coded with per-row affine 6-bit codes. One cell of one kv head takes 52 words (208 B) instead of q8_0's 68 words
(272 B), −23.5 %.

Quality (simulation, L_code 32k, KL vs **f16** KV): q8_0 0.00090, KVarN6 0.00099 (with two extra q8 roundings the real
format doesn't have), KVarN5 0.00138, KVarN4 0.00224. The simulation passed the q8_0 bar; the real-format gate is
below.

## Layout: records in place of q8_0, plus an exact window

The llama.cpp K/V tensors stay allocated as q8_0 (head-major, `kv_size × 68` words per head). With KVarN on, they
hold **records**: per kv head, tile t (cells 128t .. 128t+127) at word t × 6656:

- codes: 128 rows × 48 words. Slice s (channels 128s ..) at words 24s ..; channel c of the slice at bit 6c.
- axes: 512 words at +6144.
  - K: `Ak f32[256] | Bk f16[256] | Ck f16[2][128]`, with K[t][c] = Ck[s][t] · (Ak[c]·x + Bk[c])
  - V: `Av f16[2][128] | Bv f16[2][128] | Cv f32[256]`, with V[t][c] = Cv[c] · (Av[s][t]·x + Bv[s][t])

The newest cells live in an exact q8_0 **window ring** in the top `kvr_cw` (12544) cells of each kv-head region of the layer's own K / V tensors ("the tail"), because KVarN records use 208 B/cell instead of q8_0's 272 and never reach that region when kv_size >= ~4.25 × the ring size (~53k cells). Below that context size (or with `JI_KVR_TAIL=0`), the rings are separate buffers (`kvr_wk` / `kvr_wv`, 130 MiB for 10 layers). Rings are allocated at engine init, *before* the expert cache sizes itself from free VRAM. When they were allocated lazily as separate buffers, VRAM was over-committed, ~1.4 GB of buffers were evicted to GTT, and decode ran 3–4× slower (engine commit 4f5d347ee, rings-in-tail optimization).

**Sealing** (`kvr_seal`): once the context passes 8192 cells, after every step, all complete tiles below n − 4
(`KVR_ROLLBACK`, so speculative rollback never touches a record) are quantised from the ring into records. Each
workgroup takes one tile × kv head × slice and kind, and runs `JI_KVR_ITERS` (16) balancing iterations in LDS
(`mat[128][129]` f16, odd stride). `kvr_S` is the sealed boundary. Cells below S are records, cells in [S, n) are
in the ring.

**Unsealing** (`kvr_mat` into the ring): when a step writes below S (prompt truncated or reused), the tiles from
that point onward are expanded back into the ring, which is lossy (their dequantised values). If the new length is
≤ 8192, everything goes back. Logged as `KVarN: cells a..b back into the window`.

Where the VRAM goes: records use 52 of every 68 words. The KV-tail elastic expert slots, carved from the top of the
K/V tensors, see the context as 208/272 of its q8 size (`c_lo × 272 / 208`). The saving becomes expert-cache
capacity at depth.

## Decode attention

- `attn_kvr6q`: per sealed tile, kv head and slice, computes q · Bk (the K offset's dot product, a per-tile constant
  of the scores) into `bqt`.
- `attn_kvr6`: records only (slots 0..R−1). It unpacks 6-bit codes, computes score = Ck[s][t] · (Σ_c q[c]·Ak[c]·x[c]
  + bqt), and accumulates V with its per-token axes. Prefetching QK loop; `ABL` spec constant; bindings 0..11.
  Kernel time at 116k dense: ~1346 µs vs `attn_hm` (q8_0) 1128 µs (+19 %). On a full token this is hidden: the KV
  bytes saved go to the expert cache.
- `attn_hm` window mode: cells [S, n_kv) from the ring (`ring` push constant), slots R..R+31.
- `attn_bg` (sparse tail estimate) takes the last slots. `attn_merge` reads 256 slots (`slots256`).
- Sparse page metadata is rebuilt from a q8 copy (records → `kvr_mat` → temp, plus window rows).

Rejected on the way: a combined `attn_hmr` (LDS > 32 KB dropped occupancy to 1 workgroup per CU, 1887–2513 µs);
an unrolled QK loop (406 SGPR spills).

## Prefill

Prefill kernels read q8_0. If nothing is sealed and the batch fits the ring, prefill writes the ring directly.
Otherwise, per layer at micro-batch 0, records plus window are **materialised** into a q8 temp of p0 + T rows
(`kvr_mat`, in the compute buffer above the scratch, else `kvr_pft`). The existing prefill kernels run on the temp.
After the last micro-batch's attention, the new rows are copied out to the ring. Seal and metadata rebuild run at
the end of the prefill.

## TP2

The helper GPU keeps kv head 1 as a q8 mirror (`tp_kvm`). Before sealing, rows the helper wrote are pulled back
(`jde_kvr_head1_back`), and rows its mirror lacks are pushed to it with a metadata rebuild. Restore writes the
mirror straight from the host rows.

## State save / restore (prompt cache, slot files)

`llama_kv_cache::hm_write_rows` / `hm_read_rows` call `llama_kv_hm_get_hook` / `llama_kv_hm_set_hook` per kv head.

- Save (`jde_kvr_hm_get`) materialises q8 rows (records → `kvr_mat`, window → copy) to a host stage, so slot files
  stay plain q8_0 and portable.
- Restore (`jde_kvr_hm_set`) ingests rows from cell 0: seal [0, S), the rest into the ring, then the helper mirror.

Two bugs here made every save and restore silently wrong, and probably caused the 2026-10-23 box freeze (garbage
axes, then out-of-bounds sparse reads):

1. llama passes **view** tensors. The hook compared them against the base tensors, never matched, and llama wrote
   raw q8 rows over the records. The hook now compares `view_src`.
2. KVarN bound its hooks at the first decode. A state load before that (the KL harness restores its prefix
   first) bypassed them. `jde_kv_attach` now binds right after the memory exists.

Trace: `JI_KVR_TRACE=1` logs any row that bypasses the hook (`plain write`).

The second freeze's dmesg showed the ring hang after a TP2 process ended mid-step. TP2 GPUs wait in-queue on
host-set VkEvents, and a dead relay never sets them. Engine 2c61614d3 releases every TP event before abort, at
teardown and on fatal signals. See `features/README.md` (operational hazards).

## MTP issue and resolution

**Cause:** separate window rings (130 MiB) reduced the primary GPU's free-VRAM margin from ~412 to ~281 MiB; with MTP running, the amdgpu driver evicted 1.4–2.8 GB of buffers to GTT: card1 VRAM 7855 → 6903 MiB at load and 5473 after the prompts, GTT +1.4 / +2.8 GB; q8_0 unchanged. Per-kernel profile: MoE matvecs 14.6 → 29.6 ms/token, shared-expert 512×2048 matvec 2.6× slower; attention kernels unchanged.

**Result (TP2, 1024 tokens, rewrite / story / explain):** KVarN 32.9 / 34.1 / 34.5 → 50.6 / 60.8 / 55.3 t/s; q8_0 51.6 / 61.1 / 55.8. KL unchanged (decode from restored states): single card 32k 0.00058, 130k 0.00074; TP2 32k 0.00047, 130k 0.00081.

## Measured (TP2, 100 W, 2026-10-23)

| case | q8_0 | KVarN-6 |
|---|---|---|
| short prompts, decode (rewrite / story / explain) | 43.2 / 49.6 / 48.0 | 42.1 / 49.1 / 47.3 |
| 36k restored, decode | 36.3 | 37.6 |
| 247k full prefill / decode | — | 432 t/s / 38.0 t/s |

KL gate (`kl.sh L_code 0 JI_TP=2 JI_KVARN=1`, vs our full-attention q8_0 reference; q8 baseline 0.00020): **pending**.

## From-scratch quality (2026-10-31: defect confirmed, open)

**Earlier concern:** 130k from-scratch prefill scored KL 0.27 in the harness. Root cause: the KL harness (`~/jt/kl_v2.sh`) never set `JI_JPF=1`, so all prefills used the ggml graph-path engine, which does not build KVarN records or the window ring. A saved state from graph-path prefill had buffer corruption (rel. error 1.000 everywhere), while the tail through decode matched q8_0 to digit; the high KL was thus meaningless.

**Status (job88, 2026-10-31):** with the harness fixed (`JI_JPF=1`) the defect is real: TP2, default sparse, 130k from
scratch through the JDE prefill: q8_0 KL 0.00123, KVarN **0.118** (top1 0.906). 32k from scratch is fine (q8 0.00071,
KVarN 0.00092, single card, sparse off); restored states decode fine at every length. The graph-path runs (KL ~2,
empty saved states) were a separate harness artefact, not this defect. Open: bisect by depth (64k / 98k) with the
JDE prefill, then compare a saved from-scratch state per layer / cell (`kvrblk.py`).

## Porting notes

- Format and kernels are plain GLSL. The balancing loop is LDS-heavy (32 KB of f16 per workgroup); on RDNA2
  wave32/64 it should be faster.
- The in-place trick (records inside the q8 tensor) avoids touching llama.cpp's allocator. A clean engine would
  allocate 208 B/cell and save the whole 23.5 %.
- Keep slot files in q8_0 (save materialises), so cache files stay engine- and format-independent.
- On 8 GB cards every fixed buffer comes out of the margin the driver needs; VRAM over-commit doesn't fail, it
  silently evicts buffers to GTT and halves matvec speed — check `mem_info_vram_used` / `mem_info_gtt_used` in
  sysfs when a change slows unrelated kernels.
