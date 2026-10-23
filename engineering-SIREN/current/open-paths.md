# Open optimization paths (live list, ordered by effort)

> Every known open lever, with status, expected gain, cost and blockers. Ordered by time/effort needed (smallest
> first) within each tier. Update in place; results go to `engineering-log.md`, decisions to `candidate-ledger.md`.
> Last update: 2026-10-23. **All sparse-attention KL figures dated before 2026-10-23 were measured against corrupt
> references (graph-path layer-3 KV loss, engineering log 2026-10-23 afternoon) and are void; v2 numbers below.**

Current numbers (AUX-1, EXL3 4.0 bpw, q8_0 KV, sparse attention on): decode ~44 t/s short, ~35 at 72k, ~32 at
100k, 29.4-30.0 at 245k; prefill ~450 t/s (7k), 305-319 t/s
(132k), ~230 (262k). Combined sparse KL vs full attention ~0.00045 at 130k.
Targets: single card decode >= 50 t/s at any depth; dual card decode >= 80, prefill >= 1000 / >= 500 (262k).

Legend: gain = expected, unmeasured unless stated. "KL" = added KL vs our own EXL3 4.0 bpw full-attention output.

## Tier 0 - hours (config, env knobs, already-written code awaiting a run)

| # | Path | Status | Expected gain | Cost / risk |
|---|---|---|---|---|
| 1 | ~~245k decode regression~~ RESOLVED: page cache after the 18 GB download (24.5-26.9 t/s); after drop_caches 29.3-30.0 t/s | done | - | drop caches before speed runs |
| 2 | ~~Per-layer exact prompt tail default (JI_SPF_TAIL_LAYERS=3,7,11)~~ DONE (engine f5ddada15): prefill 132k 305.5 -> 318.7 t/s, prefill KL 0.00035 -> 0.00038 | done | - | - |
| 3 | Per-layer dense DECODE attention (JI_SP_DENSE_LAYERS) MEASURED: 3,7,11 -> decode KL 0.00004-0.00006 (harness floor) for -6 % at 100k / -13 % at 245k; 7,11 -> KL 0.00006-0.00030 for -3 % / -9 %. Opt-in quality setting; default stays sparse | user decision | quality vs speed | - |
| 4 | ~~Reboot with `iommu=pt` + `amdgpu.lockup_timeout`~~ DONE 2026-10-20: iommu=pt neutral (245k 29.4-29.7 vs 29.3-30.0, 2k 43-45.5); lockup timeout 60/120 s active (long jobs no longer need splitting - unverified) | done | - | - |
| 5 | ~~LM head IQ4_NL (from the BF16 head)~~ REJECTED 2026-10-20: decode 2k 44.6 -> 47.3 (+6 %), 245k +4 %, but KL 0.00033 -> 0.0044 (130k) / 0.00036 -> 0.0026 (99k), top-1 0.997 -> 0.976. Q5_K (333 MB) untested: at most +2-3 %. Real path: EXL3 head ~4 bpw (item 29, needs requant) | rejected | - | tools/swap_head.py kept |
| 6 | ~~Dynamic top-k (JI_TOPK_TAU, engine option)~~ REJECTED 2026-10-20: KL 99k/130k tau 0.02: 0.0016/0.0008, 0.04: 0.0070/0.0022, 0.06: 0.031/0.0062; decode 2k 44.7 -> 44.7/45.1/45.7, 245k unchanged (a layer's experts already run in parallel: dropping one does not shorten the critical path) | rejected | - | - |
| 7 | `v_sad_u8` from GLSL (GL_EXT_integer_dot_product `dotPacked4x8EXT(x, 0x01010101)`) - does RADV/ACO lower it to sad / cheap ops on gfx803? | OPEN | mul1 decode ~20 % cheaper per weight | ISA dump of a test shader |
| 8 | ~~srvpf decode 0.0 t/s~~ CLOSED: raw /completion on the docs prompt - the model predicts end-of-text after the document's last line (benign) | done | - | - |
| 9 | JI_SP_NORM=2 LSE in attn_score instead of attn_topk (2-workgroup latency) | optional | makes NORM=2's -5 % KL nearly free | small shader change |

## Sparse attention rework (2026-10-23, after the v2 references) - user priority: keep the speed, lose the loss

Measured on v2 references (KL, TP2): default (16k keys, Gaussian tail) 64k 0.0028 / 100k 0.0066 / 130k 0.0115 / 261k
0.0177; dense 0.0001-0.0002. Budget 32k / 64k / 128k at 261k: 0.0087 / 0.0036 / 0.0014. Tail diagnostics: no tail
only slightly worse than the estimate; exact page mass + mean V *worse*; estimated mass + exact page values ~ default:
both the mass and the value of the unselected pages matter. Dense attention over 4-bit KVarN-simulated KV: 0.002-0.0028.
Sources for S1-S6: Gemini deep research 2026-10-23 (share link moGCmgy2EScs; its KL predictions are estimates and run
~3x optimistic - measure everything).

| # | Path | Status | Expected gain | Cost / risk |
|---|---|---|---|---|
| S0 | Speed per budget at 200k (dense / 16k / 64k / 128k) - the trade-off curve that frames everything below | RUNNING | decision data | none |
| S1 | Monopole tail: drop the variance term (n exp(q.muK), JI_SP_BG=1); argument: the diagonal-variance term ignores intra-page key correlation and inflates unselected mass (Multipole Attention, Hooper et al. NeurIPS 2025) | QUEUED | KL, free | env knob |
| S2 | Sub-page centroids: split each 32-key page into M = 2 (-4) k-means sub-clusters with own (muK, muV, n), monopole mass each - fixes value dilution (the unweighted page-mean V) inside the same ~1.5 KB/page | OPEN | KL (claimed large; unverified) | attn_meta (2-means on page completion), attn_bg/attn_score layout |
| S3 | Error-bounded adaptive budget (Twilight top-p, Lin et al. NeurIPS 2025): select pages by descending estimated mass until p = 0.985-0.99 of the estimated total; dense fallback when that needs > ~25 % of the context; per KV head, per step | OPEN | KL bound on broad layers, keeps small budgets on peaked ones | topk -> cumulative-mass threshold (binary search on the score histogram); depends on S1/S2 for a usable mass estimate |
| S4 | Two-tier precision tail: exact q8 for the selected pages, low-bit (4 / 3 / 2-bit) K and V for every other token (exact softmax formula, quantisation noise only on tail mass) instead of a statistical summary; variant: Gemini's zero-VRAM split of q8 K into hi4 / lo4 bit planes (hi4 scores all candidates, lo4 + V only for winners) | OPEN (bound being measured: dense 4-bit everywhere 0.002-0.0028, so a tier with exact top pages should land well below) | near-lossless with ~2x less attention traffic than dense at 4-bit tail | shadow low-bit KV (VRAM: ~670 MB at 2-bit, 262k) or a KV layout change (bit planes: every K reader changes) |
| S5 | Stratified sampled correction of the tail (MagicPIG-style control variate): 32-64 unselected pages per KV head read exactly each step, their (exact - estimate) residual scaled up - unbiased tail | OPEN | KL (bias removal) | small extra reads; sampling RNG in shader |
| S6 | Prefill: XAttention strided antidiagonal block scoring (Xu et al. ICML 2025) to replace the dense-prompt-tail crutch | OPEN, after decode | prefill KL / speed | new scoring pass |
| S7 | Per-layer graded budgets on the v2 references (old JI_SP_LBUDGET verdict void) | OPEN | KL / speed balance | env knob |
| S8 | Continuum Attention (arXiv 2610.20586): NOT applicable - a different attention operator trained from scratch (5M-param models), not an approximation of softmax | rejected | - | - |

## Tier 1 - days


| # | Path | Status | Expected gain | Cost / risk |
|---|---|---|---|---|
| 10b | ~~Dense projections in mainline formats~~ REJECTED 2026-10-20 (hybrid with the old GGUF's IQ4_NL/Q8_0/Q5_K/Q6_K dense tensors): decode 44.6 -> 42.4 t/s short, 29.9 -> ~27.5 at 245k (bigger dense weights ate the VRAM cache; the mv path loses EXL3's fused prep/post), KL 0.012-0.013 (35x). EXL3 dense stays | rejected | - | - |
| 10 | **mul1 + EXL3 any width** WORKING (engine f5ddada15 + fixes): MiniCPM-V 4.7 3.74 bpw converted (16.3 GiB), correct output, decode 34 t/s short (Qwen3.6 4.0: 44), prefill 319 t/s at 20k, API suite 44/45 (model answers trivial questions inside an unclosed think block). Needs JI_UBATCH=2048 at 262k. Open: KL/PPL reference for MiniCPM, mul1 decode cost (item 7), cache slots per width (item 11) | done (follow-ups open) | - | - |
| 11 | Expert cache with byte-granular slots (mixed-width experts: 3/4/5-bit sizes differ by 1.7x; today slot = max size) | OPEN (after 10) | +20-40 % cache capacity on mixed-bit models | JDE slot allocator |
| 12 | ~~Clustered LM head (SV-Decode)~~ NOT APPLICABLE to the EXL3 head: its output Hadamard mixes 128-row blocks of consecutive token ids, so rows cannot be regrouped by cluster without re-quantization | rejected | - | - |
| 13 | Prefill MoE GEMM two tile heights - DEPRIORITISED: our MoE GEMM already skips padded columns; the remaining cost of thin experts is the per-workgroup codebook decode, which a narrower tile does not reduce. Prefill split (7.2k): dense EXL3 GEMM 43 % (3.8-3.96 TFLOPS), MoE 42 % (~2.9), FA 10 % | low expected gain | - | - |
| 14 | XOR bank swizzle in the GEMM LDS tiles | OPEN | only if conflicts exist (profile first) | small |
| 15 | Raw-basis prefill statistics without the code-gen hit (separate pipeline variant compiled only when on, or raw q written by the graph) | patch kept (scratchpad rawpf_patch/) | prefill KL 0.00035 -> 0.00026 (measured, with tail) | earlier attempt cost -8 to -15 % prefill |
| 16 | ~~Per-layer sparse budgets (JI_SP_LBUDGET)~~ REJECTED as default: late layers 8192 = +2-3 % but the 64k row doubles KL (0.00034 -> 0.00075); option kept | rejected | - | - |
| 17 | 32-key sub-page selection in prefill (report 3 P3) | OPEN, lower priority now (prefill KL small) | prefill KL | core loop change (BC=128 blocks) |
| 18 | KL harness resolution: more rows / longer tails (1024 positions/row = ~+-10 % noise) | OPEN | trustworthy decisions on <15 % effects | refs need full-attention runs |
| 19 | Direct loader for EXL3 safetensors folders (skip the GGUF repack) | OPEN (convenience) | none (UX) | loader plumbing |
| 20 | n-gram / suffix-automaton drafting for agentic edits (AgSpec), p-min gating | OPEN | big on code rewrites (n-gram 37 -> 50 t/s measured, lossy variant rejected; lossless verify cost is the limiter) | verify scaling (see 25) |

## Tier 2 - a week or more

| # | Path | Status | Expected gain | Cost / risk |
|---|---|---|---|---|
| 21 | **Dual card**: done - **JI_TP=2 full tensor parallel decode (+ per-token expert balance, 1-4 token steps): short 46.3 -> 59.1, 100k 33.8 -> 45.2, 245k 30.0 -> 41.1; code rewrite with n-gram spec 32.6 -> 50.6**. Open: (a) MTP drafts inside JDE: done, no gain (verify 2/3/4 tokens 25/33/50 ms vs 17; engineering-log 2026-10-21 later), (b) cheaper syncs, (c) router f16, (d) prefill on two cards (x4 link: pipelining) | in progress (engineering-log 2026-10-21) | (a) none | ALU-bound trellis decode ~200 G weights/s per card |
| 22 | Vision: MiniCPM-V 4.7 tower (features injected at decoder layer 6 - `insert_layer_id`), streamed from host RAM per block (HyperQwen PR #36 pattern: bulk copy per block, not zero-copy; ~300 MB VRAM) | in scope (user), after 10 | image input at low VRAM cost | mtmd / clip support for the tower + injection point |
| 23 | MTP on JDE: draft step on the decode engine, grouped (dedup) expert verify, dynamic draft vocabulary (NanoSpec), calibrated sub-4-bit MTP head | 2026-10-23: +9-17 % decode (overhead cut, p-min 0.5, profiled 40k vocab, IQ4_NL head), prefill -10 %; **BUG: garbage after a deep context** -> opt-in only (JI_MTP=1) | fix the deep-context bug, then default on | - |
| 24 | KVarN6 KV cache | IN PROGRESS (user task 4): in-place records + q8 window design, kvr_seal + attn_hmr written, harness pending; then prefill readers, state I/O, unseal, KL gate | ~0.65 GB more expert cache at 262k | kernel + integration work |
| 24b | Decode -20-25 % after a large prefill until restart (expert-fill copies ahead of each TP step run 3-5x slower) | OPEN BUG (pre-existing) | +25 % decode in real use | find what slows the fills |
| 25 | Lossless grouped verify (one read per distinct expert across verified tokens) | TRIED (JI_X3_GROUP): not faster - LDS occupancy, 16-30 % duplicates | speculation becomes profitable | MoE kernel rework |
| 26 | DuoAttention-style per-head retrieval / streaming classes for the 160 attention heads | OPEN | lower budgets on streaming heads | calibration + per-head budgets |

### Deferred by the user (2026-10-22): EXL3 GEMM kernel rewrite
TODO, to be approached differently later. pf_mm / mul_mm_gcn_exl3 run at ~45 % of FP32 peak (prefill: dense ~2.4
TFLOP/s per card at 100 W); the user estimates ~50 % headroom. Measured so far (do not repeat blindly): BK16 +7 % (in),
T128, BN128 (+BK16), 256-thread 8x8 / 8x4 register tiles (JI_PF_DV / JI_PF_IV) neutral or slower; occupancy 64 VGPR /
13 KiB LDS / 4 waves per SIMD, no spills. At the 100 W cap prefill is power-bound (80 W: 488 t/s vs 100 W: 629), so a
rewrite has to cut instructions / energy per FMA, not only stalls.

## Tier 3 - needs a quantization GPU (VENE, deferred by user) or re-quantization

| # | Path | Expected gain |
|---|---|---|
| 27 | Re-quantize Qwen3.6 with mul1 (cheaper decode ALU than mcg on gfx803, see 7) | ALU-bound matvecs faster |
| 28 | Hot / cold expert bit tiering (VRAM-hot ~3.35 bpw, cold ~2.45), down_proj > gate/up bits, GDN 3.5 bpw, attention K/V 5.5-6 bpw | report 5: big PCIe and cache gains inside the KL bar (unverified) |
| 29 | LM head EXL3 4 bpw (alternative to 5) | ~-0.7 ms |
| 30 | REAM folding of the ~12.5 % most dormant experts (256 -> 224) | cache hit rate |
| 31 | KL of EXL3 4.0 bpw vs BF16 (the denominator of every KL budget) | measurement owed |
| 32 | Calibrated (imatrix / GPTQ) MTP head below 4 bits | MTP draft cost |

## Not planned / user decision

- GPU over/under-clocking (VRAM 2000 -> 2150 MHz, refresh timing, core clocks): left alone by user decision (2026-10-20).
- RDNA2 port (6900 XT on CHERRY): estimated 1-2 days to run (wave64 forced), ~1 week to retune hot kernels; not planned.

## Rejected (measured; do not retry without a new idea)

- Sparse decode: gate-weighted ranking JI_SP_NORM=2 (-3 % KL for -4 % speed as built), dipole page mass (JI_SP_BG=3),
  variance coefficient != 0.5, global log-mass offset, log-I0 RoPE pairs (moot after the raw basis), budget 24k/32k
  (no KL gain), miss substitution (KL 0.002-0.05), speculative-position substitution (KL 0.05).
- Sparse prefill: raw-basis statistics as built (code-gen regression), semi-dense tail (32k budget: KL 0.00072 at -2 %;
  dominated by the per-layer exact tail), PV-skip (never triggers).
- System: PCIe MaxReadReq 4096, THP always, CPU share of misses (JI_X3_ZC 5-7), iommu=pt: no gain (2026-10-20).
- MTP on one card (all quants ~3x slower than no speculation), u24 multiply in the trellis decode (not the limiter),
  codebook lookup tables (64k entries: too big for LDS, TA-bound through L1), FA2 head-dim split (our kernel already
  avoids the register problem), low-rank / SVD experts (full rank), static pruning beyond ~12.5 %, IQ4_NL transcode of
  the host expert copy (does not fit in 32 GB next to EXL3).
