# Open optimization paths (live list, ordered by effort)

> Every known open lever, with status, expected gain, cost and blockers. Ordered by time/effort needed (smallest
> first) within each tier. Update in place; results go to `engineering-log.md`, decisions to `candidate-ledger.md`.
> Last update: 2026-10-23. **All sparse-attention KL figures dated before 2026-10-23 were measured against corrupt
> references (graph-path layer-3 KV loss, engineering log 2026-10-23 afternoon) and are void; v2 numbers below.**

Current numbers (AUX-1, EXL3 4.0 bpw, q8_0 KV, sparse attention on): decode ~44 t/s short, ~35 at 72k, ~32 at
100k, 29.4-30.0 at 245k; prefill ~450 t/s (7k), 305-319 t/s
(132k), ~230 (262k). Combined sparse KL vs full attention ~0.00045 at 130k.
Targets: single card decode >= 50 t/s at any depth; dual card decode >= 80, prefill >= 1000 / >= 500 (262k).

**2026-10-31: ranked actions from the deep-research reports (decode-kernel tricks, MTP speculative sampling,
register-decoded prefill GEMM, sparse scorer, Qwen4 readiness) - `research-actions.md`.**

**TOP PRIORITY (user, 2026-10-26): native EXL3 loading + MTP + vision from the checkpoint + fast mul1 decode —
`native-exl3-plan.md`.** Item 7 (v_sad_u8) is its step 6.

Legend: gain = expected, unmeasured unless stated. "KL" = added KL vs our own EXL3 4.0 bpw full-attention output.

## Highest-value next steps (2026-11-01, after write-through + placement + pinning)

State: Sage TP2 short decode ~58 t/s = ~23 ms GPU + ~3 ms host per step (~1.95 tokens); 116k 39.9 t/s; prefill ~530
t/s at 10k, ~311 t/s at 116k. Card work balanced (primary 16.7 / helper 16.0 ms per step), so expert-cache work
on the primary no longer shortens steps much (learned eviction bound: engineering-log 2026-11-01 later).

### Decode (ranked by gain / effort; estimates, not measurements)

| # | step | time it attacks (measured) | est. gain | effort |
|---|---|---|---|---|
| D1 | Host out of the step loop: MTP draft head recorded into the same GPU submission as the verify | host gap ~2.6-3 ms/step (draft ~1.35, sampling ~0.3, bookkeeping) | +5-8 % | medium-high (server / speculative step order) |
| D2 | Cheaper card-to-card syncs: GPUs poll each other's markers over BAR instead of GPU -> host relay -> event, and/or fewer sync points (~80/step) | sync floor ~1.7-2 ms/step + per-layer imbalance | +4-7 % | medium |
| D3 | Router kernel: 34 us per layer for a 1 MB mat-vec + top-k (target ~8-10 us; suspect the serial softmax/top-k tail); on both cards' critical path | ~1.4 ms/step | +3-4 % | low-medium, self-contained |
| D4 | Fewer tiny dispatches: x3_prep 371 x ~2.6 us per token - fuse into the producing kernel's output stage (fusing into the consumer, FUSEP, was slower) | ~1 ms/step | ~+2 % | medium |
| D5 | Dense EXL3 mat-vec arithmetic (research-actions #4-5: integer accumulate, v_sad_u8 via side Mesa), lab-benched first | dense x3_mv ~6 ms/step | +2-5 % if it works | medium, uncertain |
| D6 | Rebalance cards after write-through (JI_TP_FRAC / hot-set size sweep) | 16.7 vs 16.0 ms | +1-3 % | low (config) |
| D7 | Re-test 2 MTP drafts/step after D1-D2 (2.37 vs 1.95 tokens/step; lost to fixed per-step cost so far) | fixed step cost | +5-10 % conditional | low |

Together D1-D5 as estimated: ~26 -> ~21 ms per step, ~+20-25 % (~70 t/s short). Also: AUX1 hard freeze
2026-11-01 09:23 during heavy server-restart churn - check reproducibility before stress runs; prefer A/Bs inside
one server instance; move the route log's fp32 -> fp16 conversion off the decode thread (~2 % of its time).

### Prefill (at 100 W both cards sit at the power limit: only less work / energy helps - memory note)

Split (7.2k, 2026-10-20): dense EXL3 GEMM 43 %, MoE GEMM 42 %, flash attention 10 %; attention grows with depth
(10k ~530 vs 116k ~311 t/s).

| # | step | est. gain | effort |
|---|---|---|---|
| P1 | Register-decoded prefill GEMM (research-actions #6: decode each 16x16 tile once, share across a quad by DPP, B from LDS) - fewer decode ops per FMA, i.e. less energy per token | +10-20 % prefill | ~1 week |
| P2 | Fold the mul1 affine in the prefill GEMM (research-actions #7; decode already folds, KL 0.00014) - one FMA per weight less | +3-6 % (est.) | ~1 day + KL gate |
| P3 | Win back the placement cost: the corpus hot set + JI_TP_DUP=0 cost prefill 2-5 %; let prefill use its own split / ownership | +2-5 % | low-medium |
| P4 | MTP's share of prefill (item 23 measured ~10 % at long prompts): check whether prompt positions produce per-position outputs (logits / nextn rows) beyond the JI_MTP_PF_WIN window, cut what the draft head does not need | up to ~5-10 % on long prompts (verify first) | low to check |
| P5 | Long-context prefill attention: sparse prefill budget / 32-key sub-pages (item 17), cheaper KVarN seal | depth-dependent, larger at 100k+ | medium, KL-gated |
| P6 | Two-card exchange over card B's PCIe 2.0 x4: smaller wire format (8-bit exchange already +3 %), overlap | +2-5 % | medium |
| P7 | v_sad_u8 (side Mesa; research-actions #5) - fewer ops per weight in the GEMM decode stage too | a few % | 2-3 days + Mesa build |
| - | Hardware (not software): x8/x8 lanes for card B, est. +15-20 % two-card prefill; >100 W is excluded by the user | - | - |

### Research bets (2026-11-01; user: #R1 in scope; no re-quantization at this stage)

Already well known / reproduced elsewhere and earlier discarded by the user: training-free expert skipping (ACE,
"halving activated experts") and batch-aware routing (SERE, Opportunistic Expert Activation) - STRATA already does
both, crib notes from it if tried; SpecPrefill-style prompt-token dropping - rejected (never makes sense here).

| # | idea | why it could change the shape of the system | first de-risking step |
|---|---|---|---|
| R1 | **Speculative layer pipeline instead of tensor parallel**: card A layers 0-19, card B 20-39, kept busy by MTP-drafted tokens in flight (SpecPipe / PipeSpec, arXiv 2504.04104 / 2505.01572 - dense models with separate drafts; nothing found for MoE + built-in MTP on asymmetric consumer links) | ~80 card syncs per step -> one 8 KB hidden-state handoff per token over the x4 link; each card owns its 20 layers' experts (~7.5 GB: near all-resident: placement / misses / write-through mostly moot); host sampling + drafting overlap the GPUs. Est. +20-40 % decode at ~80 % pipeline efficiency (guess). Hard parts: 2-deep drafts (chained MTP), flush + DeltaNet / KV rollback on rejects | measure chained 2-deep MTP acceptance on Sage; time a 20-layer stage on one card |
| R2 | **Co-schedule complementary work on one card**: token t's expert mat-vecs (trellis ALU-bound) concurrent with token t+1's attention (KV-bandwidth-bound) on two queues (needs R1's tokens in flight) | at 116k attention is most of the 58 -> 40 t/s drop; overlapping it with ALU-bound work could hide much of it | lab microbench: x3_mv_moe + attn kernels concurrently on two queues vs back to back on gfx803 |
| R3 | **Pipelined prefill** (R1's layer split for prefill): one ~16 MB handoff per 4096-token chunk instead of per-layer exchanges over card B's PCIe 2.0 x4; each card needs mainly its own layers' experts (resident -> little per-chunk upload) | removes the exchange and most expert uploads; power-cap caveat: only work / energy reductions count | after R1; estimate upload + exchange energy share first |
| R4 | **Free deep drafts for the pipeline**: suffix-tree / retrieval drafting from prompt + previous turns + own output ("Draft Less, Retrieve More", arXiv 2605.20104), MTP for the first draft token | R1 needs >= 2-deep drafts where MTP is weakest; code / edits / tool output are where retrieval hits | n-gram spec measured 37 -> 50 t/s on a code rewrite earlier: re-measure on current engine |
| R5 | **MTP draft head on the CPU** (40k-vocab head, small) | its ~1.35 ms/step leaves the GPU critical path; counts fully in a pipeline | time the head on the Ryzen (AVX2) |
| R6 | Fallback if R1 disappoints: **duplicate the token mixer on both cards** (DeltaNet / attention computed fully on each) | halves the per-layer syncs (~80 -> ~40) at the cost of duplicated mixer work | measure mixer share of per-card time |
| R7 | (needs re-quantization - later) nested-precision experts (2-bit base + 2-bit residual trellis; precision per token by router weight); gfx8-cheap trellis codes (cf. Proteus, ICML 2026) | halves ALU + PCIe for low-weight experts / attacks the trellis-decode ALU floor | offline CPU error study of a residual trellis on a few experts |

## Tier 0 - hours (config, env knobs, already-written code awaiting a run)

| # | Path | Status | Expected gain | Cost / risk |
|---|---|---|---|---|
| 1 | ~~245k decode regression~~ RESOLVED: page cache after the 18 GB download (24.5-26.9 t/s); after drop_caches 29.3-30.0 t/s | done | - | drop caches before speed runs |
| 2 | ~~Per-layer exact prompt tail default (JI_SPF_TAIL_LAYERS=3,7,11)~~ DONE (engine f5ddada15): prefill 132k 305.5 -> 318.7 t/s, prefill KL 0.00035 -> 0.00038 | done | - | - |
| 3 | Per-layer dense DECODE attention (JI_SP_DENSE_LAYERS) MEASURED: 3,7,11 -> decode KL 0.00004-0.00006 (harness floor) for -6 % at 100k / -13 % at 245k; 7,11 -> KL 0.00006-0.00030 for -3 % / -9 %. Opt-in quality setting; default stays sparse | user decision | quality vs speed | - |
| 4 | ~~Reboot with `iommu=pt` + `amdgpu.lockup_timeout`~~ DONE 2026-10-20: iommu=pt neutral (245k 29.4-29.7 vs 29.3-30.0, 2k 43-45.5); lockup timeout 60/120 s active (long jobs no longer need splitting - unverified) | done | - | - |
| 5 | ~~LM head IQ4_NL (from the BF16 head)~~ REJECTED 2026-10-20: decode 2k 44.6 -> 47.3 (+6 %), 245k +4 %, but KL 0.00033 -> 0.0044 (130k) / 0.00036 -> 0.0026 (99k), top-1 0.997 -> 0.976. Q5_K (333 MB) untested: at most +2-3 %. Real path: EXL3 head ~4 bpw (item 29, needs requant) | rejected | - | tools/swap_head.py kept |
| 6 | ~~Dynamic top-k (JI_TOPK_TAU, engine option)~~ REJECTED 2026-10-20: KL 99k/130k tau 0.02: 0.0016/0.0008, 0.04: 0.0070/0.0022, 0.06: 0.031/0.0062; decode 2k 44.7 -> 44.7/45.1/45.7, 245k unchanged (a layer's experts already run in parallel: dropping one does not shorten the critical path) | rejected | - | - |
| 7 | `v_sad_u8` from GLSL (GL_EXT_integer_dot_product `dotPacked4x8EXT(x, 0x01010101)`) - RESOLVED: not emitted by ACO as v_sad_u8 on gfx8, but the gain came from dropping the f16 round trip (exact) + dotPacked4x8EXT byte sum (SDWA lowers it) + affine fold. Sage 21.8 -> 18.9 ms/token. | DONE 2026-10-31 (engine d44ea8d) | mul1 decode now at mcg cost (fix 1: exact); affine fold (fix 2, opt-in) reaches it without losing top-1 | — |
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
| S0 | Speed per budget at 200k (dense / 16k / 64k / 128k) | short-run data (60-80 tokens, ±5 %): dense 25.6, 16k 38.0, 64k 34.9, 128k 31.1 t/s; 1024-token rerun RUNNING | decision data | none |
| S0b | Per-layer ablation at 64k keys | DONE 2026-10-24: every layer contributes about equally (one layer dense: 261k 0.0036 -> 0.0030-0.0036); no layer-specific budgets needed | - | - |
| S1 | Monopole tail: drop the variance term (n exp(q.muK), JI_SP_BG=1); argument: the diagonal-variance term ignores intra-page key correlation and inflates unselected mass (Multipole Attention, Hooper et al. NeurIPS 2025) | REJECTED 2026-10-23: KL same as the variance term (130k 0.0114 vs 0.0115; with 64k keys 0.0016 vs 0.0009) - the tail estimator is not the lever on this model | - | - |
| S2 | Sub-page centroids: split each 32-key page into M = 2 (-4) k-means sub-clusters with own (muK, muV, n), monopole mass each - fixes value dilution (the unweighted page-mean V) inside the same ~1.5 KB/page | OPEN | KL (claimed large; unverified) | attn_meta (2-means on page completion), attn_bg/attn_score layout |
| S3 | (partly DONE as JI_SPARSE_FRAC / JI_SPARSE_PF_FRAC: budget = share of the context, default 0.5 - KL depends on the share read exactly, the layers attend broadly; a per-step top-p mass threshold is still open but the mass estimate it needs is weak, see S1) Error-bounded adaptive budget (Twilight top-p, Lin et al. NeurIPS 2025): select pages by descending estimated mass until p = 0.985-0.99 of the estimated total; dense fallback when that needs > ~25 % of the context; per KV head, per step | OPEN | KL bound on broad layers, keeps small budgets on peaked ones | topk -> cumulative-mass threshold (binary search on the score histogram); depends on S1/S2 for a usable mass estimate |
| S4 | DROPPED 2026-10-24 (cost model from measurements: at a 50 % budget the 4-bit tail still reads ~76 % of the dense bytes -> ~27 t/s at 200k vs 32.4 t/s for SEL + FRAC 0.5 at KL ~0.001; not on the Pareto front for this broad-attention model) Two-tier precision tail: exact q8 for the selected pages, low-bit (4 / 3 / 2-bit) K and V for every other token (exact softmax formula, quantisation noise only on tail mass) instead of a statistical summary; variant: Gemini's zero-VRAM split of q8 K into hi4 / lo4 bit planes (hi4 scores all candidates, lo4 + V only for winners) | OPEN (bound being measured: dense 4-bit everywhere 0.002-0.0028, so a tier with exact top pages should land well below) | near-lossless with ~2x less attention traffic than dense at 4-bit tail | shadow low-bit KV (VRAM: ~670 MB at 2-bit, 262k) or a KV layout change (bit planes: every K reader changes) |
| S5 | Stratified sampled correction of the tail (MagicPIG-style control variate): 32-64 unselected pages per KV head read exactly each step, their (exact - estimate) residual scaled up - unbiased tail | OPEN | KL (bias removal) | small extra reads; sampling RNG in shader |
| S6 | Prefill: XAttention strided antidiagonal block scoring (Xu et al. ICML 2025) to replace the dense-prompt-tail crutch | OPEN, after decode | prefill KL / speed | new scoring pass |
| S7 | Per-layer graded budgets | NOT NEEDED (S0b: error spread evenly over layers) | - | - |
| S9 | Mass-estimate page ranking: decode JI_SP_SEL=1 (re-test on v2: KL -40 % at 64k keys, and faster) and prefill JI_SPF_SEL=1 (new, -10 %) | DONE 2026-10-24, defaults | - | - |
| S10 | TP2 JDE prefill: intermittent ring timeout on the exchange queues (2 hangs in ~10 h of TP2 prefill: card 1 comp_1.0.1 2026-10-24 00:37, helper comp_1.1.0 + 1.2.0 at 12:05; no AER errors) - soak with JI_PF_XQ=0 (copies inline) to separate a sync race from a link stall on the chipset-attached x4 slot | OPEN (reliability) | uptime | long soak runs |
| S8 | Continuum Attention (arXiv 2610.20586): NOT applicable - a different attention operator trained from scratch (5M-param models), not an approximation of softmax | rejected | - | - |


### Post-MVP track: learned / better sparse-attention estimators (user, 2026-10-24)

Gate first: S11's oracle ceiling. If exact-mass ranking at a small budget is far better than our estimate, ranking
is the bottleneck and L1 pays; if the oracle also needs ~50 % of the context, only cheaper bytes help (KVarN / 4-bit).

| # | Path | Status | Expected gain | Cost / risk |
|---|---|---|---|---|
| S11 | Ranking ceiling: exact-mass oracle (JI_SP_ORACLE=2) vs mass-estimate ranking | DONE 2026-10-24 (oracle had scored rotated K with the raw-basis query since the raw-basis change - fixed: own stored-basis query binding; oracle-based diagnostics from 10-19..10-21 are suspect): oracle 16k 0.0026 / 0.0047, 32k 0.0014 / 0.0044, 64k 0.0012 / 0.0014 vs ours 0.0044 / 0.0080, 0.0023 / 0.0042, 0.0009 / 0.0022 (130k / 261k) -> perfect PAGE ranking buys <= ~1.7x; attention is genuinely broad. L1 is worth at most a ~1.5-2x smaller budget at page granularity; token-level granularity, cheaper bytes (L3, Geodesia-style precision ladders) and tail summaries are the bigger levers | - | - |
| L1 | Learned page indexer, self-distilled from the frozen model's dense attention (KL to per-page attention mass): SeerAttention / -R (arXiv 2410.13276, 2506.08889: ~0.4B tokens, 11-19 A100-h on 4-14B), DeepSeek-V3.2 lightning-indexer warm-up with the main model frozen (arXiv 2512.02556: ~2B tokens). Tiny per-page MLP over a compressed key summary ~ cost of today's formula | post-MVP | smaller budgets at equal KL (if S11 says so) | training data = dense attention maps from long prefills: 20-50M tokens = 1-2 days on AUX1 at ~300 t/s (0.4B = ~15 days) |
| L2 | Re-tune always-kept pages (sink, recent window) once the ranker is better - learned selectors may make them redundant (reported both ways) | post-MVP (cheap: env knobs) | small | none |
| L3 | Cheaper exact part for the selected pages: KVarN-6 records (-24 % bytes) or 4-bit keys for scoring / exact for winners - preferred over approximating the selected attention itself (the selected pages carry most mass) | KVarN-6 next | bigger budget at equal speed | KVarN phase |
| L5 | Mixed-precision read ladder per block (Geodesia-KV, arXiv 2608.23599: rate-distortion over 16/8/4/2-bit/centroid by attention mass) - the graded version of the dropped uniform two-tier S4 | post-MVP | KL per byte | per-block allocation at decode time |
| L6 | Token-level (finer than 32-key page) selection with a cheap token sketch (sign bits / PQ / FP8 indexer) - the oracle ceiling above is page-level only | post-MVP | unknown (research) | sketch storage + scoring pass |
| M1 | 8-token sub-centroids: each 32-key page's ~1.5 KB summary becomes four 8-token sub-blocks (mean K, mean V, scalar/low-rank spread) - finer selection unit AND query-dependent tail values (fixes the measured exact-mass x mean-V pathology: the Gaussian under-estimate damps the mean-V error on spiky pages). Gemini report 2 (2026-10-24), extends S2 | post-MVP sparse track, FIRST (training-free) | breaks the page-oracle ceiling; tail KL | attn_meta layout + attn_score / attn_bg / pf_pmeta |
| M2 | 8-token micro-block selection read in place (8 tokens q8 K = 2,176 B: still fully coalesced on gfx803); top-k over micro-block ids | post-MVP, with M1 | KL at small budgets -> speed | attn_topk / attn_hm indexing |
| M3 | V-Skip: inside the selected set, read exact keys, then skip value reads for sub-blocks whose exact weight < tau x (estimated total mass from the scoring pass - our split-K chunks have no global running sum) and use the sub-centroid V | post-MVP, after M1 | value bytes of the selected set | attn_hm change |
| M4 | Asymmetric KV bits (keys 6-bit, values 5 / 4-bit: softmax amplifies key error exponentially, value error linearly) | IN the KVarN-6 phase (variant K6V5 / K6V4) | -24..-35 % exact-part bytes | seal / record format |
| M5 | Rejected on paper: value prefix-centering (report 2 claims -20..30 % KL) - with the same estimated masses in numerator and denominator of the merge it cancels exactly (sum a(v-c)/Z + c = sum a v / Z); cross-layer index reuse (layers decorrelated by 3 DeltaNet layers) low priority | - | - | - |
| L7 | (after M1-M3, trained at 8-token micro-block granularity - at page granularity it is capped by the ~1.7x page oracle) Warm-start residual estimator (user: in scope after the MVP): train only a small correction on top of today's analytic page score / tail, from a few long prefills (labels = exact per-page mass and outputs from the frozen model), <= 6 h per iteration, <= 24 h total; validate against the S11 oracle and end-to-end KL | post-MVP, first L-item | KL at small budgets -> speed | prompt for the data-efficiency research written 2026-10-24 |
| T0 | Sparse throughput track (user goal: max speed at near-dense quality). First: per-kernel profile at 200k / 262k (attn_score, attn_topk, attn_hm on selected pages, attn_bg tail, attn_merge; prefill pf_pmeta / pf_fa_sparse), then levers: int8 / low-rank page summaries (scoring + tail read ~1/11 of dense bytes at any budget), hierarchical (coarse-to-fine) scoring, selection reuse across consecutive tokens, KVarN-6 / precision ladder for the exact part, drop the dense prompt tail once the tail estimate allows. Ceiling at 200k (2 GPUs): decode <= ~37 t/s with zero attention cost (non-attention ~27 ms/token) vs 25.6 dense / 32.4 now; prefill ~2x dense at small budgets | post-MVP | decode up to ~+45 % vs dense at 200k (more at 262k), prefill ~2x | profile first |
| L8 | Data-efficient training loop for L7 (Gemini report 3, 2026-10-24): (1) engine dump hook during long prefills - per document 256-512 stratified query positions (half from model-generated completion spans: prefill vs decode query shift) with exact per-page log-sum-exp per head, plus page statistics (incl. min / max, pre-RoPE means for the 64 RoPE dims); (2) Phase 0, training-free: closed-form ridge recalibration of the analytic score on one 131k dump (seconds, CPU) - may improve today's estimator; (3) offline CPU proxy (KL / top-k recall from the dumps, seconds per candidate) for the inner loop, GPU KL gate only for finalists (1-2k queries over 7-10 documents resolve 10-20 % effects); (4) zero-initialised residual MLP on the 192 position-free dims (RoPE dims handled analytically -> length generalisation), trained on CPU in minutes; loss: KL, then boundary-restricted KL + pairwise hinge. First iteration ~4M prefill tokens (~4 h) + ~15 GB dumps (SSD); full ~17M tokens: dumps too big for the SSD (HDD / heavier subsampling) | post-MVP | - | corpus: local-first (user decision pending); downloads need user approval |
| L9 | Value-aware ranking: rank pages by the error they would cause if left to the tail (mass x ||page value - tail substitute||), not mass alone - explains our estimator beating the mass oracle at 130k / 64k keys; testable analytically before any training | post-MVP, early (cheap) | KL at fixed budget | scorer change |
| L10 | Page key statistics for the 64 RoPE dims in the pre-RoPE frame (rotated to the page centre at scoring time): post-RoPE means partly cancel in the fast-rotating pairs over 32 tokens | post-MVP, early (cheap) | estimator quality, length generalisation | attn_meta: un-RoPE keys (invertible) |
| L4 | Learned tail summaries (a few distilled summary vectors per page, Activation-Beacon-like, arXiv 2401.03462) instead of mean key / mean value | post-MVP | bounded by S11's headroom | distillation pipeline shared with L1 |

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
| 24b | ~~Decode -20-25 % after a large prefill until restart (expert-fill copies ahead of each TP step run 3-5x slower)~~ DONE (engine 104fa96): huge-page host imports, pre-faulted; decode now within 1 % of fresh after a 21.7k prefill | done | - | - |
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
| 27 | Re-quantize Qwen3.6 with mul1 — not needed: mul1 now decodes at mcg cost (2026-10-31 engine d44ea8d); performance gain would only come from refactoring to native loading (plan step 3) | not applicable | — | — |
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
