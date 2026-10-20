# Open optimization paths (live list, ordered by effort)

> Every known open lever, with status, expected gain, cost and blockers. Ordered by time/effort needed (smallest
> first) within each tier. Update in place; results go to `engineering-log.md`, decisions to `candidate-ledger.md`.
> Last update: 2026-10-20.

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
| 6 | Dynamic top-k: drop routed experts with gate < tau * sum (ik "Smart Expert Reduction"), renormalise | OPEN (router_fused push constant) | report 5: 8 -> ~6.2 experts at tau 0.025, KL < 0.001 claimed -> fewer misses (~+10-20 % decode) | KL sweep tau 0.01-0.04; differs from the rejected miss substitution (drops by gate, not by cache state) |
| 7 | `v_sad_u8` from GLSL (GL_EXT_integer_dot_product `dotPacked4x8EXT(x, 0x01010101)`) - does RADV/ACO lower it to sad / cheap ops on gfx803? | OPEN | mul1 decode ~20 % cheaper per weight | ISA dump of a test shader |
| 8 | srvpf "decode 0.0 t/s": /completion on the 132k docs prompt returns 0 tokens | OPEN (anomaly) | correctness check | look at stop reason / first token |
| 9 | JI_SP_NORM=2 LSE in attn_score instead of attn_topk (2-workgroup latency) | optional | makes NORM=2's -5 % KL nearly free | small shader change |

## Tier 1 - days

| # | Path | Status | Expected gain | Cost / risk |
|---|---|---|---|---|
| 10 | **mul1 codebook + EXL3 any width (2-8 bits)** | engine + shaders + converter written (types, x3_mv, mul_mm_gcn_exl3, dequant, JDE CPU path, codebook flag, tokenizer / vocab padding); numpy reference validated vs Qwen3.6 (cos 0.94-0.99) | runs MiniCPM-V 4.7 3.73 bpw (and any future mul1 / mixed-bit EXL3); smaller experts (3/4-bit) -> more cache hits, fewer PCIe bytes | build, GPU-vs-CPU-reference check, convert (HDD scratch), PPL / KL sanity, speed |
| 11 | Expert cache with byte-granular slots (mixed-width experts: 3/4/5-bit sizes differ by 1.7x; today slot = max size) | OPEN (after 10) | +20-40 % cache capacity on mixed-bit models | JDE slot allocator |
| 12 | Clustered LM head (SV-Decode: h-space clusters, evaluate top 25 %, centroid fallback; exact for greedy via norm bound) | OPEN | head ~1.3 -> ~0.5 ms after item 5 (~+4 %) | offline clustering + two-stage shader |
| 13 | Prefill MoE GEMM: two tile heights (BM 64 for full experts, 16 for the tails; tile fill 81 % -> 95 %) | OPEN (report 3) | MoE GEMM +15-30 % -> prefill +8-15 % | new GEMM variant + dispatcher |
| 14 | XOR bank swizzle in the GEMM LDS tiles | OPEN | only if conflicts exist (profile first) | small |
| 15 | Raw-basis prefill statistics without the code-gen hit (separate pipeline variant compiled only when on, or raw q written by the graph) | patch kept (scratchpad rawpf_patch/) | prefill KL 0.00035 -> 0.00026 (measured, with tail) | earlier attempt cost -8 to -15 % prefill |
| 16 | Per-layer / per-head sparse budgets (decode and prefill), from the dump analysis (layer 3/7 broad, 39 sharp) | OPEN | same KL at lower cost or lower KL at same cost | uses `JI_KV_DUMP` + scratchpad ana/ tools |
| 17 | 32-key sub-page selection in prefill (report 3 P3) | OPEN, lower priority now (prefill KL small) | prefill KL | core loop change (BC=128 blocks) |
| 18 | KL harness resolution: more rows / longer tails (1024 positions/row = ~+-10 % noise) | OPEN | trustworthy decisions on <15 % effects | refs need full-attention runs |
| 19 | Direct loader for EXL3 safetensors folders (skip the GGUF repack) | OPEN (convenience) | none (UX) | loader plumbing |
| 20 | n-gram / suffix-automaton drafting for agentic edits (AgSpec), p-min gating | OPEN | big on code rewrites (n-gram 37 -> 50 t/s measured, lossy variant rejected; lossless verify cost is the limiter) | verify scaling (see 25) |

## Tier 2 - a week or more

| # | Path | Status | Expected gain | Cost / risk |
|---|---|---|---|---|
| 21 | **Dual card** (on arrival): `lspci` link of slot 2 (B450: chipset PCIe 2.0 x4), prefill layer split + pipeline, decode expert-server (hot experts resident on card 2) or layer split, MTP re-evaluation | waiting for hardware (`dual-card-design.md`) | decode toward 80 t/s, prefill ~2x | slot 2 cannot stream experts |
| 22 | Vision: MiniCPM-V 4.7 tower (features injected at decoder layer 6 - `insert_layer_id`), streamed from host RAM per block (HyperQwen PR #36 pattern: bulk copy per block, not zero-copy; ~300 MB VRAM) | in scope (user), after 10 | image input at low VRAM cost | mtmd / clip support for the tower + injection point |
| 23 | MTP on JDE: draft step on the decode engine, grouped (dedup) expert verify, dynamic draft vocabulary (NanoSpec), calibrated sub-4-bit MTP head | parked until card 2 (single card MTP ~3x slower today) | 1.3-1.8x decode if verify cost is fixed | large |
| 24 | KVarN6 KV cache | low priority (attn_hm faster on q8_0; VRAM gain ~+5 % at 247k) | ~0.8 GB VRAM at 262k | kernel work |
| 25 | Lossless grouped verify (one read per distinct expert across verified tokens) | OPEN | speculation becomes profitable | MoE kernel rework |
| 26 | DuoAttention-style per-head retrieval / streaming classes for the 160 attention heads | OPEN | lower budgets on streaming heads | calibration + per-head budgets |

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
