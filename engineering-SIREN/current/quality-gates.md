# Quality gates

Status: ACTIVE. 2026-10-15.

## The reference (made once, on VENE)

`ref/run_ref.sh` runs the official BF16 Qwen3.6-35B-A3B (llama.cpp CUDA, attention on an
RTX 3090, experts in RAM, f16 KV, flash attention) over a fixed tokenized corpus and stores, for
every scored position, the top-512 log-probs, the exact tail mass and the target log-prob.

| set | content | rows x length | scored positions |
|---|---|---|---|
| S1_wiki | wikitext-2 test | 48 x 2048 | second half of each row (49,104) |
| S2_code | llama.cpp source (eval half of the files) | 48 x 2048 | second half (49,104) |
| S3_chat | model-written replies: code, maths, tool calls, multi-turn agent, 6 languages (text sampled from UD-IQ4_NL for speed; logits from BF16) | 23 x var | the generated tokens |
| L_code | one code stream at 32k / 128k / 262k | 3 rows | last 1024 of each |
| L_prose | War and Peace at 32k / 128k / 262k | 3 rows | last 1024 of each |

Files (VENE `data/`): `corpus/*.jtok`, `refs/*.jref`, routing traces `refs/*.jrtr`
(per token, per layer: the 8 expert ids and their weights). Formats: `ref/tools/junk-ref/junk-ref.cpp`.

## The metric

KL(BF16 || candidate) over the reference's top-512 tokens plus one lumped tail bucket.
This is a lower bound of the true KL; with 512 tokens the tail mass is tiny.
Also reported: top-1 agreement, perplexity of both, KL percentiles.
`junk-ref --jr-mode compare` (llama.cpp-based engines) and `ref/tools/exl3_kld.py` (EXL3) compute it.

## The bars (filled in from the reference run)

| gate | bar | measured as |
|---|---|---|
| weights | KL mean <= EXL3 3.5 bpw on S1+S2+S3 | `exl3-3.5bpw` row of `data/eval/REPORT.md` |
| KV cache | KL <= q8_0 KV (BF16 weights) on L_code, L_prose, S3 | `kv-q8_0` rows |
| sparse attention | KL increase <= the KV-cache bar on L sets | to be measured on AUX-1 |

Numbers: pending (reference run in progress).

## Error budget of the runtime approximations (2026-10-19)

The KL figures for sparse attention are measured against our own full-attention engine (EXL3 4.0 bpw weights,
q8_0 KV), so they ADD to the weight and KV quantization error (roughly additively at these magnitudes). Total vs
BF16 ~ KL(weights) + KL(q8_0 KV, ~0.0015 vs f16 KV) + KL(sparse decode + prefill, ~0.00045 together at 130k; was ~0.0018
before the raw-basis page statistics and the exact prompt tail, 2026-10-19).
The q8_0-KV figure was used only as a yardstick for "tiny". Sparse decode and prefill are on by default by user
decision (not free, worth the cost: +44 % decode at 245k, +8 % at 72k, +30 % prefill at 80k, ~2.4x at 262k). Still owed: KL of
EXL3 4.0 bpw vs BF16 (needs a reference GPU), the denominator for judging these terms.
2026-10-22: JDE prefill's sparse attention uses 128-key statistics pages (JI_PF_SUB=128): KL at 261k with the tail
prefilled 0.00069 (= 32-key pages); with every layer sparse 0.00145 -> 0.00159. TP2 sparse decode before engine
63d680059 read wrong page metadata (KL 0.04 at 130k): TP2 depth quality figures before that commit are invalid.
