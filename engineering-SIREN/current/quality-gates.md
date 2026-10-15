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
