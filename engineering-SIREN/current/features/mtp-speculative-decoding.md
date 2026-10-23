# MTP speculative decoding (draft head on the helper GPU)

Status: **on in production** (`JI_MTP=1` in serve_prod.sh, since 2026-10-23). Engine: edf3f579f (head on JDE),
be2f7bf3d / 3153cf5ce (overhead cuts, vocab, requant), 638265c90 (CB eviction fix), b67e52d41 (attention / prompt
windows). History: engineering log 2026-10-16, 2026-10-19, 2026-10-21, 2026-10-23.

## Result

TP2, 100 W, 600-token replies, greedy-identical outputs (lossless):

| reply | no MTP | MTP |
|---|---|---|
| code rewrite | 43.2 | 49.4 (+14 %) |
| story | 49.6 | 58.5 (+18 %) |
| explain | 48.0 | 58.9 (+23 %) |

Prefill cost: −2–4 % fresh, −6 % for a 7.5k continuation on a 72k context.

## Why it was hard here

Verify cost doesn't amortise on this hardware. EXL3 routed experts are ALU-bound per row: one trellis decode per
(token, expert) pair. Cold experts also stream over PCIe. A 1/2/3/4-token TP2 step costs 17.3 / 25.3 / 33.1 /
49.8 ms. On one card speculation never paid. The win needed:

1. **A draft that costs ~1–2 ms**, not 15. The draft head runs inside JDE on the helper GPU (`jde_mtp_init`,
   `jde_mtp_decode`). The sidecar (`mtp-IQ4_NL.v2.gguf`: IQ4_NL block 40, output head, Q8_0 embedding) loads on the
   CPU (`-md … --spec-type draft-mtp -ngld 0 -devd none --no-spec-draft-backend-sampling`). JDE copies it to the
   helper's VRAM. The primary has no room at 262k.
2. **A small head vocabulary**: HyperQwen's *profiled* 40,960-token list (the tokens Qwen 3.8 27B actually emits;
   `ref/mtp/hyperqwen_draft_vocab_40960.json`, Apache-2.0, `JI_MTP_VOCAB_FILE`). The LM head is re-quantised to
   IQ4_NL (`JI_MTP_HEAD_TYPE`). Per the user's guidance this was judged by end-to-end t/s: IQ4_NL wins by being
   smaller even though its acceptance is a bit lower than Q6_K's.
3. **Cutting host overhead per step** (`JI_SPEC_PROF=N` profiler):
   - one fused draft decode per step, with catch-up rows deferred;
   - repeated draft row skipped;
   - head computed only for the output row;
   - greedy top-10 straight from the logits;
   - no 3 MB sampler clone per step (lazy clone).
4. **One draft, gated**: `--spec-draft-n-max 1`, p-min 0.5. Two drafts lose: each extra verified token costs
   8–10 ms.
5. **Windows on the draft head**:
   - Attention window: the last 8192 cells (`JI_MTP_ATT_WIN`, offset bindings). Attending everything cost ~7 ms per
     call at 72k.
   - Prompt catch-up: the last 128 rows (`JI_MTP_PF_WIN`).
   - Spec processing per 4k chunk at 72k: 0.9 → 0.15 s.
6. JDE's nextn rows are the **post-output_norm** hidden state (the graph's h_nextn). Pre-norm x was a bug.

## Acceptance data (draft 3, t=0; ggml sidecars)

Acceptance is flat from 4 bits up: IQ4_NL 0.644 equals BF16 0.637 (per position 0.78 / 0.57 / 0.44). Below 4 bits:
Q3_K 0.613, IQ3_S 0.563, Q2_K 0.550, all uncalibrated. A calibrated sub-4-bit head is open-path #32.

## Correctness traps (all fixed)

- Compute-buffer expert slots got corrupted by deep continuation prefills when MTP rows were wanted: garbage
  output after a 64k restore. Now every such prefill evicts all CB slots. The stray write is still unidentified.
- The MTP context's first verify reallocates the compute buffer. CB slots are re-bound, not retired (otherwise
  verify misses go up 3–4×).
- The draft head's KV (272 MiB at 262k) and scratch come out of the helper's margin (`JI_TP_MARGIN_MIB=1900`).

## Rejected

- Grouped expert decode in verify (`JI_X3_GROUP`): LDS occupancy cost exceeds the 16–30 % duplicate saving.
- Probabilistic drafting / rejection sampling (only +2–9 % at T ≥ 0.5 upstream).
- n ≥ 2 drafts.
- The ggml graph-path draft (15 ms per call).

## Porting notes

- On a card where verify amortises (tensor/matrix units, or a format whose decode isn't per-row ALU-bound), use
  more drafts. The n=1 / p-min 0.5 choice is specific to gfx803 + EXL3.
- The profiled-vocabulary head and the windowed draft attention transfer directly.
