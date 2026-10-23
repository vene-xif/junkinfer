# JunkInfer feature architecture

One document per feature: what it does, its data layout and kernels, measured wins (dated, with t/s before →
after), what was rejected, and porting notes for a different engine or GPU (e.g. 6900 XT + RX 580). The
chronological record stays in `../engineering-log.md`; these documents are the stable description.

| feature | doc | status |
|---|---|---|
| Sparse attention (decode + prefill, tail estimate) | [sparse-attention.md](sparse-attention.md) | default on |
| Expert memory hierarchy (pinned host, VRAM cache, elastic slots, TP2 balance) | [expert-cache.md](expert-cache.md) | default on |
| MTP speculative decoding (draft head on the helper GPU) | [mtp-speculative-decoding.md](mtp-speculative-decoding.md) | default on |
| KVarN-6 KV cache (in place) | [kvarn6-kv-cache.md](kvarn6-kv-cache.md) | opt-in, gate pending |
| JDE decode engine (one-shot tokens, fused kernels, EXL3 mat-vecs) | [jde-decode-engine.md](jde-decode-engine.md) | default on |
| Tensor-parallel decode / prefill on two GPUs | [tensor-parallel-two-gpus.md](tensor-parallel-two-gpus.md) | production |
| EXL3 kernels (trellis decode, stripe-major mat-vecs, GEMMs) | ../exl3-support.md (to be extended) | default on |
| Prompt reuse, small prompts, server / supervisor | to write | default on |
| Host / system tuning (DPM, governor, GTT, power cap) | ../physics.md + to write | — |

## Operational hazards (gfx803, two cards)

- A GPU left waiting on a TP VkEvent (process ended mid-step) hangs its ring. The amdgpu reset on Polaris then
  fails (`asic atom init failed`) and the box freezes. It happened twice on 2026-10-23; fixed by releasing events on
  abort, teardown and fatal signals (engine 2c61614d3). Still: never SIGKILL a busy TP2 process.
- One model process at a time (pinned experts: two processes run out of memory).
- CPU governor `performance` (schedutil costs 20–25 % decode after large prefills).
