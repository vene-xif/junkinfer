# Dual RX 580 design (draft, 2026-10-19, before the second card arrives)

## Hardware facts that drive the design

- Board: Gigabyte B450 AORUS ELITE. Slot 1 (PCIEX16) is CPU-attached PCIe 3.0 x16 (measured ~11-14 GB/s
  host->device, 7.9 GB/s through BAR writes). The second full-length slot (PCIEX4) hangs off the B450 chipset:
  **PCIe 2.0 x4, ~1.6-2 GB/s**, shared with the chipset uplink (SATA / USB / LAN). dmidecode reports "x8"; trust
  `lspci -vv` LnkSta after installation.
- Each card: 8 GB VRAM, ~200 GB/s. No peer-to-peer path worth using (Vulkan, consumer Polaris, chipset slot):
  card-to-card traffic goes through host memory.
- Model: 16.3 GB of routed experts (40 layers x 256 x 3 mats, EXL3 4 bpw), ~1.3 GB of dense weights, KV at 262k
  q8_0 2.7 GB (only the 10 attention layers), DeltaNet states ~63 MB.

Consequence: card 2 must not stream experts over its own link at decode time (a 1.5 MB expert miss costs ~1 ms
there vs ~0.12 ms on card 1). Whatever card 2 computes must be resident in its VRAM.

## Decode today (single card, short context, ~21.5 ms/token GPU)

Routed experts ~9 ms (mostly PCIe-bound misses: ~45 missed experts/token), dense mat-vecs ~8 ms, LM head 2.2 ms,
GDN / attention / router / misc ~3 ms. At 245k: + sparse attention ~4 ms, smaller expert cache (KV takes VRAM).

## Options

### A. Layer split (pipeline, one token at a time)
Card 1: layers 0..k-1 (+ embeddings); card 2: layers k..39 + LM head. Hidden state (8 KB) crosses through the host
twice per token (~20-50 us each). Each card caches experts of its own layers only -> twice the cache, but card 2's
misses would cross the slow link -> card 2 needs (nearly) all of its layers' experts resident: 20 layers = 8.2 GB
of experts > 8 GB. Balance with fewer layers on card 2 (e.g. 14 layers = 5.7 GB experts + their dense weights +
KV share + LM head) and card-1-side zero-copy for the rest.
- Pro: simple, decode work per card halves the miss traffic; KV/attention also split.
- Con: no parallelism within a token (cards take turns); card 2's residency limits the split point.

### B. Expert residency split ("expert server", asymmetric)
Card 2 holds a static set of the most popular experts of ALL layers (~7 GB = ~43 % of experts, by routing
frequency more like 60-70 % of routed hits) and computes those; card 1 runs everything else (dense, attention,
LM head, its own expert cache, zero-copy misses over x16). Per layer: router on card 1 -> ids to card 2 (tiny) ->
card 2 computes its experts while card 1 computes its own -> card 2's partial sums (8 KB) back to card 1.
- Pro: card 2 never streams over its slow link; per-layer work runs on both cards in parallel; card 1 keeps its
  whole VRAM budget for its cache (dense + KV + cache).
- Con: 40 host round trips per token (~40 x 30-60 us = 1.2-2.4 ms) unless card 2's work is pipelined/batched;
  needs a cross-device sync primitive per layer (host-visible flags / timeline semaphores across devices are not
  shareable on RADV without external-memory/semaphore extensions -> host polling of mapped memory).

### C. Hybrid of A and B
Layers split as in A for prefill (pipeline parallelism across ubatches, ggml's multi-device scheduler), decode as B.

## Prefill

ggml-vulkan already supports multiple devices (layer split, `-sm layer`) and the scheduler can pipeline consecutive
ubatches across devices (`n_copies`): with two cards, prefill can approach 2x if both stages are balanced and card
2's experts are resident (prefill touches nearly all experts: card 2's must not cross PCIe 2.0 x4 - 8 GB at
1.6 GB/s = 5 s per ubatch, fatal). Our custom kernels (ji_fa2, EXL3 GEMMs, sparse prefill) are per-device and should
work on either card; the host-pinned expert upload path must be restricted to card 1.

## Plan when the card arrives

1. Measure: `lspci -vv` link of slot 2, host->device bandwidth on card 2, both cards under RADV, idle power.
2. Prefill first (graph path, layer split + pipeline parallel): lowest engine risk, biggest relative gain.
3. Decode: prototype B with a static hot-expert set on card 2 (from routing statistics), host-polled per-layer
   handoff; measure the per-layer round trip; fall back to A if the handoff costs more than the misses it saves.
4. Re-evaluate MTP / speculation on the new cost structure (2-token verify scaling).
