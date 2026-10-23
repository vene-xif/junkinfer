# Expert memory hierarchy (pinned host experts, VRAM cache, elastic slots)

Status: **on by default**. History: engineering log 2026-10-16 (hybrid cache, pinned experts, elastic slots),
2026-10-18 (compute-buffer slots, one-shot tokens), 2026-10-21 (TP2 expert balance), 2026-10-23 (swaps 16, async
fills, KV-clear re-seed, guard fix, CB re-bind).

## Why

Qwen3.6-35B-A3B at EXL3 4.0 bpw has ~17 GB of routed experts (40 layers × 256 experts, 8 routed + 1 shared per
token). Each RX 580 has 8 GB. Decode needs ~360 expert reads per token, so where an expert lives decides the
speed:

| tier | where | read cost |
|---|---|---|
| bank slots | dedicated VRAM buffer | VRAM bandwidth (~200 GB/s) |
| elastic KV-tail slots | unused top cells of the attention K/V tensors | VRAM |
| elastic compute-buffer slots | idle ggml compute buffer (mostly the KQ mask) while JDE decodes | VRAM |
| miss | pinned host RAM (GTT, `--load-mode none` → Vulkan_Host) | **zero-copy over PCIe** (x16: ~14 GB/s; x4 2.0: ~1.6 GB/s) |

## Host side: pinned experts

The boot sets `ttm.pages_limit=6291456 amdgpu.gttsize=24576`, giving 24 GB of GTT. With `--load-mode none`, CPU
experts load into pinned Vulkan_Host memory (17.5 GB). Prefill uploads become async DMA instead of staged copies:
prefill +35–50 % (6.7k: 175 → 260 t/s, 2026-10-16).

EXL3 misses run **zero-copy on the GPU** (`JI_X3_ZC`): the MoE mat-vec reads the host page directly. The CPU never
computes experts. That made **one-shot tokens** possible: the whole token is one `vkQueueSubmit`, and routing
records (`ro_host`) feed the cache statistics after the fence. That replaced 81 per-layer host handshakes.

gfx8 constraint: host → GPU data must cross a submission boundary. The L2 caches system memory and isn't
invalidated mid-submission; a host-set VkEvent hung the GPU.

## VRAM cache

- Policy: persistent decayed-frequency score (decay 0.98) plus admission hysteresis, already a recency-frequency
  hybrid. Sweeps of decay, LRFU and eviction guards moved the hit rate < 1 %; it's bound by capacity.
- `JI_CACHE_SWAPS` (default 16, was 2): experts swapped in per token. Fills are `vkCmdCopyBuffer` from the pinned
  page to the slot. Short decode +5–10 %; 245k after a restore 24.1 → 30.6 t/s on the first reply.
- Slot map: a host shadow (reading the BAR slot map cost 7 ms per token in uncached PCIe reads).
- **Async fills** (`JI_ASYNC_FILL`): copies on their own queue. Keys are marked −2 while in flight and published
  when the fence signals. Neutral to +3 %.
- Budget: device-local free VRAM after the weights, the KV cache, sparse metadata, KVarN rings and a margin
  (`JI_TP_MARGIN_MIB`, 1900 in production for the helper's prefill scratch). `JI_CACHE_MIB` overrides it.

### Elastic slots (the big win)

1. **KV-tail slots**: a 262k q8_0 KV cache is 2.85 GB and mostly empty. JDE carves ~1780 slots from the top cells
   of the 20 attention K/V tensors. `jde_kv_guard` runs before every ubatch and evicts slots the context may reach:
   limit = used cells + n_tokens + 4096. Warm decode at 262k context: 42 → 46.5 t/s. With KVarN, the boundary
   scales by 272/208.
2. **Compute-buffer slots**: the ggml compute buffer (2.75 GiB at ub 4096) is idle during JDE decode.
   `GGML_VK_SUBALLOCATION_BLOCK_SIZE=3.75 GiB` makes it one chunk, giving ~1834 slots. They are evicted before any
   graph-path write (process_ubatch, graph_compute, memory updates that run a graph). After a reallocation they are
   re-bound and refilled, not retired. Any prefill with MTP rows wanted evicts them all (a stray write, not
   identified, corrupted them after deep restores).
3. KV clear (`llama_kv_cache::clear(data)` zeroes the buffers) bumps `g_kv_clear_gen`. The guard then drops and
   re-seeds KV-tail slots (`jde_note_kv_clear`).

Verification: `JI_EL_VERIFY` checks cached tensors against the source (0 mismatches required).

## TP2 (two GPUs)

Expert ownership is 0 / 1 / 2. "2" means the hottest 25 % of routing traffic (566 experts), present on both GPUs
and pinned in the primary's cache. `tp_split` gives each shared selection to the GPU with fewer selections so far,
using the same deterministic rule on both. After-MoE waits: 2.78 → 1.03 ms per token. The helper (x4 link) holds
its own hot-expert cache.

## Measured

- Hit rate: ~96–97 % short context; ~79 % at 247k (single card, before KVarN). Misses cost ~68 MB per token
  (~45 experts) over PCIe.
- History of short decode, single card: 21 t/s (CPU experts) → 39 (VRAM cache + replay) → 46.5 (elastic KV-tail) →
  42–44 (EXL3; stripe-major mat-vecs).
- With TP2: 58–59 t/s at full power; 43–49 t/s at 100 W.

## Rejected

- Miss substitution (swap a low-weight non-resident expert for a resident one): KL 0.002 at the mildest setting
  (PPL hides it).
- Dynamic top-k.
- Weight prefetcher on a second queue for prefill (no gain; prefill is compute-bound).
- CPU sharing of misses.
- Misses-first row order.
- SDMA transfer queue: RADV on gfx8 (Mesa 26.0) has none.

## Porting notes

- On a 16 GB card (6900 XT) about half the experts fit. The tiers stay; the KV-tail and compute-buffer tricks
  remain worth it because they cost nothing.
- Zero-copy misses need a host-visible pinned mapping and a fence boundary per token on gfx8. RDNA2 has the same L2
  rule unless memory is mapped uncached.
- The cache-update host work runs between tokens: pin the CPU governor to performance (schedutil cost 20–25 %
  decode after large prefills).
