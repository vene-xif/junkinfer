# Tensor-parallel decode and prefill on two GPUs (JI_TP=1/2)

Status: **on in production (JI_TP=2 via serve_prod.sh)**. Engine commits: efad9ea7e (JI_TP=1, cross-GPU sync), c4f8da270 (JI_TP=2 full), 42fcac098..9e4073234 (attention split, state sync), bcc21bd57 (expert balance), be2f7bf3d (MTP), 2c61614d3 (TP teardown safety). History: engineering log 2026-10-20 .. 2026-10-23.

## Why

Two RX 580 8GB GPUs on AUX1: card 1 (PCIe 3.0 x16, ~11 GB/s host↔device) and card 2 (PCIe 2.0 x4, ~1.3 GB/s). Single-card decode is ALU-bound at short context (~3G weights/token, ~200 G weights/s per Polaris). Full tensor-parallel (JI_TP=2: both GPUs split every layer) parallelizes the latency-critical path and feeds two GCN ALUs concurrently.

## How it works

### Work split

- **Attention**: kv-head split (since 42fcac098): the primary attends with kv head 0 over the llama.cpp KV cache, the helper with kv head 1 over its own q8_0 mirror of that head (`tp_kvm`); before the split the helper contributed a zero partial.
- **Gated DeltaNet**: key heads [8r, 8r+8) (r = rank 0/1) with their value heads; ssm_out over its K quarters; shared expert half rows [256r, +256); down projections K-half.
- **Routed experts**: ownership map 0/1/2 (2 = both GPUs: the hottest 25% of routing, 566 experts at 70% traffic share, pinned in primary's cache).
- **Per-token balance**: tp_split gives each shared-expert selection to the side with fewer selections so far; after-MoE waits: primary 2.78 → 1.03 ms/token.
- **Dense weights**: split by rank as above; the helper holds a 1 GB copy of the dense weights it needs plus a 63 MB recurrent-state mirror.

### Cross-GPU sync

- **Exchange memory**: one small pinned host allocation (`aligned_alloc`, owned by the primary) imported into both devices (VK_EXT_external_memory_host); each writer owns its cache lines; visible across dispatch boundaries. (Separate from the 24 GB GTT limit the experts need.)
- **Partial exchange**: after mixer and after MoE (~2 × 2048 f16 bytes per token through host), marked with `vkCmdWriteBufferMarkerAMD` (13.6 µs per two-GPU sync with 8 KB spread).
- **Relay thread** (`jde_tp_relay`): polls host markers every layer, submits helper's pre-recorded token program, waits for both marks, sets graphics-queue VkEvents via `vkSetEvent` on both GPUs.
- **Why graphics queues**: CP waits on host-set VkEvents only work on graphics queues here - a compute queue never sees the host write. The primary moves its TP submissions to ggml's graphics-family queue; the helper gets its own device and graphics queue. (Separately, gfx8's L2 is not invalidated mid-submission, so host -> GPU data must cross a sync point.)
- **Watchdog** (`JI_TP_TIMEOUT_MS`, default 20 s): a lost sync (marker stuck) releases both GPUs' events and aborts; the supervisor restarts the server. First decode after deep restore can exceed 2 s; default was 2 s (switched TP off for the server's lifetime).
- **Teardown safety** (`jde_tp_release_all`): every TP event on both GPUs set before process termination, preventing hung rings (on gfx8, failed reset → box freeze; happened twice 2026-10-23).

### Multi-token steps

TP2 supports 1–4 token steps (prompt tails, MTP verify windows). Program count R = nt; partials [nt][n_embd]; one command buffer per token count on both cards. Outputs identical to single-card.

### State movement

After a graph ubatch or state save, helper's recurrent state copied back (~300 ms); before first TP step, copied in (~80 ms). TP2 state split kept after a context checkpoint (partial state read, `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`); first decode after a long prompt decodes 54 → 23 ms/token (~1.5 s slow phase, likely power limiter under 100 W cap).

### TP2 prefill (JDE, JI_JPF=1)

Reuses TP2 decode layout (layer/rank split, exchange queue for all-reduces). Two micro-batches: f16 partials (~20 ms each through x4 link, must overlap). Rank 0 streams cold experts from host; rank 1's experts resident. BK16 on GEMM variants; half-width tiles for ≤32-row remainders.

## Measured

| config | short | 100k | 245k | prefill 80k |
|---|---|---|---|---|
| single card (full power, before the 100 W cap) | 46.3 | 33.8 | 30.0 | 347.8 |
| JI_TP=1 (expert-server, 262k) | 49.9 | 41.0 | 37.9 | — |
| **JI_TP=2 (full TP, 262k)** | **58.0** | **43.8** | **40.0** | **— (ggml)** |
| TP2 after attn split (2026-10-21 later) | 58.8 | 45.2 | 41.1 | — |
| TP2 + expert balance (59.1 short) | 59.1 | — | — | — |
| TP2 at the 100 W cap, CPU governor performance (2026-10-23, 600-token replies) | 43.2-49.6 | — | — | — |
| TP2 prefill (JDE, 262k, restored + 7.5k tail) | — | — | — | 213 (was 128 ggml) |
| TP2 prefill at 100 W (2026-10-22 later) | — | — | — | 229 |

Decode KL (130k): single 0.00036 = TP 0.00035 (summation order, not bit-exact). Prefill KL (1025-token tail, 130k): graph 0.00050 / JDE TP 0.00050; 261k graph 0.00045 / JDE TP 0.00065 (f16 exchange partials).

## Rejected / measured no-gain

- JI_TP_DIRECT=1 (kernels write partials straight to exchange; host writes cost more: 15.1 → 16.0 ms/token GPU).
- Per-layer dynamic balance with shared experts (all ownership scenarios 0.3–1.0: ~50 t/s short context).
- (Not rejected, for the record: the MTP draft head on the helper was slower than no speculation on 2026-10-21, then on by default from 2026-10-23 after the overhead cuts, +14-23 % - see mtp-speculative-decoding.md.)
- Grouped expert decode (LDS occupancy, duplicate accumulators cost 16–30 %).

## Operational hazards

- **Event-wait freeze** (gfx8): process killed mid-TP step leaves GPUs waiting on events forever; ring hung, reset fails ("asic atom init"), box freezes, needs power cycle. **Fix** (2026-10-23): all TP events set on process termination (before abort, at teardown, from termination/crash signals).
- **Helper falls off PCIe bus**: card 2 under prefill load at 145 W (one daisy-chained PSU cable) → "device lost from bus". **Mitigation**: 100 W power cap per card (`31-amdgpu-powercap.rules`).
- **Two llama-* processes**: both running TP2 concurrently OOM the box (pinned experts per process) → reboot.

## Knobs

`JI_TP` (0=off, 1=MoE-only helper, 2=full TP; serve.sh default 0, production 2), `JI_TP_FRAC` (expert share 0.3–1.0, default 0.7 at depth), `JI_TP_MARGIN_MIB` (helper prefill scratch, default 1900 for decode; 1500 at production 100 W), `JI_TP_TIMEOUT_MS` (watchdog, default 20000), `JI_TP_NT` (max tokens per step, default JDE_NT_MAX=4), `JI_TP_DIRECT` (write partials to exchange directly, off), `JI_TP_LOST_CONTINUE` (fallback to single GPU on lost sync, off; default aborts for clean restart).

## Porting notes (RDNA2 / other engines)

- gfx8-specific: L2 not invalidated mid-submission (compute queues broke TP). RDNA2 may be different; test compute queues with host-set VkEvents.
- Sync 13.6 µs per point already includes per-layer imbalance; schedule both GPUs' layers with equal cost (routed experts scale per-token; dense per-layer).
- State copies between GPUs and batched through staging buffers are the next bottleneck (280 ms → 38–50 ms per request after split/batching). On a 16 GB card that can hold more experts, recurrent-state mirror sizes shrink.
- Helper expert residency (zero-copy misses over x4 link too expensive) fits the asymmetry; symmetric dual would need either a fast peer-to-peer path or an expert placement strategy orthogonal to the x16/x4 split.
